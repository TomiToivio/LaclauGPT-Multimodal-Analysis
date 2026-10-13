"""Bounded, provenance-labelled Step 4 evidence selection (no input mutations).

Since issue #236 this module is a **thin adapter over the shared contract** in
:mod:`ep24_context_budget`, rather than a fourth budget policy of its own. Step 4's labels,
environment variables and defaults are unchanged; what the shared contract adds is the part
#236's goal text asks for and this stage did not previously have:

* **context reserved for the output** -- the old packet spent the entire budget on input, so a
  full budget left the model no room to answer, and that failure presented as a model problem;
* **per-field character and token accounting**, so an over-budget prompt is visible in the log
  before it is sent;
* **truncation reported per field** instead of one global marker.

The public entry point still returns the prompt body as a ``str``; :func:`build_packet_report`
returns the full :class:`~ep24_context_budget.Packet` for callers that want to log the accounting.
"""
import os

from ep24_context_budget import (
    MIN_TOTAL_CHARS,
    Channel,
    Packet,
)
from ep24_context_budget import (
    build_packet as _build_packet,
)

#: Per-channel character caps. Independently capped so one huge stream cannot starve the rest.
LIMITS = {"video": 8000, "asr_original": 5000, "asr_english": 3500,
          "frame": 3500, "ocr": 1500, "metadata": 1200,
          "memory": 500, "rag": 500}

#: Priority order. Higher wins the budget; the numbering mirrors the list order below, and a
#: test pins that the whole-video narrative still precedes the ASR streams.
PRIORITY = {"video": 60, "asr_original": 50, "asr_english": 40, "frame": 30, "ocr": 20,
            "metadata": 10, "memory": 5, "rag": 4}

META_KEYS = ("author_username", "username", "video_id", "country", "language",
             "date", "published", "caption", "description", "url", "platform",
             "entities", "themes", "research_notes")

#: Provenance labels. These are the contract: they tell the model what the text is and, just as
#: importantly, what it is not.
LABEL_VIDEO = "Whole-video narrative [video analysis; model-generated, verify against sources]"
LABEL_ASR_ORIGINAL = "Speech [ASR original language; transcription may be inaccurate]"
LABEL_ASR_ENGLISH = "Speech [English translation; secondary to original]"
LABEL_FRAME = "Visuals [sampled keyframe analysis; not a complete video]"
LABEL_OCR = "Visible text [OCR; may be inaccurate]"
LABEL_METADATA = "Source metadata [context; not proof of on-screen content]"
LABEL_MEMORY = "Researcher normalization [NOT video evidence]"
LABEL_RAG = "Previous corpus analyses [NOT video evidence]"

TOTAL_ENV = "LACLAUGPT_SUMMARY_EVIDENCE_CHARS"
RESERVE_OUTPUT_ENV = "LACLAUGPT_SUMMARY_RESERVE_OUTPUT_CHARS"
RESERVE_SCHEMA_ENV = "LACLAUGPT_SUMMARY_RESERVE_SCHEMA_CHARS"
CAP_ENV_FMT = "LACLAUGPT_SUMMARY_CAP_{channel}"

#: #236 asks for an overall budget of roughly 16-24k characters *with* room reserved for the
#: answer. These are the defaults for that shape.
DEFAULT_TOTAL = 20000
DEFAULT_RESERVE_OUTPUT = 2000
DEFAULT_RESERVE_SCHEMA = 800


def clean(value):
    """Remove missing values and refuse spurious Pandas nan/None strings."""
    if value is None:
        return ""
    raw = str(value).strip()
    return "" if raw.lower() in ("nan", "none", "nat", "<na>") else raw


def choose(row, *keys):
    for key in keys:
        value = clean(row.get(key, ""))
        if value:
            return value
    return ""


def cap(value, limit):
    """Kept for callers that cap a single value; the packet builder no longer uses it."""
    value = clean(value)
    if len(value) <= limit:
        return value
    marker = "\n[TRUNCATED: source evidence exceeds channel budget]"
    return value[:max(0, limit - len(marker))] + marker


def _env_int(name, default):
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return int(raw)  # a non-integer is the caller's error and should surface as one


def _cap_for(channel, fallback):
    return _env_int(CAP_ENV_FMT.format(channel=channel.upper()), fallback)


def _retrieval_enabled() -> bool:
    return os.getenv("LACLAUGPT_SUMMARY_INCLUDE_RETRIEVAL", "0") == "1"


def _channels(row, memory, rag):
    channels = [
        Channel("video", LABEL_VIDEO,
                choose(row, "vllm_video_analysis", "vllm_video_markdown_analysis",
                       "vllm_structured_output"),
                PRIORITY["video"], _cap_for("video", LIMITS["video"])),
        Channel("asr_original", LABEL_ASR_ORIGINAL,
                choose(row, "asr_transcript", "whisper_transcript", "whisperResult"),
                PRIORITY["asr_original"], _cap_for("asr_original", LIMITS["asr_original"])),
        Channel("asr_english", LABEL_ASR_ENGLISH,
                choose(row, "asr_translated", "whisper_translated"),
                PRIORITY["asr_english"], _cap_for("asr_english", LIMITS["asr_english"])),
        Channel("frame", LABEL_FRAME, choose(row, "frame_analysis_1"),
                PRIORITY["frame"], _cap_for("frame", LIMITS["frame"])),
        Channel("ocr", LABEL_OCR, choose(row, "ocr_1"),
                PRIORITY["ocr"], _cap_for("ocr", LIMITS["ocr"])),
        Channel("metadata", LABEL_METADATA,
                "\n".join(f"{key}: {clean(row.get(key))}" for key in META_KEYS
                          if clean(row.get(key))),
                PRIORITY["metadata"], _cap_for("metadata", LIMITS["metadata"])),
    ]
    if _retrieval_enabled():
        # Context from other sources is never evidence. It is opt-in and labelled as such.
        channels.append(Channel("memory", LABEL_MEMORY, memory,
                                PRIORITY["memory"], _cap_for("memory", LIMITS["memory"])))
        channels.append(Channel("rag", LABEL_RAG, rag,
                                PRIORITY["rag"], _cap_for("rag", LIMITS["rag"])))
    return channels


def build_packet_report(row, *, memory="", rag="", total=None):
    """Select source evidence and return the full packet, including its accounting.

    The input mapping is never changed. Raises :class:`~ep24_context_budget.BudgetError`
    (a ``ValueError``) when the requested budget cannot produce a valid prompt.
    """
    budget = _env_int(TOTAL_ENV, DEFAULT_TOTAL) if total is None else int(total)
    return _build_packet(
        _channels(row, memory, rag),
        total=budget,
        reserve_output=_env_int(RESERVE_OUTPUT_ENV, DEFAULT_RESERVE_OUTPUT),
        reserve_schema=_env_int(RESERVE_SCHEMA_ENV, DEFAULT_RESERVE_SCHEMA),
    )


def build_packet(row, *, memory="", rag="", total=None) -> str:
    """The prompt body. Unchanged signature and defaults; bounded, labelled, never mutating."""
    return build_packet_report(row, memory=memory, rag=rag, total=total).text


__all__ = [
    "DEFAULT_RESERVE_OUTPUT", "DEFAULT_RESERVE_SCHEMA", "DEFAULT_TOTAL", "LIMITS", "META_KEYS",
    "MIN_TOTAL_CHARS", "PRIORITY", "Packet", "build_packet", "build_packet_report", "cap",
    "choose", "clean",
]
