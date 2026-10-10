"""Bounded, provenance-labelled Step 4 evidence selection (no input mutations)."""
import os
import re

LIMITS = {"video": 8000, "asr_original": 5000, "asr_english": 3500,
          "frame": 3500, "ocr": 1500, "metadata": 1200,
          "memory": 500, "rag": 500}
META_KEYS = ("author_username", "username", "video_id", "country", "language",
             "date", "published", "caption", "description", "url", "platform",
             "entities", "themes", "research_notes")


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
    value = clean(value)
    if len(value) <= limit:
        return value
    # Make evidence truncation explicit; never silently imply completeness.
    marker = "\n[TRUNCATED: source evidence exceeds channel budget]"
    return value[:max(0, limit - len(marker))] + marker


def build_packet(row, *, memory="", rag="", total=None):
    """Select source evidence by priority under per-stream and global character budgets.

    The input mapping is never changed. Context from other sources is not evidence.
    """
    total = int(total if total is not None else os.getenv("LACLAUGPT_SUMMARY_EVIDENCE_CHARS", "20000"))
    if total < 2000:
        raise ValueError("Step 4 evidence budget must be at least 2000 characters")
    streams = [
        ("Whole-video narrative [video analysis; model-generated, verify against sources]",
         choose(row, "vllm_video_analysis", "vllm_video_markdown_analysis", "vllm_structured_output"), "video"),
        ("Speech [ASR original language; transcription may be inaccurate]",
         choose(row, "asr_transcript", "whisper_transcript", "whisperResult"), "asr_original"),
        ("Speech [English translation; secondary to original]",
         choose(row, "asr_translated", "whisper_translated"), "asr_english"),
        ("Visuals [sampled keyframe analysis; not a complete video]",
         choose(row, "frame_analysis_1"), "frame"),
        ("Visible text [OCR; may be inaccurate]",
         choose(row, "ocr_1"), "ocr"),
        ("Source metadata [context; not proof of on-screen content]",
         "\n".join(f"{key}: {clean(row.get(key))}" for key in META_KEYS if clean(row.get(key))), "metadata"),
    ]
    if os.getenv("LACLAUGPT_SUMMARY_INCLUDE_RETRIEVAL", "0") == "1":
        streams.extend([
            ("Researcher normalization [NOT video evidence]", memory, "memory"),
            ("Previous corpus analyses [NOT video evidence]", rag, "rag"),
        ])
    sections = []
    remaining = total
    for label, value, channel in streams:
        limit = max(0, min(int(os.getenv(f"LACLAUGPT_SUMMARY_CAP_{channel.upper()}", str(LIMITS[channel]))), remaining - len(label) - 8))
        if not clean(value) or limit <= 0:
            continue
        block = f"### {label}\n{cap(value, limit)}"
        sections.append(block)
        remaining -= len(block) + 2
    return "\n\n".join(sections) if sections else "No usable source evidence supplied. Report this limitation; do not invent a video narrative."
