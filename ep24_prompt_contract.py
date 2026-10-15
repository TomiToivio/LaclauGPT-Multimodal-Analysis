#!/usr/bin/env python3
"""Shared model-visible prompt/context contract for EP24 Steps 4-8 (issue #241).

This is the **one** builder Steps 4-8 use to turn a cumulative record into the model-visible
packet. It answers, in one place, the four questions each stage was otherwise answering
differently:

* how large the model-visible packet may be, and how much room is held back for the reply and
  the JSON schema (:class:`StageBudget`);
* which fields are capped independently, and what happens when the whole packet overflows
  (:func:`build_packet`);
* which fields are *source evidence* and which are continuity context that must never be
  presented as evidence (``EVIDENCE`` / ``CONTEXT`` and the per-field ``label``);
* what exactly the cache key covers, so a prompt/model/option change cannot reuse a stale answer
  (:func:`effective_prompt_fingerprint`).

Consolidation note (issue #241).  Four overlapping modules answered this question before:
``ep24_prompt_contract`` (this module), ``ep24_context_budget``, ``ep24_evidence_packets`` and
``ep24_summary_evidence``.  Different stages imported different ones, so a stage's effective
budget depended on which builder it happened to import.  They are now one module, and the
per-stage budgets are the ones the runbook documents
(``docs/EP24_PROMPT_EVIDENCE_RUNBOOK.md``): Step 4 20,000 / Step 6 28,000 / Step 7 20,000 /
Step 8 12,000 model-visible characters.  ``docs/EP24_PROMPT_CONTRACT_CONSOLIDATION.md`` records
each stage's effective budget before and after, field by field.

Budget accounting, stated once because it is easy to get wrong:

* ``model_visible_chars`` is the cap on the packet text the model receives.  It is the stage's
  contract number and it is what an operator sets with the stage's environment variable.
* ``output_reserve_chars`` / ``schema_reserve_chars`` state how much of the stage's context window
  is held back for the reply and the JSON schema.  They are **headroom documentation**: the
  runbook's per-stage numbers are the documented *model-visible* budgets, so a reserve does not
  subtract from them again.
* Each field is capped independently *and* by what is left of the budget, so the packet never
  exceeds ``model_visible_chars`` and one large field cannot crowd out primary evidence.

Design rules carried from the issue:

* inputs are **provenance-labelled**: every field's label says where the text came from and what
  it is not;
* source information is never *deleted* -- truncation is marked in-band, reported per field, and
  the caller's row is never mutated;
* memory / RAG / other stages' outputs are *context*, and are either labelled as such or left out
  of the packet entirely: they are never evidence for a new political claim, statement or tie;
* no stage's own earlier output is fed back to it.

Readable research logic, not a framework: the per-stage field tables below are plain tuples in the
order the stage's evidence priority, and the builder is one function.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

# ---------------------------------------------------------------------------------------------
# The vocabulary of the contract
# ---------------------------------------------------------------------------------------------

#: The two kinds a field may declare. The distinction is the point of the contract: context
#: (memory, RAG, earlier analyses) may aid continuity and normalization, but it is not evidence
#: and must never be labelled or treated as such.
EVIDENCE = "evidence"
CONTEXT = "context"

#: Appended in-band so the model cannot read a clipped field as a complete one. Deliberately short:
#: every character of the marker is a character of source text the model does not see.
TRUNCATION_MARKER = " [TRUNCATED]"

#: What a stage sends when it has no usable evidence. The limitation is stated rather than left
#: blank, because a blank packet invites invention.
NO_EVIDENCE = (
    "No usable source evidence supplied. Report this limitation; do not invent a narrative."
)

#: Characters per token for the estimate in the run log. Deliberately crude and stated: it is a
#: *budget signal* used to compare runs, not a tokenizer, and the contract never promises an exact
#: count (issue #241 asks for "token estimates").
CHARS_PER_TOKEN = 4

#: The spellings pandas invents for a missing cell. Step 4 keeps an empty JSON container (``[]``,
#: ``{}``) as information, because its metadata block renders each column as ``name: value``.
SUMMARY_MISSING_VALUES = ("nan", "none", "nat", "<na>")

#: Steps 6-8 read JSON columns, where an empty container is also "no value". This list is exactly
#: the one their previous builder used; it is kept identical so the change is not silent.
STAGE678_MISSING_VALUES = ("nan", "none", "[]", "{}")

REASON_EMPTY = "empty field"
REASON_NO_ROOM = "no budget left for this field"
REASON_TRUNCATED = "field exceeded its own cap or the remaining budget"


class BudgetError(ValueError):
    """The requested stage budget cannot produce a valid prompt."""


def clean(value: object) -> str:
    """Normalise a cell: missing values and the string spellings pandas invents become empty."""
    if value is None:
        return ""
    raw = str(value).strip()
    return "" if raw.casefold() in SUMMARY_MISSING_VALUES else raw


def estimate_tokens(text: str) -> int:
    """Crude token estimate for the budget signal. Empty text is zero tokens, not one."""
    return tokens_for_chars(len(text))


def tokens_for_chars(chars: int) -> int:
    """The same estimate applied to a character count, for a field reported by length."""
    return (chars + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN if chars > 0 else 0


def _env_int(name: str, default: int) -> int:
    if not name:
        return int(default)
    raw = os.getenv(name)
    if raw in (None, ""):
        return int(default)
    try:
        return int(raw)
    except ValueError as exc:
        raise BudgetError(f"{name} must be an integer, got {raw!r}") from exc


# ---------------------------------------------------------------------------------------------
# Per-stage budgets
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StageBudget:
    """A stage's model-visible budget, its reply/schema reserve, and its operator override.

    ``budget_env`` names the environment variable an operator sets to retune the stage without a
    code change. The reserve env vars exist only where the repository already had them (Step 4);
    where a stage has no number, the reserve stays 0 rather than being invented.
    """

    name: str
    model_visible_chars: int
    budget_env: str
    min_model_visible_chars: int = 128
    output_reserve_chars: int = 0
    output_reserve_env: str = ""
    schema_reserve_chars: int = 0
    schema_reserve_env: str = ""

    @property
    def declared_total_chars(self) -> int:
        """Packet cap plus the reply/schema headroom the stage's context window must also hold."""
        return (
            self.model_visible_chars + self.output_reserve_chars + self.schema_reserve_chars
        )


#: The documented contract values, from the runbook's per-stage table. The minimum each stage will
#: accept is the floor its own previous code enforced: Step 4 refused below 2,000 characters
#: (``ep24_context_budget.MIN_TOTAL_CHARS``), Steps 6-7 below 128 (``ep24_evidence_packets``), and
#: Step 8 floored an operator override at 3,000 (``step_8_roihu_social_network_analysis``).
STAGE_BUDGETS: dict[str, StageBudget] = {
    "step4": StageBudget(
        name="step4",
        model_visible_chars=20000,
        budget_env="LACLAUGPT_SUMMARY_EVIDENCE_CHARS",
        min_model_visible_chars=2000,
        output_reserve_chars=2000,
        output_reserve_env="LACLAUGPT_SUMMARY_RESERVE_OUTPUT_CHARS",
        schema_reserve_chars=800,
        schema_reserve_env="LACLAUGPT_SUMMARY_RESERVE_SCHEMA_CHARS",
    ),
    "step6": StageBudget(
        name="step6",
        model_visible_chars=28000,
        budget_env="LACLAUGPT_STEP6_MAX_CONTEXT_CHARS",
    ),
    "step7": StageBudget(
        name="step7",
        model_visible_chars=20000,
        budget_env="LACLAUGPT_STEP7_MAX_CONTEXT_CHARS",
    ),
    "step8": StageBudget(
        name="step8",
        model_visible_chars=12000,
        budget_env="LACLAUGPT_STEP8_MAX_EVIDENCE_CHARS",
        min_model_visible_chars=3000,
    ),
}


def normalise_stage(stage: int | str) -> str:
    """Accept ``6`` or ``"step6"`` and return the canonical ``"step6"`` key, loudly for junk."""
    key = f"step{stage}" if isinstance(stage, int) else str(stage).strip().casefold()
    if key not in STAGE_BUDGETS:
        raise KeyError(f"unknown EP24 stage {stage!r}; known: {sorted(STAGE_BUDGETS)}")
    return key


def stage_budget(stage: int | str, *, model_visible_chars: int | None = None) -> StageBudget:
    """Resolve a stage's budget: an explicit argument, then the environment, then the contract."""
    base = STAGE_BUDGETS[normalise_stage(stage)]
    visible = (
        int(model_visible_chars)
        if model_visible_chars is not None
        else _env_int(base.budget_env, base.model_visible_chars)
    )
    output_reserve = _env_int(base.output_reserve_env, base.output_reserve_chars)
    schema_reserve = _env_int(base.schema_reserve_env, base.schema_reserve_chars)
    if visible < base.min_model_visible_chars:
        raise BudgetError(
            f"EP24 {base.name} model-visible budget must be at least "
            f"{base.min_model_visible_chars} characters, got {visible} "
            f"(set {base.budget_env})"
        )
    if output_reserve < 0 or schema_reserve < 0:
        raise BudgetError("reserved output/schema context cannot be negative")
    if output_reserve + schema_reserve >= visible:
        raise BudgetError(
            f"reserved context ({output_reserve} output + {schema_reserve} schema) does not fit "
            f"inside the {visible}-character model-visible budget of {base.name}; raise "
            f"{base.budget_env} or lower the reserves"
        )
    return StageBudget(
        name=base.name,
        model_visible_chars=visible,
        budget_env=base.budget_env,
        min_model_visible_chars=base.min_model_visible_chars,
        output_reserve_chars=output_reserve,
        output_reserve_env=base.output_reserve_env,
        schema_reserve_chars=schema_reserve,
        schema_reserve_env=base.schema_reserve_env,
    )


# ---------------------------------------------------------------------------------------------
# The packet
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Stream:
    """One provenance-labelled input the model may see.

    ``name`` is the cumulative-CSV column (or the stage's name for a derived stream) and appears in
    the packet heading; ``label`` is the provenance statement next to it. ``kind`` decides whether
    the field counts as evidence. Field order is priority: the caller lists them most important
    first.
    """

    name: str
    label: str
    text: str = ""
    kind: str = EVIDENCE
    cap: int | None = None


@dataclass(frozen=True)
class FieldReport:
    """What the model actually saw for one field, so truncation is auditable, not implied."""

    name: str
    kind: str
    cap: int
    original_chars: int
    kept_chars: int
    omitted: bool
    reason: str = ""

    @property
    def truncated(self) -> bool:
        """True when part of a field that reached the model was clipped."""
        return not self.omitted and self.kept_chars < self.original_chars

    @property
    def est_tokens(self) -> int:
        """Token estimate for the characters of this field that reached the model."""
        return tokens_for_chars(self.kept_chars)


def _truncate(text: str, limit: int) -> str:
    """Clip ``text`` to at most ``limit`` characters, marking the clip in-band."""
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    keep = max(0, limit - len(TRUNCATION_MARKER))
    return text[:keep] + TRUNCATION_MARKER[: limit - keep]


@dataclass(frozen=True)
class Packet:
    """The rendered packet plus the accounting that makes it auditable."""

    stage: str
    text: str
    report: tuple[FieldReport, ...]
    model_visible_chars: int
    output_reserve_chars: int
    schema_reserve_chars: int

    @property
    def used(self) -> int:
        return len(self.text)

    @property
    def token_estimate(self) -> int:
        return estimate_tokens(self.text)

    @property
    def evidence_chars(self) -> int:
        return sum(r.kept_chars for r in self.report if r.kind == EVIDENCE)

    @property
    def context_chars(self) -> int:
        return sum(r.kept_chars for r in self.report if r.kind == CONTEXT)

    @property
    def truncated_fields(self) -> tuple[str, ...]:
        return tuple(r.name for r in self.report if r.truncated)

    @property
    def omitted_fields(self) -> tuple[str, ...]:
        return tuple(r.name for r in self.report if r.omitted)

    @property
    def any_truncation(self) -> bool:
        """Any field clipped, or left out for want of room. An empty field alone is not that."""
        return any(
            r.truncated or (r.omitted and r.reason != REASON_EMPTY) for r in self.report
        )

    def render_report(self) -> str:
        """One line per field: chars, tokens, truncation. For the stage's run log."""
        lines = [
            f"context budget: {self.model_visible_chars} chars "
            f"(reserved: {self.output_reserve_chars} output + {self.schema_reserve_chars} schema), "
            f"used {self.used}"
        ]
        for r in self.report:
            state = "omitted" if r.omitted else ("truncated" if r.truncated else "full")
            detail = f" ({r.reason})" if r.reason else ""
            lines.append(
                f"  {r.name}: {r.kept_chars}/{r.original_chars} chars, "
                f"~{r.est_tokens} tokens, {state}{detail}"
            )
        return "\n".join(lines)

    def log_line(self) -> str:
        """One line for the run log: model-visible chars, estimate, truncation by field."""
        return (
            f"stage={self.stage} chars={self.used}/{self.model_visible_chars} "
            f"tokens~{self.token_estimate} evidence={self.evidence_chars} "
            f"context={self.context_chars} truncated={list(self.truncated_fields)} "
            f"omitted={list(self.omitted_fields)}"
        )

    def summary(self) -> dict:
        """Machine-readable accounting for a stage log or a persisted column."""
        return {
            "model_visible_chars": self.model_visible_chars,
            "output_reserve_chars": self.output_reserve_chars,
            "schema_reserve_chars": self.schema_reserve_chars,
            "used": self.used,
            "est_tokens": self.token_estimate,
            "truncated": list(self.truncated_fields),
            "omitted": list(self.omitted_fields),
            "fields": [
                {
                    "name": r.name,
                    "kind": r.kind,
                    "cap": r.cap,
                    "characters_available": r.original_chars,
                    "characters_included": r.kept_chars,
                    "truncated": r.truncated,
                    "dropped": r.omitted,
                    "reason": r.reason,
                }
                for r in self.report
            ],
        }


def build_packet(
    stage: int | str,
    streams: Iterable[Stream],
    *,
    model_visible_chars: int | None = None,
    preamble: str = "",
    include_context: bool = True,
) -> Packet:
    """Lay out independently capped, provenance-labelled fields under one stage budget.

    Fields are filled in the order given -- evidence first, then context, each preserving the
    caller's order, so continuity context can never displace source evidence. A field that does not
    fit is reported per field rather than dropped silently; a field that partly fits carries an
    in-band truncation marker. The caller's data is never modified, and the packet text never
    exceeds the stage's model-visible budget.
    """
    key = normalise_stage(stage)
    budget = stage_budget(key, model_visible_chars=model_visible_chars)
    supplied = list(streams)
    for stream in supplied:
        if stream.kind not in (EVIDENCE, CONTEXT):
            raise ValueError(
                f"field {stream.name!r} kind must be {EVIDENCE!r} or {CONTEXT!r}, got "
                f"{stream.kind!r}"
            )
    ordered = [s for s in supplied if s.kind == EVIDENCE]
    if include_context:
        ordered += [s for s in supplied if s.kind == CONTEXT]

    opening = clean(preamble)[: budget.model_visible_chars]
    blocks: list[str] = [opening] if opening else []
    used = len(opening)
    reports: list[FieldReport] = []

    for stream in ordered:
        text = clean(stream.text)
        cap = budget.model_visible_chars if stream.cap is None else int(stream.cap)
        header = f"[{stream.name} | {stream.label}]"
        separator = 2 if blocks else 0
        room = budget.model_visible_chars - used - separator - len(header) - 1
        limit = min(cap, room)
        if not text or limit <= 0:
            reports.append(
                FieldReport(
                    stream.name, stream.kind, cap, len(text), 0, True,
                    REASON_EMPTY if not text else REASON_NO_ROOM,
                )
            )
            continue
        kept = _truncate(text, limit)
        used += separator + len(header) + 1 + len(kept)
        blocks.append(f"{header}\n{kept}")
        reports.append(
            FieldReport(
                stream.name, stream.kind, cap, len(text), len(kept), False,
                REASON_TRUNCATED if len(kept) < len(text) else "",
            )
        )

    text = "\n\n".join(blocks) if blocks else NO_EVIDENCE[: budget.model_visible_chars]
    return Packet(
        stage=key,
        text=text,
        report=tuple(reports),
        model_visible_chars=budget.model_visible_chars,
        output_reserve_chars=budget.output_reserve_chars,
        schema_reserve_chars=budget.schema_reserve_chars,
    )


# ---------------------------------------------------------------------------------------------
# Step 4: multimodal summary/fusion (roihu_summary.py)
# ---------------------------------------------------------------------------------------------

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

#: Step 4's metadata block, and the per-channel caps its packet has always used.
SUMMARY_META_KEYS = (
    "author_username", "username", "video_id", "country", "language", "date", "published",
    "caption", "description", "url", "platform", "entities", "themes", "research_notes",
)
SUMMARY_CAPS = {
    "video": 8000, "asr_original": 5000, "asr_english": 3500, "frame": 3500,
    "ocr": 1500, "metadata": 1200, "memory": 500, "rag": 500,
}


def choose(row, *keys: str) -> str:
    """The first non-empty value among ``keys``, or the empty string."""
    for key in keys:
        value = clean(row.get(key, ""))
        if value:
            return value
    return ""


def summary_cap(stream: str, default: int) -> int:
    """Step 4's per-stream cap, overridable per stream for one rerun."""
    return _env_int(f"LACLAUGPT_SUMMARY_CAP_{stream.upper()}", default)


def summary_retrieval_enabled() -> bool:
    """Memory/RAG context is opt-in for Step 4; it is never evidence, only continuity."""
    return os.getenv("LACLAUGPT_SUMMARY_INCLUDE_RETRIEVAL", "0") == "1"


def step4_streams(row, *, memory: str = "", rag: str = "") -> list[Stream]:
    """Step 4's fields in budget order: whole-video narrative, speech, visuals, then context."""
    return [
        Stream(
            "video", LABEL_VIDEO,
            choose(row, "vllm_video_analysis", "vllm_video_markdown_analysis",
                   "vllm_structured_output"),
            cap=summary_cap("video", SUMMARY_CAPS["video"]),
        ),
        Stream(
            "asr_original", LABEL_ASR_ORIGINAL,
            choose(row, "asr_transcript", "whisper_transcript", "whisperResult"),
            cap=summary_cap("asr_original", SUMMARY_CAPS["asr_original"]),
        ),
        Stream(
            "asr_english", LABEL_ASR_ENGLISH,
            choose(row, "asr_translated", "whisper_translated"),
            cap=summary_cap("asr_english", SUMMARY_CAPS["asr_english"]),
        ),
        Stream(
            "frame", LABEL_FRAME, choose(row, "frame_analysis_1"),
            cap=summary_cap("frame", SUMMARY_CAPS["frame"]),
        ),
        Stream(
            "ocr", LABEL_OCR, choose(row, "ocr_1"),
            cap=summary_cap("ocr", SUMMARY_CAPS["ocr"]),
        ),
        Stream(
            "metadata", LABEL_METADATA,
            "\n".join(
                f"{key}: {clean(row.get(key))}"
                for key in SUMMARY_META_KEYS
                if clean(row.get(key))
            ),
            cap=summary_cap("metadata", SUMMARY_CAPS["metadata"]),
        ),
        Stream(
            "memory", LABEL_MEMORY, memory, kind=CONTEXT,
            cap=summary_cap("memory", SUMMARY_CAPS["memory"]),
        ),
        Stream(
            "rag", LABEL_RAG, rag, kind=CONTEXT,
            cap=summary_cap("rag", SUMMARY_CAPS["rag"]),
        ),
    ]


# ---------------------------------------------------------------------------------------------
# Steps 6-8: Laclau/Palonen discourse, discourse network, social network
# ---------------------------------------------------------------------------------------------

#: A stage's packet opens by saying what it is and which parts are not direct speech. Kept exactly
#: as Steps 6-8 sent it before consolidation.
PREAMBLE_STEPS_6_8 = (
    "EP24 current-document evidence; prior model output is descriptive context, not direct speech."
)

ROLE_SOURCE = "original source/derived transcript; verify original modality"
ROLE_METADATA = "record metadata, not quotation evidence"
ROLE_PRIOR_MODEL = "prior model analysis, context only, not source quotation"
ROLE_CODEBOOK = "researcher normalization, not source quote"
ROLE_RESEARCHER = "researcher annotation or entity context, verify provenance"

#: Field order is deliberate: source speech and visible words outrank prior model output. The caps
#: are the ones Steps 6-8 were built against; the per-stage lists below are exactly those lists.
STEP67_SOURCE_FIELDS = (
    ("asr_transcript", 9000),
    ("videoDescription", 4500),
    ("whisper_transcript", 5000),
    ("asr_translation", 5500),
    ("asr_translated", 5500),
    ("whisper_translated", 3500),
    ("ocr_1", 3000),
    ("frame_analysis_1", 2500),
    ("vllm_video_analysis", 3500),
    ("summary_analysis", 4000),
)
STEP678_META_FIELDS = (
    ("video_id", 180), ("country", 90), ("platform", 100), ("url", 400),
    ("created_at", 120), ("date", 120),
)
STEP67_RESEARCHER_FIELDS = (("entities", 1800), ("themes", 1400), ("researcher_note", 1500))
STEP67_CODEBOOK_FIELDS = (("codebook_context_json", 1200),)
STEP7_DNA_FIELDS = (("laclau_structured_json", 1200),)
STEP8_SOURCE_FIELDS = (("asr_transcript", 5000), ("asr_translation", 1500), ("ocr_1", 1600))
STEP8_SNA_FIELDS = (
    ("dna_statements_json", 3800), ("ep24_entity_resolution_json", 1200),
    ("entities", 800), ("researcher_note", 600),
)

SOURCE_FIELD_NAMES = (
    "asr_transcript", "videoDescription", "whisper_transcript", "asr_translation",
    "asr_translated", "whisper_translated", "ocr_1",
)
PRIOR_MODEL_FIELD_NAMES = (
    "frame_analysis_1", "vllm_video_analysis", "summary_analysis", "laclau_structured_json",
    "dna_statements_json",
)
META_FIELD_NAMES = tuple(name for name, _cap in STEP678_META_FIELDS)


def role_for(name: str) -> str:
    """The provenance statement a Steps 6-8 field carries, decided once for all three stages."""
    if name == "codebook_context_json":
        return ROLE_CODEBOOK
    if name in SOURCE_FIELD_NAMES:
        return ROLE_SOURCE
    if name in META_FIELD_NAMES:
        return ROLE_METADATA
    if name in PRIOR_MODEL_FIELD_NAMES:
        return ROLE_PRIOR_MODEL
    return ROLE_RESEARCHER


def stage678_fields(stage: str) -> tuple[tuple[str, int], ...]:
    """The ordered (column, cap) list one of Steps 6-8 reads from the cumulative record."""
    if stage == "step6":
        return STEP67_SOURCE_FIELDS + STEP678_META_FIELDS + STEP67_RESEARCHER_FIELDS \
            + STEP67_CODEBOOK_FIELDS
    if stage == "step7":
        return STEP67_SOURCE_FIELDS + STEP678_META_FIELDS + STEP67_RESEARCHER_FIELDS \
            + STEP67_CODEBOOK_FIELDS + STEP7_DNA_FIELDS
    if stage == "step8":
        return STEP8_SOURCE_FIELDS + STEP8_SNA_FIELDS + STEP678_META_FIELDS
    raise KeyError(f"steps 6-8 field tables do not cover {stage!r}")


def _stage678_value(row, name: str) -> str:
    """A Steps 6-8 cell: pandas missing values and an empty JSON container both mean "no value"."""
    raw = row.get(name, "")
    text = "" if raw is None else str(raw).strip()
    return "" if text.casefold() in STAGE678_MISSING_VALUES else text


def stage678_streams(stage: str, row) -> list[Stream]:
    """Steps 6-8's fields in budget order, each carrying its provenance role."""
    return [
        Stream(name, role_for(name), _stage678_value(row, name), cap=cap)
        for name, cap in stage678_fields(stage)
    ]


# ---------------------------------------------------------------------------------------------
# The stage-facing entry point
# ---------------------------------------------------------------------------------------------


def stage_streams(
    stage: int | str, row, *, memory: str = "", rag: str = ""
) -> list[Stream]:
    """Materialise a stage's provenance-labelled streams from one cumulative record."""
    key = normalise_stage(stage)
    if key == "step4":
        return step4_streams(row, memory=memory, rag=rag)
    return stage678_streams(key, row)


def build_stage_packet(
    stage: int | str,
    row,
    *,
    memory: str = "",
    rag: str = "",
    max_chars: int | None = None,
) -> Packet:
    """Build one stage's bounded, provenance-labelled packet from a cumulative record.

    ``max_chars`` retunes the model-visible budget for this call only (a benchmark run);
    otherwise the stage's environment variable and then the contract default decide it. The input
    record is never modified.
    """
    key = normalise_stage(stage)
    return build_packet(
        key,
        stage_streams(key, row, memory=memory, rag=rag),
        model_visible_chars=max_chars,
        preamble="" if key == "step4" else PREAMBLE_STEPS_6_8,
        include_context=summary_retrieval_enabled() if key == "step4" else True,
    )


# ---------------------------------------------------------------------------------------------
# The cache key
# ---------------------------------------------------------------------------------------------


def effective_prompt_fingerprint(
    system_prompt: str,
    user_prompt: str,
    model: str,
    options: Mapping[str, object] | None = None,
) -> str:
    """Cache key over everything that changes the answer: prompt, model and generation options.

    A cache keyed on the source alone silently reuses answers produced by a different prompt,
    model or ``num_ctx``/``num_predict``/``temperature`` setting: the answer looks current and is
    not. Generation options were added to the Step 6/7 keys in #287 for exactly this reason.
    Options are serialised with sorted keys, so a reordered mapping is the same contract and a
    changed value is not.
    """
    payload = json.dumps(
        {
            "model": model,
            "system": system_prompt,
            "user": user_prompt,
            "options": dict(options or {}),
        },
        sort_keys=True,
        ensure_ascii=False,
        default=repr,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "BudgetError", "CHARS_PER_TOKEN", "CONTEXT", "EVIDENCE", "FieldReport", "NO_EVIDENCE",
    "PREAMBLE_STEPS_6_8", "Packet", "STAGE_BUDGETS", "StageBudget", "Stream", "TRUNCATION_MARKER",
    "build_packet", "build_stage_packet", "choose", "clean", "effective_prompt_fingerprint",
    "estimate_tokens", "normalise_stage", "role_for", "stage678_fields", "stage678_streams",
    "stage_budget", "stage_streams", "summary_cap", "summary_retrieval_enabled",
    "tokens_for_chars",
]
