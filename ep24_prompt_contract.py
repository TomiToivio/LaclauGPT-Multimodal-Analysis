"""Shared model-visible prompt/context contract for EP24 Steps 4-8 (issue #241).

One place to answer the four questions every stage was otherwise answering differently:

* how large the model-visible packet may be, and how much is held back for the reply;
* which streams are capped independently, and what happens when the whole packet overflows;
* which streams are *source evidence* and which are continuity context that must never be
  presented as evidence;
* what exactly the cache key covers, so a prompt/model/option change cannot reuse a stale answer.

The Step 4 shape landed in #244 (``ep24_summary_evidence.build_packet``) is the precedent: bounded
per-stream caps, an explicit truncation marker, and retrieval context off by default. This module
generalises that shape into a named, per-stage contract. It deliberately does **not** change any
stage's own evidence selection -- adopting it is a separate, per-stage change (#236/#238/#239/#240).
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

# The two kinds a stream may declare. The distinction is the whole point of the contract:
# context (memory, RAG, earlier analyses) may aid continuity and normalization, but it is not
# evidence and must never be labelled or treated as such (ep24_context.py says the same).
EVIDENCE = "evidence"
CONTEXT = "context"

TRUNCATION_MARKER = "\n[TRUNCATED: stream exceeded its budget]"
NO_EVIDENCE = (
    "No usable source evidence supplied. Report this limitation; do not invent a narrative."
)

# Characters per token is a coarse estimate, good enough to log and to compare runs. It is not a
# tokenizer and the contract never promises an exact count (issue #241 asks for "token estimates").
CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class StageBudget:
    """A stage's character budget, its reply reserve, and its default per-stream caps."""

    name: str
    total_chars: int
    output_reserve_chars: int
    stream_caps: Mapping[str, int] = field(default_factory=dict)

    @property
    def evidence_chars(self) -> int:
        """Chars the model may see, after holding back room for the reply/JSON schema."""
        return max(0, self.total_chars - self.output_reserve_chars)

    def cap_for(self, stream: str, total_streams: Iterable[str]) -> int:
        """A stream's cap, or an equal share of the budget when the stage names no cap."""
        if stream in self.stream_caps:
            return int(self.stream_caps[stream])
        names = list(total_streams)
        return max(500, self.evidence_chars // max(1, len(names)))


# Ranges as stated in #236 (16-24k), #239 (24-32k, 4k output), #240 (16-24k) and #238 (8-16k plus
# room for the JSON schema). Defaults sit inside each range; every value is overridable by env.
STAGE_BUDGETS: dict[str, StageBudget] = {
    "step4": StageBudget(
        "step4", 20000, 6000,
        {"video": 8000, "asr_original": 5000, "asr_english": 3500,
         "frame": 3500, "ocr": 1500, "metadata": 1200},
    ),
    "step6": StageBudget(
        "step6", 32000, 4000,
        {"speech": 12000, "signs": 6000, "fusion": 6000,
         "annotations": 3000, "codebook": 2000},
    ),
    "step7": StageBudget(
        "step7", 24000, 6000,
        {"utterances": 12000, "actors": 3000, "concepts": 3000, "codebook": 2000},
    ),
    "step8": StageBudget(
        "step8", 16000, 6000,
        {"source_context": 6000, "actor_mentions": 4000, "quotes": 3000, "dna": 2000},
    ),
}


def _env_override(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def stage_budget(stage: str) -> StageBudget:
    """The budget for a stage, with env overrides applied.

    ``LACLAUGPT_<STAGE>_EVIDENCE_CHARS`` overrides the total and
    ``LACLAUGPT_<STAGE>_OUTPUT_RESERVE`` the reserve, so a rerun can be tuned without code change.
    """
    key = stage.strip().casefold()
    if key not in STAGE_BUDGETS:
        raise KeyError(f"unknown EP24 stage {stage!r}; known: {sorted(STAGE_BUDGETS)}")
    base = STAGE_BUDGETS[key]
    upper = key.upper()
    return StageBudget(
        base.name,
        _env_override(f"LACLAUGPT_{upper}_EVIDENCE_CHARS", base.total_chars),
        _env_override(f"LACLAUGPT_{upper}_OUTPUT_RESERVE", base.output_reserve_chars),
        base.stream_caps,
    )


@dataclass(frozen=True)
class Stream:
    """One labelled input the model may see. ``kind`` decides whether it counts as evidence."""

    name: str
    text: str
    kind: str = EVIDENCE
    priority: int = 0
    cap: int | None = None


@dataclass(frozen=True)
class StreamReport:
    """What actually reached the model, so truncation is auditable rather than implied."""

    name: str
    kind: str
    cap: int
    original_chars: int
    kept_chars: int
    dropped: bool

    @property
    def truncated(self) -> bool:
        return self.kept_chars < self.original_chars and not self.dropped


@dataclass(frozen=True)
class Packet:
    """The rendered packet plus the accounting the caller logs."""

    stage: str
    text: str
    streams: tuple[StreamReport, ...]
    budget_chars: int
    reserve_chars: int

    @property
    def used_chars(self) -> int:
        return len(self.text)

    @property
    def token_estimate(self) -> int:
        return estimate_tokens(self.text)

    @property
    def evidence_chars(self) -> int:
        return sum(r.kept_chars for r in self.streams if r.kind == EVIDENCE)

    @property
    def context_chars(self) -> int:
        return sum(r.kept_chars for r in self.streams if r.kind == CONTEXT)

    @property
    def truncated_streams(self) -> tuple[str, ...]:
        return tuple(r.name for r in self.streams if r.truncated)

    @property
    def dropped_streams(self) -> tuple[str, ...]:
        return tuple(r.name for r in self.streams if r.dropped)

    def log_line(self) -> str:
        """One line for the run log: model-visible chars, estimate, truncation by field."""
        return (
            f"stage={self.stage} chars={self.used_chars}/{self.budget_chars} "
            f"reserve={self.reserve_chars} tokens~{self.token_estimate} "
            f"evidence={self.evidence_chars} context={self.context_chars} "
            f"truncated={list(self.truncated_streams)} dropped={list(self.dropped_streams)}"
        )


def estimate_tokens(text: str) -> int:
    """Coarse token estimate for logging and run-to-run comparison (not a tokenizer)."""
    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN if text else 0


def _clean(value: object) -> str:
    """Refuse spurious Pandas nan/None strings, as the Step 4 packet does."""
    if value is None:
        return ""
    raw = str(value).strip()
    return "" if raw.casefold() in ("nan", "none", "nat", "<na>") else raw


def _cap(value: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(value) <= limit:
        return value
    marker = TRUNCATION_MARKER
    return value[: max(0, limit - len(marker))] + marker[:limit]


def build_packet(
    stage: str,
    streams: Iterable[Stream],
    *,
    budget: StageBudget | None = None,
    include_context: bool = True,
) -> Packet:
    """Lay out independently capped streams under one stage budget.

    Ordering is by priority and then always evidence before context, so continuity context can
    never displace source evidence (#239 asks exactly this for primary quotation, and #240 that
    memory/RAG is never evidence). Inputs are never mutated. When a stream does not fit, context
    is dropped before evidence, and any truncation is reported per field rather than implied.
    """
    budget = budget or stage_budget(stage)
    supplied = list(streams)
    if any(s.kind not in (EVIDENCE, CONTEXT) for s in supplied):
        raise ValueError("stream kind must be 'evidence' or 'context'")
    ordered = sorted(
        supplied,
        key=lambda s: (s.kind == CONTEXT, s.priority, s.name.casefold()),
    )
    if not include_context:
        ordered = [s for s in ordered if s.kind != CONTEXT]

    reports: list[StreamReport] = []
    blocks: list[str] = []
    remaining = budget.evidence_chars
    all_names = [s.name for s in ordered]

    for stream in ordered:
        text = _clean(stream.text)
        original = len(text)
        cap = int(stream.cap if stream.cap is not None else budget.cap_for(stream.name, all_names))
        header = f"### {stream.name} [{stream.kind}]"
        room = remaining - len(header) - 2
        if not text or cap <= 0 or room <= 200:
            # Never silently drop evidence: an empty or squeezed-out stream is reported.
            reports.append(StreamReport(stream.name, stream.kind, cap, original, 0, True))
            continue
        kept = _cap(text, min(cap, room))
        block = f"{header}\n{kept}"
        blocks.append(block)
        remaining -= len(block) + 2
        reports.append(
            StreamReport(stream.name, stream.kind, cap, original, len(kept), kept == "")
        )

    # Even the empty-input notice must not exceed the model-visible budget.
    text = "\n\n".join(blocks) if blocks else NO_EVIDENCE[:budget.evidence_chars]
    return Packet(stage, text, tuple(reports), budget.evidence_chars, budget.output_reserve_chars)


def effective_prompt_fingerprint(
    system_prompt: str,
    user_prompt: str,
    model: str,
    options: Mapping[str, object] | None = None,
) -> str:
    """Cache key over everything that changes the answer: prompt, model and generation options.

    Generalises the Step 4 fix in #244, where the key had to start covering the effective Ollama
    options so a ``num_ctx``/``num_predict`` change could not reuse a stale cached answer.
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
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
