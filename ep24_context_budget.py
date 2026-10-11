#!/usr/bin/env python3
"""Shared EP24 Steps 4-8 model-input budget and prompt contract (issue #241).

Three stages already bounded their model input by hand, each its own way: Step 2 in
``roihu_frame._row_context`` (``LACLAUGPT_FRAME_CONTEXT_MAX_CHARS``, a prompt-version constant
and a ``context_sha256`` cache key), Step 4 twice (``ep24_summary_evidence.build_packet`` with
per-stream caps, and ``roihu_summary``). Each is reasonable; together they are three policies,
and the properties #241 asks for are missing from all three:

* **model-visible chars AND token estimates logged per field** -- so an over-budget prompt is
  visible before it is sent, not inferred afterwards;
* **context reserved for the output and the JSON schema** -- a budget that fills every character
  leaves the model no room to answer, and the failure looks like a model problem;
* **a cache key that versions on prompt, model AND options** -- otherwise changing the prompt
  silently reuses answers produced by the old one;
* **truncation reported per field**, not as a single global flag.

This module is the shared contract. It is deliberately pure: it does not read files, contact a
model, or mutate its inputs. Stage integrations consume it (``ep24_summary_evidence`` for Step 4,
and the frame/summary stages as they are migrated); it never imports them.

Design rules carried from the issue:

* inputs are **provenance-labelled** -- every channel's label says where the text came from and
  what it is not, because context (memory, RAG, other stages' outputs) is an aid, never evidence;
* source information is never *deleted* -- truncation is explicit and reported, and the caller
  keeps the untouched row;
* no stage's own outputs are fed back to it: excluding a channel is a caller decision that this
  module records as ``omitted``, so the omission is visible rather than silent.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

#: Characters per token used for the estimate. Deliberately crude and stated: it is a *budget
#: signal*, not a tokenizer. A real tokenizer would add a dependency to every stage, and the
#: number exists to catch an over-budget prompt early, not to bill for it.
CHARS_PER_TOKEN = 4

TRUNCATION_MARKER = "\n[TRUNCATED: field exceeds its budget; the source is unchanged]"

TOTAL_ENV = "LACLAUGPT_CONTEXT_MAX_CHARS"
RESERVE_OUTPUT_ENV = "LACLAUGPT_CONTEXT_RESERVE_OUTPUT_CHARS"
RESERVE_SCHEMA_ENV = "LACLAUGPT_CONTEXT_RESERVE_SCHEMA_CHARS"
CAP_ENV_PREFIX = "LACLAUGPT_CONTEXT_CAP_"

#: A prompt below this is not worth sending: the reserve and labels alone would consume it.
MIN_TOTAL_CHARS = 2000

#: What a stage must keep in every packet so its own omission is visible to the reader.
NOTHING_USABLE = (
    "No usable source evidence was supplied for this record. Report that limitation and do not "
    "invent content: no channel carried provenance-labelled evidence within the configured budget."
)

_MISSING = ("nan", "none", "nat", "<na>")


class BudgetError(ValueError):
    """The requested budget cannot produce a valid prompt."""


def clean(value: Any) -> str:
    """Normalise a cell: missing values and the string spellings pandas invents become empty."""
    if value is None:
        return ""
    raw = str(value).strip()
    return "" if raw.lower() in _MISSING else raw


def estimate_tokens(text: str) -> int:
    """Crude token estimate for a budget signal. Empty text is zero tokens, not one."""
    text = clean(text)
    if not text:
        return 0
    return max(1, (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN)


@dataclass(frozen=True, slots=True)
class Channel:
    """One provenance-labelled input field.

    ``label`` is part of the contract, not decoration: it is what tells the model (and the
    reader of the log) whether the text is direct evidence, a secondary rendering of it, or
    context that must not be treated as evidence at all.
    """

    name: str
    label: str
    value: str = ""
    priority: int = 0
    cap: int | None = None

    def cap_for(self, default: int) -> int:
        if self.cap is not None:
            return int(self.cap)
        env = os.getenv(f"{CAP_ENV_PREFIX}{self.name.upper()}")
        return int(env) if env not in (None, "") else int(default)


@dataclass(frozen=True, slots=True)
class FieldReport:
    """What the model actually saw for one field."""

    channel: str
    chars: int
    est_tokens: int
    truncated: bool
    omitted: bool
    reason: str = ""


@dataclass(frozen=True, slots=True)
class Packet:
    """The prompt body plus the accounting that makes it auditable."""

    text: str
    report: tuple[FieldReport, ...]
    budget: int
    reserve_output: int
    reserve_schema: int

    @property
    def used(self) -> int:
        return len(self.text)

    @property
    def truncated_fields(self) -> tuple[str, ...]:
        return tuple(r.channel for r in self.report if r.truncated)

    @property
    def omitted_fields(self) -> tuple[str, ...]:
        return tuple(r.channel for r in self.report if r.omitted)

    def render_report(self) -> str:
        """One compact line per field: chars, tokens, truncation. For the stage log."""
        lines = [
            f"context budget: {self.budget} chars "
            f"(reserved: {self.reserve_output} output + {self.reserve_schema} schema), "
            f"used {self.used}"
        ]
        for r in self.report:
            state = "omitted" if r.omitted else ("truncated" if r.truncated else "full")
            detail = f" ({r.reason})" if r.reason else ""
            lines.append(
                f"  {r.channel}: {r.chars} chars, ~{r.est_tokens} tokens, {state}{detail}"
            )
        return "\n".join(lines)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise BudgetError(f"{name} must be an integer, got {raw!r}") from exc


def resolve_budget(
    total: int | None = None,
    *,
    reserve_output: int | None = None,
    reserve_schema: int | None = None,
) -> tuple[int, int, int]:
    """Resolve the three budget numbers from arguments, then the environment, then defaults."""
    total = int(total) if total is not None else _env_int(TOTAL_ENV, 20000)
    reserve_output = (
        int(reserve_output) if reserve_output is not None
        else _env_int(RESERVE_OUTPUT_ENV, 2000)
    )
    reserve_schema = (
        int(reserve_schema) if reserve_schema is not None
        else _env_int(RESERVE_SCHEMA_ENV, 800)
    )
    if total < MIN_TOTAL_CHARS:
        raise BudgetError(
            f"context budget must be at least {MIN_TOTAL_CHARS} characters, got {total}"
        )
    if reserve_output < 0 or reserve_schema < 0:
        raise BudgetError("reserved output/schema context cannot be negative")
    if reserve_output + reserve_schema >= total:
        raise BudgetError(
            f"reserved context ({reserve_output} output + {reserve_schema} schema) leaves no room "
            f"for input within a {total}-character budget; raise {TOTAL_ENV} or lower the reserves"
        )
    return total, reserve_output, reserve_schema


def build_packet(
    channels: Iterable[Channel],
    *,
    total: int | None = None,
    reserve_output: int | None = None,
    reserve_schema: int | None = None,
) -> Packet:
    """Assemble a bounded, provenance-labelled prompt body and report what was included.

    Channels are filled by priority (descending), then by the order given. A channel that does
    not fit is recorded as ``omitted`` with its reason rather than dropped silently; a channel
    that partly fits is truncated with an in-band marker, so the model cannot read a clipped
    field as a complete one. The caller's data is never modified.
    """
    budget, reserve_output, reserve_schema = resolve_budget(
        total, reserve_output=reserve_output, reserve_schema=reserve_schema
    )
    available = budget - reserve_output - reserve_schema
    ordered = sorted(list(channels), key=lambda c: -c.priority)

    sections: list[str] = []
    report: list[FieldReport] = []
    remaining = available
    for channel in ordered:
        value = clean(channel.value)
        if not value:
            report.append(FieldReport(channel.name, 0, 0, False, True, "empty"))
            continue
        head = f"### {channel.label}\n"
        room = min(channel.cap_for(remaining), remaining - len(head) - 2)
        if room <= 0:
            report.append(
                FieldReport(channel.name, 0, 0, False, True,
                            "no budget left after higher-priority fields")
            )
            continue
        truncated = len(value) > room
        if truncated:
            body = value[:max(0, room - len(TRUNCATION_MARKER))] + TRUNCATION_MARKER
        else:
            body = value
        block = head + body
        sections.append(block)
        report.append(
            FieldReport(channel.name, len(body), estimate_tokens(body), truncated, False,
                        "exceeds its channel budget" if truncated else "")
        )
        remaining -= len(block) + 2

    text = "\n\n".join(sections) if sections else NOTHING_USABLE
    return Packet(text, tuple(report), budget, reserve_output, reserve_schema)


def prompt_contract_hash(
    prompt: str,
    model: str,
    options: Mapping[str, Any] | None = None,
    *,
    extra: Sequence[Any] = (),
) -> str:
    """A cache key that changes when the prompt, the model OR the options change.

    A cache keyed on the source alone silently reuses answers produced by a different prompt or
    model -- the answer looks current and is not. Options are serialised with sorted keys so a
    reordered mapping is the same contract and a changed value is not.
    """
    payload = {
        "prompt": prompt,
        "model": model,
        "options": dict(sorted((options or {}).items())),
        "extra": list(extra),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=repr)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def summarise(packet: Packet) -> dict[str, Any]:
    """Machine-readable accounting for a stage log or a run report."""
    return {
        "budget": packet.budget,
        "reserve_output": packet.reserve_output,
        "reserve_schema": packet.reserve_schema,
        "used": packet.used,
        "est_tokens": estimate_tokens(packet.text),
        "truncated": list(packet.truncated_fields),
        "omitted": list(packet.omitted_fields),
        "fields": [
            {"channel": r.channel, "chars": r.chars, "est_tokens": r.est_tokens,
             "truncated": r.truncated, "omitted": r.omitted, "reason": r.reason}
            for r in packet.report
        ],
    }


__all__ = [
    "BudgetError", "CHARS_PER_TOKEN", "CAP_ENV_PREFIX", "Channel", "FieldReport",
    "MIN_TOTAL_CHARS", "NOTHING_USABLE", "Packet", "RESERVE_OUTPUT_ENV", "RESERVE_SCHEMA_ENV",
    "TOTAL_ENV", "TRUNCATION_MARKER", "build_packet", "clean", "estimate_tokens",
    "prompt_contract_hash", "resolve_budget", "summarise",
]

# re-exported for callers that only need to strip the marker when re-reading a packet
strip_truncation_marker = re.compile(re.escape(TRUNCATION_MARKER)).sub
