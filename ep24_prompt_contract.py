"""Cross-stage prompt/context contract for the numbered EP24 Roihu steps (issue #241).

Issue #241 asks for one policy shared by Steps 4-8 rather than five separate
conventions: direct-source evidence first, stage-specific context packets, no stage
reading its own earlier output back in, no source information deletion, no unmanaged
RAG flood, logged model-visible size and truncation, and a cache key that changes when
the prompt, the model or the options change.

Steps 5-8 each grew a bounded builder independently and they agree on most of that by
convention. This module makes the *shared* part explicit and testable, and leaves the
genuinely stage-specific part -- which fields matter, and in what order -- in the step,
where it belongs.

What lives here:

``STEPS``
    The registry: for each step, the module that implements it, the environment
    variable that overrides its budget, and the documented default. Tests import the
    builders through this registry, so a step that stops honouring the contract fails
    there rather than quietly shipping a different policy.

``budget(step)``
    The effective character budget, environment first, documented default second.

``prompt_hash(step, version, model, system_prompt, context)``
    The cache fingerprint. It covers the step's prompt version, the selected model, the
    system prompt and the effective context -- so a change to any of them changes the
    key. Hashing the context alone would let a system-prompt edit reuse stale results.

``EXCLUDED_COLUMNS``
    Columns that must never reach a model prompt: another stage's generated analysis,
    the RAG context, and the memory context. A step may *read* them for provenance, but
    feeding them back is what #241 rules out ("no own outputs fed back", "no unmanaged
    RAG flood").

``truncation_entry(...)`` and ``truncation_report(...)``
    The per-field truncation record. #241 asks for truncation "by field", which a bare
    boolean cannot express, so steps that adopt this emit one entry per field that lost
    characters -- enough to answer "what did the model not see?" after the fact. The
    original column is never modified; the record describes the packet, not the data.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class StepContract:
    """One numbered step's place in the shared prompt/context policy."""

    step: int
    module: str
    builder: str
    env: str
    default_budget: int
    #: Fields that carry direct or explicitly-labelled evidence for this step.
    primary_fields: tuple[str, ...] = ()
    notes: str = field(default="")


#: The documented budgets. Keep the defaults in step with the step modules; the test
#: asserts the registry and the module agree rather than trusting this table.
STEPS: Mapping[int, StepContract] = {
    5: StepContract(
        step=5, module="roihu_postprocess", builder="build_postprocess_prompt",
        env="LACLAUGPT_STEP5_MAX_PROMPT_CHARS", default_budget=12000,
        notes="Evidence-grounded postprocess extraction.",
    ),
    6: StepContract(
        step=6, module="roihu_populism", builder="build_prompt_context",
        env="LACLAUGPT_STEP6_MAX_CONTEXT_CHARS", default_budget=28000,
        notes="Theory locked (Laclau/Palonen); evidence selection only.",
    ),
    7: StepContract(
        step=7, module="step_7_roihu_discourse_network_analysis", builder="build_prompt_context",
        env="LACLAUGPT_STEP7_MAX_CONTEXT_CHARS", default_budget=28000,
        notes="DNA statement-level evidence; theory locked.",
    ),
    8: StepContract(
        step=8, module="step_8_roihu_social_network_analysis", builder="build_sna_evidence",
        env="LACLAUGPT_STEP8_MAX_EVIDENCE_CHARS", default_budget=12000,
        notes="Actor-relation evidence only; no inferred hidden ties.",
    ),
}

#: Columns that must never be fed to a model. Another stage's generated analysis is
#: circularity; the RAG and memory contexts are the unmanaged flood #241 names.
EXCLUDED_COLUMNS: tuple[str, ...] = (
    "rag_context_json",
    "memory_context_json",
    "laclau_analysis_json",
    "formula_of_populism_analysis",
    "dna_analysis_json",
    "sna_analysis_json",
    "postprocess_analysis",
)


def budget(step: int) -> int:
    """The effective character budget for ``step``: environment, else the default."""
    contract = STEPS[step]
    raw = os.getenv(contract.env)
    if raw is None or str(raw).strip() == "":
        return contract.default_budget
    try:
        value = int(str(raw).strip())
    except ValueError:
        return contract.default_budget
    return value if value > 0 else contract.default_budget


def prompt_hash(
    step: int,
    version: str,
    model: str,
    system_prompt: str,
    context: str,
) -> str:
    """Cache fingerprint covering the prompt version, model, system prompt and context.

    Deliberately not a hash of the context alone: #241 requires the key to move when
    the model or the settings change, otherwise a model swap silently reuses results
    produced by the previous one.
    """
    payload = f"step={step}\nversion={version}\nmodel={model}\n{system_prompt}\n{context}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def truncation_entry(field_name: str, had: int, taken: int) -> dict[str, Any]:
    """One field's truncation record. ``had`` and ``taken`` are character counts."""
    return {
        "field": field_name,
        "chars_available": int(had),
        "chars_included": int(taken),
        "truncated": int(taken) < int(had),
    }


def truncation_report(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collapse per-field entries into the record stored beside a result."""
    lost = [e for e in entries if e.get("truncated")]
    return {
        "fields_included": len(entries),
        "fields_truncated": len(lost),
        "characters_lost": sum(
            int(e["chars_available"]) - int(e["chars_included"]) for e in lost
        ),
        "by_field": {str(e["field"]): dict(e) for e in entries},
    }
