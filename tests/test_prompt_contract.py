"""The cross-stage prompt/context contract holds for every numbered step (issue #241).

Each test drives the step's real builder through the registry, so a step that stops
honouring the shared policy fails here instead of shipping a different one.
"""
from __future__ import annotations

import importlib
import re
from pathlib import Path

import pandas as pd
import pytest

from ep24_prompt_contract import (
    EXCLUDED_COLUMNS,
    STEPS,
    budget,
    prompt_hash,
    truncation_entry,
    truncation_report,
)

ROOT = Path(__file__).resolve().parents[1]

#: Distinctive markers. Never assert on the bare words "RAG" or "memory": a step is
#: allowed to *say* it excluded them (Step 8's header does exactly that).
RAG_MARKER = "ZZRAGMARKER"
MEMORY_MARKER = "ZZMEMMARKER"
OWN_OUTPUT_MARKER = "ZZOWNOUTPUT"


def _builder(step: int):
    contract = STEPS[step]
    return getattr(importlib.import_module(contract.module), contract.builder)


def _text(result) -> str:
    """Builders return either the text or (text, ...) -- take the text."""
    return result[0] if isinstance(result, tuple) else result


def _fat_row() -> pd.Series:
    return pd.Series(
        {
            "asr_transcript": "Alice objected to Bob. " * 400,
            "summary_analysis": "A conversation between two actors. " * 400,
            "ocr_1": "PROTEST SIGN " * 200,
            "researcher_note": "researcher is uncertain " * 100,
            "entities": "Alice, Bob",
            "country": "Finland",
            "rag_context_json": f"{RAG_MARKER} " * 5000,
            "memory_context_json": f"{MEMORY_MARKER} " * 5000,
            "formula_of_populism_analysis": f"{OWN_OUTPUT_MARKER} " * 5000,
        }
    )


@pytest.mark.parametrize("step", sorted(STEPS))
def test_the_registry_names_the_env_var_the_module_actually_reads(step: int) -> None:
    """Guards the registry against drift: it must describe real code."""
    contract = STEPS[step]
    source = (ROOT / f"{contract.module}.py").read_text(encoding="utf-8")
    assert contract.env in source, (
        f"{contract.module} does not read {contract.env}; the contract registry is stale"
    )


@pytest.mark.parametrize("step", sorted(STEPS))
def test_each_builder_respects_an_environment_budget(step: int, monkeypatch) -> None:
    contract = STEPS[step]
    monkeypatch.setenv(contract.env, "4000")
    assert budget(step) == 4000
    out = _text(_builder(step)(_fat_row()))
    assert len(out) <= 4000, f"step {step} exceeded its 4000-char budget ({len(out)})"


@pytest.mark.parametrize("step", sorted(STEPS))
def test_no_builder_feeds_rag_memory_or_another_stages_analysis_back(step: int) -> None:
    out = _text(_builder(step)(_fat_row()))
    for marker in (RAG_MARKER, MEMORY_MARKER, OWN_OUTPUT_MARKER):
        assert marker not in out, f"step {step} leaked {marker} into the model input"


@pytest.mark.parametrize("step", sorted(STEPS))
def test_no_builder_mutates_the_original_row(step: int) -> None:
    row = _fat_row()
    before = row.copy()
    _builder(step)(row)
    assert row.equals(before), f"step {step} mutated its input row"


def test_the_excluded_set_covers_rag_memory_and_own_outputs() -> None:
    assert "rag_context_json" in EXCLUDED_COLUMNS
    assert "memory_context_json" in EXCLUDED_COLUMNS
    assert any("analysis" in c for c in EXCLUDED_COLUMNS), "no stage output is excluded"


def test_the_budget_falls_back_to_the_documented_default(monkeypatch) -> None:
    contract = STEPS[8]
    monkeypatch.delenv(contract.env, raising=False)
    assert budget(8) == contract.default_budget
    monkeypatch.setenv(contract.env, "not-a-number")
    assert budget(8) == contract.default_budget
    monkeypatch.setenv(contract.env, "-5")
    assert budget(8) == contract.default_budget


def test_the_cache_hash_covers_prompt_model_system_and_context() -> None:
    digest = prompt_hash(8, "v1", "m", "s", "c")
    assert digest == prompt_hash(8, "v1", "m", "s", "c"), "not deterministic"
    assert prompt_hash(8, "v2", "m", "s", "c") != digest, "version did not move the key"
    assert prompt_hash(8, "v1", "other", "s", "c") != digest, "model did not move the key"
    assert prompt_hash(8, "v1", "m", "s2", "c") != digest, "system prompt did not move the key"
    assert prompt_hash(8, "v1", "m", "s", "c2") != digest, "context did not move the key"


def test_the_truncation_report_says_what_the_model_did_not_see() -> None:
    report = truncation_report(
        [
            truncation_entry("asr_transcript", 10_000, 3_000),
            truncation_entry("entities", 40, 40),
        ]
    )
    assert report["fields_included"] == 2
    assert report["fields_truncated"] == 1
    assert report["characters_lost"] == 7_000
    assert report["by_field"]["entities"]["truncated"] is False
    assert report["by_field"]["asr_transcript"]["truncated"] is True
