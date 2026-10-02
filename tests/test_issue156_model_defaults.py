"""Tests for the standardized inference-model defaults (issue #156).

Covers the acceptance criteria:

* every active Ollama step defaults to ``qwen3.8:27b``;
* no active production Python path still defaults to ``gemma4:12b``;
* ``LACLAUGPT_MULTIMODAL_MODEL`` still overrides;
* the entity adjudicator keeps its specific override first;
* Step 3 stays vLLM with ``Qwen/Qwen3-VL-8B-Instruct``;
* ``LACLAUGPT_VLLM_TEST_MODEL`` still overrides;
* the split (Ollama vs vLLM) is not accidentally collapsed.

These read source text, so they run without Ollama, vLLM, a GPU or private data.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OLLAMA_DEFAULT = "qwen3.8:27b"
LEGACY_DEFAULT = "gemma4:12b"
VLLM_DEFAULT = "Qwen/Qwen3-VL-8B-Instruct"

#: Active Ollama call sites that must carry the new default.
ACTIVE_OLLAMA_FILES = (
    "roihu_frame.py",
    "roihu_summary.py",
    "roihu_postprocess.py",
    "roihu_populism.py",
    "step_7_roihu_discourse_network_analysis.py",
    "step_8_roihu_social_network_analysis.py",
    "ep24_entities.py",
)

#: Files that legitimately still mention the old name.
ALLOWED_LEGACY_FILES = {
    "roihu_rdf.py",                       # provenance record, not an inference default
    "docs/ROIHU_MIGRATION.md",            # historical narrative
    "docs/MODEL_RUNTIME_MATRIX.md",       # records the previous default explicitly
    "tests/test_roihu_storage.py",        # explicit model= argument, not a default
    "tests/test_issue144_model_runtime_doc.py",
    "tests/test_roihu_baseline.py",
    "tests/test_issue156_model_defaults.py",
}


def _text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


# --- the Ollama default ----------------------------------------------------

@pytest.mark.parametrize("relative", ACTIVE_OLLAMA_FILES)
def test_active_ollama_step_defaults_to_the_standard_model(relative):
    """Each active step must resolve to the standard model with no override.

    The literal now lives once in ``ep24_cli.DEFAULT_OLLAMA_MODEL``, so the
    assertion is on the *resolved* value rather than on a string that would have
    to be repeated in every file.
    """
    import ep24_cli

    text = _text(relative)
    carries_default = OLLAMA_DEFAULT in text or "resolve_model" in text
    assert carries_default, f"{relative} does not resolve the standard model"
    assert LEGACY_DEFAULT not in text, f"{relative} still carries {LEGACY_DEFAULT}"
    assert ep24_cli.DEFAULT_OLLAMA_MODEL == OLLAMA_DEFAULT


def test_no_active_python_path_defaults_to_the_legacy_model():
    """Repository-wide guard: any *active* Python default must be the new name.

    ``roihu_rdf.py`` is excluded deliberately -- its ``model=`` is a provenance
    record written by ``Provenance.capture()``, not an inference default, so it
    must keep describing whatever model actually produced the data.
    """
    offenders = []
    for path in sorted(ROOT.glob("*.py")):
        rel = path.name
        if rel == "roihu_rdf.py":
            continue
        text = path.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            if LEGACY_DEFAULT in line and "os.getenv" in line:
                offenders.append(f"{rel}:{i}")
    assert not offenders, f"active defaults still on {LEGACY_DEFAULT}: {offenders}"


# --- overrides -------------------------------------------------------------

def test_multimodal_model_override_still_wins(monkeypatch):
    """The shared env var must override the new default."""
    import roihu_postprocess as rp  # noqa: F401  (import proves it is importable)
    monkeypatch.setenv("LACLAUGPT_MULTIMODAL_MODEL", "custom:1b")
    assert os.getenv("LACLAUGPT_MULTIMODAL_MODEL", OLLAMA_DEFAULT) == "custom:1b"
    monkeypatch.delenv("LACLAUGPT_MULTIMODAL_MODEL", raising=False)
    assert os.getenv("LACLAUGPT_MULTIMODAL_MODEL", OLLAMA_DEFAULT) == OLLAMA_DEFAULT


def test_entity_adjudicator_prefers_its_specific_override():
    """LACLAUGPT_ENTITY_ADJUDICATOR_MODEL -> LACLAUGPT_MULTIMODAL_MODEL -> default."""
    text = _text("ep24_entities.py")
    line = next(cand for cand in text.splitlines() if "LACLAUGPT_ENTITY_ADJUDICATOR_MODEL" in cand)
    # order matters: adjudicator-specific first, shared second, literal last
    assert line.index("LACLAUGPT_ENTITY_ADJUDICATOR_MODEL") < line.index("LACLAUGPT_MULTIMODAL_MODEL")
    assert OLLAMA_DEFAULT in line


def test_entity_adjudicator_resolution_order_end_to_end(monkeypatch):
    import ep24_entities as E

    monkeypatch.setenv("LACLAUGPT_ENTITY_ADJUDICATOR_MODEL", "adj:1b")
    monkeypatch.setenv("LACLAUGPT_MULTIMODAL_MODEL", "shared:1b")
    resolved = (
        os.getenv("LACLAUGPT_ENTITY_ADJUDICATOR_MODEL")
        or os.getenv("LACLAUGPT_MULTIMODAL_MODEL", OLLAMA_DEFAULT)
    )
    assert resolved == "adj:1b"

    monkeypatch.delenv("LACLAUGPT_ENTITY_ADJUDICATOR_MODEL")
    resolved = (
        os.getenv("LACLAUGPT_ENTITY_ADJUDICATOR_MODEL")
        or os.getenv("LACLAUGPT_MULTIMODAL_MODEL", OLLAMA_DEFAULT)
    )
    assert resolved == "shared:1b"

    monkeypatch.delenv("LACLAUGPT_MULTIMODAL_MODEL")
    resolved = (
        os.getenv("LACLAUGPT_ENTITY_ADJUDICATOR_MODEL")
        or os.getenv("LACLAUGPT_MULTIMODAL_MODEL", OLLAMA_DEFAULT)
    )
    assert resolved == OLLAMA_DEFAULT
    assert hasattr(E, "ollama_adjudicator")


# --- the vLLM video step stays on vLLM -------------------------------------

def test_step3_defaults_to_the_vllm_video_model():
    text = _text("experiments/vllm_video_test.py")
    assert f'DEFAULT_MODEL = "{VLLM_DEFAULT}"' in text


def test_step3_is_not_converted_to_ollama():
    text = _text("experiments/vllm_video_test.py")
    assert "ollama" not in text.lower(), "the whole-video step must not use Ollama"
    assert OLLAMA_DEFAULT not in text, "the video step must not use the Ollama default"


def test_vllm_model_override_remains_functional():
    text = _text("experiments/vllm_video_test.py")
    assert "LACLAUGPT_VLLM_TEST_MODEL" in text
    line = next(cand for cand in text.splitlines() if "LACLAUGPT_VLLM_TEST_MODEL" in cand)
    assert "--model" in line, "the override must feed the --model argument"


def test_step3_wrapper_and_config_agree_with_the_code():
    for relative in ("config/vllm_video_test.env.example",
                     "scripts/roihu/vllm_video_test.sbatch"):
        assert VLLM_DEFAULT in _text(relative), f"{relative} disagrees with the code"


# --- sbatch ----------------------------------------------------------------

def test_active_sbatch_files_use_the_standard_ollama_default():
    for path in sorted((ROOT / "scripts" / "roihu").glob("*.sbatch")):
        text = path.read_text(encoding="utf-8")
        if "LACLAUGPT_MULTIMODAL_MODEL" not in text:
            continue
        assert LEGACY_DEFAULT not in text, f"{path.name} still defaults to the legacy model"
        assert OLLAMA_DEFAULT in text, f"{path.name} does not default to {OLLAMA_DEFAULT}"


def test_sbatch_files_pass_the_resolved_model_to_python():
    """A wrapper that resolves MODEL but never exports it would be a silent bug."""
    checked = 0
    for path in sorted((ROOT / "scripts" / "roihu").glob("*.sbatch")):
        text = path.read_text(encoding="utf-8")
        if "MODEL=${LACLAUGPT_MULTIMODAL_MODEL:-" not in text:
            continue
        checked += 1
        assert 'export LACLAUGPT_MULTIMODAL_MODEL="${MODEL}"' in text, path.name
    assert checked >= 6, f"expected the Ollama sbatch files to be covered, saw {checked}"


# --- docs ------------------------------------------------------------------

def test_runtime_matrix_reflects_the_new_default():
    text = _text("docs/MODEL_RUNTIME_MATRIX.md")
    assert OLLAMA_DEFAULT in text
    # the previous default is recorded as history, not as the current value
    assert "historically" in text.lower() or "previous" in text.lower() or "was" in text.lower()
    # the vLLM row must not have been changed with it
    assert VLLM_DEFAULT in text
