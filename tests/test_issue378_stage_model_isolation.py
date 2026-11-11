"""Stage-specific Ollama model isolation (#378).

Model size changes are opt-in until exact Roihu tags and quality have passed
the research approval gate. Step 6 never inherits a smaller shared override.
"""
import pytest

from ep24_models import (
    DEFAULT_OLLAMA_MODEL,
    stage_ollama_model,
    stage_ollama_model_source,
)


def test_step6_remains_pinned_when_global_model_is_small(monkeypatch):
    monkeypatch.setenv("LACLAUGPT_MULTIMODAL_MODEL", "small-test:8b")
    monkeypatch.delenv("LACLAUGPT_STEP6_MODEL", raising=False)
    assert stage_ollama_model(6) == DEFAULT_OLLAMA_MODEL
    assert stage_ollama_model_source(6) == "step6_pinned_27b_default"


def test_all_other_ollama_stages_allow_independent_overrides(monkeypatch):
    monkeypatch.setenv("LACLAUGPT_MULTIMODAL_MODEL", "shared-test:8b")
    for step in (2, 4, 5, 7, 8):
        monkeypatch.setenv(f"LACLAUGPT_STEP{step}_MODEL", f"stage{step}-test:4b")
        assert stage_ollama_model(step) == f"stage{step}-test:4b"
        assert stage_ollama_model_source(step) == f"LACLAUGPT_STEP{step}_MODEL"
        monkeypatch.delenv(f"LACLAUGPT_STEP{step}_MODEL")
        assert stage_ollama_model(step) == "shared-test:8b"


def test_explicit_step6_model_is_auditable(monkeypatch):
    monkeypatch.setenv("LACLAUGPT_STEP6_MODEL", "qwen3.8:27b")
    assert stage_ollama_model(6) == "qwen3.8:27b"
    assert stage_ollama_model_source(6) == "LACLAUGPT_STEP6_MODEL"


def test_unknown_stage_fails_closed():
    with pytest.raises(ValueError):
        stage_ollama_model(9)
