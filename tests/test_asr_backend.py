"""Contract tests for the backend-neutral EP24 ASR adapter (issue #128)."""

import importlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def backend(monkeypatch):
    for key in (
        "LACLAUGPT_ASR_ENGINE",
        "LACLAUGPT_ASR_MODEL",
        "LACLAUGPT_ASR_DEVICE",
        "LACLAUGPT_ASR_COMPUTE_TYPE",
        "LACLAUGPT_ASR_DOWNLOAD_ROOT",
    ):
        monkeypatch.delenv(key, raising=False)
    import asr_backend

    return importlib.reload(asr_backend)


def test_default_backend_is_modern_canary_candidate(backend):
    assert backend.describe_backend() == {
        "engine": "canary",
        "model": "nvidia/canary-1b-v2",
    }


def test_ep24_country_language_hints_cover_all_ten_countries(backend):
    expected = {
        "finland": "fi",
        "poland": "pl",
        "portugal": "pt",
        "germany": "de",
        "spain": "es",
        "hungary": "hu",
        "croatia": "hr",
        "france": "fr",
        "bulgaria": "bg",
        "sweden": "sv",
    }
    assert backend.COUNTRY_LANGUAGE_HINTS == expected


def test_whisper_temperature_ladder_is_preserved_for_baseline_only(backend):
    assert tuple(backend.LEGACY_TEMPERATURE) == (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def test_backend_selection_is_provenance_only_until_model_load(backend, monkeypatch):
    monkeypatch.setenv("LACLAUGPT_ASR_ENGINE", "faster-whisper")
    monkeypatch.setenv("LACLAUGPT_ASR_MODEL", "large-v3-turbo")
    assert backend.describe_backend() == {
        "engine": "faster-whisper",
        "model": "large-v3-turbo",
    }


def test_unknown_engine_fails_loudly_without_importing_a_model(backend, monkeypatch):
    monkeypatch.setenv("LACLAUGPT_ASR_ENGINE", "whisper-large")
    with pytest.raises(ValueError, match="Unknown LACLAUGPT_ASR_ENGINE"):
        backend.describe_backend()


def test_preprocess_uses_backend_abstraction_not_hardcoded_whisper():
    text = (ROOT / "roihu_preprocess.py").read_text(encoding="utf-8")
    assert "load_asr_model" in text
    assert not re.search(r"^import whisper$", text, re.M)
    assert "whisper.load_model(" not in text


def test_preprocess_writes_generic_asr_fields_and_not_whisper_fields():
    text = (ROOT / "roihu_preprocess.py").read_text(encoding="utf-8")
    for field in ("asr_transcript", "asr_language", "asr_translated", "asr_backend", "asr_model"):
        assert field in text
    for legacy in ("whisperResult", "whisper_transcript", "whisper_language", "whisper_translated"):
        assert legacy not in text
