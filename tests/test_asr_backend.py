"""Contract tests for the configurable ASR backend (transcription stage).

These tests are deterministic and require no GPU, no network, no private EP24
data and no model download. They pin the properties that matter for the
"preserve the legacy behaviour by default" rule:

1. the default backend configuration reproduces the historical call;
2. the temperature fallback ladder is the historical one;
3. unsupported engines fail loudly rather than silently falling back;
4. the preprocess script no longer hard-codes a single ASR engine.
"""

import importlib
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def backend(monkeypatch):
    """Import asr_backend with a clean ASR environment."""
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


def test_default_backend_is_the_historical_whisper_large(backend):
    """No env vars must mean: openai-whisper, 'large', './whisper/' cache."""
    assert backend.describe_backend() == {
        "engine": "whisper",
        "model": "large",
        "download_root": "./whisper/",
    }


def test_temperature_ladder_is_preserved_verbatim(backend):
    """The legacy fallback ladder must not drift."""
    assert tuple(backend.LEGACY_TEMPERATURE) == (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def test_faster_whisper_selected_by_env(backend, monkeypatch):
    monkeypatch.setenv("LACLAUGPT_ASR_ENGINE", "faster-whisper")
    monkeypatch.setenv("LACLAUGPT_ASR_MODEL", "large-v3-turbo")
    cfg = backend.describe_backend()
    assert cfg["engine"] == "faster-whisper"
    assert cfg["model"] == "large-v3-turbo"
    assert cfg["compute_type"] == "int8_float16"


def test_unknown_engine_fails_loudly(backend, monkeypatch):
    """A typo must raise, not silently transcribe with a different engine."""
    monkeypatch.setenv("LACLAUGPT_ASR_ENGINE", "whisper-large")
    with pytest.raises(ValueError, match="Unknown LACLAUGPT_ASR_ENGINE"):
        backend.describe_backend()


def test_preprocess_uses_the_backend_and_not_a_hardcoded_engine():
    """Guard against a regression back to a hard-coded whisper.load_model."""
    text = (ROOT / "roihu_preprocess.py").read_text(encoding="utf-8")
    assert "load_asr_model" in text
    assert not re.search(r"^import whisper$", text, re.M), "hard-coded whisper import returned"
    assert "whisper.load_model(" not in text, "hard-coded model load returned"


def test_preprocess_preserves_legacy_output_fields():
    """The three whisper_* columns written to CSV/SQLite must survive."""
    text = (ROOT / "roihu_preprocess.py").read_text(encoding="utf-8")
    for field in ("whisper_transcript", "whisper_language", "whisper_translated"):
        assert field in text, field
