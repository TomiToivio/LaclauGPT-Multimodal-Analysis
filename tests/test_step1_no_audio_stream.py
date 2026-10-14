"""Regressions for audio-free EP24 analysis clips on Roihu."""
from __future__ import annotations

import subprocess

import asr_backend
import pytest

# Both symbols are resolved through the module on purpose. tests/test_asr_backend.py
# reloads asr_backend to re-read env-dependent state, and importlib.reload rebinds
# NoAudioStreamError in the module dict in place: a name imported here would keep
# pointing at the pre-reload class, so the raise would no longer match the
# pytest.raises type and this test would fail only when that file ran first.


def test_no_audio_stream_has_distinct_error(monkeypatch, tmp_path):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 1, "", "Output file #0 does not contain any stream")
    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(asr_backend.NoAudioStreamError, match="no audio stream"):
        asr_backend._nemo_audio_path("silent.mp4", str(tmp_path))


def test_corrupt_media_stays_processing_error(monkeypatch, tmp_path):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 1, "", "moov atom not found")
    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="moov atom not found") as exc:
        asr_backend._nemo_audio_path("corrupt.mp4", str(tmp_path))
    assert not isinstance(exc.value, asr_backend.NoAudioStreamError)
