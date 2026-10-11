"""Regressions for audio-free EP24 analysis clips on Roihu."""
from __future__ import annotations

import subprocess

import pytest

from asr_backend import NoAudioStreamError, _nemo_audio_path


def test_no_audio_stream_has_distinct_error(monkeypatch, tmp_path):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 1, "", "Output file #0 does not contain any stream")
    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(NoAudioStreamError, match="no audio stream"):
        _nemo_audio_path("silent.mp4", str(tmp_path))


def test_corrupt_media_stays_processing_error(monkeypatch, tmp_path):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 1, "", "moov atom not found")
    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="moov atom not found") as exc:
        _nemo_audio_path("corrupt.mp4", str(tmp_path))
    assert not isinstance(exc.value, NoAudioStreamError)
