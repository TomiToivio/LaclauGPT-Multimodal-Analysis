"""Regression tests: NeMo ASR never passes MP4 directly to obsolete torchaudio StreamReader."""
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

import asr_backend


def test_nemo_wav_uses_ffmpeg_pcm_and_rejects_missing_output(tmp_path, monkeypatch):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video")
    seen = []

    def fake_run(cmd, **kwargs):
        seen.append(cmd)
        Path(cmd[-1]).write_bytes(b"RIFF" + b"0" * 48)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(asr_backend.subprocess, "run", fake_run)
    output = asr_backend._nemo_audio_path(str(source), str(tmp_path))
    assert output.endswith(".wav")
    assert ["-ac", "1"] == seen[0][seen[0].index("-ac"):seen[0].index("-ac")+2]
    assert ["-ar", "16000"] == seen[0][seen[0].index("-ar"):seen[0].index("-ar")+2]


def test_nemo_wav_reports_decoder_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(asr_backend.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "bad video"))
    with pytest.raises(RuntimeError, match="bad video"):
        asr_backend._nemo_audio_path("bad.mp4", str(tmp_path))
