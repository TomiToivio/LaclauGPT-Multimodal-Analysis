"""Regression tests for interrupted Roihu analysis-clip writes."""
from pathlib import Path
from types import SimpleNamespace

import ep24_video


def test_truncated_existing_clip_is_rebuilt_atomically(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source-video")
    target = ep24_video.analysis_clip_path(source, tmp_path / "clips")
    target.parent.mkdir(parents=True)
    target.write_bytes(b"partial mp4 missing moov atom")
    original_target = target.read_bytes()
    calls = []

    def probe(command, **kwargs):
        assert command[0] == "ffprobe"
        candidate = Path(command[-1])
        valid = candidate.read_bytes() == b"complete mp4"
        return SimpleNamespace(returncode=0 if valid else 1,
                               stdout="4.0\n" if valid else "",
                               stderr="moov atom not found" if not valid else "")

    monkeypatch.setattr(ep24_video.subprocess, "run", probe)

    def encode(command, **kwargs):
        calls.append(command)
        assert target.read_bytes() == original_target
        assert command[-1] != str(target)
        Path(command[-1]).write_bytes(b"complete mp4")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    result = ep24_video.prepare_analysis_clip(source, target.parent, runner=encode)
    assert result == target
    assert target.read_bytes() == b"complete mp4"
    assert len(calls) == 1
    assert list(target.parent.iterdir()) == [target]
    # A valid cache should now be reused without invoking the encoder.
    ep24_video.prepare_analysis_clip(source, target.parent, runner=lambda *a, **k: 1)


def test_failed_trim_preserves_existing_clip_and_cleans_temporary(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    target = ep24_video.analysis_clip_path(source, tmp_path / "clips")
    target.parent.mkdir(parents=True)
    target.write_bytes(b"broken")
    monkeypatch.setattr(
        ep24_video.subprocess, "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout="", stderr="moov atom not found"),
    )

    def failed_encoder(command, **kwargs):
        Path(command[-1]).write_bytes(b"partial")
        return SimpleNamespace(returncode=1, stdout="", stderr="encoding stopped")

    import pytest
    with pytest.raises(RuntimeError, match="encoding stopped"):
        ep24_video.prepare_analysis_clip(source, target.parent, runner=failed_encoder)
    assert target.read_bytes() == b"broken"
    assert list(target.parent.iterdir()) == [target]
