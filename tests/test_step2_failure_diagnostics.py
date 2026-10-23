"""Regression checks for persistent Step 2 failures and stale keyframes."""
from pathlib import Path

import pytest

from roihu_frame import _validate_keyframe, OUTPUT_COLUMNS


def test_missing_keyframe_reports_original_path(tmp_path):
    path = tmp_path / "missing.jpg"
    with pytest.raises(FileNotFoundError, match="Step 2 keyframe does not exist"):
        _validate_keyframe(str(path), 1.0)


def test_frame_failure_includes_persisted_diagnostic():
    assert "frame_analysis_error" in OUTPUT_COLUMNS
    frame = Path("roihu_frame.py").read_text(encoding="utf-8")
    assert 'df.at[index, "frame_analysis_error"] = f"{type(exc).__name__}: {exc}"[:2000]' in frame


def test_stale_step1_keyframe_cache_is_regenerated():
    source = Path("roihu_preprocess.py").read_text(encoding="utf-8")
    assert "cache_missing_keyframe" in source
    assert "if not frame.is_file() or frame.stat().st_size == 0:" in source


def test_orchestrator_keeps_failure_csv():
    source = Path("ep24_stage_orchestrator.py").read_text(encoding="utf-8")
    assert 'f"step_{step:02d}_failed_batch.csv"' in source
    assert "bad.to_csv(report, index=False" in source
