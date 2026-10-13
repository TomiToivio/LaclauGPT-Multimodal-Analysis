"""Regression checks for reprocessing a previously analyzed EP24 frame."""
from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd
import pytest

from ep24_pipeline import assert_source_metadata_preserved


def test_frame_stage_explicitly_allows_its_own_output_columns():
    """Protect the actual Step 2 call site against reverting to QUALITY_COLUMNS."""
    tree = ast.parse((Path(__file__).resolve().parents[1] / "roihu_frame.py").read_text())
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "assert_source_metadata_preserved"
    ]
    assert len(calls) == 1
    mutable = next(k.value for k in calls[0].keywords if k.arg == "mutable_columns")
    assert isinstance(mutable, ast.Name)
    assert mutable.id == "OUTPUT_COLUMNS"


def test_reprocessing_can_refresh_frame_output_but_not_source_metadata():
    before = pd.DataFrame([{
        "video_id": "source-1",
        "asr_transcript": "Original transcript",
        "frame_analysis_1": "Old analysis",
        "frame_analysis_status": "ok",
        "processing_status": "OK",
    }])
    after = before.copy(deep=True)
    after.loc[0, "frame_analysis_1"] = "New analysis"
    after.loc[0, "frame_analysis_status"] = "cached"
    assert_source_metadata_preserved(
        before, after,
        mutable_columns=("frame_analysis_1", "frame_analysis_status"),
    )
    after.loc[0, "asr_transcript"] = "Changed transcript"
    with pytest.raises(AssertionError, match="asr_transcript"):
        assert_source_metadata_preserved(
            before, after,
            mutable_columns=("frame_analysis_1", "frame_analysis_status"),
        )
