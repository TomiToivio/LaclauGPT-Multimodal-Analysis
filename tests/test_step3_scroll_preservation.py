"""Regression: Step 3 must not replace cumulative scroll metadata."""
import logging

import pandas as pd

from ep24_pipeline import assert_source_metadata_preserved
from experiments.vllm_video_test import OUTPUT_COLUMNS, write_local_checkpoint


def test_checkpoint_preserves_upstream_scroll_values(tmp_path):
    source = pd.DataFrame([{
        "video_id": "HEPP24-PT-CR-IG-20240601-1053000",
        "SCROLL": "TRUE",
        "SCROLL_SECONDS": "[1.5]",
        "needs_resplit": "TRUE",
        "video_initial_skip_seconds": "2.0",
    }])
    result = {key: "" for key in OUTPUT_COLUMNS}
    result.update({
        "SCROLL": "FALSE", "SCROLL_SECONDS": "[]",
        "needs_resplit": "FALSE", "video_initial_skip_seconds": "1.0",
        "vllm_video_status": "ok",
    })
    output = write_local_checkpoint(
        source, [0], [result], tmp_path / "out.csv",
        tmp_path / "out.sqlite3", logging.getLogger(__name__),
    )
    assert_source_metadata_preserved(source, output)
    for col in ("SCROLL", "SCROLL_SECONDS", "needs_resplit", "video_initial_skip_seconds"):
        assert output.at[0, col] == source.at[0, col]
        assert output.at[0, "vllm_video_" + col.lower()] == result[col]


def test_checkpoint_remains_additive_without_upstream_scroll(tmp_path):
    source = pd.DataFrame([{"video_id": "test-video"}])
    result = {key: "" for key in OUTPUT_COLUMNS}
    result.update({"SCROLL": "FALSE", "SCROLL_SECONDS": "[]", "vllm_video_status": "ok"})
    output = write_local_checkpoint(
        source, [0], [result], tmp_path / "out.csv",
        tmp_path / "out.sqlite3", logging.getLogger(__name__),
    )
    assert_source_metadata_preserved(source, output)
    assert output.at[0, "SCROLL"] == "FALSE"
    assert output.at[0, "SCROLL_SECONDS"] == "[]"
