"""Regression coverage for Step 5 parse failure followed by enrichment."""
from pathlib import Path

import pandas as pd

from ep24_pipeline import ensure_columns


def test_unparsed_step5_video_status_survives_csv_roundtrip(tmp_path):
    frame = pd.DataFrame({"video_id": ["ok", "parse-failed"],
                          "video_status": ["", ""]})
    ensure_columns(frame, ("video_status",))
    frame.at[0, "video_status"] = "OK"
    path = tmp_path / "step5.csv"
    frame.to_csv(path, index=False)
    restored = pd.read_csv(path, dtype=str, keep_default_na=False)
    for column in frame.columns:
        assert restored[column].fillna("").astype(str).tolist() == (
            frame[column].fillna("").astype(str).tolist()
        )
    assert restored["video_status"].tolist() == ["OK", ""]


def test_step5_initializes_video_status_and_comparison_is_null_safe():
    source = (Path(__file__).resolve().parents[1] / "roihu_postprocess.py").read_text()
    initialization = source.split("ensure_columns(\n        df,", 1)[1].split("    df[\"processing_status\"]", 1)[0]
    assert '"video_status",' in initialization
    assert 'normalized[column].fillna("").astype(str).tolist()' in source
    assert 'df[column].fillna("").astype(str).tolist()' in source
