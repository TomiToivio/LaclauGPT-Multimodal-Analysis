"""Stage CSVs must carry JSON, not Python ``repr`` (#333 D3).

Measured on the delivered FI/PL/PT export: 6,832 cells across `ep24_human_seed_context`,
`step4_summary_provenance`, `_pipeline_step_6` and `_pipeline_step_7` hold `{'key': 'value'}` text that
`json.loads` rejects. The cause is `DataFrame.to_csv`: a dict/list cell is rendered with `str()`, so a
provenance dict leaves Python as a repr, and every later consumer treats "looks structured" as "is JSON"
and fails.

`write_cumulative_csv` is the single place stage CSVs are written, so the repair lives there and no
stage has to remember it.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from ep24_pipeline import json_safe_cell, write_cumulative_csv


def test_containers_are_serialized_as_json():
    assert json.loads(json_safe_cell({"a": 1})) == {"a": 1}
    assert json.loads(json_safe_cell([1, 2])) == [1, 2]
    assert json.loads(json_safe_cell(("x", "y"))) == ["x", "y"]
    assert json.loads(json_safe_cell({"s"})) == ["s"]  # JSON has no set: sorted list


def test_scalars_pass_through_unchanged():
    for value in ("text", 3, 2.5, True, None):
        assert json_safe_cell(value) is value


def test_written_csv_round_trips_structured_cells(tmp_path):
    before = pd.DataFrame([{"video_id": "v1", "provenance": ""}])
    after = pd.DataFrame([{
        "video_id": "v1",
        "provenance": {"model": "stub", "prompt_version": "v3"},
        "nested": [{"statement_id": "s1"}],
        "SCROLL": "TRUE",
    }])
    path = tmp_path / "stage.csv"
    write_cumulative_csv(before, after, path, mutable_columns=["provenance", "nested"])

    row = pd.read_csv(path, dtype=str, keep_default_na=False).iloc[0]
    assert json.loads(row["provenance"]) == {"model": "stub", "prompt_version": "v3"}
    assert json.loads(row["nested"]) == [{"statement_id": "s1"}]
    assert row["SCROLL"] == "TRUE"


def test_the_callers_frame_is_not_mutated(tmp_path):
    before = pd.DataFrame([{"video_id": "v1"}])
    after = pd.DataFrame([{"video_id": "v1", "provenance": {"model": "stub"}}])
    write_cumulative_csv(before, after, tmp_path / "o.csv")
    assert after.iloc[0]["provenance"] == {"model": "stub"}


def test_a_repr_cell_would_have_failed_before_this_fix():
    # The regression this guards: repr text looks structured and is not parseable.
    repr_cell = str({"model": "stub"})
    with pytest.raises(json.JSONDecodeError):
        json.loads(repr_cell)
    assert json.loads(json_safe_cell({"model": "stub"})) == {"model": "stub"}
