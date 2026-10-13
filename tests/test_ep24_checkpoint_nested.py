"""Regression tests for Mongo-derived nested values in EP24 checkpoints."""
import json
import sqlite3

import pandas as pd

from ep24_backups import write_checkpoint


def test_checkpoint_serializes_nested_values_without_mutating_source(tmp_path):
    nested = {"actor": "UNSA", "scores": [1, 2], "metadata": {"ok": True}}
    df = pd.DataFrame({
        "_storage_id": ["a", "b"],
        "analysis": [nested, {"actor": "Other"}],
        "tags": [["a", "b"], []],
        "count": [1, 2],
    })
    path = tmp_path / "step_02_finland.csv"

    manifest = write_checkpoint(df, path, stage="step_02", country="finland")

    assert manifest["rows"] == 2
    assert path.exists()
    assert path.with_suffix(".sqlite3").exists()
    assert path.with_suffix(".csv.manifest.json").exists()
    with sqlite3.connect(path.with_suffix(".sqlite3")) as db:
        records = pd.read_sql_query("SELECT * FROM records ORDER BY _storage_id", db)
    assert json.loads(records.loc[0, "analysis"]) == nested
    assert json.loads(records.loc[0, "tags"]) == ["a", "b"]
    assert json.loads(records.loc[1, "tags"]) == []
    assert records.loc[0, "count"] == 1
    assert df.loc[0, "analysis"] is nested
    assert isinstance(df.loc[0, "tags"], list)


def test_checkpoint_supports_scalar_and_null_object_cells(tmp_path):
    df = pd.DataFrame({
        "mixed": [{"nested": [1]}, None, "plain text", 5],
    })
    path = tmp_path / "mixed.csv"
    write_checkpoint(df, path, stage="step_02", country="finland")
    with sqlite3.connect(path.with_suffix(".sqlite3")) as db:
        rows = db.execute("SELECT mixed FROM records ORDER BY rowid").fetchall()
    assert json.loads(rows[0][0]) == {"nested": [1]}
    assert rows[1][0] is None
    assert rows[2][0] == "plain text"
    assert rows[3][0] == "5" or rows[3][0] == 5
