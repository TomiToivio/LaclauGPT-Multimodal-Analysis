"""Regression tests for the opt-in completed Monday500 dataframe export."""
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from ep24_stage_orchestrator import export_completed_monday500


class FakeCollection:
    def __init__(self, rows):
        self.rows = rows

    def count_documents(self, query):
        if not query:
            return len(self.rows)
        return sum(row.get("_pipeline", {}).get("step_07", {}).get("status") == "complete"
                   for row in self.rows)

    def find(self, query):
        return [row for row in self.rows
                if row.get("_pipeline", {}).get("step_07", {}).get("status") == "complete"]


class FakeStorage:
    def __init__(self, rows):
        self.collection = FakeCollection(rows)
        self.db = {"sample_dataframe": self.collection}

    def collection_name(self, suffix):
        assert suffix == "dataframe"
        return "sample_dataframe"


def sample_rows(size=500, completed=500):
    return [{"_storage_id": f"monday500:finland:{i:04d}",
             "source_field": f"raw-{i}",
             "_pipeline": {"step_07": {"status": "complete" if i < completed else "error"}}}
            for i in range(size)]


def test_export_gated_by_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("LACLAUGPT_MONDAY500_AUTO_EXPORT", raising=False)
    assert export_completed_monday500(FakeStorage(sample_rows()), "finland", tmp_path) is None
    assert not list(tmp_path.rglob("*.csv"))


def test_export_refuses_partial_sample(monkeypatch, tmp_path):
    monkeypatch.setenv("LACLAUGPT_MONDAY500_AUTO_EXPORT", "1")
    assert export_completed_monday500(FakeStorage(sample_rows(completed=499)), "finland", tmp_path) is None


def test_export_preserves_all_rows_and_is_repeatable(monkeypatch, tmp_path):
    monkeypatch.setenv("LACLAUGPT_MONDAY500_AUTO_EXPORT", "1")
    storage = FakeStorage(sample_rows())
    path = export_completed_monday500(storage, "finland", tmp_path)
    assert path == tmp_path / "monday500_exports" / "finland_steps_1_7.csv"
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert len(frame) == 500
    assert len(set(frame["_storage_id"])) == 500
    assert "raw-0" in frame["source_field"].values
    contents = path.read_bytes()
    assert export_completed_monday500(storage, "finland", tmp_path) == path
    assert path.read_bytes() == contents
