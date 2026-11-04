"""Synthetic-only regression checks for issue #373."""
from pathlib import Path

import pandas as pd
import pytest

from ep24_backups import write_checkpoint
from scripts.ep24.ep24_rollout import reconcile, stage_counts, discover
from scripts.ep24.export_full_corpus import frame_from_docs, verified_checkpoint, coverage, completed


def record(n, **extras):
    return {"video_id": f"video-{n}", "allas_filename": f"private/{n}.mp4",
            "_storage_id": f"stable-{n}", **extras}


def test_sample_resumes_by_media_not_first_500_rows():
    source = [record(n) for n in range(503)]
    # Historical sample identities may be different from current row-number hashes.
    sample = [dict(record(n), _storage_id=f"sample-{n}",
                   _pipeline={"step_01": {"status": "complete"},
                              "step_07": {"status": "complete"}}) for n in range(500)]
    new, matched, stats = reconcile(source, sample)
    assert len(matched) == 500
    assert len(new) == 3
    assert stats["new_to_insert"] == 3
    assert stage_counts(matched)["7"]["complete"] == 500


def test_partially_complete_sample_is_not_assumed_finished():
    source = [record(1), record(2)]
    sample = [dict(record(1), _pipeline={"step_01": {"status": "complete"}})]
    new, matched, _ = reconcile(source, sample)
    assert len(new) == 1
    assert stage_counts(matched)["1"]["complete"] == 1
    assert stage_counts(matched)["7"]["pending"] == 1


def test_ambiguous_or_orphaned_existing_records_fail_closed():
    with pytest.raises(ValueError, match="ambiguous"):
        reconcile([record(1)], [record(1), dict(record(1), _storage_id="different")])
    with pytest.raises(ValueError, match="absent"):
        reconcile([record(1)], [record(2)])


def test_source_duplicates_and_invalid_are_audited():
    new, matched, stats = reconcile([record(1), record(1), {"video_id": "no-media"}], [])
    assert len(new) == 1 and not matched
    assert stats["duplicate_input_rows"] == 1
    assert stats["invalid_input_rows"] == 1


def test_fresh_country_discovery(tmp_path):
    for name in ("finland", "poland", "portugal", "germany", "sweden"):
        (tmp_path / f"ep24_{name}.csv").write_text("video_id\nx\n")
    assert [p.stem for p in discover(tmp_path, "fresh")] == ["ep24_germany", "ep24_sweden"]
    assert [p.stem for p in discover(tmp_path, "continue")] == [
        "ep24_finland", "ep24_poland", "ep24_portugal"]


def test_export_preserves_partial_rows_and_nested_json(tmp_path):
    docs = [
        record(2, country="finland", _pipeline={"step_01": {"status": "complete"},
                                              "step_07": {"status": "error"}},
               _provenance={"model": "test"}, labels=["a", "b"]),
        record(1, country="finland", _pipeline={"step_07": {"status": "complete"}},
               _provenance={"model": "test"}, labels=["c"]),
    ]
    assert not completed(docs)
    assert coverage(docs)["step_07"] == {"error": 1, "complete": 1}
    df = frame_from_docs(docs)
    assert df["_storage_id"].tolist() == ["stable-1", "stable-2"]
    manifest = verified_checkpoint(df, tmp_path / "full.csv", country="finland")
    reread = pd.read_csv(tmp_path / "full.csv", dtype=str, keep_default_na=False)
    assert len(reread) == 2
    assert reread.loc[1, "labels"] == '["a", "b"]'
    assert manifest["rows"] == 2
    assert (tmp_path / "full.sqlite3").is_file()
    assert (tmp_path / "full.csv.manifest.json").is_file()


def test_checkpoint_nested_csv_is_json_not_python_repr(tmp_path):
    import json
    df = pd.DataFrame([{"_storage_id": "a", "nested": {"key": ["value"]}}])
    write_checkpoint(df, tmp_path / "c.csv", stage="test", country="test")
    assert json.loads(pd.read_csv(tmp_path / "c.csv").loc[0, "nested"]) == {"key": ["value"]}
