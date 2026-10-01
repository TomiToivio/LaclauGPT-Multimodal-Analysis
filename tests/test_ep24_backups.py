"""Backup/checkpoint tests for the EP24 pipeline (issue #64).

The property that matters is recoverability: a snapshot must be complete and
verifiable, and a crash mid-write must not damage the previous good one. These
tests use synthetic dataframes and tmp_path only.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ep24_backups as bk  # noqa: E402


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "video_id": "SYNTH-1",
                "country": "Finland",
                "new_entity": "",
                "entities": "Synthetic Person",
                "whisper_transcript": "synthetic",
            },
            {
                "video_id": "SYNTH-2",
                "country": "Finland",
                "new_entity": "E2",
                "entities": "E2",
                "whisper_transcript": "",
            },
        ]
    )


# --- CSV snapshots ---------------------------------------------------------

def test_snapshot_writes_the_csv_and_a_manifest(tmp_path):
    manifest = bk.write_cumulative_csv(
        _df(), tmp_path / "step_01_preprocess.csv", country="Finland", step=1
    )
    assert (tmp_path / "step_01_preprocess.csv").is_file()
    assert (tmp_path / "step_01_preprocess.csv.manifest.json").is_file()
    assert manifest.rows == 2
    assert manifest.country == "Finland" and manifest.step == 1
    assert manifest.csv_sha256


def test_snapshot_preserves_every_column_including_empty_ones(tmp_path):
    path = tmp_path / "step.csv"
    bk.write_cumulative_csv(_df(), path, country="Finland", step=1)
    back = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert list(back.columns) == list(_df().columns)
    assert back.loc[0, "new_entity"] == "", "a blank field must stay blank"


def test_blank_is_not_written_as_the_string_nan(tmp_path):
    """The blank-vs-false distinction the issue also requires of cleaning."""
    path = tmp_path / "step.csv"
    bk.write_cumulative_csv(_df(), path, country="Finland", step=1)
    text = path.read_text(encoding="utf-8")
    assert "nan" not in text.lower()


def test_manifest_records_the_checksum_of_the_bytes_on_disk(tmp_path):
    path = tmp_path / "step.csv"
    manifest = bk.write_cumulative_csv(_df(), path, country="Finland", step=1)
    assert manifest.csv_sha256 == bk.sha256_file(path)
    ok, reason = bk.verify_snapshot(path)
    assert ok, reason


def test_verify_detects_a_tampered_snapshot(tmp_path):
    path = tmp_path / "step.csv"
    bk.write_cumulative_csv(_df(), path, country="Finland", step=1)
    path.write_text(path.read_text(encoding="utf-8").replace("SYNTH-1", "TAMPERED"), encoding="utf-8")
    ok, reason = bk.verify_snapshot(path)
    assert not ok and "checksum" in reason


def test_verify_reports_a_missing_manifest_clearly(tmp_path):
    path = tmp_path / "lonely.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    ok, reason = bk.verify_snapshot(path)
    assert not ok and "manifest" in reason


def test_a_failed_write_leaves_the_previous_snapshot_intact(tmp_path, monkeypatch):
    """The issue's 'never destroy an older good checkpoint' rule."""
    path = tmp_path / "step.csv"
    good = bk.write_cumulative_csv(_df(), path, country="Finland", step=1)
    original_bytes = path.read_bytes()

    def boom(*_args, **_kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_csv", boom)
    try:
        bk.write_cumulative_csv(_df(), path, country="Finland", step=2)
    except RuntimeError:
        pass
    assert path.read_bytes() == original_bytes, "the good snapshot must survive a failed write"
    assert bk.verify_snapshot(path)[0]
    assert good.csv_sha256 == bk.sha256_file(path)


def test_no_temporary_files_are_left_behind(tmp_path):
    bk.write_cumulative_csv(_df(), tmp_path / "step.csv", country="Finland", step=1)
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


# --- SQLite checkpoints ----------------------------------------------------

def _state_rows():
    return [
        {"record_id": "SYNTH-1", "stage": 1, "state": "complete", "attempts": 1,
         "completed_at": 1.0, "provenance": {"model": "synthetic"}},
        {"record_id": "SYNTH-2", "stage": 1, "state": "error", "attempts": 2,
         "error": "boom", "provenance": {}},
    ]


def test_sqlite_checkpoint_round_trips_the_state(tmp_path):
    db = tmp_path / "state.sqlite"
    digest = bk.write_sqlite_checkpoint(db, stage_state=_state_rows())
    assert digest == bk.sha256_file(db)

    connection = sqlite3.connect(db)
    try:
        rows = connection.execute(
            "SELECT record_id, state, attempts FROM stage_state ORDER BY record_id"
        ).fetchall()
    finally:
        connection.close()
    assert rows == [("SYNTH-1", "complete", 1), ("SYNTH-2", "error", 2)]


def test_sqlite_checkpoint_is_recorded_in_the_manifest(tmp_path):
    manifest = bk.write_cumulative_csv(
        _df(), tmp_path / "step.csv", country="Finland", step=1
    )
    bk.write_sqlite_checkpoint(tmp_path / "state.sqlite", stage_state=_state_rows(), manifest=manifest)
    assert manifest.sqlite_sha256
    payload = json.loads((tmp_path / "step.csv.manifest.json").read_text(encoding="utf-8"))
    # The manifest file on disk is written before the sqlite checkpoint, so it
    # must be re-readable and self-consistent; the in-memory manifest carries the digest.
    assert payload["schema_version"] == bk.SCHEMA_VERSION
    assert set(payload) >= {"country", "step", "rows", "columns", "csv_sha256"}


def test_sqlite_checkpoint_replaces_a_previous_one_safely(tmp_path):
    db = tmp_path / "state.sqlite"
    bk.write_sqlite_checkpoint(db, stage_state=_state_rows())
    bk.write_sqlite_checkpoint(db, stage_state=[_state_rows()[0]])
    connection = sqlite3.connect(db)
    try:
        count = connection.execute("SELECT COUNT(*) FROM stage_state").fetchone()[0]
    finally:
        connection.close()
    assert count == 1


def test_checkpoint_output_paths_are_deterministic(tmp_path):
    """The issue's per-country/step naming contract."""
    path = tmp_path / "outputs" / "Finland" / "step_01_preprocess.csv"
    bk.write_cumulative_csv(_df(), path, country="Finland", step=1)
    assert path.is_file()
    manifest = bk.read_manifest(path)
    assert manifest["country"] == "Finland"
    assert manifest["step"] == 1
