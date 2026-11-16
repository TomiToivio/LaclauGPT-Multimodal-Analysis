"""Synthetic tests for the exact-LFS EP24 QA runner (#405)."""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/ep24/audit_issue405_sources.py"
spec = importlib.util.spec_from_file_location("audit_issue405_sources", PATH)
audit = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(audit)


def test_rejects_lfs_pointer_as_csv(tmp_path):
    path = tmp_path / "finland.csv"
    path.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:123\nsize 10\n")
    result = audit.audit_country(path, "finland")
    assert result["errors"] == ["git_lfs_pointer_not_dataset"]
    assert result["rows"] == 0


def test_counts_stage_status_and_invalid_json_without_printing_rows(tmp_path):
    path = tmp_path / "finland.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "country", "video_id", "_storage_id", "dna_status", "sna_status",
            "dna_statements_json", "asr_transcript"
        ])
        writer.writeheader()
        writer.writerow({"country": "finland", "video_id": "private-1",
                         "_storage_id": "private-id-1", "dna_status": "ok",
                         "sna_status": "error", "dna_statements_json": '[{"a":1}]',
                         "asr_transcript": "Private transcript"})
        writer.writerow({"country": "poland", "video_id": "private-2",
                         "_storage_id": "private-id-1", "dna_status": "error",
                         "sna_status": "", "dna_statements_json": "[{bad json",
                         "asr_transcript": "Sensitive evidence"})
    result = audit.audit_country(path, "finland", verify_fixture=False)
    assert result["rows"] == 2
    assert result["wrong_country_rows"] == 1
    assert result["duplicate_storage_ids"] == 1
    assert result["stages"]["7"]["success"] == 1
    assert result["stages"]["7"]["errors"] == 1
    assert result["stages"]["8"]["missing"] == 1
    assert result["invalid_structured_json"] == {"dna_statements_json": 1}
    assert "Private transcript" not in str(result)
    assert "private-id-1" not in str(result)
    assert "Sensitive evidence" not in str(result)


def test_row_width_mismatch_is_reported(tmp_path):
    path = tmp_path / "finland.csv"
    path.write_text("country,video_id\nfinland,1,extra\n", encoding="utf-8")
    result = audit.audit_country(path, "finland", verify_fixture=False)
    assert "csv_field_count_mismatch" in result["errors"]
