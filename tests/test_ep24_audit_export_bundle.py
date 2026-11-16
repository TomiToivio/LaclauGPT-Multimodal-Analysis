"""The export-bundle audit must find what a pointer file and a merge badge cannot (#405).

Fixtures here are synthetic and public: the real country CSVs live in the private repository and never
appear in this test file, in its assertions, or in its output.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "ep24"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import audit_export_bundle as bundle  # noqa: E402


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]], bom: bool = True) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig" if bom else "utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def export(tmp_path: Path) -> Path:
    columns = ["source_key", "_storage_id", "video_id", "country", "dataset",
               "preprocess_status", "ocr_1", "asr_transcript", "asr_language",
               "laclau_status", "laclau_analysis"]
    rows = [
        # a real result
        {"source_key": "k1", "_storage_id": "ep24:finland:1", "video_id": "v1", "country": "finland",
         "dataset": "ep2024_reprocess", "preprocess_status": "ok", "ocr_1": "frame text",
         "asr_transcript": "spoken words", "asr_language": "fi", "laclau_status": "ok",
         "laclau_analysis": "analysis one"},
        # status says success, nothing written -- the defect the audit exists for
        {"source_key": "k2", "_storage_id": "ep24:finland:2", "video_id": "v2", "country": "finland",
         "dataset": "ep2024_reprocess", "preprocess_status": "ok", "ocr_1": "",
         "asr_transcript": "", "asr_language": "", "laclau_status": "ok", "laclau_analysis": ""},
        # an honest failure
        {"source_key": "k3", "_storage_id": "ep24:finland:3", "video_id": "v3", "country": "finland",
         "dataset": "ep2024_reprocess", "preprocess_status": "error", "ocr_1": "",
         "asr_transcript": "", "asr_language": "", "laclau_status": "", "laclau_analysis": ""},
    ]
    digest = write_csv(tmp_path / "finland" / "finland.csv", columns, rows)
    (tmp_path / "manifest.json").write_text(json.dumps({
        "version": "test-export-v1", "exported_at": "2026-11-15T19:00:00+00:00",
        "producer_revision": "monday500_20261031",
        "files": {"finland/finland.csv": digest},
    }), encoding="utf-8")
    write_csv(tmp_path / "children" / "dna_statements.csv",
              ["child_id", "source_key", "country"],
              [{"child_id": "c1", "source_key": "k1", "country": "finland"},
               {"child_id": "c2", "source_key": "orphan", "country": "finland"}])
    return tmp_path


def test_the_bom_in_the_exported_header_is_measured_not_silently_stripped(export: Path):
    report = bundle.audit_export(export)
    assert report["countries"]["finland"]["file"]["header_bom"] is True
    # ... and the reader still keys the first column correctly, so the audit itself is sound
    assert report["countries"]["finland"]["identity"]["missing_key_columns"] == []


def test_status_ok_with_no_output_is_a_counted_defect(export: Path):
    report = bundle.audit_export(export, samples=True)
    stages = {stage["stage"]: stage for stage in report["countries"]["finland"]["stages"]}
    assert stages[1]["completed"] == 1
    assert stages[1]["failed"] == 1
    assert stages[1]["claimed_but_empty"] == 1
    assert report["private_samples"]["finland"]["1"] == ["ep24:finland:2"]
    # a stage with no status and no output is not_started, not a failure
    assert stages[6]["not_started"] >= 1


def test_identity_counts_are_reported_per_country(export: Path):
    identity = bundle.audit_export(export)["countries"]["finland"]["identity"]
    assert identity["rows"] == 3
    assert identity["identity_present"] == 3
    assert identity["identity_missing"] == 0
    assert identity["identity_duplicates"] == 0
    assert identity["countries_in_frame"] == ["finland"]
    assert identity["dataset_labels"] == ["ep2024_reprocess"]


def test_manifest_hash_is_compared_with_the_bytes_on_disk(export: Path):
    manifest = bundle.audit_export(export)["manifest"]
    assert manifest["present"] is True
    assert manifest["all_match"] is True
    assert manifest["files_compared"]["finland/finland.csv"]["match"] is True


def test_a_tampered_csv_fails_the_manifest_comparison(export: Path):
    path = export / "finland" / "finland.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    manifest = bundle.audit_export(export)["manifest"]
    assert manifest["all_match"] is False


def test_child_tables_report_orphan_keys(export: Path):
    children = {child["table"]: child for child in bundle.audit_export(export)["children"]}
    statements = children["dna_statements"]
    assert statements["rows"] == 2
    assert statements["unmatched_source_keys"] == 1
    assert statements["videos_with_children"] == {"finland": 1}
    assert children["sna_edges"]["present"] is False


def test_the_markdown_reports_counts_without_row_content(export: Path):
    text = bundle.markdown(bundle.audit_export(export))
    assert "Stage coverage (per country)" in text
    assert "claimed/empty" in text
    for secret in ("frame text", "spoken words", "analysis one", "ep24:finland:2"):
        assert secret not in text


def test_an_absent_manifest_is_stated_not_assumed(export: Path):
    (export / "manifest.json").unlink()
    manifest = bundle.audit_export(export)["manifest"]
    assert manifest == {"present": False}


def test_missing_countries_are_marked_missing(tmp_path: Path):
    report = bundle.audit_export(tmp_path)
    assert report["countries"]["poland"] == {"present": False}
    assert report["manifest"] == {"present": False}
