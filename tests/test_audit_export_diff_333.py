"""Export-to-export diff: the audit must distinguish a re-package from a new run (#333)."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "ep24"))

import audit_export_diff as diff  # noqa: E402

HEADER = ["record_id", "asr_transcript", "frame_analysis_1", "vllm_structured_output",
          "summary_analysis", "postprocess_entities", "laclau_structured_json",
          "dna_statements_json", "sna_nodes_json"]


def write_export(root: Path, country: str, rows: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"{country}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in HEADER})


def test_identical_exports_are_reported_as_frozen(tmp_path: Path) -> None:
    # "{}" is treated as empty by the tooling, so use values that are unambiguously present.
    rows = [{"record_id": "r1", "asr_transcript": "hello"},
            {"record_id": "r2", "asr_transcript": "world"}]
    write_export(tmp_path / "a", "finland", rows)
    write_export(tmp_path / "b", "finland", rows)
    report = diff.compare(tmp_path / "a", tmp_path / "b")
    kinds = {f["kind"] for f in report["findings"]}
    assert "frozen_stage" in kinds
    assert report["countries"]["finland"]["stages"]["1_preprocess"]["content_changed"] == 0


def test_changed_values_are_detected_even_when_presence_is_identical(tmp_path: Path) -> None:
    write_export(tmp_path / "a", "finland", [{"record_id": "r1", "asr_transcript": "before"}])
    write_export(tmp_path / "b", "finland", [{"record_id": "r1", "asr_transcript": "after"}])
    report = diff.compare(tmp_path / "a", tmp_path / "b")
    stage = report["countries"]["finland"]["stages"]["1_preprocess"]
    assert stage["with_output_older"] == stage["with_output_newer"] == 1
    assert stage["content_changed"] == 1
    assert any(f["kind"] == "content_changed" for f in report["findings"])


def test_a_stage_regression_is_an_error(tmp_path: Path) -> None:
    write_export(tmp_path / "a", "poland", [
        {"record_id": "r1", "laclau_structured_json": '{"statements": 3}'},
        {"record_id": "r2", "laclau_structured_json": '{"statements": 4}'},
    ])
    write_export(tmp_path / "b", "poland", [
        {"record_id": "r1", "laclau_structured_json": '{"statements": 3}'},
        {"record_id": "r2", "laclau_structured_json": ""},
    ])
    report = diff.compare(tmp_path / "a", tmp_path / "b")
    errors = [f for f in report["findings"] if f["kind"] == "stage_regression"]
    assert errors and errors[0]["severity"] == "error"
    assert errors[0]["older"] == 2 and errors[0]["newer"] == 1


def test_a_disappearing_child_table_is_an_error(tmp_path: Path) -> None:
    for name in ("a", "b"):
        write_export(tmp_path / name, "finland", [{"record_id": "r1"}])
    child = tmp_path / "a" / "children" / "finland_dna_statements.csv"
    child.parent.mkdir(parents=True)
    child.write_text("statement_id,parent_record_id\ns1,r1\n", encoding="utf-8")
    report = diff.compare(tmp_path / "a", tmp_path / "b")
    assert any(f["kind"] == "child_table_disappeared" for f in report["findings"])


def test_records_added_and_removed_are_counted(tmp_path: Path) -> None:
    write_export(tmp_path / "a", "portugal", [{"record_id": "r1"}])
    write_export(tmp_path / "b", "portugal", [{"record_id": "r2"}])
    entry = diff.compare(tmp_path / "a", tmp_path / "b")["countries"]["portugal"]
    assert entry["records_added"] == 1
    assert entry["records_removed"] == 1


def test_output_never_contains_cell_contents(tmp_path: Path) -> None:
    secret = "TRANSCRIPT-CONTENT-MUST-NOT-APPEAR"
    write_export(tmp_path / "a", "finland", [{"record_id": "r1", "asr_transcript": secret}])
    write_export(tmp_path / "b", "finland", [{"record_id": "r1", "asr_transcript": secret + "!"}])
    rendered = diff.render(diff.compare(tmp_path / "a", tmp_path / "b"))
    assert secret not in rendered
