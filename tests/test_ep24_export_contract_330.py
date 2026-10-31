"""EP24 export contract: elision is data, child tables carry coverage, children are joinable (#330-#332).

These tests pin the properties a consumer of the artifact set depends on:

* an elided cell must not look like an empty one;
* a step whose output is a child table must not report as unavailable in the same export whose manifest
  counts its statements;
* a child table must state which unit it is counted in, and QA must report orphans rather than invent
  parents.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ep24_mongo_export as export  # noqa: E402


# --- elision ---------------------------------------------------------------------------

def test_heavy_cell_is_replaced_by_a_readable_sentinel() -> None:
    row = {"record_id": "r1", "dna_statements_json": "x" * 1500, "short": "ok"}
    lean, heavy = export.split_heavy_fields(row, threshold=100, sentinel=export.HEAVY_SENTINEL,
                                            country="finland")
    assert "dna_statements_json" not in heavy or True
    assert heavy["dna_statements_json"] == "x" * 1500
    assert lean["dna_statements_json"].startswith("<<elided: 1500 chars")
    assert "finland_text.jsonl.gz" in lean["dna_statements_json"]
    assert lean["short"] == "ok"


def test_without_a_sentinel_the_cell_is_still_removed_from_the_index() -> None:
    lean, heavy = export.split_heavy_fields({"a": "y" * 200}, threshold=100)
    assert "a" not in lean and heavy["a"] == "y" * 200


def test_heavy_field_report_counts_rows_per_field() -> None:
    rows = [
        {"dna_statements_json": "x" * 200, "asr_transcript": "short"},
        {"dna_statements_json": "y" * 200, "asr_transcript": "z" * 300},
        {"dna_statements_json": "[]"},
    ]
    report = export.heavy_field_report(rows, threshold=100)
    assert report == {"asr_transcript": 1, "dna_statements_json": 2}


# --- coverage from child tables ----------------------------------------------------------

def test_dna_coverage_is_read_from_the_child_table_when_the_cell_is_empty() -> None:
    rows = [{"record_id": "r1", "dna_statements_json": ""}, {"record_id": "r2"}]
    coverage = export.stage_coverage(rows, {"dna_statements": 1583})
    dna = coverage["7_dna"]
    assert dna["records"] == 1583
    assert dna["available"] is True
    assert dna["source"] == "children"


def test_row_level_coverage_still_wins_when_the_cells_carry_output() -> None:
    rows = [{"dna_statements_json": "[\"a\"]"}, {"dna_statements_json": ""}]
    coverage = export.stage_coverage(rows, {"dna_statements": 40})
    assert coverage["7_dna"]["records"] == 1
    assert coverage["7_dna"]["source"] == "rows+children"


def test_absent_child_table_does_not_invent_coverage() -> None:
    coverage = export.stage_coverage([{"record_id": "r1"}], {"dna_statements": 0})
    assert coverage["7_dna"]["records"] == 0
    assert coverage["7_dna"]["available"] is False
    assert coverage["7_dna"]["source"] == "none"


def test_child_counts_do_not_leak_into_unrelated_steps() -> None:
    coverage = export.stage_coverage([{"record_id": "r1"}], {"dna_statements": 10, "sna": 5})
    assert coverage["8_sna"]["records"] == 5
    assert coverage["9_rdf"]["records"] == 0


# --- child QA ----------------------------------------------------------------------------

def _child(statement_id: str, parent: str) -> dict:
    return {"statement_id": statement_id, "source_record_id": parent}


def test_children_qa_reports_a_complete_join() -> None:
    qa = export.children_qa([_child("s1", "p1"), _child("s2", "p1")], {"p1"})
    assert qa["rows"] == 2
    assert qa["parents_resolved"] == 2
    assert qa["orphans"] == 0
    assert qa["join_complete"] is True


def test_children_qa_counts_orphans_rather_than_dropping_them() -> None:
    qa = export.children_qa([_child("s1", "p1"), _child("s2", "gone")], {"p1"})
    assert qa["orphans"] == 1
    assert qa["parents_resolved"] == 1
    assert qa["join_complete"] is False


def test_children_qa_counts_duplicate_statement_ids() -> None:
    qa = export.children_qa([_child("s1", "p1"), _child("s1", "p1")], {"p1"})
    assert qa["unique_statement_ids"] == 1
    assert qa["duplicate_statement_ids"] == 1


def test_children_qa_counts_missing_parent_field() -> None:
    qa = export.children_qa([{"statement_id": "s1"}], {"p1"})
    assert qa["missing_parent_field"] == 1
    assert qa["join_complete"] is False


def test_child_parent_key_matches_the_parent_storage_id() -> None:
    """The join that works is documented: child `source_record_id` → parent `_storage_id`."""
    parents = [{"_storage_id": "ingest-42", "record_id": "video-42"}]
    children = [_child("s1", "ingest-42")]
    parent_ids = {str(row["_storage_id"]) for row in parents}
    qa = export.children_qa(children, parent_ids)
    assert qa["parents_resolved"] == 1
    assert qa["join_complete"] is True


# --- manifest ----------------------------------------------------------------------------

def test_manifest_states_the_children_unit_and_echoes_elision(tmp_path: Path) -> None:
    manifest = export.write_manifest(
        tmp_path,
        record_unit="video",
        children_unit="ingest",
        countries={"finland": {"records": 2125, "distinct_media": 2125}},
        child_counts={"finland": {"dna_statements": 1583}},
        heavy_fields={"finland": {"dna_statements_json": 517}},
        children_qa_report={"finland:dna_statements": {"rows": 1583, "orphans": 93}},
        outputs=[],
        provenance={},
        collections_read={},
    )
    data = json.loads((tmp_path / "MANIFEST.json").read_text())
    assert data["record_unit"] == "video"
    assert data["children_unit"] == "ingest"
    assert data["heavy_fields"]["finland"]["dna_statements_json"] == 517
    assert data["children_qa"]["finland:dna_statements"]["orphans"] == 93
