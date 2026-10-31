"""#330/#331/#332: the export must not report "nothing produced" for data it shipped.

Three separate silences are covered here:

* DNA coverage reading a cell the exporter blanks (#332),
* heavy cells vanishing from the index CSV with no trace (#331),
* and the child tables being the only place a statement survives.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import ep24_mongo_export as E


def _row(**kw):
    base = {"record_id": "r1", "dna_statements_json": "", "dna_statement_count": 0}
    base.update(kw)
    return base


def test_dna_coverage_counts_a_row_whose_json_cell_is_blank():
    """The regression: coverage said 'unavailable' while the row carried statements."""
    rows = [_row(dna_statement_count=3), _row(dna_statement_count=0)]
    coverage = E.stage_coverage(rows)
    assert coverage["7_dna"]["available"] is True, "a blanked JSON cell must not hide real statements"
    assert coverage["7_dna"]["records"] == 1


def test_dna_coverage_is_still_unavailable_when_nothing_was_produced():
    rows = [_row(), _row()]
    coverage = E.stage_coverage(rows)
    assert coverage["7_dna"]["available"] is False


def test_the_blanked_json_cell_no_longer_decides_alone():
    """Both shapes count: older exports carry the JSON, child-table exports carry the count."""
    json_shape = E.stage_coverage([_row(dna_statements_json="[{\"x\":1}]")])
    count_shape = E.stage_coverage([_row(dna_statement_count=1)])
    assert json_shape["7_dna"]["records"] == 1
    assert count_shape["7_dna"]["records"] == 1


def test_a_moved_cell_is_named_in_the_lean_row():
    lean, heavy = E.split_heavy_fields({"record_id": "r1", "asr_transcript": "x" * 1200})
    assert "asr_transcript" not in lean
    assert "asr_transcript" in heavy
    assert json.loads(lean["_text_bundle_fields"]) == ["asr_transcript"]


def test_several_moved_cells_are_all_named_and_sorted():
    lean, _ = E.split_heavy_fields({"record_id": "r1", "b_text": "x" * 1200, "a_text": "y" * 1200})
    assert json.loads(lean["_text_bundle_fields"]) == ["a_text", "b_text"]


def test_a_row_with_nothing_moved_gets_no_marker():
    lean, heavy = E.split_heavy_fields({"record_id": "r1", "short": "ok"})
    assert "_text_bundle_fields" not in lean
    assert heavy == {}


def _audit_module():
    spec = importlib.util.spec_from_file_location(
        "audit_export_compat",
        Path(__file__).resolve().parents[1] / "scripts/ep24/audit_export.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_audit_reads_the_current_bundle_marker():
    audit = _audit_module()
    row = {"_text_bundle_fields": json.dumps(["summary", "asr_transcript"])}
    assert audit._bundle_fields(row) == ["summary", "asr_transcript"]


def test_audit_still_reads_exports_written_before_the_rename():
    """#353 renamed the marker; an export made before it must not look empty to the audit.

    Reading only the new name is how #331/#332 came back: the stage shipped its prose, the audit saw
    nothing, and the coverage report called it a defect.
    """
    audit = _audit_module()
    row = {"heavy_fields_in_text_bundle": "summary,asr_transcript"}
    assert audit._bundle_fields(row) == ["summary", "asr_transcript"]


def test_audit_reports_nothing_for_a_row_with_no_bundle_marker():
    audit = _audit_module()
    assert audit._bundle_fields({"record_id": "r1"}) == []
