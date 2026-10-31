"""The export contract a consumer can rely on (#330, #331, #332).

Three failures motivated these tests, all observed on a real `ep24-mongo-export-1` artifact set:

* child tables are exported at pipeline-run level, and their `_storage_id` is the *statement's* id,
  so the obvious join returns nothing and a naive one double counts (1,583 ingest-level statements
  against 1,490 attached to a video-level parent);
* cells longer than the text threshold leave `combined.csv` silently, so a stage that produced
  output looks empty to anyone reading only the CSV;
* coverage and the manifest's child counts describe the same step in different units without saying
  so, so two files of one export give two answers.

Nothing here invents data: orphans keep `parent_record_id == ""` and are counted, an elided field is
declared in `heavy_fields`, and a disagreement becomes a `contract_warnings` entry rather than a
silently corrected number.
"""
from __future__ import annotations

import json
from pathlib import Path

import ep24_mongo_export as export


def parent(record_id: str, storage: str) -> dict:
    return {"record_id": record_id, "_storage_id": storage, "allas_filename": f"{record_id}.mp4"}


PARENTS = [parent("video:a.mp4", "hash-a"), parent("video:b.mp4", "hash-b")]


class TestChildParentIdentity:
    def test_child_source_record_id_attaches_to_the_video_parent(self):
        children = [
            {"_storage_id": "stmt-1", "statement_id": "stmt-1", "source_record_id": "hash-a"},
            {"_storage_id": "stmt-2", "statement_id": "stmt-2", "source_record_id": "hash-a"},
            {"_storage_id": "stmt-3", "statement_id": "stmt-3", "source_record_id": "gone"},
        ]
        attached, counts = export.attach_parent_ids(children, PARENTS)
        assert [row["parent_record_id"] for row in attached] == ["video:a.mp4", "video:a.mp4", ""]
        assert counts == {"records": 3, "matched": 2, "orphan": 1}

    def test_statement_level_storage_id_is_not_treated_as_a_parent(self):
        # The child's own _storage_id is the statement id; matching it against a parent id would be
        # a coincidence, not a link.
        attached, counts = export.attach_parent_ids(
            [{"_storage_id": "stmt-9", "statement_id": "stmt-9"}], PARENTS)
        assert attached[0]["parent_record_id"] == ""
        assert counts["orphan"] == 1

    def test_parent_index_prefers_the_collapsed_video_identity(self):
        index = export.parent_index(PARENTS)
        assert index["hash-a"] == "video:a.mp4"
        assert index["video:a.mp4"] == "video:a.mp4"


def contract(**overrides) -> dict:
    """A minimal manifest shaped like the real one, then overridden per test."""
    base = {
        "record_unit": "video",
        "children_unit": "video",
        "record_counts": {"finland": {"exported": 2}},
        "stage_coverage": {
            "finland": {
                "7_dna": {"records": 2, "available": True},
                "8_sna": {"records": 0, "available": True},
            }
        },
        "children_detail": {
            "finland": {"dna_statements": {"records": 3, "matched": 2, "orphan": 1}}
        },
        "heavy_fields": {"finland": {"dna_statements_json": 2}},
        "text_bundle": {"suffix": "_text.jsonl.gz", "key": "record_id", "threshold": 1000},
    }
    base.update(overrides)
    return base


class TestCoverageReconciliation:
    def test_an_export_whose_children_reach_their_parents_is_quiet(self):
        assert export.coverage_reconciliation(contract()) == []

    def test_coverage_claiming_records_without_one_attached_child_is_reported(self):
        manifest = contract(children_detail={
            "finland": {"dna_statements": {"records": 3, "matched": 0, "orphan": 3}}})
        warnings = export.coverage_reconciliation(manifest)
        assert [w["kind"] for w in warnings] == ["coverage-without-children"]
        assert "not exported" in warnings[0]["detail"]

    def test_children_attached_while_coverage_claims_nothing_is_reported(self):
        manifest = contract(stage_coverage={"finland": {"7_dna": {"records": 0, "available": True}}})
        assert [w["kind"] for w in export.coverage_reconciliation(manifest)] == [
            "children-without-coverage"]

    def test_a_step_marked_unavailable_despite_proven_output_is_reported(self):
        manifest = contract(stage_coverage={
            "finland": {"7_dna": {"records": 0, "available": False}}},
            children_detail={"finland": {"dna_statements": {"records": 2, "matched": 2,
                                                            "orphan": 0}}})
        kinds = {w["kind"] for w in export.coverage_reconciliation(manifest)}
        assert "children-but-unavailable" in kinds


class TestCoverageCounters:
    def test_a_zero_statement_counter_is_not_output(self):
        rows = [{"dna_statement_count": "0"}, {"dna_statement_count": "3"}]
        coverage = export.stage_coverage(rows)["7_dna"]
        assert coverage["records"] == 1
        assert coverage["available"] is True

    def test_a_counter_survives_the_text_threshold_where_the_json_does_not(self):
        row = {"dna_statement_count": "4", "dna_statements_json": "x" * 5000}
        lean, heavy = export.split_heavy_fields(row)
        assert "dna_statements_json" in heavy and lean["dna_statement_count"] == "4"
        assert export.stage_coverage([row])["7_dna"]["records"] == 1


class TestManifestStatesTheContract:
    def test_split_and_children_units_are_written_into_the_manifest(self, tmp_path: Path):
        manifest = export.write_manifest(
            tmp_path,
            record_unit="video",
            countries={"finland": {"records": 2, "distinct_media": 2}},
            child_counts={"finland": {"dna_statements": 3}},
            collections_read={"finland": {"dna_statements": "c"}},
            provenance={},
            outputs=[],
            contract={"stage_coverage": {"finland": {"7_dna": {"records": 2, "available": True}}},
                      "record_counts": {"finland": {"exported": 2}}},
            child_detail={"finland": {"dna_statements": {"records": 3, "matched": 2, "orphan": 1}}},
            heavy_fields={"finland": {"dna_statements_json": 2}},
            text_threshold=1000,
        )
        assert manifest["children_unit"] == "video"
        assert manifest["children_detail"]["finland"]["dna_statements"]["orphan"] == 1
        assert manifest["heavy_fields"]["finland"]["dna_statements_json"] == 2
        assert manifest["text_bundle"] == {"suffix": "_text.jsonl.gz", "key": "record_id",
                                           "threshold": 1000}
        assert manifest["contract_warnings"] == []
        on_disk = json.loads((tmp_path / "MANIFEST.json").read_text(encoding="utf-8"))
        assert on_disk["children_detail"] == manifest["children_detail"]
