"""Synthetic tests for the read-only EP24 Mongo export (#306). No database required."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import ep24_mongo_export as export


def row(**overrides):
    base = {"_storage_id": "s1", "allas_filename": "a.mp4", "video_id": "v1", "country": "finland"}
    base.update(overrides)
    return base


class TestIdentity:
    def test_media_key_prefers_allas_filename(self):
        assert export.media_key(row()) == "a.mp4"
        assert export.media_key(row(allas_filename="")) == "s1"
        assert export.media_key(row(allas_filename="", _storage_id="")) == "v1"
        assert export.media_key({}) == ""

    def test_video_and_ingest_units_are_namespaced_apart(self):
        assert export.record_identity(row(), "video") == "video:a.mp4"
        assert export.record_identity(row(), "ingest") == "ingest:s1"
        assert export.record_identity(row(), "video") != export.record_identity(row(), "ingest")

    def test_unknown_unit_is_rejected(self):
        with pytest.raises(export.ExportError):
            export.record_identity(row(), "episode")

    def test_a_row_with_no_id_is_quarantined_not_invented(self):
        empty = {"allas_filename": "", "_storage_id": "", "video_id": ""}
        assert export.record_identity(empty, "video") == ""
        assert export.quarantine_reason(empty, "video") == "no_identity"
        assert export.quarantine_reason(row(), "video") is None


class TestStageScore:
    def test_empty_values_do_not_count(self):
        assert export.stage_score({}) == 0
        assert export.stage_score({"asr_transcript": "   "}) == 0
        assert export.stage_score({"asr_transcript": "hello"}) == 1

    def test_lists_and_dicts_count_only_when_non_empty(self):
        assert export.stage_score({"entities": []}) == 0
        assert export.stage_score({"entities": [{"x": 1}]}) == 1


class TestSelectCanonical:
    def test_one_row_per_video_and_the_most_complete_wins(self):
        rows = [
            row(_storage_id="s1", asr_transcript="t1"),
            row(_storage_id="s2", asr_transcript="t2", ocr_1="text"),
            row(_storage_id="s3"),
        ]
        kept, stats = export.select_canonical(rows, "video")
        assert len(kept) == 1
        assert kept[0]["_storage_id"] == "s2", "the row with the most stage output wins"
        assert stats["duplicates_collapsed"] == 2
        assert stats["records"] == 1

    def test_ingest_unit_keeps_every_run(self):
        rows = [
            row(_storage_id="s1", asr_transcript="t1"),
            row(_storage_id="s2", asr_transcript="t2"),
            row(_storage_id="s3"),
        ]
        kept, stats = export.select_canonical(rows, "ingest")
        assert len(kept) == 3
        assert stats["duplicates_collapsed"] == 0
        assert stats["distinct_media"] == 1

    def test_selection_is_deterministic_regardless_of_input_order(self):
        rows = [row(_storage_id=f"s{i}", asr_transcript="x") for i in range(5)]
        first, _ = export.select_canonical(rows, "video")
        second, _ = export.select_canonical(list(reversed(rows)), "video")
        assert first == second
        assert first[0]["_storage_id"] == "s0", "ties break on storage id, not cursor order"

    def test_quarantined_rows_are_counted(self):
        rows = [row(), {"allas_filename": "", "_storage_id": "", "video_id": ""}]
        kept, stats = export.select_canonical(rows, "video")
        assert len(kept) == 1
        assert stats["quarantined"] == 1


class TestTextSplit:
    def test_long_strings_leave_the_row_and_short_ones_stay(self):
        lean, heavy = export.split_heavy_fields(
            {"record_id": "video:a", "short": "abc", "long": "x" * 20}, threshold=10)
        # The marker is deliberate (#331): a cell that moved must not look like a cell that was never
        # written. Everything else about the split is unchanged.
        assert lean == {"record_id": "video:a", "short": "abc",
                        "heavy_fields_in_text_bundle": "long"}
        assert heavy == {"long": "x" * 20}

    def test_the_bundle_is_gzipped_jsonl_keyed_by_record_id(self, tmp_path):
        import gzip
        import json
        path = tmp_path / "text.jsonl.gz"
        rows = [{"record_id": "video:a", "asr_transcript": "y" * 50},
                {"record_id": "video:b", "small": 1}]
        assert export.write_text_bundle(path, rows, threshold=10) == 1
        line = gzip.open(path, "rt", encoding="utf-8").read().strip()
        payload = json.loads(line)
        assert payload["record_id"] == "video:a"
        assert payload["asr_transcript"] == "y" * 50


class TestOutput:
    def test_csv_columns_are_sorted_and_json_cells_are_stable(self, tmp_path: Path):
        rows = [{"b": 1, "a": {"y": 1, "x": 2}}, {"a": None, "b": 2}]
        path = tmp_path / "out" / "rows.csv"
        assert export.write_rows(path, rows) == 2
        text = path.read_text(encoding="utf-8")
        assert text.splitlines()[0] == "a,b"
        assert '"{""x"": 2, ""y"": 1}"' in text, "dict cells are JSON with sorted keys"

    def test_project_row_drops_mongo_id_only(self):
        projected = export.project_row({"_id": "abc", "keep": 1})
        assert projected == {"keep": 1}

    def test_manifest_states_the_unit_and_both_counts(self, tmp_path: Path):
        manifest = export.write_manifest(
            tmp_path,
            record_unit="video",
            countries={"finland": {"records": 3, "distinct_media": 2}},
            child_counts={"finland": {"entities": 10, "sna": 0}},
            collections_read={"finland": {"canonical": "c", "sna": "absent"}},
            provenance={"finland": {"pipeline_markers": [], "newest_timestamp": None}},
            outputs=[],
        )
        on_disk = json.loads((tmp_path / "MANIFEST.json").read_text(encoding="utf-8"))
        assert on_disk["record_unit"] == "video"
        assert on_disk["records"] == 3
        assert on_disk["distinct_media"] == 2
        assert on_disk["read_only"] is True
        assert on_disk["children"]["finland"]["sna"] == 0, "an absent collection is a zero, not a crash"
        assert manifest["records"] == 3


class TestConfig:
    def test_missing_config_raises_without_echoing_anything(self):
        with pytest.raises(export.ExportError) as excinfo:
            export.config_from_env({"LACLAUGPT_MONGODB_URI": "", "LACLAUGPT_MONGODB_DATABASE": ""})
        assert "LACLAUGPT_MONGODB_URI" in str(excinfo.value)

    def test_config_reads_env(self):
        uri, db = export.config_from_env({
            "LACLAUGPT_MONGODB_URI": "mongodb://example", "LACLAUGPT_MONGODB_DATABASE": "ep24"})
        assert uri == "mongodb://example" and db == "ep24"


class TestCollections:
    def test_collection_naming(self):
        assert export.collection_name("finland", "_dataframe") == \
            "laclaugpt_ep2024_reprocess_finland_dataframe"
        assert export.child_collection("poland", "dna_statements") == \
            "laclaugpt_ep2024_reprocess_poland_dna_statements"

    def test_unknown_country_and_role_raise(self):
        with pytest.raises(export.ExportError):
            export.collection_name("sweden")
        with pytest.raises(export.ExportError):
            export.child_collection("finland", "nonsense")


class TestDashboardContract:
    """The keys the researcher dashboard (#195) and the coverage report (#314) actually read."""

    def test_coverage_counts_output_and_names_the_status_that_failed(self):
        rows = [
            {"record_id": "video:a.mp4", "laclau_status": "done",
             "laclau_structured_json": '{"nodal_points": ["kansa"]}'},
            {"record_id": "video:b.mp4", "laclau_status": "error", "laclau_structured_json": ""},
        ]
        coverage = export.stage_coverage(rows)
        assert coverage["6_discourse"]["records"] == 1
        assert coverage["6_discourse"]["failed_or_other"] == 1
        assert coverage["6_discourse"]["available"] is True
        assert coverage["6_discourse"]["status_field"] == "laclau_status"

    def test_a_step_the_source_cannot_express_is_unavailable_not_zero(self):
        rows = [{"record_id": "video:a.mp4", "asr_transcript": "puhetta"}]
        coverage = export.stage_coverage(rows)
        assert coverage["8_sna"]["records"] == 0
        assert coverage["8_sna"]["available"] is False
        assert coverage["1_preprocess"]["available"] is True

    def test_the_contract_uses_the_names_the_dashboard_reads(self):
        stats = {"finland": {"records": 2, "duplicates_collapsed": 3, "quarantined": 1,
                            "distinct_media": 2, "source_documents": 6}}
        rows = {"finland": [{"record_id": "video:a.mp4", "laclau_status": "done",
                             "laclau_structured_json": "{}"}]}
        contract = export.dashboard_contract(stats, rows)
        assert contract["record_counts"]["finland"] == {
            "read": 6, "exported": 2, "duplicates": 3, "invalid_ids": 1, "distinct_media": 2}
        assert contract["record_counts"]["exported_total"] == 2
        assert contract["inputs"]["finland"]["profile"] == "mongo"
        assert "8_sna" in contract["inputs"]["finland"]["stage_unavailable"]
        assert contract["stage_coverage"]["finland"]["6_discourse"]["records"] == 1

    def test_the_manifest_carries_the_contract_without_losing_its_own_provenance(self, tmp_path):
        stats = {"finland": {"records": 1, "duplicates_collapsed": 0, "quarantined": 0,
                             "distinct_media": 1, "source_documents": 1}}
        manifest = export.write_manifest(
            tmp_path, record_unit="video", countries=stats, child_counts={},
            collections_read={"finland": {"canonical": "c"}}, provenance={"finland": {}},
            outputs=[], contract=export.dashboard_contract(
                stats, {"finland": [{"record_id": "video:a.mp4", "asr_transcript": "x"}]}))
        assert manifest["read_only"] is True
        assert manifest["collections_read"]["finland"]["canonical"] == "c"
        assert manifest["record_counts"]["finland"]["exported"] == 1
        assert manifest["stage_coverage"]["finland"]["1_preprocess"]["records"] == 1
        on_disk = json.loads((tmp_path / "MANIFEST.json").read_text(encoding="utf-8"))
        assert on_disk["inputs"]["finland"]["stage_unavailable"]


class TestIterDocumentsResumes:
    """A long read must survive the server closing an idle cursor."""

    class Collection:
        def __init__(self, documents, fail_after=None):
            self.documents = list(documents)
            self.fail_after = fail_after
            self.calls = []

        def find(self, query, batch_size=0):
            self.calls.append(query)
            start = 0
            if query:
                threshold = query["_id"]["$gt"]
                start = next(i for i, d in enumerate(self.documents) if d["_id"] == threshold) + 1
            documents = self.documents[start:]

            class Cursor:
                def __init__(self, docs, fail_after):
                    self.docs = docs
                    self.fail_after = fail_after
                    self.no_cursor_timeout = False

                def sort(self, *_args):
                    return self

                def __iter__(self):
                    for index, doc in enumerate(self.docs):
                        if self.fail_after is not None and index == self.fail_after:
                            raise type("_OperationCancelled", (Exception,), {})("operation cancelled")
                        yield doc

            return Cursor(documents, self.fail_after)

    def test_a_cancelled_cursor_resumes_from_the_last_id(self):
        import importlib.util
        import sys as _sys
        spec = importlib.util.spec_from_file_location(
            "export_mongo_results", Path(__file__).resolve().parents[1]
            / "scripts" / "ep24" / "export_mongo_results.py")
        module = importlib.util.module_from_spec(spec)
        _sys.modules["export_mongo_results"] = module
        spec.loader.exec_module(module)

        documents = [{"_id": i, "v": i} for i in range(5)]
        collection = self.Collection(documents, fail_after=2)
        got = list(module.iter_documents(collection, batch_size=2, limit=None, retries=3))
        assert [d["_id"] for d in got] == [0, 1, 2, 3, 4]
        assert collection.calls[1] == {"_id": {"$gt": 1}}


class TestChildTableContract:
    """#330/#331: child tables are ingest-level and must expose a deterministic video parent."""

    def parents(self):
        return [
            {"record_id": "video:a.mp4", "_storage_id": "s1", "video_id": "v1",
             "allas_filename": "a.mp4"},
            {"record_id": "video:b.mp4", "_storage_id": "s2", "video_id": "v2",
             "allas_filename": "b.mp4"},
        ]

    def children(self):
        return [
            {"_storage_id": "st1", "source_record_id": "s1", "statement_id": "st1"},
            {"_storage_id": "st2", "source_record_id": "missing"},
            {"_storage_id": "st1", "source_record_id": "s1"},
        ]

    def test_children_get_the_exported_video_key_not_the_statement_id(self):
        joined = export.attach_child_parents(self.children(), self.parents())
        assert [row["parent_record_id"] for row in joined] == [
            "video:a.mp4", "", "video:a.mp4"]
        assert [row["parent_join"] for row in joined] == [
            "source_record_id", "unjoined", "source_record_id"]

    def test_the_statement_id_survives_and_is_not_used_as_the_parent(self):
        joined = export.attach_child_parents(self.children(), self.parents())
        assert joined[0]["_storage_id"] == "st1"
        assert joined[0]["parent_record_id"] != joined[0]["_storage_id"]

    def test_unjoined_children_are_counted_not_dropped(self):
        joined = export.attach_child_parents(self.children(), self.parents())
        contract = export.child_contract(joined, country="finland", role="dna_statements")
        assert contract["rows"] == 3
        assert contract["joined_rows"] == 2
        assert contract["unjoined_rows"] == 1
        assert contract["duplicate_statement_ids"] == 1
        assert contract["parent_key_field"] == "parent_record_id"

    def test_missing_source_record_id_is_an_orphan(self):
        joined = export.attach_child_parents(
            [{"_storage_id": "st9", "statement_id": "st9"}], self.parents())
        assert joined[0]["parent_join"] == "unjoined"
        assert joined[0]["parent_record_id"] == ""


class TestNormalizedCoverage:
    """#332: coverage must read the count that survives a child-table export."""

    def test_statement_count_makes_dna_measurable_when_the_json_is_blank(self):
        rows = [{"dna_statement_count": 3, "dna_statements_json": ""},
                {"dna_statement_count": 0, "dna_statements_json": ""}]
        coverage = export.stage_coverage(rows)
        assert coverage["7_dna"]["available"] is True
        assert coverage["7_dna"]["records"] == 1
        assert coverage["7_dna"]["pct_of_exported"] == 50.0

    def test_coverage_matches_the_rows_that_report_statements(self):
        rows = [{"dna_statement_count": n} for n in (0, 2, 0, 5)]
        coverage = export.stage_coverage(rows)
        assert coverage["7_dna"]["records"] == sum(
            1 for row in rows if row["dna_statement_count"] > 0)

    def test_truthy_zero_count_is_not_output(self):
        assert export._has_output(0) is False
        assert export._has_output(1) is True
        assert export._has_output("[]") is True
        assert export._has_output("{}") is True
        assert export._has_output("0") is False
