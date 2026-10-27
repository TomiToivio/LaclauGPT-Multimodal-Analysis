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
        assert lean == {"record_id": "video:a", "short": "abc"}
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
