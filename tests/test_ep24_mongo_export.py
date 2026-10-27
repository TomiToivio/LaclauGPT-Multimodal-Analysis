"""Issue #306 section 1: the read-only EP24 MongoDB export contract.

Synthetic documents only — this repository never holds research data. The tests pin what makes the
export trustworthy rather than merely runnable:

* read-only access (no write path), and the URI never reaches the manifest;
* one row per video, most complete run wins, with the collapse counted;
* unusable documents quarantined and counted, never given a synthetic id;
* a step that the source cannot carry is ``available: false``, not ``records: 0``;
* the artifact names and manifest shape are the #195 contract, so an existing dashboard can read
  the export without a second schema.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ep24_mongo_export as export  # noqa: E402


def doc(**overrides) -> dict:
    base = {
        "video_id": "v1", "_storage_id": "s1", "allas_filename": "https://a3s.fi/a/v1.mp4",
        "source_type": "tiktok", "country": "finland",
        "processing_status": "done", "preprocess_status": "done",
        "asr_transcript": "puhutaan populismista", "frame_analysis_status": "done",
        "frame_analysis_1": "kuvaus", "vllm_video_status": "done",
        "vllm_video_analysis": "video kertoo", "summary_quality_status": "done",
        "summary_analysis": "yhteenveto", "postprocess_entities": "[]",
        "laclau_status": "done", "laclau_structured_json": '{"nodal_points": ["kansa"]}',
        "dna_statements_json": "[]", "sna_nodes_json": "[]", "sna_edges_json": "[]",
    }
    base.update(overrides)
    return base


class FakeCollection:
    def __init__(self, name: str, docs: list[dict]):
        self.name = name
        self._docs = docs
        self.writes = 0

    def find(self, query=None, projection=None, batch_size=0):
        # Mirror pymongo's signature exactly: (filter, projection). Recording both is what lets a
        # test prove the projection is not being passed where the filter belongs.
        self.last_query, self.last_projection = query, projection
        return iter(self._docs)

    def insert_one(self, *_args, **_kwargs):  # pragma: no cover - must never be called
        self.writes += 1
        raise AssertionError("the export must not write")


class FakeDatabase:
    def __init__(self, collections: dict[str, list[dict]]):
        self.name = "fake_db"
        self._collections = {name: FakeCollection(name, docs)
                             for name, docs in collections.items()}

    def list_collection_names(self) -> list[str]:
        return list(self._collections)

    def __getitem__(self, name: str) -> FakeCollection:
        return self._collections[name]

    @property
    def collections(self) -> dict[str, FakeCollection]:
        return self._collections


class ConnectionTests(unittest.TestCase):
    def test_both_uri_spellings_are_accepted_and_only_the_variable_name_is_returned(self) -> None:
        uri, database, source = export.connection_settings(
            {"LACLAUGPT_MONGODB_URI": "mongodb://user:secret@host/db", "LACLAUGPT_MONGODB_DATABASE": "x"})
        self.assertEqual(uri, "mongodb://user:secret@host/db")
        self.assertEqual((database, source), ("x", "LACLAUGPT_MONGODB_URI"))

        _, _, producer_source = export.connection_settings(
            {"LACLAUGPT_MONGO_URI": "mongodb://host/db"})
        self.assertEqual(producer_source, "LACLAUGPT_MONGO_URI")

    def test_a_missing_uri_is_a_clear_error(self) -> None:
        with self.assertRaises(export.ExportError):
            export.connection_settings({})

    def test_collection_names_come_from_the_producer_naming_contract(self) -> None:
        self.assertEqual(export.collection_name("finland", "dataframe"),
                         "laclaugpt_ep2024_reprocess_finland_dataframe")
        self.assertEqual(export.collection_name("poland", "dna_statements"),
                         "lclauGPT".replace("lclauGPT", "laclaugpt_ep2024_reprocess_poland_dna_statements"))


class IdentityTests(unittest.TestCase):
    def test_a_document_needs_a_video_id_and_a_media_key(self) -> None:
        self.assertTrue(export.has_usable_identity(doc()))
        self.assertFalse(export.has_usable_identity(doc(video_id="")))
        self.assertFalse(export.has_usable_identity(doc(allas_filename="", _storage_id="")))
        self.assertEqual(export.record_key(doc(video_id=""), "fi"), "")

    def test_identity_is_country_scoped(self) -> None:
        fi = export.record_key(doc(country="finland"), "fi")
        pl = export.record_key(doc(country="finland"), "pl")
        self.assertTrue(fi.startswith("fi/v1#"))
        self.assertNotEqual(fi, pl)

    def test_two_segments_of_one_file_stay_apart(self) -> None:
        first = export.record_key(doc(sequence_number="1"), "fi")
        second = export.record_key(doc(sequence_number="2"), "fi")
        self.assertNotEqual(first, second)


class RowTests(unittest.TestCase):
    def test_the_most_complete_run_wins_and_the_collapse_is_counted(self) -> None:
        rows, stats, _columns = export.canonical_rows([
            doc(laclau_structured_json="", summary_analysis=""),
            doc(_storage_id="s2", summary_analysis="täydempi"),
        ], "fi")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["summary_analysis"], "täydempi")
        self.assertEqual((stats["read"], stats["duplicates"], stats["invalid_ids"]), (2, 1, 0))

    def test_unusable_documents_are_counted_and_absent_from_the_rows(self) -> None:
        rows, stats, _columns = export.canonical_rows([doc(), doc(video_id="", allas_filename="")], "fi")
        self.assertEqual(len(rows), 1)
        self.assertEqual(stats["invalid_ids"], 1)

    def test_a_row_carries_every_stage_field_the_contract_names(self) -> None:
        rows, _stats, _columns = export.canonical_rows([doc()], "fi")
        for field in export.stage_fields():
            self.assertIn(field, rows[0])

    def test_json_values_become_text_but_stay_machine_readable(self) -> None:
        rows, _stats, _columns = export.canonical_rows([doc(laclau_structured_json={"nodal_points": ["kansa"]})], "fi")
        self.assertEqual(json.loads(rows[0]["laclau_structured_json"]), {"nodal_points": ["kansa"]})


class CoverageTests(unittest.TestCase):
    def test_a_step_with_output_is_present(self) -> None:
        rows, _stats, _columns = export.canonical_rows([doc()], "fi")
        coverage = export.stage_coverage(rows)
        self.assertGreater(coverage["6_discourse"]["records"], 0)
        self.assertTrue(coverage["6_discourse"]["available"])

    def test_a_failed_status_is_counted_as_failed_not_as_coverage(self) -> None:
        rows, _stats, _columns = export.canonical_rows([doc(laclau_status="failed", laclau_structured_json="")], "fi")
        coverage = export.stage_coverage(rows)
        self.assertEqual(coverage["6_discourse"]["records"], 0)
        self.assertEqual(coverage["6_discourse"]["failed_or_other"], 1)


class ExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.output = Path(self._tmp.name) / "out"
        self.db = FakeDatabase({
            export.collection_name("finland", "dataframe"): [
                doc(), doc(_storage_id="s2", summary_analysis=""), doc(video_id=""),
            ],
            export.collection_name("finland", "dna_statements"): [{"statement_id": "d1"}],
            # The Polish frame deliberately carries no SNA columns: the step cannot be measured there.
            export.collection_name("poland", "dataframe"): [
                {k: v for k, v in doc(video_id="p1").items() if not k.startswith("sna_")}],
        })

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_the_artifacts_are_the_contract_a_dashboard_already_reads(self) -> None:
        export.export_countries(self.db, ["fi", "pl"], self.output, uri_source="TEST_URI_VAR")
        for name in ("fi.csv", "pl.csv", "combined.csv", "records.jsonl", "MANIFEST.json",
                     "DATA_DICTIONARY.md", "dna_statements.csv", "sna_nodes.csv"):
            self.assertTrue((self.output / name).is_file(), name)
        manifest = json.loads((self.output / "MANIFEST.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["record_unit"], "video")
        self.assertIn("exported", manifest["record_counts"]["fi"])
        self.assertIn("6_discourse", manifest["stage_coverage"]["fi"])
        self.assertIn("stage_unavailable", manifest["inputs"]["fi"])

    def test_the_uri_never_reaches_the_manifest(self) -> None:
        export.export_countries(self.db, ["fi"], self.output, uri_source="LACLAUGPT_MONGODB_URI")
        text = (self.output / "MANIFEST.json").read_text(encoding="utf-8")
        self.assertIn("LACLAUGPT_MONGODB_URI", text)
        self.assertNotIn("mongodb://", text)

    def test_a_country_without_collections_is_unavailable_not_empty(self) -> None:
        export.export_countries(self.db, ["pt"], self.output, uri_source="X")
        manifest = json.loads((self.output / "MANIFEST.json").read_text(encoding="utf-8"))
        self.assertFalse(manifest["inputs"]["pt"]["available"])
        self.assertEqual(manifest["stage_coverage"]["pt"]["6_discourse"]["records"], 0)
        self.assertFalse(manifest["stage_coverage"]["pt"]["6_discourse"]["available"])

    def test_a_step_with_no_column_anywhere_is_marked_unavailable(self) -> None:
        # The Polish frame here carries no SNA columns at all.
        export.export_countries(self.db, ["pl"], self.output, uri_source="X")
        manifest = json.loads((self.output / "MANIFEST.json").read_text(encoding="utf-8"))
        self.assertIn("8_sna", manifest["inputs"]["pl"]["stage_unavailable"])

    def test_the_export_never_writes_to_the_database(self) -> None:
        export.export_countries(self.db, ["fi", "pl"], self.output, uri_source="X")
        self.assertTrue(all(collection.writes == 0 for collection in self.db.collections.values()))

    def test_the_projection_is_sent_as_a_projection_not_as_a_filter(self) -> None:
        db = FakeDatabase({export.collection_name("finland", "dataframe"): [doc()]})
        output = Path(self._tmp.name) / "proj"
        export.export_countries(db, ["fi"], output, uri_source="X")
        collection = db.collections[export.collection_name("finland", "dataframe")]
        self.assertEqual(collection.last_query, {})
        self.assertIn("allas_filename", collection.last_projection)
        self.assertNotIn("_id", collection.last_projection)

    def test_a_country_row_file_is_not_empty_when_the_source_has_rows(self) -> None:
        db = FakeDatabase({export.collection_name("finland", "dataframe"): [doc()]})
        output = Path(self._tmp.name) / "nonempty"
        export.export_countries(db, ["fi"], output, uri_source="X")
        self.assertGreater(len((output / "fi.csv").read_text(encoding="utf-8").splitlines()), 1)

    def test_child_collections_are_flattened_with_provenance(self) -> None:
        export.export_countries(self.db, ["fi"], self.output, uri_source="X")
        header = (self.output / "dna_statements.csv").read_text(encoding="utf-8").splitlines()[0]
        self.assertIn("statement_id", header)
        self.assertIn("country", header)


if __name__ == "__main__":
    unittest.main()
