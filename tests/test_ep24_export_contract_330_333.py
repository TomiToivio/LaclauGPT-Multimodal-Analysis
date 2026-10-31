"""Synthetic export contract regression checks for #330, #331, #332 and #333."""
import ep24_mongo_export as export


def test_dna_coverage_uses_positive_count_without_embedded_json():
    rows = [
        {"dna_statement_count": 3, "dna_statements_json": "", "dna_analysis_markdown": ""},
        {"dna_statement_count": 0, "dna_statements_json": "[]", "dna_analysis_markdown": ""},
        {"dna_statement_count": 0, "dna_statements_json": "", "dna_analysis_markdown": ""},
    ]
    result = export.stage_coverage(rows)["7_dna"]
    assert result["available"] is True
    assert result["records"] == 1


def test_heavy_fields_are_accounted_for_without_losing_identity():
    rows = [{"record_id": "video:alpha", "dna_statements_json": "x" * 1100},
            {"record_id": "video:beta", "dna_statements_json": "[]"}]
    assert export.heavy_field_counts(rows, threshold=1000) == {"dna_statements_json": 1}
    lean, heavy = export.split_heavy_fields(rows[0], threshold=1000)
    assert lean["record_id"] == "video:alpha"
    assert heavy["dna_statements_json"] == "x" * 1100


def test_canonical_video_parent_join_rejects_collapsed_ingest_rows():
    selected = [{"_storage_id": "ingest-current", "record_id": "video:a"}]
    children = [
        {"_storage_id": "statement-one", "source_record_id": "ingest-current", "statement_id": "one"},
        {"_storage_id": "statement-old", "source_record_id": "ingest-previous", "statement_id": "old"},
    ]
    matched, qa = export.select_parent_children(children, selected, country="finland")
    assert qa["matched"] == 1
    assert qa["excluded_unmatched"] == 1
    assert matched[0]["parent_record_id"] == "video:a"
    assert matched[0]["_storage_id"] == "statement-one"



def test_join_quarantines_ambiguous_missing_duplicate_and_foreign_keys():
    parents = [{"_storage_id": "a", "record_id": "video:a"},
               {"_storage_id": "amb", "record_id": "video:b"},
               {"_storage_id": "amb", "record_id": "video:c"}]
    rows = [{"source_record_id": "a", "statement_id": "dup"}] * 2 + [
        {"source_record_id": "amb", "statement_id": "s"},
        {"source_record_id": "a", "statement_id": "foreign", "country": "pl"},
        {"statement_id": "missing-parent"}, {"source_record_id": "a"},
        {"source_record_id": "a", "statement_id": "ok", "country": "fi"}]
    matched, qa = export.select_parent_children(rows, parents, country="finland")
    assert [r["statement_id"] for r in matched] == ["ok"]
    assert qa["duplicate_statement_id"] == 2
    assert all(qa[k] == 1 for k in ("ambiguous_parent", "cross_country", "missing_parent_key", "missing_statement_id"))


def test_long_identity_never_moves_to_bundle():
    key = "video:" + "a" * 2000
    lean, heavy = export.split_heavy_fields({"record_id": key, "summary": "x" * 2000})
    assert lean["record_id"] == key
    assert "record_id" not in heavy


def test_status_matrix_does_not_infer_completion_or_failure():
    rows = [{"laclau_status": status, "laclau_structured_json": "{}"}
            for status in ("done", "pending", "retry", "skipped", "failed", "running", "novel")]
    rows += [{}, {"laclau_structured_json": '{"claim":"synthetic"}'}]
    counts = export.stage_states(rows)["6_discourse"]["counts"]
    assert sum(counts.values()) == len(rows)
    assert counts["complete_but_empty"] == counts["not_run"] == counts["retry"] == 1
    assert counts["output_status_unknown"] == counts["unavailable"] == 1


def test_cli_normalized_layout_roundtrip(tmp_path, monkeypatch):
    import csv
    import gzip
    import importlib.util
    import json
    import sys
    import types
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("export_contract_cli", Path(__file__).resolve().parents[1] / "scripts/ep24/export_mongo_results.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    collections = {}
    for country in export.COUNTRY_SLUGS:
        # Deliberately reuse identities across countries. Join must include country.
        collections[export.collection_name(country, "_dataframe")] = [
            {"_storage_id": "selected", "allas_filename": "same.mp4", "summary": "x" * 1100,
             "dna_statement_count": 1, "dna_statements_json": "x" * 1100},
            {"_storage_id": "old", "allas_filename": "same.mp4"}]
        collections[export.child_collection(country, "dna_statements")] = [
            {"_storage_id": "statement", "statement_id": "s", "source_record_id": "selected",
             "country": country, "evidence_quote": "Synthetic quote", "provenance": {"test": True}},
            {"statement_id": "old", "source_record_id": "old"}]

    class Cursor(list):
        def sort(self, *args):
            return self
    class Collection:
        def __init__(self, rows):
            self.rows = rows
        def find(self, *args, **kwargs):
            return Cursor(self.rows)
    class Database:
        def list_collection_names(self):
            return list(collections)
        def __getitem__(self, name):
            return Collection(collections[name])
    class Client:
        def __init__(self, *args, **kwargs):
            pass
        def __getitem__(self, name):
            return Database()
    monkeypatch.setitem(sys.modules, "pymongo", types.SimpleNamespace(MongoClient=Client))
    monkeypatch.setenv("LACLAUGPT_MONGODB_URI", "synthetic")
    monkeypatch.setenv("LACLAUGPT_MONGODB_DATABASE", "synthetic")
    assert cli.main(["--output", str(tmp_path), "--children", "dna_statements"]) == 0
    manifest = json.loads((tmp_path / "MANIFEST.json").read_text())
    for country in export.COUNTRY_SLUGS:
        with (tmp_path / f"{country}.csv").open() as f:
            parents = list(csv.DictReader(f))
        with (tmp_path / "children" / f"{country}_dna_statements.csv").open() as f:
            children = list(csv.DictReader(f))
        with gzip.open(tmp_path / f"{country}_text.jsonl.gz", "rt") as f:
            bundle = [json.loads(line) for line in f]
        # The parent CSV is per-country and carries no `country` column: the country is the
        # file's, not the row's. Reading it from the row raised KeyError and blocked this test.
        parent_keys = {(country, r["record_id"]) for r in parents}
        assert all((r["parent_country"], r["parent_record_id"]) in parent_keys for r in children)
        assert len(children) == manifest["children"][country]["dna_statements"] == 1
        assert manifest["stage_coverage"][country]["7_dna"]["records"] == 1
        assert manifest["stage_coverage"][country]["7_dna"]["parent_count_mismatches"] == 0
        assert manifest["heavy_fields"][country]["summary"] == len(bundle) == 1
        assert bundle[0]["record_id"] == parents[0]["record_id"]
        assert "summary" in json.loads(parents[0]["_text_bundle_fields"])
    assert len(manifest["outputs"]) == 10
    audit_spec = importlib.util.spec_from_file_location(
        "audit_contract_cli", Path(__file__).resolve().parents[1] / "scripts/ep24/audit_export.py")
    audit = importlib.util.module_from_spec(audit_spec)
    audit_spec.loader.exec_module(audit)
    assert audit.main(["--export", str(tmp_path), "--quiet"]) == 0


def test_child_coverage_distinguishes_absent_empty_and_counter_mismatch():
    parents = [{"record_id": "video:a", "dna_statement_count": 3}]
    coverage = export.stage_coverage(parents)
    export.reconcile_dna_coverage(coverage, [], parents, available=True)
    assert coverage["7_dna"]["records"] == 0
    assert coverage["7_dna"]["available"] is True
    assert coverage["7_dna"]["source_output_records"] == 1
    assert coverage["7_dna"]["parent_count_mismatches"] == 1
    coverage = export.stage_coverage(parents)
    export.reconcile_dna_coverage(coverage, [], parents, available=False)
    assert coverage["7_dna"]["available"] is False
