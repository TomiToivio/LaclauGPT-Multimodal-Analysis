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
    assert qa == {"matched": 1, "excluded_unmatched": 1}
    assert matched[0]["parent_record_id"] == "video:a"
    assert matched[0]["_storage_id"] == "statement-one"
