"""#342: the retry planner must select failed/missing work and never disturb a good result."""
from __future__ import annotations

import ep24_mongo_export as exporter
from scripts.ep24.plan_ep24_retry import select


def _row(record_id, **kw):
    base = {"record_id": record_id, "country": "finland"}
    for step in exporter.COVERAGE_STEPS:
        status_field, outputs = exporter.COVERAGE_STEPS[step]
        if status_field:
            base.setdefault(status_field, "")
        for field in outputs:
            base.setdefault(field, "")
    base.update(kw)
    return base


def test_a_failed_status_is_selected():
    row = _row("v1", preprocess_status="failed")
    plan = select([row], ("1_preprocess",))
    assert plan["1_preprocess"]["missing"] == ["v1"]


def test_a_record_with_output_is_reported_but_not_retried():
    """Output is evidence: a failure status on top of real output is a human question, not a rerun."""
    row = _row("v1", preprocess_status="needs_review", asr_transcript="hello")
    plan = select([row], ("1_preprocess",))
    assert plan["1_preprocess"]["missing"] == []
    assert plan["1_preprocess"]["failed_with_output"] == ["v1"]


def test_a_record_with_no_output_is_selected():
    row = _row("v1")
    plan = select([row], ("6_discourse",))
    assert plan["6_discourse"]["missing"] == ["v1"]


def test_a_zero_counter_does_not_count_as_output():
    """The #332 semantics, reused: dna_statement_count=0 is not an output."""
    row = _row("v1", laclau_status="done", dna_statement_count=0)
    plan = select([row], ("7_dna",))
    assert plan["7_dna"]["missing"] == ["v1"]


def test_a_successful_record_is_left_alone():
    row = _row("v1", dna_statement_count=4, dna_status="done")
    plan = select([row], ("7_dna",))
    assert plan["7_dna"]["missing"] == []


def test_ids_are_deduplicated():
    rows = [_row("v1"), _row("v1"), _row("v2")]
    plan = select(rows, ("6_discourse",))
    assert plan["6_discourse"]["missing"] == ["v1", "v2"]


def test_elided_fields_are_country_scoped(tmp_path):
    import json
    import gzip
    from scripts.ep24.plan_ep24_retry import elided_fields

    (tmp_path / "MANIFEST.json").write_text(json.dumps({
        "heavy_fields": {"finland": ["asr_transcript"], "poland": ["vllm_video_analysis"]}
    }), encoding="utf-8")
    with gzip.open(tmp_path / "portugal_text.jsonl.gz", "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({"record_id": "p1", "video_summary": "available"}) + "\n")
    assert elided_fields(tmp_path, "finland") == {"asr_transcript"}
    assert elided_fields(tmp_path, "poland") == {"vllm_video_analysis"}
    assert elided_fields(tmp_path, "portugal") == {"video_summary"}
    assert "vllm_video_analysis" not in elided_fields(tmp_path, "finland")


def test_country_scoped_retry_plan_does_not_hide_missing_output(tmp_path):
    import csv
    import json
    from scripts.ep24.plan_ep24_retry import plan_for_export

    (tmp_path / "MANIFEST.json").write_text(json.dumps({
        "heavy_fields": {"poland": ["vllm_video_analysis"]}
    }), encoding="utf-8")
    with (tmp_path / "combined.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["record_id", "country", "vllm_video_analysis"])
        writer.writeheader()
        writer.writerow({"record_id": "fi1", "country": "finland", "vllm_video_analysis": ""})
        writer.writerow({"record_id": "pl1", "country": "poland", "vllm_video_analysis": ""})
    result = plan_for_export(tmp_path, ("3_video",), None)
    assert result["countries"]["finland"]["steps"]["3_video"]["missing"] == 1
    assert result["countries"]["finland"]["steps"]["3_video"]["unverifiable"] == 0
    assert result["countries"]["poland"]["steps"]["3_video"]["missing"] == 0
    assert result["countries"]["poland"]["steps"]["3_video"]["unverifiable"] == 1
