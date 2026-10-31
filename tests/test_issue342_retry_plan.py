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
