"""Non-blocking stage eligibility and selective reprocessing (#342).

The contract these tests pin:

* usable output is **never** reselected, whatever a stale flag says;
* usable input with no output **is** selected, even when the predecessor is called partial;
* a stage that legitimately produced nothing is not rerun unless asked;
* the manifest is deterministic and contains no research content.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ep24_eligibility as el  # noqa: E402


def row(**fields) -> dict:
    return {"record_id": "r1", "country": "finland", **fields}


# --- status classification ---------------------------------------------------------------

def test_usable_output_is_success_and_never_retried() -> None:
    verdict = el.assess(row(asr_transcript="text"), 1)
    assert verdict.execution_status == el.SUCCEEDED
    assert verdict.output_available is True
    assert verdict.retry_eligible is False


def test_old_failure_flag_does_not_override_present_output() -> None:
    """The flag says failed; the artifact exists. The artifact wins."""
    verdict = el.assess(row(laclau_status="error", laclau_summary_md="summary"), 6)
    assert verdict.output_available is True
    assert verdict.retry_eligible is False


def test_failure_without_output_is_retried() -> None:
    verdict = el.assess(row(laclau_status="error"), 6)
    assert verdict.execution_status == el.FAILED
    assert verdict.retry_eligible is True


def test_success_without_output_is_partial_and_retry_is_opt_in() -> None:
    verdict = el.assess(row(summary_quality_status="ok"), 4)
    assert verdict.execution_status == el.PARTIAL
    assert verdict.retry_eligible is False  # could be a legitimately empty result
    opted_in = el.assess(row(summary_quality_status="ok"), 4, include_empty=True)
    assert opted_in.retry_eligible is True


def test_no_input_is_a_recorded_skip_not_a_retry() -> None:
    verdict = el.assess(row(), 3)  # no frame_file, no timestamps
    assert verdict.execution_status == el.SKIPPED
    assert verdict.retry_eligible is False
    assert "no usable input" in verdict.diagnostic


def test_usable_input_and_no_record_means_never_run() -> None:
    verdict = el.assess(row(asr_transcript="text"), 6)
    assert verdict.execution_status == el.NOT_RUN
    assert verdict.retry_eligible is True


# --- non-blocking inputs -----------------------------------------------------------------

def test_text_only_record_reaches_discourse_without_video() -> None:
    """The old gate demanded predecessor 'complete'; a transcript alone is enough for Step 6."""
    ok, note = el.usable_input(row(asr_transcript="a transcript"), 6)
    assert ok is True
    assert "asr_transcript" in note


def test_dna_can_run_on_partial_discourse() -> None:
    ok, _ = el.usable_input(row(laclau_summary_md="partial discourse"), 7)
    assert ok is True


def test_video_stage_requires_frames_not_merely_a_status() -> None:
    ok, _ = el.usable_input(row(frame_analysis_status="ok"), 3)
    assert ok is False
    ok, _ = el.usable_input(row(frame_file="f.jpg", frame_timestamp_seconds="12"), 3)
    assert ok is True


def test_downstream_usability_follows_partial_status() -> None:
    verdict = el.assess(row(summary_quality_status="REPROCESS", summary_analysis="half"), 4)
    assert verdict.usable_by_downstream is True  # output exists; downstream may proceed


# --- manifest ----------------------------------------------------------------------------

def test_manifest_separates_selected_from_skipped_with_reasons() -> None:
    rows = [
        # already has video output -> must not be reselected
        row(record_id="done", frame_file="f", frame_timestamp_seconds="1",
            vllm_structured_output='{"analysis": "x"}'),
        # frames present, no output -> the work to do
        row(record_id="ready", frame_file="f", frame_timestamp_seconds="1"),
        # no frames at all -> a recorded skip, not a retry
        row(record_id="noframes", asr_transcript="text"),
    ]
    manifest = el.select_reprocess(rows, [3])
    reasons = {entry["record_id"]: entry["reason_code"] for entry in manifest["selected"]}
    assert reasons == {"ready": "not_run"}
    skipped = {entry["record_id"] for entry in manifest["skipped"]}
    assert skipped == {"done", "noframes"}


def test_country_filter_applies() -> None:
    rows = [
        {"record_id": "a", "country": "finland", "asr_transcript": "x"},
        {"record_id": "b", "country": "poland", "asr_transcript": "y"},
    ]
    manifest = el.select_reprocess(rows, [6], countries=["poland"])
    assert {entry["record_id"] for entry in manifest["selected"]} == {"b"}


def test_manifest_is_deterministic() -> None:
    rows = [{"record_id": f"r{i}", "country": "finland", "asr_transcript": "x"} for i in range(5)]
    first = el.select_reprocess(rows, [6, 7])
    second = el.select_reprocess(list(reversed(rows)), [6, 7])
    assert first["selected"] == second["selected"]


def test_summary_line_is_one_line() -> None:
    manifest = el.select_reprocess([row(asr_transcript="x")], [6])
    line = el.summary_line(manifest)
    assert "\n" not in line
    assert "selected" in line


def test_records_without_identity_are_ignored() -> None:
    manifest = el.select_reprocess([{"asr_transcript": "x"}], [6])
    assert manifest["selected"] == [] and manifest["skipped"] == []


# --- prefilter ---------------------------------------------------------------------------

def test_nonblocking_query_does_not_require_a_complete_predecessor() -> None:
    query = el.nonblocking_query(6)
    assert "step5_status.status" not in str(query)
    assert "DELETE" in str(query)


def test_video_prefilter_requires_frames() -> None:
    query = el.nonblocking_query(3)
    assert "frame_file" in query and "frame_timestamp_seconds" in query
