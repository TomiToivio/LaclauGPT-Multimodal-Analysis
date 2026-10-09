"""Protect downstream analysis from historic false-complete Step 1 rows."""
from ep24_stage_orchestrator import eligible_query


def test_step2_requires_valid_preprocessed_frame():
    query = eligible_query(2, retry_errors=True, force=False)
    assert query["_pipeline.step_01.status"] == "complete"
    assert query["preprocess_status"] == {"$in": ["ok", "cached"]}
    assert query["frame_file"]["$nin"] == ["", None]
    assert query["frame_timestamp_seconds"]["$nin"] == ["", None]
    assert "error" in query["$or"][1]["_pipeline.step_02.status"]["$in"]


def test_step3_requires_successful_frame_analysis():
    query = eligible_query(3, retry_errors=False, force=False)
    assert query["frame_analysis_status"] == {"$in": ["ok", "cached"]}


def test_step1_import_is_not_blocked_by_frame_constraints():
    query = eligible_query(1, retry_errors=True, force=False)
    assert "frame_file" not in query
    assert "preprocess_status" not in query
