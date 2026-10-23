"""Step 5 readiness reporting regression tests."""
from pathlib import Path

from ep24_stage_orchestrator import eligible_query, status_path


def test_completed_step5_records_not_reeligible_for_retry():
    query = eligible_query(5, retry_errors=True, force=False)
    assert query[f"{status_path(4)}.status"] == "complete"
    statuses = query["$or"][1][f"{status_path(5)}.status"]["$in"]
    assert "error" in statuses
    assert "complete" not in statuses


def test_stage_complete_message_reports_upstream_backlog():
    text = (Path(__file__).resolve().parents[1] / "ep24_stage_orchestrator.py").read_text()
    assert '"STAGE_COMPLETE country=%s step=%d upstream_complete=%d "' in text
    assert "remaining_upstream = max(0, total_rows - upstream)" in text
    assert 'if total_rows > 0 and ready_total == stage_counts["complete"]' in text
    assert '"ready_input=%d awaiting_upstream=%d stage_not_started=%d "' in text
