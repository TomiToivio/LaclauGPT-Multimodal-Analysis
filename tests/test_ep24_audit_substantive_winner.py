"""Regression for issue #306: audit winner must contain analysis, not just statuses."""

import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/ep24/audit_stage_coverage.py"
SPEC = importlib.util.spec_from_file_location("audit_stage_coverage_script", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_video_dedupe_prefers_substantive_output_over_status_only_retry():
    successful = {
        "_storage_id": "a", "allas_filename": "same.mp4",
        "summary_analysis": "Grounded synthetic summary",
    }
    failed_retry = {
        "_storage_id": "z", "allas_filename": "same.mp4",
        "summary_quality_status": "error", "summary_quality_reason": "timeout",
        "vllm_video_status": "error", "vllm_structured_status": "error",
        "laclau_status": "error", "dna_status": "error", "sna_status": "error",
    }
    assert MODULE.dedupe([successful, failed_retry], "video") == [successful]
    assert MODULE.dedupe([failed_retry, successful], "video") == [successful]
    assert MODULE.dedupe([successful, failed_retry], "ingest") == [successful, failed_retry]


def test_video_dedupe_tie_breaker_is_deterministic():
    a = {"_storage_id": "a", "allas_filename": "same.mp4", "summary_analysis": "A"}
    b = {"_storage_id": "b", "allas_filename": "same.mp4", "summary_analysis": "B"}
    assert MODULE.dedupe([a, b], "video") == [b]
    assert MODULE.dedupe([b, a], "video") == [b]
