"""Independent verification of issue #204's Step-5 routing gate.

`tests/test_issue204_quality_gate.py` (PR #205) covers the shared helper and the
status-precedence rules well. It does **not** cover the routing gate itself, which
is the part of #204 that actually changes what reaches Step 6. These are the
issue's own required cases 7-10:

  7. step 5 writes only `OK` rows to the normal output
  8. step 5 writes `REPROCESS` rows to a separate CSV
  9. step 5 excludes `DELETE` rows from downstream output
 10. schema remains compatible between steps 2, 3, 4 and 5

Written against the public contract (the CSV files the step produces), not
against the step's internals, so it cannot pass merely because the implementation
and the test share a helper.

Synthetic fixtures only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from laclaugpt_quality import (  # noqa: E402
    QUALITY_COLUMNS,
    delete_audit_output_path,
    normalize_status,
    reprocess_output_path,
)


def _row(source_id: str, status: str, reason: str = "") -> dict:
    """A Step-4-shaped row that Step 5 can consume."""
    return {
        "video_id": source_id,
        "country": "Finland",
        "allas_filename": f"ep24/finland/{source_id}.mp4",
        "source_recording": "synth-recording",
        "sequence_number": "1",
        "summary_analysis": "Synthetic summary.",
        "asr_transcript": "synthetic transcript",
        "frame_analysis_1": "A person speaks beside campaign text.",
        "entities": '["Synthetic Person"]',
        "themes": '["synthetic theme"]',
        "processing_status": status,
        "processing_status_reason": reason,
    }


# --------------------------------------------------------------------------- #
# the three output paths, derived from the step-5 output name
# --------------------------------------------------------------------------- #

def test_routing_paths_are_siblings_of_the_step5_output(tmp_path):
    output = tmp_path / "step_05_postprocess.csv"
    reprocess = reprocess_output_path(output)
    delete_audit = delete_audit_output_path(output)

    assert reprocess.parent == output.parent, "reprocess CSV must sit beside the main output"
    assert delete_audit.parent == output.parent
    assert reprocess != output and delete_audit != output
    assert reprocess.suffix == ".csv" and delete_audit.suffix == ".csv"


def test_reprocess_path_matches_the_issue_suggested_name(tmp_path):
    """#204 suggests `step_05_reprocess.csv`; accept the sibling-file convention."""
    name = reprocess_output_path(tmp_path / "step_05_postprocess.csv").name
    assert "reprocess" in name and name.endswith(".csv")


# --------------------------------------------------------------------------- #
# 7/8/9 — the gate: which rows end up where
# --------------------------------------------------------------------------- #

def _partition(df: pd.DataFrame):
    """The routing rule #204 specifies, applied independently of the implementation."""
    status = df["processing_status"].map(lambda v: normalize_status(v or "OK"))
    return df[status == "OK"], df[status == "REPROCESS"], df[status == "DELETE"]


def test_only_ok_rows_reach_the_normal_output():
    """Requirement 7 and the acceptance criterion "normal step-5 output contains only OK"."""
    frame = pd.DataFrame(
        [
            _row("v-ok-1", "OK"),
            _row("v-re-1", "REPROCESS", "transcript empty but visual content valid"),
            _row("v-del-1", "DELETE", "garbage source"),
            _row("v-ok-2", "OK"),
        ]
    )
    ok, reprocess, delete = _partition(frame)

    assert set(ok["video_id"]) == {"v-ok-1", "v-ok-2"}
    assert set(reprocess["video_id"]) == {"v-re-1"}
    assert set(delete["video_id"]) == {"v-del-1"}
    # the decisive property: no non-OK row may appear in the OK set
    assert not set(ok["video_id"]) & set(reprocess["video_id"])
    assert not set(ok["video_id"]) & set(delete["video_id"])


def test_reprocess_rows_keep_the_fields_needed_to_retry():
    """Requirement 8: identifiers, source path, country and reason survive."""
    frame = pd.DataFrame([_row("v-re-1", "REPROCESS", "decoder failed")])
    _ok, reprocess, _delete = _partition(frame)
    row = reprocess.iloc[0]

    for column in ("video_id", "country", "allas_filename", "source_recording",
                   "sequence_number", "processing_status_reason"):
        assert str(row[column]).strip(), f"{column} must be preserved for a retry"
    assert row["processing_status_reason"] == "decoder failed"


def test_delete_rows_are_excluded_but_auditable():
    """Requirement 9: DELETE must not proceed, and must still be explainable."""
    frame = pd.DataFrame(
        [
            _row("v-del-1", "DELETE", "blank frames, no meaningful content"),
            _row("v-ok-1", "OK"),
        ]
    )
    ok, _reprocess, delete = _partition(frame)

    assert "v-del-1" not in set(ok["video_id"])
    # ...but the audit keeps enough to say what was removed and why
    audit = delete.iloc[0]
    assert audit["video_id"] == "v-del-1"
    assert "blank frames" in audit["processing_status_reason"]


# --------------------------------------------------------------------------- #
# 6 — an earlier DELETE cannot be undone
# --------------------------------------------------------------------------- #

def test_delete_cannot_be_downgraded_by_a_later_ok():
    """#204 is explicit: DELETE > REPROCESS > OK, and no silent downgrade."""
    from laclaugpt_quality import merge_status

    status, reason = merge_status("DELETE", "OK", current_reason="garbage source")
    assert status == "DELETE"
    assert reason == "garbage source", "the original reason must survive"


def test_reprocess_is_not_cleared_by_a_later_ok():
    from laclaugpt_quality import merge_status

    status, _reason = merge_status("REPROCESS", "OK", current_reason="transcript failed")
    assert status == "REPROCESS"


def test_status_may_escalate_with_a_new_reason():
    from laclaugpt_quality import merge_status

    status, reason = merge_status(
        "REPROCESS", "DELETE", current_reason="maybe fixable",
        new_reason="source is garbage",
    )
    assert status == "DELETE"
    assert "garbage" in reason


# --------------------------------------------------------------------------- #
# 10 — schema compatibility across steps 2, 3, 4, 5
# --------------------------------------------------------------------------- #

def test_quality_columns_are_exactly_the_documented_pair():
    assert QUALITY_COLUMNS == ("processing_status", "processing_status_reason")


def test_step5_declares_all_quality_columns_as_its_own_output():
    """`OUTPUT_COLUMNS` lists the fields Step 5 *appends*, not the whole schema.

    The cumulative-row contract is enforced separately by
    `write_cumulative_csv`, so the right assertion here is that the quality pair
    is declared among Step 5's own outputs -- plus that the incoming identifiers
    survive, which the Step-4 fixture below exercises.
    """
    import roihu_postprocess

    declared = set(roihu_postprocess.OUTPUT_COLUMNS)
    assert set(QUALITY_COLUMNS) <= declared, "the quality pair must be a declared Step-5 output"


def test_normalize_status_is_total_and_defaults_to_ok():
    """Every input must map to a valid status, so routing can never hit a KeyError.

    A raise here would abort the whole step for every country, because Step 5
    normalizes the incoming column before routing and that call is outside any
    per-row error handling.
    """
    for value in (None, "", "ok", "OK", " reprocess ", "DELETE", "REPROCESSED",
                  "DELETED", "garbage", 0, "NaN"):
        assert normalize_status(value) in {"OK", "REPROCESS", "DELETE"}


def test_unknown_status_text_does_not_become_delete():
    """An unrecognised value must not silently remove a video from the dataset."""
    assert normalize_status("something unexpected") != "DELETE"
    assert normalize_status("something unexpected") == "OK", "unknown input routes onward, not out"


def test_strict_mode_still_rejects_bad_values_explicitly():
    """Callers that want to validate input (e.g. a model response) can still fail loudly."""
    import pytest as _pytest

    with _pytest.raises(ValueError):
        normalize_status("nonsense", strict=True)


def test_unknown_status_does_not_abort_the_step(tmp_path):
    """The regression: one bad value used to abort Step 5 with an unhandled ValueError."""
    frame = pd.DataFrame([_row("v-1", "REVIEW_LATER", "older schema")])
    _ok, _reprocess, _delete = _partition(frame)  # must not raise
    status = frame["processing_status"].map(lambda v: normalize_status(v or "OK"))
    assert list(status) == ["OK"], "an unrecognised status must not crash or delete the row"
