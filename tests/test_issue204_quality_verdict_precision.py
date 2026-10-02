"""Regression tests for the prose-verdict false positive in the quality gate.

The first version of `status_from_text` scanned prose for the bare words
OK / REPROCESS / DELETE and took the strongest match. On the EP24 corpus that is
a false-positive generator: describing an editing cut ("the original audio is
deleted and replaced by music") or quoting a deleted social-media post is
ordinary video description, and every such sentence escalated the row to DELETE.

That mattered more than a dropped column, because a DELETE row is removed from
the step-5 output *and* has its local frame and video copy deleted, so the video
silently disappears from the research dataset.

These tests pin both directions: ordinary description stays OK, and genuine
verdicts are still detected.
"""
from __future__ import annotations

import pytest

from laclaugpt_quality import quality_decision_from_analysis, status_from_text

INNOCENT_PROSE = [
    "The video shows a screenshot of a deleted social-media post by the candidate.",
    "The original audio is deleted and replaced by music.",
    "A graphic displays a deleted comment from a news article about the parliament.",
    "Auto-generated subtitles were deleted in editing, leaving only the on-screen title.",
    "The watermark was deleted from the corner of the frame.",
    "The video is OK in the sense that the framing is steady.",
]


@pytest.mark.parametrize("text", INNOCENT_PROSE)
def test_ordinary_prose_is_not_read_as_a_delete_verdict(text):
    assert status_from_text(text, default="OK") == "OK"
    status, _ = quality_decision_from_analysis(text)
    assert status == "OK", f"{status!r} from descriptive prose: {text!r}"


VERDICTS = [
    ("Video problems: meaningless content, garbage. DELETE.", "DELETE"),
    ("This clip is **DELETE**.", "DELETE"),
    ("STATUS: REPROCESS", "REPROCESS"),
    ("The splitter failed here; mark for REPROCESS.", "REPROCESS"),
    ("The user scrolls mid-video; this should be deleted.", "DELETE"),
    ("Some description.\n\nDELETE\n", "DELETE"),
    ("The clip must be reprocessed and cut again.", "REPROCESS"),
    ("The video is essentially garbage.", "DELETE"),
    ("`REPROCESS`", "REPROCESS"),
    ("The source should be reprocessed.", "REPROCESS"),
    ("The frame is blank; meaningless content.", "DELETE"),
]


@pytest.mark.parametrize("text,expected", VERDICTS)
def test_real_verdicts_are_still_detected(text, expected):
    assert status_from_text(text, default="OK") == expected


def test_delete_verdict_wins_over_a_weaker_label_in_the_same_text():
    text = "Some parts look OK, but the source is garbage. DELETE."
    assert status_from_text(text, default="OK") == "DELETE"
