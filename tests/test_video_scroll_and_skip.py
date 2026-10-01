"""Tests for the EP24 video rules from issue #22.

Two pipeline-wide behaviours are covered:

1. the mandatory first-second exclusion reaches every analysis path;
2. scroll-detection failures are reported as ``SCROLL`` / ``SCROLL_SECONDS`` and
   turn into a deterministic, bounded re-split plan.

No GPU, no Whisper and no real media: the media-facing parts are exercised
through the pure helpers (command building, offset sampling, parsing, planning),
which is where the logic that could silently corrupt an analysis actually lives.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import video_config  # noqa: E402
import video_scroll  # noqa: E402

# --------------------------------------------------------------------------- #
# 1-3. the first-second skip reaches each analysis path
# --------------------------------------------------------------------------- #

def test_skip_default_and_central_definition():
    """The duration is defined once, not repeated as a literal per stage."""
    assert video_config.VIDEO_INITIAL_SKIP_SECONDS == 1.0
    assert video_config.initial_skip_seconds() == 1.0


def test_skip_is_overridable_by_environment(monkeypatch):
    monkeypatch.setenv(video_config.SKIP_SECONDS_ENV, "2.5")
    assert video_config.initial_skip_seconds() == 2.5
    monkeypatch.setenv(video_config.SKIP_SECONDS_ENV, "not-a-number")
    assert video_config.initial_skip_seconds() == video_config.VIDEO_INITIAL_SKIP_SECONDS
    monkeypatch.setenv(video_config.SKIP_SECONDS_ENV, "-3")
    assert video_config.initial_skip_seconds() == video_config.VIDEO_INITIAL_SKIP_SECONDS


def test_skip_applied_to_frame_extraction():
    """Frame sampling must start after the skip, never at 0 s."""
    offsets = video_config.sample_offsets_seconds(180)
    assert offsets, "a 3-minute clip must yield samples"
    assert 0 not in offsets, "the transition frame at 0 s must not be sampled"
    assert min(offsets) >= video_config.VIDEO_INITIAL_SKIP_SECONDS
    assert offsets == sorted(offsets)
    assert len(offsets) <= 6, "legacy cadence caps the sample count"


def test_skip_applied_to_audio_extraction():
    """The ffmpeg command must seek past the skip before reading input."""
    command = video_config.audio_extract_command("in.mp4", "out.wav", 60.0)
    assert "-ss" in command
    ss_index = command.index("-ss")
    assert float(command[ss_index + 1]) == video_config.VIDEO_INITIAL_SKIP_SECONDS
    # -ss must precede -i so ffmpeg seeks before decoding.
    assert ss_index < command.index("-i")
    # And the analysed length excludes the skipped part.
    assert "-t" in command
    length = float(command[command.index("-t") + 1])
    assert length == pytest.approx(60.0 - video_config.VIDEO_INITIAL_SKIP_SECONDS)


def test_skip_applied_to_vlm_prompt():
    """The VLM prompt must carry the skip rule, sourced from the same constant."""
    text = video_config.scroll_rule_text()
    assert "Ignore the first 1.0 second" in text
    assert "SCROLL: TRUE or FALSE" in text
    assert "SCROLL_SECONDS" in text


def test_analysis_interval_starts_at_skip_and_ends_at_clip_end():
    start, end = video_config.analysis_interval(42.0)
    assert start == video_config.VIDEO_INITIAL_SKIP_SECONDS
    assert end == 42.0


# --------------------------------------------------------------------------- #
# 9. clips at or below the skip fail gracefully
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("duration", [0.6, 1.0])
def test_clip_at_or_below_skip_is_not_analysable(duration):
    assert video_config.is_analysable(duration) is False
    assert video_config.sample_offsets_seconds(duration) == []


def test_unknown_duration_is_not_treated_as_analysable():
    assert video_config.is_analysable(None) is False


# --------------------------------------------------------------------------- #
# 4, 5. SCROLL / SCROLL_SECONDS schema
# --------------------------------------------------------------------------- #

def test_scroll_true_with_timestamps_parses():
    text = "Some narrative.\n\nSCROLL: TRUE\nSCROLL_SECONDS: [8.4, 17.9]\n"
    report = video_scroll.parse_scroll_report(text, 60.0)
    assert report.scroll is True
    assert report.scroll_seconds == [8.4, 17.9]


def test_scroll_false_gives_empty_timestamp_list():
    text = "Narrative only.\nSCROLL: FALSE\nSCROLL_SECONDS: []\n"
    report = video_scroll.parse_scroll_report(text, 60.0)
    assert report.scroll is False
    assert report.scroll_seconds == []


def test_missing_scroll_block_defaults_to_false():
    """An unhelpful model must not flag every clip for re-splitting."""
    report = video_scroll.parse_scroll_report("just prose, no fields", 60.0)
    assert report.scroll is False
    assert report.scroll_seconds == []
    assert report.parse_error


def test_scroll_false_ignores_stray_numbers():
    text = "SCROLL: FALSE\nSCROLL_SECONDS: [5.0]\n"
    report = video_scroll.parse_scroll_report(text, 60.0)
    assert report.scroll is False
    assert report.scroll_seconds == []


# --------------------------------------------------------------------------- #
# 6/8. the known first-second transition must not count as a boundary
# --------------------------------------------------------------------------- #

def test_initial_transition_never_becomes_a_boundary():
    text = "SCROLL: TRUE\nSCROLL_SECONDS: [0.0, 0.9, 1.0, 12.5]\n"
    report = video_scroll.parse_scroll_report(text, 60.0)
    assert report.scroll is True
    assert report.scroll_seconds == [12.5], "only the genuine in-clip scroll survives"


def test_boundaries_are_deduplicated_and_clamped():
    text = "SCROLL: TRUE\nSCROLL_SECONDS: [12.5, 12.6, 900.0]\n"
    report = video_scroll.parse_scroll_report(text, 60.0)
    assert report.scroll_seconds == [12.5], "near-duplicates collapse, out-of-range drops"


# --------------------------------------------------------------------------- #
# 6. detected scrolls trigger the re-split path
# --------------------------------------------------------------------------- #

def test_scroll_true_plans_a_deterministic_resplit():
    report = video_scroll.parse_scroll_report("SCROLL: TRUE\nSCROLL_SECONDS: [20.0]", 60.0)
    plan = video_scroll.plan_resplit(report, 60.0, "source.mp4")
    assert plan.status == "needs_resplit"
    assert plan.boundaries == [20.0]
    assert plan.segments == [(1.0, 20.0), (20.0, 60.0)]
    assert len(plan.derived_names) == 2


def test_resplit_is_reproducible():
    text = "SCROLL: TRUE\nSCROLL_SECONDS: [20.0]"
    a = video_scroll.plan_resplit(video_scroll.parse_scroll_report(text, 60.0), 60.0, "s.mp4")
    b = video_scroll.plan_resplit(video_scroll.parse_scroll_report(text, 60.0), 60.0, "s.mp4")
    assert a.derived_names == b.derived_names, "same input must give the same clip names"


def test_scroll_true_without_timestamps_needs_review_not_a_guess():
    report = video_scroll.parse_scroll_report("SCROLL: TRUE\nSCROLL_SECONDS: []", 60.0)
    plan = video_scroll.plan_resplit(report, 60.0, "s.mp4")
    assert plan.status == "needs_resplit"
    assert plan.segments == [], "no split is invented without timestamps"
    assert report.flagged_without_timestamps is True


def test_scroll_false_needs_no_action():
    report = video_scroll.parse_scroll_report("SCROLL: FALSE", 60.0)
    assert video_scroll.plan_resplit(report, 60.0, "s.mp4").status == "ok"


# --------------------------------------------------------------------------- #
# 7. provenance survives re-splitting
# --------------------------------------------------------------------------- #

def test_derived_names_are_deterministic_and_traceable():
    name = video_scroll.derived_clip_name("dir/myclip.mp4", 1, 1.0, 20.0)
    assert name.startswith("myclip_scroll01_")
    assert name.endswith(".mp4")
    # The derived name records the boundaries, so a clip maps back to its cut.
    assert "000010" in name and "000200" in name
    assert video_scroll.derived_clip_name("dir/myclip.mp4", 1, 1.0, 20.0) == name


def test_source_clip_name_is_preserved_in_derived_names():
    plan = video_scroll.plan_resplit(
        video_scroll.parse_scroll_report("SCROLL: TRUE\nSCROLL_SECONDS: [15.0]", 40.0),
        40.0,
        "EP24/PL/keep-me.mp4",
    )
    for name in plan.derived_names:
        assert name.startswith("keep-me_scroll"), name
        assert "/" not in name, "derived names must not escape the output directory"


def test_resplit_does_not_overwrite_the_source():
    """The plan only proposes new names; it never names the original clip."""
    plan = video_scroll.plan_resplit(
        video_scroll.parse_scroll_report("SCROLL: TRUE\nSCROLL_SECONDS: [15.0]", 40.0),
        40.0,
        "keep-me.mp4",
    )
    assert "keep-me.mp4" not in plan.derived_names


# --------------------------------------------------------------------------- #
# 10. no infinite recursive split loop is possible
# --------------------------------------------------------------------------- #

def test_depth_limit_stops_recursion():
    report = video_scroll.parse_scroll_report("SCROLL: TRUE\nSCROLL_SECONDS: [10.0]", 60.0)
    at_limit = video_scroll.plan_resplit(
        report, 60.0, "s.mp4", depth=video_scroll.MAX_RESPLIT_DEPTH
    )
    assert at_limit.status == "refused"
    assert "depth" in at_limit.reason


def test_excessive_boundaries_are_refused_as_a_model_artefact():
    many = ", ".join(str(5 + i) for i in range(video_scroll.MAX_SCROLL_BOUNDARIES + 3))
    report = video_scroll.parse_scroll_report(f"SCROLL: TRUE\nSCROLL_SECONDS: [{many}]", 300.0)
    plan = video_scroll.plan_resplit(report, 300.0, "s.mp4")
    assert plan.status == "refused"


def test_short_clip_is_never_split():
    report = video_scroll.parse_scroll_report("SCROLL: TRUE\nSCROLL_SECONDS: [0.5]", 0.8)
    plan = video_scroll.plan_resplit(report, 0.8, "s.mp4")
    assert plan.status == "not_analysable"


# --------------------------------------------------------------------------- #
# 8. legacy compatibility
# --------------------------------------------------------------------------- #

def test_legacy_pipeline_and_fields_are_untouched():
    """The skip is added around the stages; no legacy stage or field is removed."""
    contract = (ROOT / "docs/LEGACY_PIPELINE_CONTRACT.md").read_text(encoding="utf-8")
    for stage in ("roihu_preprocess.py", "roihu_frame.py", "roihu_summary.py",
                  "roihu_postprocess.py", "roihu_populism.py"):
        assert stage in contract or stage.replace("roihu", "puhti") in contract
    for field in ("whisperResult", "frame_files", "ocr_1", "summary_analysis",
                  "formula_of_populism_analysis"):
        assert field in contract, field

    preprocess = (ROOT / "roihu_preprocess.py").read_text(encoding="utf-8")
    for legacy_column in ("'frame_files'", "'ocr_1'", "'whisper_transcript'",
                          "'whisper_language'", "'whisper_translated'"):
        assert legacy_column in preprocess, legacy_column


def test_preprocess_uses_the_shared_skip_not_a_literal():
    """No stage may reintroduce the bare range(0, ...) frame loop."""
    preprocess = (ROOT / "roihu_preprocess.py").read_text(encoding="utf-8")
    assert "sample_offsets_seconds" in preprocess
    assert "audio_extract_command" in preprocess
    assert "range(0, duration_seconds, 30)" not in preprocess
    assert "from video_config import" in preprocess
