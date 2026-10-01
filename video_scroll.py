"""Scroll-transition handling: parse the VLM report and plan a re-split.

The scroll-detection splitter that produced the EP24 clips occasionally fails,
leaving a clip that contains more than one TikTok/Instagram item separated by
extra scrolling transitions. The VLM analysis reports those; this module turns
the report into a deterministic, provenance-preserving re-split plan.

Nothing here runs a model or touches media: it is pure logic, so it is testable
without a GPU.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from video_config import analysis_interval, initial_skip_seconds, is_analysable

# A clip should never be split more times than this, whatever the model reports.
# The bound is what makes an infinite re-split loop impossible: each re-split
# round increments the depth, and a plan at MAX_RESPLIT_DEPTH is refused.
MAX_RESPLIT_DEPTH = 3

# More boundaries than this in one clip is treated as a model artefact rather
# than a real feed scroll, and is rejected instead of acted on.
MAX_SCROLL_BOUNDARIES = 12

# Two detected boundaries closer than this are the same transition reported
# twice; they are collapsed so a jittery model cannot inflate the split count.
MIN_BOUNDARY_SEPARATION_SECONDS = 0.5

SCROLL_LINE_RE = re.compile(r"^\s*SCROLL\s*[:=]\s*(TRUE|FALSE)\s*$", re.IGNORECASE | re.MULTILINE)
SCROLL_SECONDS_RE = re.compile(
    r"^\s*SCROLL_SECONDS\s*[:=]\s*(\[[^\]]*\]|\([^\)]*\))?", re.IGNORECASE | re.MULTILINE
)
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class ScrollReport:
    """Structured result of the scroll check for one clip."""

    scroll: bool = False
    scroll_seconds: list[float] = field(default_factory=list)
    # True when the model said SCROLL: TRUE but no usable timestamps came back.
    # The clip is still flagged for review rather than silently accepted.
    flagged_without_timestamps: bool = False
    raw_block: str = ""
    parse_error: str = ""


def parse_scroll_report(model_text: str, duration_seconds: float | None = None) -> ScrollReport:
    """Parse ``SCROLL`` / ``SCROLL_SECONDS`` out of a VLM response.

    Rules that matter downstream:

    * a missing or unparseable block is ``SCROLL = FALSE`` with an empty list,
      so an unhelpful model cannot mark every clip for re-splitting;
    * ``SCROLL_SECONDS`` is normalised — deduplicated, sorted, clamped to the
      clip, and stripped of anything inside the mandatory skip interval, so the
      known first-second transition can never appear as a boundary;
    * ``SCROLL = TRUE`` with no usable timestamps sets
      ``flagged_without_timestamps`` instead of being silently dropped.
    """
    report = ScrollReport()
    if not model_text:
        return report

    match = SCROLL_LINE_RE.search(model_text)
    if not match:
        report.parse_error = "no SCROLL line found"
        return report
    report.raw_block = match.group(0).strip()
    report.scroll = match.group(1).upper() == "TRUE"

    seconds_match = SCROLL_SECONDS_RE.search(model_text)
    raw_seconds = ""
    if seconds_match and seconds_match.group(1):
        raw_seconds = seconds_match.group(1)
        report.raw_block = f"{report.raw_block}\n{seconds_match.group(0).strip()}"

    values: list[float] = []
    if raw_seconds:
        inner = raw_seconds.strip().strip("[]()")
        values = [float(number) for number in NUMBER_RE.findall(inner)]

    report.scroll_seconds = normalise_boundaries(values, duration_seconds)

    if report.scroll and not report.scroll_seconds:
        report.flagged_without_timestamps = True
    if not report.scroll:
        # FALSE must always carry an empty list, even if stray numbers appeared.
        report.scroll_seconds = []
    return report


def normalise_boundaries(
    values: list[float],
    duration_seconds: float | None = None,
) -> list[float]:
    """Clean detected boundary timestamps.

    Drops anything at or before the skip (the transition we always ignore),
    collapses near-duplicates, sorts, and clamps to the clip end when known.
    """
    skip = initial_skip_seconds()
    start, end = analysis_interval(duration_seconds)

    cleaned: list[float] = []
    for value in values:
        seconds = round(float(value), 1)
        # At or below the skip is the known first-second transition, which must
        # never be reported as an additional scroll: the rule is that the first
        # second is always ignored by design, so its edge cannot be a boundary.
        if seconds <= skip:
            continue
        if duration_seconds is not None and seconds > end:
            continue
        if seconds <= start:
            continue
        cleaned.append(seconds)

    cleaned.sort()
    deduped: list[float] = []
    for seconds in cleaned:
        if deduped and abs(seconds - deduped[-1]) < MIN_BOUNDARY_SEPARATION_SECONDS:
            continue
        deduped.append(seconds)
    return deduped


@dataclass
class ResplitPlan:
    """A deterministic re-split decision for one clip."""

    status: str  # ok | needs_resplit | refused | not_analysable
    reason: str = ""
    boundaries: list[float] = field(default_factory=list)
    segments: list[tuple[float, float]] = field(default_factory=list)
    derived_names: list[str] = field(default_factory=list)


def derived_clip_name(source_name: str, segment_index: int, start: float, end: float) -> str:
    """Deterministic name for one derived segment.

    The same source and the same boundaries always produce the same name, so a
    re-run cannot create a second, differently-named set of clips.
    """
    stem = source_name.rsplit("/", 1)[-1]
    if stem.lower().endswith(".mp4"):
        stem = stem[:-4]
    digest = hashlib.sha256(f"{source_name}|{start:.1f}|{end:.1f}".encode()).hexdigest()[:8]
    return f"{stem}_scroll{segment_index:02d}_{int(start * 10):06d}_{int(end * 10):06d}_{digest}.mp4"


def plan_resplit(
    report: ScrollReport,
    duration_seconds: float | None,
    source_name: str,
    depth: int = 0,
) -> ResplitPlan:
    """Turn a scroll report into a re-split plan, or refuse to.

    Guards, in order:

    * a clip with no analysable interval is never split;
    * ``SCROLL = FALSE`` needs no action;
    * ``SCROLL = TRUE`` without timestamps produces ``needs_resplit`` and is
      left for a human/deterministic follow-up rather than guessed at;
    * more boundaries than ``MAX_SCROLL_BOUNDARIES`` is refused as a model
      artefact;
    * ``depth >= MAX_RESPLIT_DEPTH`` is refused, which is what bounds the
      recursion.
    """
    if not is_analysable(duration_seconds):
        return ResplitPlan(status="not_analysable", reason="clip has no analysable interval")

    if not report.scroll:
        return ResplitPlan(status="ok", reason="no additional scroll detected")

    if report.flagged_without_timestamps or not report.scroll_seconds:
        return ResplitPlan(
            status="needs_resplit",
            reason="SCROLL reported TRUE without usable timestamps; needs deterministic review",
        )

    if len(report.scroll_seconds) > MAX_SCROLL_BOUNDARIES:
        return ResplitPlan(
            status="refused",
            reason=(
                f"{len(report.scroll_seconds)} boundaries exceeds the limit of "
                f"{MAX_SCROLL_BOUNDARIES}; treated as a model artefact"
            ),
        )

    if depth >= MAX_RESPLIT_DEPTH:
        return ResplitPlan(
            status="refused",
            reason=f"re-split depth {depth} has reached the limit of {MAX_RESPLIT_DEPTH}",
        )

    start, end = analysis_interval(duration_seconds)
    boundaries = [b for b in report.scroll_seconds if start < b < end]
    if not boundaries:
        return ResplitPlan(
            status="needs_resplit",
            reason="all detected boundaries fall outside the analysis interval",
        )

    # Each segment runs from the previous cut (or the first analysable moment)
    # to the next cut (or the clip end). Every segment therefore also begins
    # after the skip where that applies, and no frame is lost at a cut: the
    # boundary is the end of one segment and the start of the next.
    edges = [start, *boundaries, end]
    segments = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    names = [
        derived_clip_name(source_name, i + 1, seg_start, seg_end)
        for i, (seg_start, seg_end) in enumerate(segments)
    ]
    return ResplitPlan(
        status="needs_resplit",
        reason=f"{len(boundaries)} additional scroll transition(s) detected",
        boundaries=boundaries,
        segments=segments,
        derived_names=names,
    )
