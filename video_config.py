"""Canonical video-analysis settings for the EP24 pipeline.

One source of truth for the pipeline-wide video rules. Do not reintroduce a
literal ``1.0`` or a bare ``range(0, ...)`` in a stage script: import from here,
so a change to the rule changes every consumer at once.

Why the skip exists
-------------------
The EP24 mobile videos were recorded on GrapheneOS phones by researchers doing
digital ethnography on TikTok/Instagram during the 2024 European Parliament
elections. A scroll-detection algorithm split the continuous screen recordings
into per-item clips, which means **the first second of every clip contains the
scrolling transition from the previous feed item**, not the item itself.

Every analysis path therefore starts at ``VIDEO_INITIAL_SKIP_SECONDS`` and runs
to the end of the clip. The source video is never modified; the skip is applied
by the reader (frame sampler, audio extractor, VLM request).
"""

from __future__ import annotations

import math
import os

# The mandatory first-second exclusion. One definition, used everywhere.
VIDEO_INITIAL_SKIP_SECONDS = 1.0

# Environment override, for testing and for the rare clip that needs a
# different offset. Kept deliberately short: this is a pipeline invariant, not a
# tuning knob.
SKIP_SECONDS_ENV = "LACLAUGPT_VIDEO_INITIAL_SKIP_SECONDS"

# A clip at or below this many seconds leaves no analysable interval once the
# skip is applied. Such clips are marked, not silently analysed as empty.
MIN_ANALYSABLE_SECONDS = 0.05


def initial_skip_seconds() -> float:
    """Return the effective skip, honouring the environment override."""
    raw = os.getenv(SKIP_SECONDS_ENV)
    if raw is None or str(raw).strip() == "":
        return VIDEO_INITIAL_SKIP_SECONDS
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return VIDEO_INITIAL_SKIP_SECONDS
    # A negative skip would move the start before the file; a skip past any
    # plausible clip would silently blank every analysis. Guard both.
    return value if value >= 0 else VIDEO_INITIAL_SKIP_SECONDS


def analysis_start_seconds(duration_seconds: float | None = None) -> float:
    """Start of the logical analysis interval.

    Capped at the clip end when a duration is known, so a clip shorter than the
    skip yields start == end rather than a negative-length interval.
    """
    skip = initial_skip_seconds()
    if duration_seconds is None:
        return skip
    try:
        duration = float(duration_seconds)
    except (TypeError, ValueError):
        return skip
    if duration <= 0:
        return 0.0
    return min(skip, duration)


def analysis_interval(duration_seconds: float | None = None) -> tuple[float, float]:
    """Return ``(start, end)`` of the logical analysis interval, in seconds.

    ``end`` is the original clip end — the skip only moves the start.
    """
    start = analysis_start_seconds(duration_seconds)
    if duration_seconds is None:
        return start, 0.0
    try:
        duration = float(duration_seconds)
    except (TypeError, ValueError):
        return start, 0.0
    return start, max(0.0, duration)


def is_analysable(duration_seconds: float | None) -> bool:
    """True when the clip is long enough to hold an analysable interval."""
    start, end = analysis_interval(duration_seconds)
    return (end - start) > MIN_ANALYSABLE_SECONDS


def sample_offsets_seconds(
    duration_seconds: float | None,
    every_seconds: int = 30,
    limit: int = 6,
) -> list[int]:
    """Whole-second sample offsets inside the analysis interval.

    Mirrors the legacy keyframe cadence (one frame every 30 s, at most six) but
    starts at the skip offset instead of at zero, so the first sample is no
    longer the scroll transition. Returns an empty list for a clip too short to
    hold an analysable interval.
    """
    if not is_analysable(duration_seconds):
        return []
    start, end = analysis_interval(duration_seconds)
    first = math.ceil(start)
    return list(range(first, int(end), every_seconds))[:limit]


def audio_extract_command(
    video_path: str,
    audio_path: str,
    duration_seconds: float | None = None,
) -> list[str]:
    """Build the ffmpeg command that extracts the analysed audio interval.

    ``-ss`` is placed before ``-i`` so ffmpeg seeks before decoding, which is
    both faster and exact enough for a 1 s offset. The interval starts at the
    skip and runs to the original clip end, so the transition audio is never
    transcribed or embedded. Returned as a list rather than executed here, so
    the command is testable without ffmpeg installed.
    """
    start, _ = analysis_interval(duration_seconds)
    command = ["ffmpeg", "-nostdin", "-y", "-ss", f"{start:.3f}", "-i", video_path]
    if duration_seconds is not None:
        try:
            length = max(0.0, float(duration_seconds) - start)
        except (TypeError, ValueError):
            length = 0.0
        if length > 0:
            command += ["-t", f"{length:.3f}"]
    # 16 kHz mono WAV is what Whisper expects; converting here avoids a second
    # decode inside the ASR engine.
    command += ["-vn", "-ac", "1", "-ar", "16000", "-f", "wav", audio_path]
    return command


def scroll_rule_text() -> str:
    """The skip/scroll instruction shared by VLM prompts.

    Kept here so the system prompt and the parser cannot drift apart.
    """
    skip = initial_skip_seconds()
    return (
        f"Ignore the first {skip:.1f} second(s) of the video: that interval "
        "contains the scrolling transition from the previous social-media item "
        "and is never part of the analysed content.\n"
        "In the remaining video, look for scrolling between distinct TikTok or "
        "Instagram feed items. Distinguish normal motion inside one post from an "
        "actual feed scroll that moves to a different post.\n"
        "Then report, on their own lines:\n"
        "SCROLL: TRUE or FALSE\n"
        "SCROLL_SECONDS: a bracketed list of approximate seconds, e.g. [8.4, 17.9]\n"
        "Set SCROLL: TRUE only when at least one additional feed-scroll "
        f"transition occurs after the first {skip:.1f} second(s). The known "
        f"first-{skip:.1f}-second transition must NOT set SCROLL: TRUE. When no "
        "additional feed scroll is present, write SCROLL: FALSE and "
        "SCROLL_SECONDS: []."
    )
