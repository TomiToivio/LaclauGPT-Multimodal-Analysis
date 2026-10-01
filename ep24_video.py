"""Shared EP24 video-analysis rules.

EP24 screen recordings were split from longer TikTok/Instagram feed captures by
scroll detection. Every split clip therefore begins with the tail of the scroll
from the previous feed item. Analysis must ignore that known transition.

This module is intentionally dependency-light so every media stage can share the
same rule without duplicating magic numbers.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

VIDEO_INITIAL_SKIP_SECONDS = float(
    os.getenv("LACLAUGPT_VIDEO_INITIAL_SKIP_SECONDS", "1.0")
)
MAX_RESPLIT_DEPTH = int(os.getenv("LACLAUGPT_VIDEO_MAX_RESPLIT_DEPTH", "1"))


def analysis_start_seconds() -> float:
    """Return the canonical start of analyzable content in every EP24 clip."""
    return VIDEO_INITIAL_SKIP_SECONDS


def is_too_short(duration_seconds: float | int | None) -> bool:
    """True when no analyzable media remains after the mandatory initial skip."""
    if duration_seconds is None:
        return False
    return float(duration_seconds) <= VIDEO_INITIAL_SKIP_SECONDS


def analysis_frame_times(
    duration_seconds: float,
    interval_seconds: float = 30.0,
    limit: int = 6,
) -> list[float]:
    """Return frame timestamps, always beginning after the known scroll artifact."""
    duration = float(duration_seconds)
    if is_too_short(duration):
        return []
    times: list[float] = []
    t = VIDEO_INITIAL_SKIP_SECONDS
    while t < duration and len(times) < limit:
        times.append(round(t, 3))
        t += interval_seconds
    return times


def analysis_clip_path(source: str | Path, output_dir: str | Path) -> Path:
    """Deterministic path for a non-destructive trimmed analysis clip."""
    source_path = Path(source)
    suffix = source_path.suffix or ".mp4"
    return Path(output_dir) / (
        f"{source_path.stem}.analysis-from-{VIDEO_INITIAL_SKIP_SECONDS:g}s{suffix}"
    )


def build_trim_command(source: str | Path, destination: str | Path) -> list[str]:
    """Build the ffmpeg command used to create the derived analysis clip."""
    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{VIDEO_INITIAL_SKIP_SECONDS:g}",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(destination),
    ]


def prepare_analysis_clip(
    source: str | Path,
    output_dir: str | Path,
    *,
    runner=subprocess.run,
) -> Path:
    """Create/reuse a derived clip after the mandatory initial artifact.

    The original source is never modified. The derived clip is transcoded so the
    1.0-second boundary is exact even when it does not land on a source keyframe.
    Frame extraction may seek the source directly at the
    canonical analysis start for frame-accurate sampling.
    """
    source_path = Path(source)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    destination = analysis_clip_path(source_path, output_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        return destination

    completed = runner(
        build_trim_command(source_path, destination),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "ffmpeg could not create EP24 analysis clip: "
            + (completed.stderr or completed.stdout or "").strip()
        )
    if not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError(f"ffmpeg produced no usable analysis clip: {destination}")
    return destination


def normalize_scroll_metadata(scroll: bool, seconds) -> dict[str, object]:
    """Normalize VLM scroll output and discard the known initial transition."""
    parsed: list[float] = []
    if seconds is None:
        seconds = []
    if isinstance(seconds, str):
        try:
            seconds = json.loads(seconds)
        except json.JSONDecodeError:
            seconds = re.findall(r"\d+(?:\.\d+)?", seconds)
    for value in seconds:
        try:
            timestamp = float(value)
        except (TypeError, ValueError):
            continue
        if timestamp > VIDEO_INITIAL_SKIP_SECONDS:
            parsed.append(round(timestamp, 3))
    parsed = sorted(set(parsed))
    if not scroll or not parsed:
        return {"SCROLL": False, "SCROLL_SECONDS": []}
    return {"SCROLL": True, "SCROLL_SECONDS": parsed}


def parse_scroll_metadata(text: str) -> dict[str, object]:
    """Extract SCROLL and SCROLL_SECONDS from a VLM response."""
    if not text:
        return {"SCROLL": False, "SCROLL_SECONDS": []}

    candidates = re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)
    for candidate in reversed(candidates):
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if "SCROLL" in data or "SCROLL_SECONDS" in data:
            raw_scroll = data.get("SCROLL", False)
            scroll = (
                raw_scroll
                if isinstance(raw_scroll, bool)
                else str(raw_scroll).strip().lower() == "true"
            )
            return normalize_scroll_metadata(scroll, data.get("SCROLL_SECONDS", []))

    scroll_match = re.search(
        r"\bSCROLL\s*:\s*(TRUE|FALSE)\b",
        text,
        flags=re.IGNORECASE,
    )
    seconds_match = re.search(
        r"\bSCROLL_SECONDS\s*:\s*(\[[^\]]*\])",
        text,
        flags=re.IGNORECASE,
    )
    scroll = bool(scroll_match and scroll_match.group(1).upper() == "TRUE")
    seconds = seconds_match.group(1) if seconds_match else []
    return normalize_scroll_metadata(scroll, seconds)


def needs_resplit(metadata: dict[str, object], *, resplit_depth: int = 0) -> bool:
    """Return whether a clip should enter the deterministic re-split queue."""
    return bool(
        metadata.get("SCROLL")
        and metadata.get("SCROLL_SECONDS")
        and int(resplit_depth) < MAX_RESPLIT_DEPTH
    )


def split_plan(
    duration_seconds: float,
    scroll_seconds,
    *,
    resplit_depth: int = 0,
) -> list[tuple[float, float]]:
    """Plan non-overlapping derived segments without touching source media."""
    duration = float(duration_seconds)
    if int(resplit_depth) >= MAX_RESPLIT_DEPTH or is_too_short(duration):
        return []
    meta = normalize_scroll_metadata(True, scroll_seconds)
    boundaries = [
        float(t)
        for t in meta["SCROLL_SECONDS"]
        if VIDEO_INITIAL_SKIP_SECONDS < float(t) < duration
    ]
    if not boundaries:
        return []
    points = [0.0, *boundaries, duration]
    return [
        (round(a, 3), round(b, 3))
        for a, b in zip(points, points[1:])
        if b > a
    ]


def derived_segment_id(
    parent_id: str,
    index: int,
    start: float,
    end: float,
) -> str:
    """Stable identifier for a derived clip, preserving parent provenance."""
    safe_parent = re.sub(r"[^A-Za-z0-9._-]+", "_", str(parent_id)).strip("_") or "clip"
    return f"{safe_parent}__resplit-{index:02d}__{start:.3f}-{end:.3f}s"


def derived_rows(
    source_row: dict,
    duration_seconds: float,
    scroll_seconds,
    *,
    parent_id: str,
    resplit_depth: int = 0,
) -> list[dict]:
    """Copy source metadata into deterministic child-row plans.

    This function plans the CSV/Pandas provenance contract without editing media.
    Every child keeps all original fields, including source URL and legacy
    columns, and receives additive resplit metadata.
    """
    plan = split_plan(
        duration_seconds,
        scroll_seconds,
        resplit_depth=resplit_depth,
    )
    children: list[dict] = []
    for index, (start, end) in enumerate(plan, start=1):
        child = dict(source_row)
        child["resplit_parent_id"] = str(parent_id)
        child["resplit_depth"] = int(resplit_depth) + 1
        child["resplit_segment_index"] = index
        child["resplit_start_seconds"] = start
        child["resplit_end_seconds"] = end
        child["resplit_derived_id"] = derived_segment_id(
            parent_id,
            index,
            start,
            end,
        )
        children.append(child)
    return children
