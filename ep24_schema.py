"""Canonical EP24 reprocessing input schema.

The active EP24 pipeline consumes researcher-recorded feed clips that were split
from mobile ethnography recordings. These rows are not scraper exports.

Source of truth: analysis/ep24_reprocess/data/to_reprocess/ep24_<country>.csv
as produced by roihu_clean.py.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

EP24_REPROCESS_COLUMNS: tuple[str, ...] = (
    "country",
    "author_username",
    "account_type",
    "source_type",
    "source_recording",
    "video_id",
    "sequence_number",
    "political_preference",
    "allas_filename",
    "new_entity",
    "new_theme",
    "video_duration",
    "researcher_new_persons",
    "researcher_new_themes",
    "researcher_note",
)

REQUIRED_MEDIA_COLUMNS: tuple[str, ...] = ("video_id", "allas_filename")

# Legacy aliases are read-only compatibility only. New outputs and docs should
# use the canonical names above.
LEGACY_ALIASES: dict[str, tuple[str, ...]] = {
    "country": ("scrapedCountry",),
    "author_username": ("authorUniqueId",),
    "video_id": ("videoId",),
    "video_duration": ("videoDuration",),
}


def value(row: Mapping[str, Any], column: str, default: str = "") -> str:
    """Read a canonical field, falling back to a legacy alias if necessary."""
    candidates = (column, *LEGACY_ALIASES.get(column, ()))
    for candidate in candidates:
        raw = row.get(candidate, "")
        text = "" if raw is None else str(raw).strip()
        if text and text.lower() != "nan":
            return text
    return default


def stable_source_id(row: Mapping[str, Any]) -> str:
    """Human-readable stable identifier for logs/provenance."""
    return "|".join(
        item for item in (
            value(row, "country"),
            value(row, "author_username"),
            value(row, "video_id"),
            value(row, "source_recording"),
            value(row, "allas_filename"),
        )
        if item
    )
