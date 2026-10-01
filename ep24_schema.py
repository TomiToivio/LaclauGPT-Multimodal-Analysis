"""Canonical EP24 reprocessing metadata helpers.

The active reprocessing corpus is researcher-recorded/split feed data, not a
scraper export. The 15-field keep-schema below is authoritative for every
country. Legacy aliases exist only for backwards-compatible reads.
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

LEGACY_ALIASES: dict[str, tuple[str, ...]] = {
    "country": ("scrapedCountry",),
    "author_username": ("authorUniqueId",),
    "video_id": ("videoId",),
    "video_duration": ("videoDuration",),
}


def value(row: Mapping[str, Any], column: str, default: str = "") -> str:
    """Read a canonical field, falling back to a legacy alias when necessary."""
    candidates = (column, *LEGACY_ALIASES.get(column, ()))
    for candidate in candidates:
        raw = row.get(candidate, "")
        text = "" if raw is None else str(raw).strip()
        if text and text.lower() != "nan":
            return text
    return default


def source_metadata(row: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Return non-empty canonical source metadata in authoritative schema order."""
    items: list[tuple[str, str]] = []
    for column in EP24_REPROCESS_COLUMNS:
        text = value(row, column)
        if text:
            items.append((column, text))
    return items


def stable_source_id(row: Mapping[str, Any]) -> str:
    """Human-readable stable identifier for logs/provenance."""
    return "|".join(
        item
        for item in (
            value(row, "country"),
            value(row, "author_username"),
            value(row, "video_id"),
            value(row, "source_recording"),
            value(row, "allas_filename"),
        )
        if item
    )
