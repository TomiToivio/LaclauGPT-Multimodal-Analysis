"""EP24 reprocessing metadata helpers.

The per-country to_reprocess CSV is the source of truth. Researcher-feed files
are not scraper exports, and the VLLM experiment must not require synthetic
scraper-era fields or a guessed fixed metadata schema.

Only media identity is mandatory for the native-video test:
- video_id
- allas_filename

All other source columns are preserved verbatim and carried forward as metadata.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

REQUIRED_MEDIA_COLUMNS: tuple[str, ...] = ("video_id", "allas_filename")

# Fields currently observed in the Finland to_reprocess input. This list is
# descriptive, not a validator: extra/new fields must flow through automatically.
OBSERVED_FINLAND_COLUMNS: tuple[str, ...] = (
    "new_id",
    "old_id",
    "video_id",
    "allas_filename",
    "puhti_filename",
    "video_filename",
    "new_filename",
    "country",
    "source_type",
    "recording_date",
    "recording_datetime",
    "whisper_language",
    "political_preference",
    "sequence_number",
    "video_duration",
)

# Legacy aliases are compatibility readers only.
LEGACY_ALIASES: dict[str, tuple[str, ...]] = {
    "video_id": ("videoId",),
    "country": ("scrapedCountry",),
    "video_duration": ("videoDuration",),
}


def value(row: Mapping[str, Any], column: str, default: str = "") -> str:
    """Read a field, with legacy aliases only as a fallback."""
    candidates = (column, *LEGACY_ALIASES.get(column, ()))
    for candidate in candidates:
        raw = row.get(candidate, "")
        text = "" if raw is None else str(raw).strip()
        if text and text.lower() != "nan":
            return text
    return default


def source_metadata(row: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Return every non-empty source field in original row order."""
    items: list[tuple[str, str]] = []
    for key in row.keys():
        raw = row.get(key, "")
        text = "" if raw is None else str(raw).strip()
        if text and text.lower() != "nan":
            items.append((str(key), text))
    return items


def stable_source_id(row: Mapping[str, Any]) -> str:
    """Stable human-readable identifier using fields actually present."""
    preferred = ("new_id", "video_id", "country", "allas_filename")
    values = [value(row, field) for field in preferred]
    return "|".join(v for v in values if v)
