"""EP24 researcher-feed metadata helpers.

The active EP24 reprocessing inputs are researcher-recorded/split feed data.
The private country CSVs are canonicalized before analysis: human annotations
arrive in `entities` and `themes`, and the four legacy annotation columns are
removed by the verified migration in LaclauGPT-Private.

Pipeline stages must preserve every incoming column dynamically and append new
analysis fields. Only media identity is mandatory for media access:
- video_id
- allas_filename

EP24_REPROCESS_COLUMNS pins the canonical post-migration source order used by
contracts/tests. It is not a projection rule: extra columns must still survive.
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
    "entities",
    "themes",
    "video_duration",
    "researcher_note",
)

REQUIRED_MEDIA_COLUMNS: tuple[str, ...] = ("video_id", "allas_filename")

LEGACY_ALIASES: dict[str, tuple[str, ...]] = {
    "video_id": ("videoId",),
    "country": ("scrapedCountry",),
    "video_duration": ("videoDuration",),
    "author_username": ("authorUniqueId",),
}


def value(row: Mapping[str, Any], column: str, default: str = "") -> str:
    """Read a field, falling back to scraper-era identity aliases when needed."""
    for candidate in (column, *LEGACY_ALIASES.get(column, ())):
        raw = row.get(candidate, "")
        text = "" if raw is None else str(raw).strip()
        if text and text.lower() != "nan":
            return text
    return default


def source_metadata(row: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Return every non-empty field in the incoming row, in row order."""
    items: list[tuple[str, str]] = []
    for key in row.keys():
        raw = row.get(key, "")
        text = "" if raw is None else str(raw).strip()
        if text and text.lower() != "nan":
            items.append((str(key), text))
    return items


def stable_source_id(row: Mapping[str, Any]) -> str:
    """Stable readable ID using fields that actually exist."""
    candidates = ("new_id", "video_id", "country", "allas_filename")
    return "|".join(value(row, key) for key in candidates if value(row, key))
