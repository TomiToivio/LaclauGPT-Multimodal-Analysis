"""EP24 researcher-feed metadata helpers.

The active EP24 reprocessing inputs are researcher-recorded/split feed data.
The input CSV itself is authoritative. Pipeline stages must preserve every
incoming column dynamically and append new analysis fields.

Only media identity is mandatory for media stages:
- video_id
- allas_filename

Every other incoming column flows through dynamically.

EP24_REPROCESS_COLUMNS describes the canonical researcher-feed field order. It is
deliberately NOT a validator: newer inputs may carry extra columns (and older
ones may miss some), and those must still be preserved and forwarded. It is kept
because downstream stages (for example roihu_rdf.py) rely on it to order the
identity columns, and because it pins the canonical contract in tests.
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

# The canonical source columns of the researcher-feed reprocess input
# (analysis/ep24_reprocess/data/to_reprocess/ep24_<country>.csv).
#
# This is the KEEP-SCHEMA as a contract constant, not a filter. The active
# pipeline is deliberately dynamic: every incoming column is preserved as-is and
# analysis fields are appended, so a country whose CSV carries extra or slightly
# differently named columns still flows through untouched (see
# ``source_metadata``). This tuple exists so stages and tests can name, document
# and validate the canonical schema in one place instead of re-declaring the
# column names independently. Do not use it to project a row into a fixed shape.
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

LEGACY_ALIASES: dict[str, tuple[str, ...]] = {
    "video_id": ("videoId",),
    "country": ("scrapedCountry",),
    "video_duration": ("videoDuration",),
    "author_username": ("authorUniqueId",),
}


def value(row: Mapping[str, Any], column: str, default: str = "") -> str:
    """Read a field, falling back to legacy aliases only when needed."""
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
