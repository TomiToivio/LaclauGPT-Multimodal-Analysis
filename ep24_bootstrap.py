"""EP24 bootstrap: canonical `entities`/`themes` seeds before Step 1 (#64).

The issue makes this non-negotiable:

    Before Step 1 begins, bootstrap must create canonical `entities` by merging
    `new_entity` + `researcher_new_persons`, and canonical `themes` by merging
    `new_theme` + `researcher_new_themes`. Every analysis step must receive these
    merged fields. Keep the four original researcher/source columns unchanged
    alongside the merged canonical fields for provenance.

Two details from the real Finland input (verified against the LFS object, not
assumed) shape this module:

- ``researcher_new_persons`` and ``researcher_new_themes`` are the literal
  string ``'[]'`` when empty, not an empty string. A naive ``+`` join would put
  a literal ``[]`` into the canonical field, so empty-container literals must be
  treated as empty.
- ``new_entity`` is frequently empty while ``new_theme`` is populated, so the two
  merges are computed independently rather than assumed to move together.

The merges are additive and lossless: the source columns are never modified or
removed, and every original label is preserved in the canonical field as written.
Normalization against a country codebook is applied on top where given, without
replacing the original label.
"""
from __future__ import annotations

import ast
import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

SEED_SEPARATOR = "; "

#: Values that mean "this field carries no values". `'[]'` and `'{}'` are what the
#: real input uses for an empty list/dict, and `nan`/`None` appear after a pandas
#: round trip with `keep_default_na=False` off.
_EMPTY_LITERALS = frozenset({"", "[]", "{}", "null", "none", "nan"})


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _looks_empty(text: str) -> bool:
    return text.strip().lower() in _EMPTY_LITERALS


def split_seed_values(raw: Any) -> list[str]:
    """Split a researcher seed field into individual labels.

    Handles the shapes the real data uses: a JSON list literal (``'["A", "B"]'``),
    an empty list literal (``'[]'``), a ``;``/`,`-separated string, and a plain
    single label. Order is preserved and duplicates are dropped.
    """
    text = _as_text(raw)
    if _looks_empty(text):
        return []

    parsed: list[str] | None = None
    if text[:1] in "[{":
        try:
            loaded = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            try:
                loaded = json.loads(text)
            except (ValueError, json.JSONDecodeError):
                loaded = None
        if isinstance(loaded, (list, tuple)):
            parsed = [_as_text(item) for item in loaded]
        elif isinstance(loaded, dict):
            parsed = [_as_text(item) for item in loaded.values()]
        elif isinstance(loaded, str):
            parsed = [loaded]

    if parsed is None:
        # Not a literal: treat separators as boundaries.
        parsed = [part for part in re.split(r"[;,]", text)]

    out: list[str] = []
    for value in parsed:
        cleaned = _as_text(value)
        if cleaned and not _looks_empty(cleaned) and cleaned not in out:
            out.append(cleaned)
    return out


def merge_seed_fields(*fields: Any, separator: str = SEED_SEPARATOR) -> str:
    """Merge several seed fields into one canonical, human-readable value.

    Lossless: every original label appears in the result verbatim, in the order
    the fields and their values were given. An empty result is the empty string
    (not ``'[]'``), so a downstream stage sees a genuinely empty field.
    """
    seen: list[str] = []
    for field in fields:
        for value in split_seed_values(field):
            if value not in seen:
                seen.append(value)
    return separator.join(seen)


def build_entities(row: Mapping[str, Any]) -> str:
    """Canonical `entities` = `new_entity` + `researcher_new_persons`."""
    return merge_seed_fields(row.get("new_entity", ""), row.get("researcher_new_persons", ""))


def build_themes(row: Mapping[str, Any]) -> str:
    """Canonical `themes` = `new_theme` + `researcher_new_themes`."""
    return merge_seed_fields(row.get("new_theme", ""), row.get("researcher_new_themes", ""))


def apply_codebook_labels(
    canonical: str,
    codebook: Mapping[str, Any] | None,
    *,
    registry_key: str = "labels",
) -> str:
    """Return the canonical value with codebook spelling applied where known.

    Original labels are preserved: a label the codebook does not know stays
    exactly as the researcher wrote it, and a known label is replaced by the
    codebook's canonical spelling only when the two differ in spelling rather
    than meaning. Never drops a label.
    """
    if not canonical:
        return canonical
    if not codebook:
        return canonical

    known = codebook.get(registry_key) or {}
    if not isinstance(known, Mapping):
        return canonical

    # casefolded alias -> canonical spelling
    lookup = {_as_text(alias).casefold(): _as_text(canon) for alias, canon in known.items()}
    out: list[str] = []
    for value in split_seed_values(canonical):
        replacement = lookup.get(value.casefold())
        out.append(replacement if replacement else value)
    return SEED_SEPARATOR.join(out)


def build_bootstrap_row(
    row: Mapping[str, Any],
    *,
    country: str,
    source_row_index: int,
    entity_codebook: Mapping[str, Any] | None = None,
    theme_codebook: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a copy of `row` with the canonical merge fields added.

    The input mapping is not mutated. Every source column is carried through
    unchanged, which is what the tests assert.
    """
    from ep24_db import record_id_for

    enriched = dict(row)
    video_id = _as_text(row.get("video_id", ""))
    enriched["record_id"] = record_id_for(country, video_id, source_row_index)
    enriched["entities"] = apply_codebook_labels(build_entities(row), entity_codebook)
    enriched["themes"] = apply_codebook_labels(build_themes(row), theme_codebook)
    return enriched


def bootstrap_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    country: str,
    entity_codebook: Mapping[str, Any] | None = None,
    theme_codebook: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Bootstrap a whole country's rows, numbering them by source order."""
    return [
        build_bootstrap_row(
            row,
            country=country,
            source_row_index=index,
            entity_codebook=entity_codebook,
            theme_codebook=theme_codebook,
        )
        for index, row in enumerate(rows)
    ]


def provenance_note() -> dict[str, str]:
    """What bootstrap did, for the audit trail the issue asks for."""
    return {
        "entities_from": "new_entity + researcher_new_persons",
        "themes_from": "new_theme + researcher_new_themes",
        "sources_preserved": "new_entity, researcher_new_persons, new_theme, researcher_new_themes",
        "researcher_note": "context/provenance, not automatic ground truth",
    }
