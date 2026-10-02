"""Public-context codebooks nest authoritative URLs under ``provenance.sources``.

Regression guard for the loader defect where those URLs were dropped silently,
leaving ``entry.sources[].url`` empty and making every URL-driven resolver
(e.g. the EP24 english_label backfill) see zero sources.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roihu_codebooks import _normalize_item, load_codebook  # noqa: E402

URLS = [
    "https://en.wikipedia.org/wiki/Social_Democratic_Party_of_Croatia",
    "https://results.elections.europa.eu/en/national-results/croatia/2024-2029",
]


def _public_context_item(**overrides):
    item = {
        "kind": "entity",
        "label": "Socijaldemokratska partija Hrvatske",
        "aliases": ["Social Democratic Party of Croatia"],
        "country": "HR",
        "language": "hr",
        "status": "public-context",
        "provenance": {
            "origin": "web-research",
            "scope": "public-context",
            "retrieved_at": "2026-09-25",
            "sources": list(URLS),
        },
        "metadata": {"provenance_class": "public_context"},
        "id": "ep24.hr.entity.socijaldemokratska_partija_hrvatske",
    }
    item.update(overrides)
    return item


def test_provenance_sources_urls_survive_load():
    entry = _normalize_item(_public_context_item(), kind="entity",
                            default_country="HR", layer="country")
    assert entry is not None
    urls = [s.url for s in entry.sources if s.url]
    assert urls == URLS, f"provenance.sources URLs dropped: {urls!r}"


def test_provenance_metadata_is_still_preserved():
    entry = _normalize_item(_public_context_item(), kind="entity",
                            default_country="HR", layer="country")
    # the dict ref itself must survive so provenance metadata is not lost
    kinds = {s.source_type for s in entry.sources}
    assert "web-research" in kinds


def test_top_level_sources_still_win_and_are_not_duplicated():
    item = _public_context_item(sources=["https://example.org/a"])
    entry = _normalize_item(item, kind="entity", default_country="HR", layer="country")
    urls = [s.url for s in entry.sources if s.url]
    assert urls == ["https://example.org/a"]


def test_resolver_sees_a_url_for_a_provenance_only_entry(tmp_path):
    """The shape the EP24 backfill helper depends on: entry.sources[].url non-empty."""
    import json
    book = {
        "schema": "ep24-private-codebook-v4",
        "country_code": "HR",
        "language": "hr",
        "entries": [_public_context_item()],
    }
    path = tmp_path / "ep24_hr_private.json"
    path.write_text(json.dumps(book, ensure_ascii=False), encoding="utf-8")
    entries, _ = load_codebook(path, layer="country")
    entry = next(e for e in entries if e.label == "Socijaldemokratska partija Hrvatske")
    assert len([s for s in entry.sources if s.url]) == len(URLS)
