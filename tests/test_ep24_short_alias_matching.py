#!/usr/bin/env python3
"""Regression tests: short party acronyms must resolve, safely (issue #72).

Context
-------
The EP24 country-codebook build drops any surface form shorter than 5
characters (``--min-length`` default 5 in ``analysis/ep24/codebooks/
build_all_country_codebooks.py``, and the same ``min_length=5`` applied three
times in ``merge_research_codebooks.py`` — on the label and twice on aliases).

Across the 10 study countries that silently removed 938 surface forms,
including the main parties of every country: ``HDZ``, ``SDP``, ``DP``, ``PP``,
``PSOE``, ``Vox``, ``PNV``, ``ERC``, ``Most`` … The audit (docs/
ep24_country_audits/HR.md, Pass 2 addendum) argues the guard is unnecessary,
because the runtime matcher already handles short forms safely.

These tests pin that runtime behaviour, so the guard can be removed without
reintroducing the substring hazard it was guarding against.

This file is PUBLIC: synthetic fixtures only, no private codebook content.

Run:  python -m pytest tests/test_ep24_short_alias_matching.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roihu_codebooks import context_block, load_codebook, score_entry  # noqa: E402


def _book(tmp_path: Path, entries: list[dict]) -> Path:
    path = tmp_path / "ep24_es_private.json"
    path.write_text(
        json.dumps(
            {
                "schema": "test-fixture",
                "country_code": "ES",
                "country": "Spain",
                "language": "es",
                "entries": entries,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


# The canonical shape the audit prescribes: a long, readable label carrying the
# acronym as an ALIAS. The alias is what the build guard currently deletes.
PARTIES = [
    {
        "kind": "entity",
        "label": "Partido Popular",
        "english_label": "People's Party",
        "aliases": ["PP", "populares"],
        "status": "researcher-grounded",
    },
    {
        "kind": "entity",
        "label": "Vox",
        "aliases": ["VOX"],
        "status": "researcher-grounded",
    },
    {
        "kind": "entity",
        "label": "Partido Socialista Obrero Español",
        "english_label": "Spanish Socialist Workers' Party",
        "aliases": ["PSOE", "Psoe"],
        "status": "researcher-grounded",
    },
]


def _selected(query: str, entries) -> list[str]:
    _block, prov = context_block(query, entries, country="ES")
    return [s["label"] for s in prov["selected"]]


def test_a_two_letter_acronym_matches_as_a_word(tmp_path: Path) -> None:
    """ES-01: `PP` must retrieve Partido Popular when it appears as a word."""
    entries, _meta = load_codebook(_book(tmp_path, PARTIES))
    assert "Partido Popular" in _selected("El PP gana en Madrid", entries)


def test_a_three_letter_acronym_matches(tmp_path: Path) -> None:
    """ES-01: `Vox` (3 chars) must retrieve Vox."""
    entries, _meta = load_codebook(_book(tmp_path, PARTIES))
    assert "Vox" in _selected("Hoy habla Vox en el mitin", entries)


def test_a_five_letter_acronym_matches(tmp_path: Path) -> None:
    """ES-01: the guard's own threshold boundary — a 4-char form must also work."""
    entries, _meta = load_codebook(_book(tmp_path, PARTIES))
    assert "Partido Socialista Obrero Español" in _selected("PSOE sube en las encuestas", entries)


def test_a_two_letter_alias_does_not_match_inside_an_unrelated_word(tmp_path: Path) -> None:
    """ES-02 (the reason the guard existed): `PP` inside a word must NOT match.

    This is the negative control. If the guard is removed and this test fails,
    the substring hazard has been reintroduced and short aliases are unsafe.
    """
    entries, _meta = load_codebook(_book(tmp_path, PARTIES))
    assert _selected("aproposito de nada", entries) == []


def test_a_two_letter_alias_does_not_match_across_a_word_boundary(tmp_path: Path) -> None:
    """ES-02: `PP` glued to other letters must not match, on either side."""
    entries, _meta = load_codebook(_book(tmp_path, PARTIES))
    assert _selected("PPx y xPP sin espacios", entries) == []


def test_short_alias_matching_stays_inside_its_country(tmp_path: Path) -> None:
    """ES-03: a Spanish acronym must not retrieve context for another country.

    Short acronyms collide across countries (`PP`, `DP`, `PSOE`), so scoping is
    the other half of the fix: the alias must survive, but only inside its own
    country.
    """
    entries, _meta = load_codebook(_book(tmp_path, PARTIES))
    block, _prov = context_block("El PP gana en Madrid", entries, country="PL", language="pl")
    assert block == "", "a Spanish acronym must not produce Polish context"


def test_long_label_still_matches_by_substring(tmp_path: Path) -> None:
    """Control: the >=3 substring rule must remain intact for long forms."""
    entries, _meta = load_codebook(_book(tmp_path, PARTIES))
    assert "Partido Popular" in _selected("el partido popular de España", entries)


def test_score_entry_prefers_exact_alias_over_token_overlap(tmp_path: Path) -> None:
    """The alias path returns a full score, not a partial token-overlap guess."""
    entries, _meta = load_codebook(_book(tmp_path, PARTIES))
    popular = next(e for e in entries if e.label == "Partido Popular")
    assert score_entry("El PP gana", popular) == 1.0, (
        "an exact alias hit must score 1.0; a partial score means the alias was "
        "matched by token overlap rather than as an alias"
    )


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
