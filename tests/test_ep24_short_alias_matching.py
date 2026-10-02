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


# Swedish regression fixtures for issue #95. One-letter aliases are valid only
# when they resolve unambiguously inside the active country scope.
def _swedish_book(tmp_path: Path, entries: list[dict]) -> Path:
    path = tmp_path / "ep24_se_private.json"
    path.write_text(
        json.dumps(
            {
                "schema": "test-fixture",
                "country_code": "SE",
                "country": "Sweden",
                "language": "sv",
                "entries": entries,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


SWEDISH_PARTIES = [
    {
        "kind": "entity",
        "label": "Socialdemokraterna",
        "english_label": "Swedish Social Democratic Party",
        "aliases": ["S"],
        "status": "researcher-grounded",
    },
    {
        "kind": "entity",
        "label": "Moderaterna",
        "english_label": "Moderate Party",
        "aliases": ["M"],
        "status": "researcher-grounded",
    },
    {
        "kind": "entity",
        "label": "Sverigedemokraterna",
        "english_label": "Sweden Democrats",
        "aliases": ["SD"],
        "status": "researcher-grounded",
    },
]


def test_swedish_one_letter_alias_matches_only_as_a_token(tmp_path: Path) -> None:
    entries, _meta = load_codebook(_swedish_book(tmp_path, SWEDISH_PARTIES))
    block, prov = context_block("S går framåt i mätningen", entries, country="SE", language="sv")
    assert "Socialdemokraterna" in block
    assert any(item["label"] == "Socialdemokraterna" for item in prov["selected"])
    block_inside, _ = context_block("Stockholm växer", entries, country="SE", language="sv")
    assert block_inside == ""


def test_three_and_four_character_forms_do_not_match_inside_words(tmp_path: Path) -> None:
    entries, _meta = load_codebook(_book(tmp_path, PARTIES))
    assert _selected("voxpopuli diskuteras") == []
    assert _selected("psoriasis nämns") == []


def test_same_country_ambiguous_one_letter_alias_abstains(tmp_path: Path) -> None:
    ambiguous = [
        *SWEDISH_PARTIES,
        {
            "kind": "entity",
            "label": "Synthetic S Party",
            "aliases": ["S"],
            "status": "researcher-grounded",
        },
    ]
    entries, _meta = load_codebook(_swedish_book(tmp_path, ambiguous))
    block, prov = context_block("S går framåt", entries, country="SE", language="sv")
    assert block == ""
    assert prov["ambiguous_short_forms"] == ["s"]


def test_short_alias_collision_is_resolved_by_country_scope(tmp_path: Path) -> None:
    se_entries, _ = load_codebook(_swedish_book(tmp_path, SWEDISH_PARTIES))
    es_entries, _ = load_codebook(_book(tmp_path, [
        {
            "kind": "entity",
            "label": "Synthetic Spanish Social Party",
            "aliases": ["S"],
            "status": "researcher-grounded",
        }
    ]))
    combined = [*se_entries, *es_entries]
    block, prov = context_block("S går framåt", combined, country="SE", language="sv")
    assert "Socialdemokraterna" in block
    assert "Synthetic Spanish Social Party" not in block
    assert prov["ambiguous_short_forms"] == []


def test_unreviewed_short_alias_is_not_used_for_retrieval(tmp_path: Path) -> None:
    provisional = [
        {
            "kind": "entity",
            "label": "Synthetic Movement",
            "aliases": ["SM"],
            "review_state": "PROVISIONAL",
        }
    ]
    entries, _meta = load_codebook(_swedish_book(tmp_path, provisional))
    block, _prov = context_block("SM går framåt", entries, country="SE", language="sv")
    assert block == ""


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
