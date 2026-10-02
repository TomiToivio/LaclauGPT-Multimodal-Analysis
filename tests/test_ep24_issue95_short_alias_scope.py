#!/usr/bin/env python3
"""Issue #95: safe country-scoped short political aliases in EP24 retrieval.

Public synthetic fixtures only. No private codebook content.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roihu_codebooks import CodebookEntry, context_block, score_entry  # noqa: E402
from roihu_identity import resolve_surface  # noqa: E402


def _party(
    label: str,
    alias: str,
    *,
    country: str = "SE",
    language: str = "sv",
    reviewed: bool = True,
    metadata: dict | None = None,
) -> CodebookEntry:
    return CodebookEntry(
        entry_id=f"{country}:{label}",
        kind="entity",
        label=label,
        aliases=[alias],
        country=country,
        source_languages=[language],
        entity_type="party",
        review_state="CANONICAL" if reviewed else "PROVISIONAL",
        origin="researcher" if reviewed else "public_context",
        locked=reviewed,
        metadata=metadata or {},
    )


SWEDISH_PARTIES = [
    _party("Socialdemokraterna", "S"),
    _party("Moderaterna", "M"),
    _party("Vänsterpartiet", "V"),
    _party("Centerpartiet", "C"),
    _party("Liberalerna", "L"),
    _party("Kristdemokraterna", "KD"),
    _party("Miljöpartiet", "MP"),
    _party("Sverigedemokraterna", "SD"),
]


def _labels(query: str, entries, *, country: str = "SE", language: str = "sv", election: str = "") -> set[str]:
    _block, provenance = context_block(
        query,
        entries,
        country=country,
        language=language,
        election=election,
        limit=20,
    )
    return {item["label"] for item in provenance["selected"]}


def test_swedish_reviewed_party_abbreviations_retrieve_on_boundaries() -> None:
    expected = {
        "S": "Socialdemokraterna",
        "M": "Moderaterna",
        "V": "Vänsterpartiet",
        "C": "Centerpartiet",
        "L": "Liberalerna",
        "KD": "Kristdemokraterna",
        "MP": "Miljöpartiet",
        "SD": "Sverigedemokraterna",
    }
    for alias, label in expected.items():
        assert label in _labels(f"{alias} presenterar sin EU-politik", SWEDISH_PARTIES)


def test_one_letter_aliases_do_not_match_inside_words() -> None:
    assert "Socialdemokraterna" not in _labels("Sverige röstar i EU-valet", SWEDISH_PARTIES)
    assert "Moderaterna" not in _labels("migration och ekonomi", SWEDISH_PARTIES)
    assert "Liberalerna" not in _labels("liberal politik diskuteras", SWEDISH_PARTIES)


def test_three_and_four_character_short_forms_are_boundary_matched_too() -> None:
    hdz = _party("Hrvatska demokratska zajednica", "HDZ", country="HR", language="hr")
    most = _party("Most nezavisnih lista", "Most", country="HR", language="hr")
    assert score_entry("HDZ je pobijedio", hdz, language="hr") == 1.0
    assert score_entry("Most je osvojio glasove", most, language="hr") == 1.0
    assert score_entry("HDZx nije stranka", hdz, language="hr") < 1.0
    assert score_entry("Mostar je grad", most, language="hr") < 1.0


def test_cross_country_collision_is_scoped_before_scoring() -> None:
    swedish_s = _party("Socialdemokraterna", "S", country="SE")
    foreign_s = _party("Synthetic Foreign Party", "S", country="FI", language="fi")
    assert _labels("S presenterar sin EU-politik", [swedish_s, foreign_s], country="SE") == {
        "Socialdemokraterna"
    }


def test_unreviewed_short_label_is_not_promoted_into_retrieval() -> None:
    provisional = CodebookEntry(
        entry_id="SE:synthetic-short-label",
        kind="entity",
        label="S",
        aliases=[],
        country="SE",
        source_languages=["sv"],
        entity_type="party",
        review_state="PROVISIONAL",
        origin="public_context",
        locked=False,
        metadata={},
    )
    assert score_entry("S presenterar sin EU-politik", provisional, language="sv") == 0.0
    provisional.metadata["reviewed_short_aliases"] = ["S"]
    assert score_entry("S presenterar sin EU-politik", provisional, language="sv") == 1.0


def test_optional_language_scope_can_restrict_a_reviewed_short_alias() -> None:
    entry = _party(
        "Socialdemokraterna",
        "S",
        metadata={"short_alias_languages": {"S": ["sv"]}},
    )
    assert score_entry("S presenterar sin EU-politik", entry, language="sv") == 1.0
    assert score_entry("S presents its EU policy", entry, language="en") == 0.0


def test_optional_election_scope_can_restrict_a_reviewed_short_alias() -> None:
    entry = _party(
        "Socialdemokraterna",
        "S",
        metadata={"short_alias_elections": {"S": ["EP2024-SE"]}},
    )
    assert score_entry("S presenterar sin EU-politik", entry, election="EP2024-SE") == 1.0
    assert score_entry("S presenterar sin EU-politik", entry, election="RIX2026-SE") == 0.0


def test_ambiguous_one_letter_identity_abstains_instead_of_picking_a_winner() -> None:
    first = _party("Synthetic A", "S")
    second = _party("Synthetic B", "S")
    result = resolve_surface("S", [first, second], country="SE", kinds=["entity"])
    assert result["decision"] == "AMBIGUOUS"


def test_background_context_remains_explicitly_non_evidence() -> None:
    _block, provenance = context_block("SD talar om EU", SWEDISH_PARTIES, country="SE", language="sv")
    assert provenance["evidence_role"] == "background_context_not_source_evidence"
    assert provenance["selection_method"] == "deterministic_lexical_v3_scoped_short_aliases"
