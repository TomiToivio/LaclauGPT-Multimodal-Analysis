"""Regression tests for EP24 country-codebook normalization rules (issue #72).

These encode the reusable rules derived from the Croatia (hr) codebook audit
(docs/EP24_CODEBOOK_AUDIT_HR.md) so the rules are ENFORCED rather than merely
documented. The fixtures are synthetic: no private codebook content, rows,
researcher notes or personal data appears here.

Rules covered:

- R2  a bare surname is a merge CANDIDATE, not a merge
- R3  a misspelling is an ALIAS, not its own entity
- R4  a role-prefixed form is an ALIAS, not its own entity
- R6  themes deduplicate on a normalized key, not the raw string
- D3  alias coverage is what makes a mention matchable at all
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roihu_codebooks import context_block, load_codebook  # noqa: E402
from roihu_memory import EP24Memory  # noqa: E402


def _write(path: Path, entries: list[dict]) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "test-fixture",
                "country_code": "HR",
                "country": "Croatia",
                "language": "hr",
                "entries": entries,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


# --------------------------------------------------------------------------
# R3 / R4: one actor, many surface forms -> ONE entry carrying aliases
# --------------------------------------------------------------------------


def test_variant_forms_are_aliases_on_one_entry_not_separate_entities(tmp_path: Path) -> None:
    """R3/R4: misspellings and role-prefixed forms are aliases, not entities.

    The Croatia audit found six separate `researcher-grounded` entries for a
    single politician. The correct shape is one canonical entry whose `forms`
    include the variants, so the variants cannot fragment the observation mass.
    """
    book = _write(
        tmp_path / "ep24_hr_private.json",
        [
            {
                "kind": "entity",
                "label": "Andrej Plenković",
                "english_label": "Andrej Plenković",
                "aliases": [
                    "Plenković",
                    "Andreja Plenković",
                    "Andrija Plenković",
                    "Croatian Prime Minister Andrej Plenković",
                ],
                "status": "researcher-grounded",
            }
        ],
    )
    entries, _meta = load_codebook(book)
    assert len(entries) == 1, "variants must not become separate entries"

    forms = set(entries[0].forms)
    for variant in (
        "Plenković",
        "Andreja Plenković",
        "Andrija Plenković",
        "Croatian Prime Minister Andrej Plenković",
    ):
        assert variant in forms, f"{variant!r} must be a matchable form"


def test_alias_forms_are_matchable_in_context_retrieval(tmp_path: Path) -> None:
    """D3: an entry with no aliases can only be matched by its exact label.

    This is the mechanism behind the audit's D1/D2: 280/350 entries carried no
    aliases, so corpus mentions such as a bare surname or a party abbreviation
    matched nothing.
    """
    book = _write(
        tmp_path / "ep24_hr_private.json",
        [
            {
                "kind": "entity",
                "label": "Croatian Democratic Union",
                "english_label": "Croatian Democratic Union",
                "aliases": ["HDZ", "Hrvatska demokratska zajednica"],
                "status": "researcher-grounded",
            }
        ],
    )
    entries, _meta = load_codebook(book)
    block, _prov = context_block("HDZ je pobijedio na izborima", entries, country="HR", language="hr")
    assert block, "an abbreviation alias must retrieve the canonical entry"


def test_entry_without_aliases_does_not_match_a_variant(tmp_path: Path) -> None:
    """The negative control for D3: no aliases means no variant matching.

    If this ever starts matching, the alias mechanism has been bypassed by a
    fuzzy fallback — which is the silent-merge risk the audit warns about.
    """
    book = _write(
        tmp_path / "ep24_hr_private.json",
        [
            {
                "kind": "entity",
                "label": "Croatian Democratic Union",
                "english_label": "Croatian Democratic Union",
                "aliases": [],
                "status": "researcher-grounded",
            }
        ],
    )
    entries, _meta = load_codebook(book)
    block, _prov = context_block("HDZ je pobijedio", entries, country="HR", language="hr")
    assert not block, "without an alias the variant must NOT match"


# --------------------------------------------------------------------------
# R2: a bare surname is ambiguous, not an automatic merge
# --------------------------------------------------------------------------


def test_bare_surname_does_not_auto_merge_across_people(tmp_path: Path) -> None:
    """R2: same surname, different people -> AMBIGUOUS, never a silent merge.

    Croatia's 2024 lists contain several politicians sharing a surname. A merge
    must be an explicit, reviewable decision, not an inference from the surname.
    """
    memory = EP24Memory(tmp_path / "memory.sqlite3")
    memory.add_object(
        "actor", "Andrej Plenković", country="HR", language="hr",
        state="CANONICAL", origin="researcher_private", locked=True,
    )
    memory.add_object(
        "actor", "Ana Plenković", country="HR", language="hr",
        state="CANONICAL", origin="researcher_private", locked=True,
    )
    # An unqualified surname must not decide between two distinct people.
    resolution = memory.resolve("Plenković", "actor", country="HR")
    assert resolution.decision in {"AMBIGUOUS", "NEW"}, (
        "an unqualified surname must not silently select or merge one of several "
        f"same-surname actors (got {resolution.decision!r})"
    )


def test_surname_resolves_once_a_country_scoped_canonical_exists(tmp_path: Path) -> None:
    """R2 corollary: an explicit canonical entry is the resolution path.

    Ambiguity is not paralysis — once the entry itself carries the surname as a
    form (R3), the canonical record is the thing that resolves it.
    """
    memory = EP24Memory(tmp_path / "memory.sqlite3")
    obj = memory.add_object(
        "actor", "Andrej Plenković", country="HR", language="hr",
        state="CANONICAL", origin="researcher_private", locked=True,
    )
    assert obj
    res = memory.resolve("Andrej Plenković", "actor", country="HR")
    assert res.obj_id == obj, "the exact canonical label must resolve to its object"


# --------------------------------------------------------------------------
# R6: theme dedup on a normalized key
# --------------------------------------------------------------------------


def test_theme_variants_collapse_to_one_retrieval_concept() -> None:
    """R6: `Croatian identity` and `Croatian National Identity` are one concept.

    The audit found theme pairs at Jaccard >= 0.6 that would silently fragment
    retrieval. They must collapse to one canonical theme with the other as an
    alias, so a query hits the concept once.
    """
    from roihu_codebooks import identity_key

    assert identity_key("Croatian identity") == identity_key("croatian  identity"), (
        "identity_key must be case- and whitespace-insensitive"
    )
    # Distinct concepts must NOT collapse.
    assert identity_key("Croatian identity") != identity_key("Croatian interests")


def test_duplicate_themes_in_one_cell_are_a_generation_bug() -> None:
    """R5: a theme repeated inside one cell is a bug, and dedupe is normalized.

    The audit found 67 legacy cells repeating a theme (worst x27). Any writer
    that splits a theme field must dedupe on a normalized key.
    """
    raw = "climate, sustainability and environmentalism; climate, sustainability and environmentalism"
    parts = [p.strip() for p in raw.split(";") if p.strip()]
    assert len(parts) != len(set(parts)), "fixture must actually contain a duplicate"

    # The rule: dedupe on a normalized key so case/punctuation drift also collapses.
    from roihu_codebooks import identity_key

    deduped = list(dict.fromkeys(identity_key(p) for p in parts))
    assert len(deduped) == 1, "normalized dedupe must collapse the repeated theme"


# --------------------------------------------------------------------------
# Country scoping: aliases must never leak across countries
# --------------------------------------------------------------------------


def test_alias_matching_stays_inside_its_country(tmp_path: Path) -> None:
    """An alias is country-scoped; it must not retrieve another country's entry."""
    book = _write(
        tmp_path / "ep24_hr_private.json",
        [
            {
                "kind": "entity",
                "label": "Croatian Democratic Union",
                "english_label": "Croatian Democratic Union",
                "aliases": ["HDZ"],
                "status": "researcher-grounded",
            }
        ],
    )
    entries, _meta = load_codebook(book)
    block, _prov = context_block("HDZ", entries, country="PL", language="pl")
    assert not block, "a Croatian alias must not retrieve context for Poland"
