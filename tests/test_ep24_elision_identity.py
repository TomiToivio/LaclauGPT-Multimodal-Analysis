#!/usr/bin/env python3
"""Regression tests: apostrophe-elision identity folding must be safe (#102).

Context
-------
`identity_key` normalises NFC, casefolds and collapses whitespace, and nothing
else — deliberately, because punctuation is analytically meaningful and
stripping it was the false-merge defect (`tests/test_identity_normalization.py`
pins that). But French/Italian elision is written with either the ASCII
apostrophe (keyboards, ASR) or U+2019 (word processors, iOS autocorrect, much
news web output), and the two are canonically unrelated (`Po` vs `Pf`), so NFC
cannot unify them:

    identity_key("Besoin d'Europe") != identity_key("Besoin d\u2019Europe")

That makes one entity two identities: two `CB-` entry ids, two memory objects,
both CANONICAL. Measured in the real FR corpus at 5,826 typographic occurrences
(merged PR #100, docs/ep24_country_audits/FR.md Finding 4).

The fix is a **separate, lossy comparison key** (`elision_key`) used where a
lossy comparison is safe — the merge/duplicate path — while `identity_key`,
which is the **id seed**, does not move. This file pins both halves plus the
safety boundary the issue asks for.

This file is PUBLIC: synthetic fixtures only, no private codebook content.

Run:  python -m pytest tests/test_ep24_elision_identity.py -v
"""
from __future__ import annotations

import json
import sys
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import roihu_codebooks as cb  # noqa: E402

ASCII = "\u0027"  # '
RIGHT = "\u2019"  # ’
LEFT = "\u2018"  # ‘
MODIFIER = "\u02bc"  # ʼ

SAFE_FAMILY = (ASCII, RIGHT, LEFT, MODIFIER)


def _write(root: Path, name: str, entries: list[dict]) -> None:
    (root / "codebooks").mkdir(parents=True, exist_ok=True)
    (root / "codebooks" / name).write_text(
        json.dumps({"schema": "fixture", "entries": entries}, ensure_ascii=False),
        encoding="utf-8",
    )


def _merged_labels(root: Path, country: str = "FR") -> list[str]:
    _entries, _meta = cb.load_profile(root, country)
    return [e.label for e in _entries]


# --------------------------------------------------------------------------- #
# the defect, stated as an executable claim
# --------------------------------------------------------------------------- #


def test_the_defect_reproduces_before_the_fix() -> None:
    """FR-01: the two spellings are two identity keys — this is why #102 exists."""
    a, b = f"Besoin d{ASCII}Europe", f"Besoin d{RIGHT}Europe"
    assert cb.identity_key(a) != cb.identity_key(b)
    assert cb.elision_key(a) == cb.elision_key(b)


@pytest.mark.parametrize("variant", SAFE_FAMILY)
def test_every_elision_apostrophe_folds_to_one_key(variant: str) -> None:
    """FR-02: all four elision apostrophes denote the same elision."""
    assert cb.elision_key(f"l{ASCII}Union") == cb.elision_key(f"l{variant}Union")


def test_fold_picks_the_project_apostrophe_not_the_typographic_one() -> None:
    """FR-03: the fold normalises ONTO the project's ASCII apostrophe.

    Which form is canonical is a researcher decision the FR audit left open, but
    the project's own labels already write ASCII, so folding must converge there
    — otherwise a fold would *introduce* typographic forms into a clean book.
    """
    assert cb.elision_key(f"l{RIGHT}Union") == cb.identity_key(f"l{ASCII}Union")
    assert cb.elision_key(f"l{RIGHT}Union") == f"l{ASCII}union"


# --------------------------------------------------------------------------- #
# the safety boundary — what must NOT be folded
# --------------------------------------------------------------------------- #


def test_prime_is_not_an_apostrophe() -> None:
    """FR-04: `′` means minutes/feet. Folding it would be a semantic change.

    This is the boundary the hygiene module already draws; the fold must agree
    with it rather than being a broader punctuation stripper.
    """
    assert cb.elision_key("5\u20325") == cb.identity_key("5\u20325")
    assert cb.elision_key("5\u20325") != cb.elision_key(f"5{ASCII}5")


def test_grave_accent_is_not_an_apostrophe() -> None:
    """` is a mark, not an elision."""
    assert cb.elision_key("`quote`") == cb.identity_key("`quote`")


def test_double_quotes_are_not_folded_into_apostrophes() -> None:
    """The guillemet/quote family is a different character class."""
    for char in ("\u201c", "\u201d"):
        assert cb.elision_key(f"{char}x{char}") == cb.identity_key(f"{char}x{char}")


def test_dash_family_is_not_folded() -> None:
    """A hyphen can join a double-barrelled surname, so dashes stay distinct."""
    for dash in ("\u002d", "\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212"):
        assert cb.elision_key(f"a{dash}b") == cb.identity_key(f"a{dash}b")


def test_no_broad_punctuation_stripping_is_introduced() -> None:
    """FR-05: the acceptance criterion — the fold is narrow by measurement.

    Folding must change *only* strings that contain an elision apostrophe.
    Anything else must be byte-identical to `identity_key`, or a broad stripper
    has been smuggled in.
    """
    controls = [
        "Grüne",
        "Łukasz Köhler",
        "Možemo!",
        "Åsa (MEP)",
        "Sanchez",
        "Sánchez",
        "People's Party",   # English possessive: has an apostrophe, still folds
        "Election",
        "Straße",
        "a-b",
        "l'union",          # already ASCII: folds to itself
    ]
    for value in controls:
        folded = cb.elision_key(value)
        if any(c in SAFE_FAMILY for c in value):
            # contains an elision apostrophe: may differ only by that character
            assert folded == cb.identity_key(value.replace(RIGHT, ASCII).replace(LEFT, ASCII).replace(MODIFIER, ASCII))
        else:
            assert folded == cb.identity_key(value), value


def test_accents_and_qualifiers_are_still_not_stripped() -> None:
    """The pre-existing rule is untouched: apostrophes and accents are different questions."""
    for a, b in (("Grüne", "Grune"), ("Možemo", "Mozemo"), ("Åsa (MEP)", "Asa"), ("Puolue (A)", "Puolue (B)")):
        assert cb.elision_key(a) != cb.elision_key(b)
        assert cb.identity_key(a) != cb.identity_key(b)


def test_nfc_and_casefolding_still_apply_to_the_fold() -> None:
    """The fold composes with the existing normalisation, not instead of it."""
    composed, decomposed = "Café", "Cafe\u0301"
    assert cb.elision_key(composed) == cb.elision_key(decomposed)
    assert cb.elision_key("L'UNION") == cb.elision_key(f"l{RIGHT}union")


# --------------------------------------------------------------------------- #
# the id must not move — the migration requirement
# --------------------------------------------------------------------------- #


def test_identity_key_is_unchanged_for_apostrophe_labels() -> None:
    """FR-06: `identity_key` is the id seed, so it must stay byte-compatible.

    This is the migration criterion: if folding were done inside `identity_key`,
    every stored id whose label carries a typographic apostrophe would move.
    Pinning the old value here is what makes that a deliberate decision.
    """
    assert cb.identity_key(f"Besoin d{RIGHT}Europe") == f"besoin d{RIGHT}europe"
    assert cb.identity_key(f"Besoin d{ASCII}Europe") == f"besoin d{ASCII}europe"
    # and they still differ, i.e. no id movement happened as a side effect
    assert cb.identity_key(f"Besoin d{RIGHT}Europe") != cb.identity_key(f"Besoin d{ASCII}Europe")


def test_entry_id_does_not_move_for_a_typographic_label() -> None:
    """FR-07: an entry whose label carries U+2019 keeps the id it had before #102."""
    typographic = f"Besoin d{RIGHT}Europe"
    eid = cb._entry_id({}, "entity", typographic, "FR")
    seed = "|".join(("FR", "entity", typographic.casefold(), ""))
    import hashlib

    assert eid == "CB-" + hashlib.sha256(seed.encode()).hexdigest()[:18]


def test_explicit_id_is_still_never_recomputed() -> None:
    """The escape hatch survives: an entry carrying its own id keeps it."""
    import tempfile

    root = Path(tempfile.mkdtemp())
    _write(root, "ep24_common_private.json", [])
    _write(root, "ep24_fr_private.json", [
        {"kind": "entity", "id": "fr-human-1", "label": f"Besoin d{RIGHT}Europe", "country": "FR"},
    ])
    entries, _meta = cb.load_profile(root, "FR")
    assert [e.entry_id for e in entries] == ["fr-human-1"]


# --------------------------------------------------------------------------- #
# the merge actually collapses the pair, and says so
# --------------------------------------------------------------------------- #


def test_merge_collapses_the_two_spellings_to_one_entry() -> None:
    """FR-08: the point of the change — one entity, one entry, not two."""
    import tempfile

    root = Path(tempfile.mkdtemp())
    _write(root, "ep24_common_private.json", [])
    _write(root, "ep24_fr_private.json", [
        {"kind": "entity", "label": f"Besoin d{ASCII}Europe", "country": "FR", "definition": "a"},
        {"kind": "entity", "label": f"Besoin d{RIGHT}Europe", "country": "FR", "definition": "b"},
    ])
    labels = _merged_labels(root)
    assert len(labels) == 1, f"the two elision spellings must collapse, got {labels}"


def test_merge_records_the_crosswalk_instead_of_merging_silently() -> None:
    """FR-09: the acceptance criterion — migration behaviour is documented and tested."""
    import tempfile

    root = Path(tempfile.mkdtemp())
    _write(root, "ep24_common_private.json", [])
    _write(root, "ep24_fr_private.json", [
        {"kind": "entity", "label": f"Besoin d{ASCII}Europe", "country": "FR"},
        {"kind": "entity", "label": f"Besoin d{RIGHT}Europe", "country": "FR"},
    ])
    _entries, meta = cb.load_profile(root, "FR")
    merges = meta["elision_merges"]
    assert merges, "an apostrophe-only merge must be recorded, not silent"
    pair = merges[0]
    assert pair["reason"] == "elision_apostrophe"
    assert {pair["kept"], pair["folded"]} == {f"Besoin d{ASCII}Europe", f"Besoin d{RIGHT}Europe"}


def test_a_distinct_entity_is_not_merged_by_the_fold() -> None:
    """FR-10: the fold must not merge genuinely different entities.

    `Besoin d'Europe` and `Besoin d'Europe` differ only by apostrophe; `Besoin
    d'Europe` and `Besoin de France` do not and must stay apart.
    """
    import tempfile

    root = Path(tempfile.mkdtemp())
    _write(root, "ep24_common_private.json", [])
    _write(root, "ep24_fr_private.json", [
        {"kind": "entity", "label": f"Besoin d{RIGHT}Europe", "country": "FR"},
        {"kind": "entity", "label": "Besoin de France", "country": "FR"},
    ])
    assert len(_merged_labels(root)) == 2


def test_elision_merges_is_empty_for_a_clean_book() -> None:
    """A book with no typographic apostrophes records no crosswalk noise."""
    import tempfile

    root = Path(tempfile.mkdtemp())
    _write(root, "ep24_common_private.json", [])
    _write(root, "ep24_fr_private.json", [
        {"kind": "entity", "label": f"l{ASCII}Union europeenne", "country": "FR"},
        {"kind": "entity", "label": "Rassemblement National", "country": "FR"},
    ])
    _entries, meta = cb.load_profile(root, "FR")
    assert meta["elision_merges"] == []
    assert meta["elision_merge_count"] == 0


# --------------------------------------------------------------------------- #
# non-French: Italian elision, and the English possessive negative case
# --------------------------------------------------------------------------- #


def test_italian_elision_folds_the_same_way() -> None:
    """ELISION-01: any language using apostrophe elision must behave alike.

    Italy is **not** one of the 10 EP24 countries in scope (COUNTRY_PROFILES:
    BG DE ES FI FR HR HU PL PT SE), so this cannot be tested through
    `load_profile`. The function-level claim is still worth pinning, and it is
    the honest version of the issue's "at least one non-French language"
    requirement: the fold is key-level and language-independent, so Italian
    elision (same orthographic mechanism) folds, while English (a different use
    of the character) folds only at the codepoint level, not semantically.
    """
    assert cb.elision_key(f"l{ASCII}Italia") == cb.elision_key(f"l{RIGHT}Italia")
    assert cb.elision_key(f"un{ASCII}altra") == cb.elision_key(f"un{RIGHT}altra")
    assert cb.elision_merged(f"l{ASCII}Italia", f"l{RIGHT}Italia") is True


def test_elision_merge_works_in_a_second_in_scope_country() -> None:
    """ELISION-02: the merge behaviour is not wired to France.

    Uses Portugal (PT, in scope) with an apostrophe-carrying label. The merge
    path has no country special-case, and this proves it rather than assuming.
    """
    import tempfile

    root = Path(tempfile.mkdtemp())
    _write(root, "ep24_common_private.json", [])
    _write(root, "ep24_pt_private.json", [
        {"kind": "entity", "label": f"d{ASCII}Orta", "country": "PT"},
        {"kind": "entity", "label": f"d{RIGHT}Orta", "country": "PT"},
    ])
    assert len(_merged_labels(root, "PT")) == 1


def test_elision_merge_is_country_scoped() -> None:
    """ELISION-03: country isolation still holds — the fold must not breach it.

    The FR and DE layers must stay separate even when a label folds to the same
    key, because the merge key includes the country.
    """
    import tempfile

    root = Path(tempfile.mkdtemp())
    _write(root, "ep24_common_private.json", [])
    _write(root, "ep24_fr_private.json", [
        {"kind": "entity", "label": f"l{RIGHT}Europe", "country": "FR"},
    ])
    _write(root, "ep24_de_private.json", [
        {"kind": "entity", "label": f"l{RIGHT}Europe", "country": "DE"},
    ])
    assert cb.elision_key(f"l{RIGHT}Europe") == cb.elision_key(f"l{ASCII}Europe")
    # same folded key, but different countries -> still distinct entries per book
    assert len(_merged_labels(root, "FR")) == 1
    assert len(_merged_labels(root, "DE")) == 1


def test_english_possessive_and_quotation_apostrophes_still_fold_but_not_across_words() -> None:
    """EN-01: the honest non-French case.

    English uses `'` as a possessive and as a quotation mark, not as elision.
    The fold still normalises the character (it is the same codepoint question),
    but it must not make two different English words equal.
    """
    assert cb.elision_key(f"People{RIGHT}s Party") == cb.elision_key(f"People{ASCII}s Party")
    # and a genuinely different word stays different
    assert cb.elision_key(f"People{RIGHT}s Party") != cb.elision_key("Peoples Party")
    assert cb.elision_key(f"People{RIGHT}s Party") != cb.elision_key("People Party")


def test_two_different_apostrophe_positions_do_not_converge() -> None:
    """EN-02: folding the character must not fold away WHERE it is."""
    assert cb.elision_key(f"d{ASCII}Europe") != cb.elision_key(f"d Europe")
    assert cb.elision_key("don\u2019t") != cb.elision_key("dont")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
