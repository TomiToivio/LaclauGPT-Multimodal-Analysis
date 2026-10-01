"""Tests for the French apostrophe/typography hygiene detector (issue #91, FR pass).

The detector exists for one measured French problem: elision is written with
either the ASCII apostrophe (``l'Union``) or U+2019 (``l’Union``), both are
attested in the real FR corpus, and ``identity_key`` preserves punctuation on
purpose. So two spellings of one entity become two identities.

The tests pin three things:

* the detector FINDS real apostrophe-family variants and reports them;
* it correctly identifies which pairs are safe to unify (apostrophe family) and
  which are NOT (prime, grave accent, dashes) — a detector that conflates those
  would license a false merge;
* ``identity_split`` reports only pairs that genuinely differ *only* in
  apostrophe style. A pair that ``identity_key`` already collapses (case,
  whitespace) is not a finding, and reporting it would be noise.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "apostrophe_hygiene", ROOT / "scripts" / "ep24" / "apostrophe_hygiene.py"
)
assert _SPEC and _SPEC.loader
hygiene = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(hygiene)

ASCII_APOSTROPHE = "\u0027"        # '
TYPOGRAPHIC = "\u2019"             # ’
PRIME = "\u2032"                   # ′
GRAVE = "\u0060"                   # `


# --------------------------------------------------------------------------
# Detection: real apostrophes are found
# --------------------------------------------------------------------------


def test_a_typographic_apostrophe_is_detected_and_named() -> None:
    finding = hygiene.audit_text("Besoin d\u2019Europe")
    assert finding is not None
    assert finding["typographic_elision"] is True
    assert finding["apostrophes"][0]["codepoint"] == "U+2019"
    assert finding["apostrophes"][0]["unicode_name"] == "RIGHT SINGLE QUOTATION MARK"


def test_an_ascii_apostrophe_is_detected_but_not_flagged_as_typographic() -> None:
    finding = hygiene.audit_text("l'Union européenne")
    assert finding is not None
    assert finding["apostrophes"][0]["codepoint"] == "U+0027"
    assert finding["typographic_elision"] is False


def test_a_label_without_any_apostrophe_is_clean() -> None:
    assert hygiene.audit_text("Rassemblement National") is None
    assert hygiene.audit_text("Élysée") is None


# --------------------------------------------------------------------------
# The negative half: things that LOOK like apostrophes but are not
# --------------------------------------------------------------------------


def test_a_prime_is_reported_but_not_treated_as_an_apostrophe() -> None:
    """A prime means minutes/feet. Folding it onto ' would corrupt a measurement."""
    finding = hygiene.audit_text(f"5{PRIME}30")
    assert finding is not None
    assert finding["apostrophes"] == [], "a prime must not enter the apostrophe family"
    assert finding["not_apostrophes"][0]["codepoint"] == "U+2032"


def test_a_grave_accent_is_reported_but_not_treated_as_an_apostrophe() -> None:
    finding = hygiene.audit_text("l`Union")
    assert finding is not None
    assert finding["apostrophes"] == []
    assert finding["not_apostrophes"][0]["codepoint"] == "U+0060"


def test_normalisation_leaves_primes_and_accents_untouched() -> None:
    assert hygiene.normalise_elision(f"5{PRIME}") == f"5{PRIME}"
    assert hygiene.normalise_elision("l" + GRAVE + "Union") == "l" + GRAVE + "Union"
    assert hygiene.normalise_elision("Élysée – Paris") == "Élysée – Paris"


# --------------------------------------------------------------------------
# Normalisation: narrow and reversible-in-effect
# --------------------------------------------------------------------------


def test_normalisation_folds_every_apostrophe_family_member_onto_ascii() -> None:
    for variant in ("\u0027", "\u2019", "\u2018", "\u02bc"):
        assert hygiene.normalise_elision(f"l{variant}Union") == "l'Union", repr(variant)


def test_normalisation_is_idempotent() -> None:
    for text in ("l\u2019Union", "Besoin d’Europe", "Élysée", "Rassemblement National"):
        once = hygiene.normalise_elision(text)
        assert hygiene.normalise_elision(once) == once


def test_normalisation_makes_the_two_spellings_share_one_identity_key() -> None:
    """The point of the fix: one entity, one identity."""
    from roihu_codebooks import identity_key

    assert identity_key("l'Union") != identity_key("l\u2019Union")
    assert identity_key(hygiene.normalise_elision("l\u2019Union")) == identity_key("l'Union")
    assert identity_key(hygiene.normalise_elision("Besoin d\u2019Europe")) == identity_key("Besoin d'Europe")


def test_normalisation_does_not_ascii_fold_accents() -> None:
    """Unifying apostrophes must not become ASCII-folding the whole language."""
    assert hygiene.normalise_elision("Élysée") == "Élysée"
    assert hygiene.normalise_elision("Raphaël Glucksmann") == "Raphaël Glucksmann"


# --------------------------------------------------------------------------
# identity_split: only genuine apostrophe-only splits are reported
# --------------------------------------------------------------------------


def test_a_genuine_apostrophe_only_split_is_reported() -> None:
    splits = hygiene.identity_split(["Besoin d'Europe", "Besoin d\u2019Europe"])
    assert len(splits) == 1
    assert splits[0]["labels"] == sorted(["Besoin d'Europe", "Besoin d\u2019Europe"])
    assert len(splits[0]["distinct_identity_keys"]) == 2


def test_a_case_only_difference_is_not_a_split() -> None:
    """identity_key already casefolds — reporting this would be noise."""
    assert hygiene.identity_split(["Alliance Rurale", "Alliance rurale"]) == []


def test_a_whitespace_only_difference_is_not_a_split() -> None:
    assert hygiene.identity_split(["Les  Républicains", "Les Républicains"]) == []


def test_two_different_entities_are_not_a_split() -> None:
    assert hygiene.identity_split(["Rassemblement National", "Les Républicains"]) == []


def test_repeated_identical_forms_are_not_a_split() -> None:
    assert hygiene.identity_split(["Renaissance", "Renaissance", "Renaissance"]) == []


# --------------------------------------------------------------------------
# Codebook audit + CLI
# --------------------------------------------------------------------------


def _book(tmp_path: Path) -> Path:
    path = tmp_path / "ep24_fr_private.json"
    path.write_text(
        json.dumps(
            {
                "schema": "test-fixture",
                "country_code": "FR",
                "country": "France",
                "language": "fr",
                "entries": [
                    {"kind": "entity", "label": "Besoin d\u2019Europe", "aliases": ["Besoin d'Europe"]},
                    {"kind": "entity", "label": "Rassemblement National", "aliases": ["RN"]},
                    {"kind": "topic", "label": "l\u2019Union européenne"},
                    {"kind": "entity", "label": "Élysée", "aliases": ["Elysee"]},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def test_codebook_audit_counts_typographic_forms(tmp_path: Path) -> None:
    _book(tmp_path)
    report = hygiene.audit(tmp_path, "FR")
    assert report["label_count"] == 7
    assert report["typographic_count"] == 2
    assert report["split_group_count"] == 1
    assert report["split_groups"][0]["labels"] == sorted(["Besoin d\u2019Europe", "Besoin d'Europe"])


def test_codebook_audit_reports_an_unreadable_layer(tmp_path: Path) -> None:
    (tmp_path / "ep24_fr_private.json").write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 10\n", encoding="utf-8"
    )
    report = hygiene.audit(tmp_path, "FR")
    assert "ep24_fr_private.json" in report["skipped_unreadable"]
    assert report["label_count"] == 0


def test_cli_text_mode_exits_1_on_a_finding() -> None:
    assert hygiene.main(["--text", "Besoin d\u2019Europe"]) == 1


def test_cli_text_mode_exits_0_on_a_clean_string() -> None:
    assert hygiene.main(["--text", "Rassemblement National"]) == 0


def test_cli_requires_root_or_text() -> None:
    assert hygiene.main([]) == 2


def test_cli_rejects_a_missing_root() -> None:
    assert hygiene.main(["--root", "/nonexistent-codebook-root"]) == 2
