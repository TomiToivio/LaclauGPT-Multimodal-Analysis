"""Regression tests for two defects found in the EP24 codebook coverage auditor.

Both defects were found while auditing a country that the original tests did not
exercise, which is exactly why they survived #81's verification:

D1.  Codebook resolution assumed the file is named after the ISO2 code.
     `ep24_hr_private.json` and `ep24_es_private.json` are, but
     `ep24_finland_private.json` and `ep24_poland_private.json` are not, so the
     auditor could not open Finland or Poland at all -- it exited 2 with
     "no codebook found". #81 verified against Croatia, whose filename matches
     the assumption.

D2.  `entity_fragmentation` keyed on the last token longer than 3 characters.
     For English-labelled entries that token is frequently a *kind* word, so
     every party collapsed into one group keyed `party`. The reported headline
     for Finland was `obs=209 n=25` covering `Brothers of Italy party`,
     `Centre Party`, `Finnish Social Democratic Party` and 22 others -- a false
     merge that also *hid* the real fragmentation underneath it.

Fixtures are synthetic: no private codebook content, rows, notes or personal data.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_SPEC = importlib.util.spec_from_file_location(
    "codebook_coverage", ROOT / "scripts" / "ep24" / "codebook_coverage.py"
)
assert _SPEC and _SPEC.loader
coverage = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(coverage)


def _book(root: Path, name: str, *, country_code: str, country: str, entries: list[dict]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(
        json.dumps(
            {
                "schema": "test-fixture",
                "country_code": country_code,
                "country": country,
                "entries": entries,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------- D1: resolution


def _name_only_fixture(tmp_path: Path) -> Path:
    """A book named after the COUNTRY, the shape that broke resolution."""
    root = tmp_path / "codebooks"
    _book(
        root,
        "ep24_finland_private.json",
        country_code="FI",
        country="Finland",
        entries=[{"kind": "entity", "label": "Kokoomus", "aliases": [], "metadata": {"observations": 3}}],
    )
    return root


def test_book_named_after_the_country_resolves_from_its_iso2_code(tmp_path: Path) -> None:
    """`--country FI` must find `ep24_finland_private.json`.

    This is the exact call that failed before the fix: the real Finland book is
    named after the country, so the ISO2 form found nothing.
    """
    root = _name_only_fixture(tmp_path)
    report = coverage.audit(root, "FI")
    assert "ep24_finland_private.json" in report["layers"]


def test_book_named_after_the_country_resolves_from_its_name(tmp_path: Path) -> None:
    """The country-name form must keep working: it is what agents were told to use."""
    root = _name_only_fixture(tmp_path)
    report = coverage.audit(root, "Finland")
    assert "ep24_finland_private.json" in report["layers"]


def test_iso2_book_still_resolves(tmp_path: Path) -> None:
    """The original ISO2-named shape must not regress."""
    root = tmp_path / "codebooks"
    _book(
        root,
        "ep24_hr_private.json",
        country_code="HR",
        country="Croatia",
        entries=[{"kind": "entity", "label": "HDZ", "aliases": [], "metadata": {"observations": 1}}],
    )
    report = coverage.audit(root, "HR")
    assert "ep24_hr_private.json" in report["layers"]


def test_country_name_is_accepted_case_insensitively(tmp_path: Path) -> None:
    root = _name_only_fixture(tmp_path)
    assert "ep24_finland_private.json" in coverage.audit(root, "finland")["layers"]


def test_common_book_is_never_treated_as_a_country_layer(tmp_path: Path) -> None:
    """`ep24_common_private.json` is a shared base, not a country; it has no country_code."""
    root = tmp_path / "codebooks"
    _book(root, "ep24_common_private.json", country_code="", country="", entries=[
        {"kind": "entity", "label": "Shared Entity", "aliases": [], "metadata": {"observations": 9}}
    ])
    _book(root, "ep24_finland_private.json", country_code="FI", country="Finland", entries=[
        {"kind": "entity", "label": "Kokoomus", "aliases": [], "metadata": {"observations": 3}}
    ])
    layers = coverage.audit(root, "FI")["layers"]
    assert "ep24_common_private.json" not in layers
    assert "ep24_finland_private.json" in layers


def test_missing_country_error_lists_what_is_available(tmp_path: Path) -> None:
    """The error must help the caller: the old one only echoed the guessed paths."""
    root = _name_only_fixture(tmp_path)
    with pytest.raises(FileNotFoundError) as excinfo:
        coverage.audit(root, "ZZ")
    assert "ep24_finland_private.json" in str(excinfo.value), "should name the books that do exist"


# ------------------------------------------------------- D2: role-word grouping


def test_party_labels_do_not_all_collapse_under_the_kind_word(tmp_path: Path) -> None:
    """Three distinct parties must not become one fragmentation group.

    Before the fix every label ending in the literal word `party` produced the
    surname key `party`, so this fixture reported a single 3-entry group.
    """
    root = tmp_path / "codebooks"
    _book(
        root,
        "ep24_xx_private.json",
        country_code="XX",
        country="Example",
        entries=[
            {"kind": "entity", "label": "Centre Party", "metadata": {"observations": 5}},
            {"kind": "entity", "label": "Finns Party", "metadata": {"observations": 7}},
            {"kind": "entity", "label": "Brothers of Italy party", "metadata": {"observations": 2}},
        ],
    )
    frag = coverage.audit(root, "XX")["layers"]["ep24_xx_private.json"]["entity_fragmentation"]
    keys = {row["key"] for row in frag}
    assert "party" not in keys, "a kind word must never be a grouping key"
    # Each label ends in a different real name, so no group should form at all.
    assert frag == [], f"distinct parties were falsely merged: {frag}"


def test_real_fragmentation_is_still_detected_beneath_the_kind_word(tmp_path: Path) -> None:
    """The fix must expose the real signal, not just suppress the false one.

    `Perussuomalaiset` and `Perussuomalaiset party` are the same actor twice, so
    they must group; `Centre Party` must not join them.
    """
    root = tmp_path / "codebooks"
    _book(
        root,
        "ep24_xx_private.json",
        country_code="XX",
        country="Example",
        entries=[
            {"kind": "entity", "label": "Perussuomalaiset", "metadata": {"observations": 30}},
            {"kind": "entity", "label": "Perussuomalaiset party", "metadata": {"observations": 12}},
            {"kind": "entity", "label": "Centre Party", "metadata": {"observations": 5}},
        ],
    )
    frag = coverage.audit(root, "XX")["layers"]["ep24_xx_private.json"]["entity_fragmentation"]
    by_key = {row["key"]: row for row in frag}
    assert "perussuomalaiset" in by_key, "the same actor under two labels must still group"
    assert by_key["perussuomalaiset"]["count"] == 2
    assert by_key["perussuomalaiset"]["observations"] == 42
    assert not any("centre" in k for k in by_key), "an unrelated party must not be merged in"


def test_a_genuine_surname_key_is_unaffected_by_the_role_word_skip(tmp_path: Path) -> None:
    """Skipping kind words must not change ordinary surname grouping."""
    root = tmp_path / "codebooks"
    _book(
        root,
        "ep24_xx_private.json",
        country_code="XX",
        country="Example",
        entries=[
            {"kind": "entity", "label": "Andrej Plenković", "metadata": {"observations": 64}},
            {"kind": "entity", "label": "Plenković", "metadata": {"observations": 26}},
        ],
    )
    frag = coverage.audit(root, "XX")["layers"]["ep24_xx_private.json"]["entity_fragmentation"]
    assert frag and frag[0]["key"] == "plenkovic"
    assert frag[0]["observations"] == 90


def test_role_words_do_not_hide_a_trailing_surname(tmp_path: Path) -> None:
    """`X Party` with a name before the kind word keeps that name as the key."""
    root = tmp_path / "codebooks"
    _book(
        root,
        "ep24_xx_private.json",
        country_code="XX",
        country="Example",
        entries=[
            {"kind": "entity", "label": "Vihreät Party", "metadata": {"observations": 4}},
            {"kind": "entity", "label": "Vihreät", "metadata": {"observations": 6}},
        ],
    )
    frag = coverage.audit(root, "XX")["layers"]["ep24_xx_private.json"]["entity_fragmentation"]
    assert frag and frag[0]["key"] == "vihreat"


def test_kind_words_are_still_kept_when_they_are_the_only_token(tmp_path: Path) -> None:
    """A label that is *only* a kind word has no name; it must key to nothing, not crash."""
    assert coverage._surname_key("Party") == ""
    assert coverage._surname_key("") == ""


def test_fix_is_read_only(tmp_path: Path) -> None:
    """Resolution change must not make the auditor write."""
    root = _name_only_fixture(tmp_path)
    path = root / "ep24_finland_private.json"
    before = path.read_bytes()
    coverage.audit(root, "FI")
    assert path.read_bytes() == before
