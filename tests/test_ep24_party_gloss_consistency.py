"""Tests for the party-gloss contradiction checker (issue #91).

A label of the form `NativeName (Gloss)` asserts the gloss is that entity's name
in another language. When the gloss belongs to a *different* entity the label is
a false identity, and because labels are retrieval match targets, a query about
one party surfaces the other.

Found in the Sweden pass: `Socialdemokraterna (Sweden Democrats)`. The Social
Democrats (S) and the Sweden Democrats (SD) are different parties, and a query
for "Sweden Democrats" returned the Social Democrats entry.

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
    "check_party_gloss", ROOT / "scripts" / "ep24" / "check_party_gloss_consistency.py"
)
assert _SPEC and _SPEC.loader
checker = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(checker)

PARTIES = {
    "Socialdemokraterna": ["Social Democratic Party", "Social Democrats"],
    "Sverigedemokraterna": ["Sweden Democrats"],
}


def _entries(*labels: str) -> list[dict]:
    return [{"kind": "entity", "label": lab, "country": "XX"} for lab in labels]


# ------------------------------------------------------- the contradiction itself

def test_cross_party_gloss_is_reported() -> None:
    """The real Sweden defect, in miniature."""
    found = checker.find_contradictions(
        _entries("Socialdemokraterna (Sweden Democrats)"), PARTIES
    )
    assert len(found) == 1
    row = found[0]
    assert row["label"] == "Socialdemokraterna (Sweden Democrats)"
    assert row["native_part_names"] == "Socialdemokraterna"
    assert row["gloss_actually_names"] == "Sverigedemokraterna"


def test_correct_glosses_are_not_reported() -> None:
    """Every legitimate way of naming the same party must pass clean."""
    labels = [
        "Socialdemokraterna (S)",
        "Socialdemokraterna (Social Democratic Party)",
        "Socialdemokraterna (Social Democrats)",
        "Socialdemokraterna (the Social Democratic Party)",
        "Sverigedemokraterna (SD)",
        "Sverigedemokraterna (Sweden Democrats)",
    ]
    assert checker.find_contradictions(_entries(*labels), PARTIES) == []


def test_a_gloss_containing_the_correct_name_is_not_reported() -> None:
    """`Sweden's Social Democratic Party` names the right party, so it is fine.

    The gloss *contains* another party's name ("Sweden ..."), so a naive
    substring check would flag it. It must not be.
    """
    found = checker.find_contradictions(
        _entries("Socialdemokraterna (Sweden's Social Democratic Party)"), PARTIES
    )
    assert found == []


def test_single_word_gloss_matching_a_party_is_not_reported() -> None:
    found = checker.find_contradictions(_entries("Sverigedemokraterna (Sweden Democrats)"), PARTIES)
    assert found == []


def test_entry_without_a_gloss_is_ignored() -> None:
    assert checker.find_contradictions(_entries("Socialdemokraterna"), PARTIES) == []


def test_entry_naming_no_known_party_is_ignored() -> None:
    assert checker.find_contradictions(_entries("Some Other Org (Something Else)"), PARTIES) == []


def test_a_trailing_kind_word_after_the_gloss_still_detects() -> None:
    """`Sverigedemokraterna (Sweden Democrats) party` — the tail must not hide it."""
    found = checker.find_contradictions(
        _entries("Socialdemokraterna (Sweden Democrats) party"), PARTIES
    )
    assert len(found) == 1


def test_diacritics_and_case_are_folded_for_matching() -> None:
    """`Miljöpartiet`/`miljopartiet` and case differences must match."""
    parties = {"Miljöpartiet": ["Green Party"]}
    assert checker.find_contradictions(_entries("Miljopartiet (Green Party)"), parties) == []
    assert checker.find_contradictions(_entries("MILJÖPARTIET (GREEN PARTY)"), parties) == []


# ------------------------------------------------------------------ table loading

def test_documentation_keys_are_not_treated_as_parties(tmp_path: Path) -> None:
    """`_note` / `_election` metadata must be skipped.

    Treating them as party stems made every gloss look contradictory on the first
    run of this checker against the real table, which is why this is pinned. The
    assertion is on the *loaded table*, not on a detection, because detection
    needs both parties present — a one-party table cannot express a contradiction.
    """
    path = tmp_path / "parties.json"
    path.write_text(
        json.dumps(
            {
                "_note": "public source note",
                "_election": "2024 EP election in Sweden",
                "_source_urls": ["https://example.invalid"],
                "Socialdemokraterna": ["Social Democratic Party"],
                "Sverigedemokraterna": ["Sweden Democrats"],
            }
        ),
        encoding="utf-8",
    )
    parties = checker.load_parties(path)
    assert set(parties) == {"Socialdemokraterna", "Sverigedemokraterna"}
    assert not any(k.startswith("_") for k in parties)
    # With both parties loaded, the real defect detects through the table path.
    assert len(checker.find_contradictions(_entries("Socialdemokraterna (Sweden Democrats)"), parties)) == 1


def test_a_metadata_key_alone_does_not_inflate_the_table(tmp_path: Path) -> None:
    """A table of only metadata must load as empty, not as a party named `_note`."""
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"_note": "x", "_election": "y"}), encoding="utf-8")
    assert checker.load_parties(path) == {}


def test_a_string_value_is_accepted_as_one_name(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"Miljöpartiet": "Green Party"}), encoding="utf-8")
    assert checker.load_parties(path) == {"Miljöpartiet": ["Green Party"]}


def test_empty_list_values_are_dropped(tmp_path: Path) -> None:
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"X": [], "Y": ["", "  "], "Z": ["Ok"]}), encoding="utf-8")
    assert checker.load_parties(path) == {"Z": ["Ok"]}


# ------------------------------------------------------------------------ CLI

def test_cli_exits_one_when_a_contradiction_exists(tmp_path: Path) -> None:
    book = tmp_path / "book.json"
    book.write_text(json.dumps({"entries": _entries("Socialdemokraterna (Sweden Democrats)")}), encoding="utf-8")
    table = tmp_path / "parties.json"
    table.write_text(json.dumps(PARTIES), encoding="utf-8")
    assert checker.main(["--codebook", str(book), "--parties", str(table)]) == 1


def test_cli_exits_zero_when_clean(tmp_path: Path) -> None:
    book = tmp_path / "book.json"
    book.write_text(json.dumps({"entries": _entries("Socialdemokraterna (S)")}), encoding="utf-8")
    table = tmp_path / "parties.json"
    table.write_text(json.dumps(PARTIES), encoding="utf-8")
    assert checker.main(["--codebook", str(book), "--parties", str(table)]) == 0


def test_cli_reports_a_missing_file_cleanly(tmp_path: Path) -> None:
    assert checker.main(["--codebook", str(tmp_path / "no.json"), "--parties", str(tmp_path / "no2.json")]) == 2


def test_checker_is_read_only(tmp_path: Path) -> None:
    book = tmp_path / "book.json"
    book.write_text(json.dumps({"entries": _entries("Socialdemokraterna (Sweden Democrats)")}), encoding="utf-8")
    table = tmp_path / "parties.json"
    table.write_text(json.dumps(PARTIES), encoding="utf-8")
    before = book.read_bytes()
    checker.main(["--codebook", str(book), "--parties", str(table)])
    assert book.read_bytes() == before, "the checker modified the codebook"
