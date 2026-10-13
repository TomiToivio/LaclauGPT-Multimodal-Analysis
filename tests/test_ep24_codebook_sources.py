"""The researcher codebook_source workbooks as a registry source for entity resolution.

Covers the loader itself and its integration with the merged resolver in
``ep24_entities.py``: the workbooks must actually drive resolution, and persons
must become reachable through them.

All fixtures are INVENTED. They reproduce the *structure* of the private
resources -- an ``old`` -> ``new`` replacement sheet with the disambiguation in
parentheses, alias groups, duplicate people across files, cross-country homonyms
-- without containing private research data, per issue #142.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from roihu_codebook_sources import (
    find_source_workbooks,
    group_rows,
    load_registry_from_dir,
    load_workbook_registry,
    parse_canonical,
    read_replacement_rows,
)
from roihu_codebooks import CodebookEntry


def _sheet(path: Path, rows: list[tuple[str, str]]) -> Path:
    pd.DataFrame(rows, columns=["old", "new"]).to_excel(path, index=False)
    return path


@pytest.fixture()
def sources(tmp_path: Path) -> Path:
    _sheet(tmp_path / "persons.xlsx", [
        ("petteri orpo", "Petteri Orpo (National Coalition Party; Finland)"),
        ("orpo", "Petteri Orpo (National Coalition Party; Finland)"),
        ("pääministeri orpo", "Petteri Orpo (National Coalition Party; Finland)"),
        ("orpon", "Petteri Orpo (National Coalition Party; Finland)"),
        ("marta kowalska", "Marta Kowalska (Civic Platform; Poland)"),
        ("piotr kowalska", "Piotr Kowalska (Law and Justice; Poland)"),
        ("jan novak", "Jan Novak (Independent; Croatia)"),
        ("jan novak de", "Jan Novak (Independent; Germany)"),
    ])
    _sheet(tmp_path / "entities.xlsx", [
        ("national coalition party", "National Coalition Party (Finland)"),
        ("kansallinen kokoomus", "National Coalition Party (Finland)"),
        ("coalition party", "National Coalition Party (Finland)"),
        # a person also listed here, exactly as in the real workbooks
        ("petteri orpo", "Petteri Orpo (National Coalition Party; Finland)"),
    ])
    _sheet(tmp_path / "themes.xlsx", [
        ("activists", "activism"),
        ("grassroots movements", "activism"),
        ("protesters", "activism"),
    ])
    return tmp_path


# --------------------------------------------------------------------------
# Parsing the researcher format
# --------------------------------------------------------------------------

def test_parse_canonical_splits_name_party_and_country():
    assert parse_canonical("Petteri Orpo (National Coalition Party; Finland)") == (
        "Petteri Orpo", "National Coalition Party", "FI", [],
    )


def test_parse_canonical_without_parentheses_is_the_name():
    assert parse_canonical("Afghanistan") == ("Afghanistan", "", "", [])


def test_parse_canonical_keeps_unknown_parts_as_extra():
    """Researcher annotation must not be silently dropped."""
    name, party, country, extra = parse_canonical("Thing (Some Party; Poland; reviewed 2024)")
    assert (name, country) == ("Thing", "PL")
    assert extra == ["reviewed 2024"]


def test_read_replacement_rows_accepts_a_headerless_sheet(tmp_path: Path):
    path = tmp_path / "odd.xlsx"
    pd.DataFrame([["a", "A"], ["b", "B"]]).to_excel(path, index=False, header=False)
    rows = read_replacement_rows(path)
    assert [r.surface for r in rows] == ["a", "b"], "a headerless pair was dropped"


def test_read_replacement_rows_accepts_csv(tmp_path: Path):
    path = tmp_path / "x.csv"
    path.write_text("old,new\norpo,Petteri Orpo (Party; Finland)\n", encoding="utf-8")
    rows = read_replacement_rows(path)
    assert rows[0].surface == "orpo"
    assert rows[0].canonical_name == "Petteri Orpo"


def test_unknown_workbook_name_is_rejected(tmp_path: Path):
    path = _sheet(tmp_path / "random.xlsx", [("a", "A")])
    with pytest.raises(ValueError, match="unknown codebook_source workbook"):
        load_workbook_registry(path)


def test_find_source_workbooks_accepts_alternative_names(tmp_path: Path):
    _sheet(tmp_path / "themed.xlsx", [("a", "A")])
    found = find_source_workbooks(tmp_path)
    assert "themes.xlsx" in found


# --------------------------------------------------------------------------
# Grouping: aliases, country, duplicate people
# --------------------------------------------------------------------------

def _row(surface: str, new: str):
    from roihu_codebook_sources import ReplacementRow
    name, party, country, extra = parse_canonical(new)
    return ReplacementRow(surface=surface, canonical_raw=new, canonical_name=name,
                          party=party, country=country, extra=extra)


def test_grouping_builds_one_entity_per_canonical_name_with_aliases():
    groups = group_rows([
        _row("orpo", "Petteri Orpo (Party; Finland)"),
        _row("orpon", "Petteri Orpo (Party; Finland)"),
        _row("petteri orpo", "Petteri Orpo (Party; Finland)"),
    ])
    assert len(groups) == 1
    assert groups[0].canonical_name == "Petteri Orpo"
    assert groups[0].country == "FI"
    assert groups[0].party == "Party"
    assert sorted(groups[0].aliases) == ["Petteri Orpo", "orpo", "orpon", "petteri orpo"]


def test_the_same_name_in_two_countries_stays_two_entities():
    """The issue requires this explicitly; keying on name alone merged them."""
    groups = group_rows([
        _row("jan novak", "Jan Novak (Independent; Croatia)"),
        _row("jan novak de", "Jan Novak (Independent; Germany)"),
    ])
    assert len(groups) == 2
    assert {g.country for g in groups} == {"HR", "DE"}


def test_a_person_listed_in_two_workbooks_becomes_one_person(sources: Path):
    """Measured on the real files: persons.xlsx is a strict subset of entities.xlsx.

    A naive per-file load gives two entries per person -- one typed ORGANIZATION
    from the filename rule -- which splits the aliases and makes people
    unreachable under a person-scoped lookup.
    """
    entries, report = load_registry_from_dir(sources)
    orpo = [e for e in entries if e.label == "Petteri Orpo"]
    assert len(orpo) == 1, "the same person was loaded twice"
    assert orpo[0].entity_type == "PERSON"
    assert orpo[0].kind == "person"
    assert report["totals"]["by_type"]["PERSON"] == 5  # Orpo, 2x Kowalska, 2x Novak


def test_the_merge_records_both_contributing_workbooks(sources: Path):
    entries, _ = load_registry_from_dir(sources)
    orpo = next(e for e in entries if e.label == "Petteri Orpo")
    sources_seen = orpo.metadata.get("source_workbook")
    sources_seen = [sources_seen] if isinstance(sources_seen, str) else sources_seen
    assert "persons.xlsx" in sources_seen
    assert "entities.xlsx" in sources_seen


def test_types_come_from_the_workbook_role(sources: Path):
    entries, report = load_registry_from_dir(sources)
    by_type = report["totals"]["by_type"]
    assert by_type["PERSON"] == 5      # Orpo, 2x Kowalska, 2x Novak
    assert by_type["THEME"] == 1
    assert by_type["ORGANIZATION"] >= 1


# --------------------------------------------------------------------------
# Integration with the merged resolver (ep24_entities.py)
# --------------------------------------------------------------------------

def test_persons_are_reachable_only_through_the_workbook_layer(sources: Path):
    """The JSON codebooks carry no persons at all, so the workbooks are required."""
    import ep24_entities

    wb_entries, _ = load_registry_from_dir(sources)
    with_wb = ep24_entities.EntityRegistry.from_codebooks(wb_entries)
    without = ep24_entities.EntityRegistry.from_codebooks([])

    assert without._records == {}

    result = with_wb.resolve("Petteri Orpo", country="FI", language="fi")
    assert result["decision"] == "RESOLVED"
    assert result["canonical_name"] == "Petteri Orpo"


def test_the_workbook_registry_resolves_the_issue_example(sources: Path):
    import ep24_entities

    entries, _ = load_registry_from_dir(sources)
    registry = ep24_entities.EntityRegistry.from_codebooks(entries)
    for mention in ("Petteri Orpo", "orpo", "pääministeri orpo"):
        result = registry.resolve(mention, country="FI", language="fi")
        assert result["decision"] == "RESOLVED", f"{mention} did not resolve"
        assert result["canonical_name"] == "Petteri Orpo"
        assert result["surface_form"] == mention, "the original mention was discarded"


def test_the_registry_keeps_the_mention_and_the_entity_separate(sources: Path):
    import ep24_entities

    entries, _ = load_registry_from_dir(sources)
    registry = ep24_entities.EntityRegistry.from_codebooks(entries)
    result = registry.resolve("pääministeri orpo", country="FI", language="fi")
    assert result["surface_form"] == "pääministeri orpo"
    assert result["canonical_name"] == "Petteri Orpo"
    assert result["entity_id"]


def test_an_ambiguous_surname_is_not_silently_merged(sources: Path):
    """Two different people share a surname: abstain, do not pick one."""
    import ep24_entities

    entries, _ = load_registry_from_dir(sources)
    registry = ep24_entities.EntityRegistry.from_codebooks(entries)
    result = registry.resolve("Kowalska", country="PL", language="pl")
    assert result["decision"] != "RESOLVED", "an ambiguous surname was auto-merged"


def test_workbook_entries_are_plain_codebook_entries(sources: Path):
    """The loader must produce the shared type, so any consumer can take it."""
    entries, _ = load_registry_from_dir(sources)
    assert entries
    for entry in entries:
        assert isinstance(entry, CodebookEntry)
        assert entry.entry_id
        assert entry.label
        assert entry.kind


def test_researcher_workbook_entries_are_locked(sources: Path):
    """A researcher-maintained canonical mapping is not auto-editable."""
    entries, _ = load_registry_from_dir(sources)
    assert all(e.locked for e in entries)
    assert all(e.review_state == "CANONICAL" for e in entries)


def test_a_broken_workbook_does_not_abort_the_rest(tmp_path: Path):
    (tmp_path / "persons.xlsx").write_bytes(b"not a workbook")
    _sheet(tmp_path / "themes.xlsx", [("activists", "activism")])
    entries, report = load_registry_from_dir(tmp_path)
    assert entries, "a bad workbook suppressed the good ones"
    assert any("error" in r for r in report["workbooks"])
