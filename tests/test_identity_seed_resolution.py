"""Synthetic fixtures for issue #33's "signal-garden-seed-resolver" slice.

Covers:
* conservative fuzzy candidate generation that only proposes, never resolves;
* ambiguous surnames still abstain even when a fuzzy candidate exists;
* party-abbreviation / politician-title alias resolution through the shared
  identity layer;
* researcher-note seed extraction (``new_entity``/``researcher_new_persons``/
  ``new_theme``/``researcher_new_themes``) as PROVISIONAL codebook entries;
* sentiment-target (positive/neutral/negative) resolution through the same
  identity layer as entities/topics, additive to legacy CSV columns.

No real private political/person/researcher data is used anywhere below.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roihu_codebooks import seed_entries_from_research_notes  # noqa: E402
from roihu_memory import EP24Memory  # noqa: E402


# --------------------------------------------------------------------------- #
# fuzzy candidates propose, never resolve
# --------------------------------------------------------------------------- #

def test_fuzzy_candidate_is_proposed_but_not_resolved(tmp_path):
    memory = EP24Memory(tmp_path / "memory.sqlite3")
    canonical = memory.add_object(
        "actor", "Prime Minister Alice Example", country="FI", language="en",
        state="CANONICAL", origin="researcher_private", locked=True,
    )
    # One character of spelling drift: not an exact alias, so resolve() alone
    # would abstain with NEW. The fuzzy layer must only *propose* this as a
    # candidate, never silently attach it to the canonical object.
    identity = memory.resolve_identity("Prime Minster Alice Example", "actor", country="FI")
    assert identity["decision"] == "FUZZY_CANDIDATE"
    assert identity["obj_id"] == ""  # never silently canonized
    candidate_ids = [c["obj_id"] for c in identity["candidates"]]
    assert canonical in candidate_ids


def test_unrelated_label_has_no_fuzzy_candidate(tmp_path):
    memory = EP24Memory(tmp_path / "memory.sqlite3")
    memory.add_object("actor", "Prime Minister Alice Example", country="FI", state="CANONICAL", locked=True)
    identity = memory.resolve_identity("Completely Unrelated Topic", "actor", country="FI")
    assert identity["decision"] == "NEW"
    assert identity["candidates"] == []


def test_ambiguous_surname_abstains_even_with_fuzzy_candidates(tmp_path):
    """Two synthetic politicians sharing a surname must stay AMBIGUOUS.

    A fuzzy near-miss on top of an already-ambiguous exact match must not be
    used to arbitrarily pick a side.
    """
    memory = EP24Memory(tmp_path / "memory.sqlite3")
    first = memory.add_object("actor", "Example Minister A", country="FI", state="CANONICAL", locked=True)
    second = memory.add_object("actor", "Example Minister B", country="FI", state="CANONICAL", locked=True)
    memory.add_alias(first, "Minister Example", country="FI")
    memory.add_alias(second, "Minister Example", country="FI")
    identity = memory.resolve_identity("Minister Example", "actor", country="FI")
    assert identity["decision"] == "AMBIGUOUS"
    assert identity["obj_id"] == ""


# --------------------------------------------------------------------------- #
# party abbreviation / politician-title alias resolution
# --------------------------------------------------------------------------- #

def test_party_abbreviation_and_colloquial_alias_resolve_to_one_canonical_party(tmp_path):
    memory = EP24Memory(tmp_path / "memory.sqlite3")
    party = memory.add_object(
        "actor", "Example Party", country="FI", language="fi",
        english_label="Example Party", state="CANONICAL", origin="researcher_private", locked=True,
    )
    for alias in ("EP", "Esimerkkipuolue", "Puolue"):
        memory.add_alias(party, alias, country="FI")
    for surface_form in ("EP", "Esimerkkipuolue", "Puolue"):
        identity = memory.resolve_identity(surface_form, "actor", country="FI")
        assert identity["decision"] == "EXISTING"
        assert identity["obj_id"] == party


def test_politician_title_and_bare_surname_resolve_to_one_canonical_person(tmp_path):
    memory = EP24Memory(tmp_path / "memory.sqlite3")
    person = memory.add_object(
        "actor", "Prime Minister Alice Example", country="FI", language="fi",
        english_label="Prime Minister Alice Example", state="CANONICAL", origin="researcher_private", locked=True,
    )
    for alias in ("PM", "Alice", "Pääministeri Alice Example"):
        memory.add_alias(person, alias, country="FI")
    for surface_form in ("PM", "Alice", "Pääministeri Alice Example"):
        identity = memory.resolve_identity(surface_form, "actor", country="FI")
        assert identity["decision"] == "EXISTING"
        assert identity["obj_id"] == person


# --------------------------------------------------------------------------- #
# researcher-note seed extraction
# --------------------------------------------------------------------------- #

def test_research_note_seeds_build_provisional_unlocked_entries():
    rows = [
        {
            "country": "FI",
            "entities": "Example Movement; Example Youth Wing; Minister Example",
            "themes": "synthetic cost-of-living grievance",
        },
        {
            "country": "FI",
            "entities": "Minister Example",  # duplicate across rows
            "themes": "synthetic EU sovereignty debate; synthetic trust in institutions",
        },
    ]
    entries = seed_entries_from_research_notes(rows, country="FI", language="fi")
    labels = {(entry.kind, entry.label) for entry in entries}
    assert ("entity", "Example Movement") in labels
    assert ("entity", "Example Youth Wing") in labels
    assert ("entity", "Minister Example") in labels
    assert ("topic", "synthetic cost-of-living grievance") in labels
    assert ("topic", "synthetic EU sovereignty debate") in labels
    assert ("topic", "synthetic trust in institutions") in labels
    # The duplicate "Minister Example" person seed across two rows must not
    # produce two entries for the same (kind, label) pair.
    assert sum(1 for entry in entries if entry.label == "Minister Example") == 1
    for entry in entries:
        assert entry.review_state == "PROVISIONAL"
        assert entry.locked is False
        assert entry.origin == "research_notes_seed"
        assert entry.layer == "researcher"
        assert entry.country == "FI"


def test_research_note_seed_metadata_preserves_source_field_provenance():
    rows = [{"entities": "Example Movement"}]
    entries = seed_entries_from_research_notes(rows, country="FI")
    assert len(entries) == 1
    # The provenance names the canonical column the seed now comes from.
    assert entries[0].metadata["source_field"] == "entities"
    assert entries[0].metadata["source_row_index"] == 0


# --------------------------------------------------------------------------- #
# sentiment targets flow through the same identity layer, additively
# --------------------------------------------------------------------------- #

def test_sentiment_targets_resolve_through_identity_layer_without_touching_legacy_columns(tmp_path):
    import pytest

    pytest.importorskip("pandas")
    import pandas as pd

    from roihu_enrich import enrich_file

    codebooks = tmp_path / "codebooks"
    codebooks.mkdir()
    (codebooks / "ep24_common_private.json").write_text(
        json.dumps({"schema": "fixture", "country_code": "COMMON", "entries": []}), encoding="utf-8",
    )
    (codebooks / "ep24_finland_private.json").write_text(
        json.dumps({"schema": "fixture", "country_code": "FI", "language": "fi", "entries": []}), encoding="utf-8",
    )
    csv_path = tmp_path / "ep24_fi.csv"
    original = pd.DataFrame([{
        "video_filename": "a/1",
        "summary_analysis": "Minister Example is mentioned",
        "entities": "Minister Example",
        "topics": "demokratia",
        "positive": "Minister Example",
        "neutral": "",
        "negative": "Opposing Example Party",
    }])
    original.to_csv(csv_path, index=False)

    memory = EP24Memory(tmp_path / "memory.sqlite3")
    minister_id = memory.add_object("target", "Minister Example", country="FI", state="CANONICAL", locked=True)

    enrich_file(csv_path, country="FI", language="fi", private_root=tmp_path, memory=memory)
    out = pd.read_csv(csv_path)

    # Legacy columns are untouched.
    assert out.loc[0, "entities"] == "Minister Example"
    assert out.loc[0, "positive"] == "Minister Example"
    assert out.loc[0, "negative"] == "Opposing Example Party"

    sentiment_ids = json.loads(out.loc[0, "ep24_memory_sentiment_target_ids_json"])
    assert sentiment_ids["positive"] == [minister_id]
    assert sentiment_ids["neutral"] == []
    assert sentiment_ids["negative"] == []  # unresolved, never silently canonized

    unresolved = json.loads(out.loc[0, "ep24_memory_unresolved_json"])
    unresolved_targets = [u for u in unresolved if u.get("valence") == "negative"]
    assert len(unresolved_targets) == 1
    assert unresolved_targets[0]["raw"] == "Opposing Example Party"
    assert unresolved_targets[0]["decision"] in {"NEW", "FUZZY_CANDIDATE", "AMBIGUOUS"}
