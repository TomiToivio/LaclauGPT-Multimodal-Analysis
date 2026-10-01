from __future__ import annotations

import json

import pytest

from roihu_codebooks import CodebookEntry
from roihu_identity import (
    ENTITY_KINDS,
    SENTIMENT_KINDS,
    THEME_KINDS,
    canonical_labels,
    resolve_many,
    resolve_surface,
)


def entry(entry_id: str, kind: str, label: str, *, aliases=(), english="", country="FI"):
    return CodebookEntry(
        entry_id=entry_id,
        kind=kind,
        label=label,
        aliases=list(aliases),
        english_label=english,
        country=country,
        review_state="CANONICAL",
        origin="researcher_private",
        locked=True,
    )


def test_exact_alias_resolves_to_shared_canonical_identity():
    entries = [
        entry("person-alice", "person", "Alice Esimerkki", aliases=("Minister Alice", "Alice"), english="Alice Example"),
        entry("party-ep", "party", "Esimerkkipuolue", aliases=("EP", "Esimerkit"), english="Example Party"),
    ]
    person = resolve_surface("Alice", entries, country="FI", kinds=ENTITY_KINDS)
    party = resolve_surface("EP", entries, country="FI", kinds=ENTITY_KINDS)
    assert person["decision"] == "EXISTING"
    assert person["entry_id"] == "person-alice"
    assert person["match_method"] == "exact_alias"
    assert party["entry_id"] == "party-ep"


def test_ambiguous_alias_abstains_instead_of_picking_actor():
    entries = [
        entry("alice-a", "person", "Alice North", aliases=("Alice",)),
        entry("alice-b", "person", "Alice South", aliases=("Alice",)),
    ]
    result = resolve_surface("Alice", entries, country="FI", kinds=ENTITY_KINDS)
    assert result["decision"] == "AMBIGUOUS"
    assert {candidate["entry_id"] for candidate in result["candidates"]} == {"alice-a", "alice-b"}


def test_fuzzy_similarity_is_candidate_only_never_auto_canonical():
    entries = [entry("party-ep", "party", "Esimerkkipuolue", aliases=("ExampleParty",))]
    result = resolve_surface("ExampleParti", entries, country="FI", kinds=ENTITY_KINDS, fuzzy_threshold=0.7)
    assert result["decision"] == "CANDIDATE"
    assert result["match_method"] == "fuzzy_candidate_only"
    assert result["candidates"][0]["entry_id"] == "party-ep"
    assert canonical_labels([result]) == []


def test_country_scope_prevents_cross_country_resolution():
    entries = [
        entry("fi-party", "party", "Example Party", country="FI"),
        entry("pl-party", "party", "Example Party", country="PL"),
    ]
    result = resolve_surface("Example Party", entries, country="FI", kinds=ENTITY_KINDS)
    assert result["decision"] == "EXISTING"
    assert result["entry_id"] == "fi-party"


def test_theme_and_sentiment_target_share_same_identity_layer():
    entries = [
        entry("topic-costs", "topic", "elinkustannukset", aliases=("hintojen nousu",), english="cost of living"),
        entry("party-ep", "party", "Esimerkkipuolue", aliases=("EP",), english="Example Party"),
    ]
    theme = resolve_surface("hintojen nousu", entries, country="FI", kinds=THEME_KINDS)
    target = resolve_surface("EP", entries, country="FI", kinds=SENTIMENT_KINDS)
    assert theme["entry_id"] == "topic-costs"
    assert target["entry_id"] == "party-ep"


def test_resolve_many_deduplicates_normalized_surface_forms():
    entries = [entry("party-ep", "party", "Esimerkkipuolue", aliases=("EP",))]
    results = resolve_many([" EP ", "ep", "EP"], entries, country="FI", kinds=ENTITY_KINDS)
    assert len(results) == 1
    assert results[0]["entry_id"] == "party-ep"


def test_enrichment_recycles_human_seed_columns_without_overwriting_legacy(tmp_path):
    pd = pytest.importorskip("pandas")
    from roihu_enrich import enrich_file
    from roihu_memory import EP24Memory

    codebooks = tmp_path / "codebooks"
    codebooks.mkdir()
    (codebooks / "ep24_common_private.json").write_text(
        json.dumps({"schema": "fixture", "country_code": "COMMON", "entries": []}),
        encoding="utf-8",
    )
    (codebooks / "ep24_finland_private.json").write_text(
        json.dumps({
            "schema": "fixture",
            "country_code": "FI",
            "language": "fi",
            "entries": [
                {
                    "id": "person-alice",
                    "kind": "person",
                    "label": "Alice Esimerkki",
                    "english_label": "Alice Example",
                    "aliases": ["Alice"],
                    "status": "researcher-grounded",
                    "locked": True,
                },
                {
                    "id": "party-ep",
                    "kind": "party",
                    "label": "Esimerkkipuolue",
                    "english_label": "Example Party",
                    "aliases": ["EP"],
                    "status": "researcher-grounded",
                    "locked": True,
                },
                {
                    "id": "topic-costs",
                    "kind": "topic",
                    "label": "elinkustannukset",
                    "english_label": "cost of living",
                    "aliases": ["hintojen nousu"],
                    "status": "researcher-grounded",
                    "locked": True,
                },
            ],
        }),
        encoding="utf-8",
    )
    csv_path = tmp_path / "ep24_fi.csv"
    pd.DataFrame([{
        "summary_analysis": "Synthetic fixture",
        "entities": "OLD MODEL ENTITY",
        "topics": "OLD MODEL TOPIC",
        "new_entity": "EP",
        "researcher_new_persons": "Alice",
        "new_theme": "hintojen nousu",
        "researcher_new_themes": "cost of living",
        "positive": "EP",
        "neutral": "",
        "negative": "Alice",
    }]).to_csv(csv_path, index=False)

    enrich_file(
        csv_path,
        country="FI",
        language="fi",
        private_root=tmp_path,
        memory=EP24Memory(tmp_path / "memory.sqlite3"),
    )
    out = pd.read_csv(csv_path)
    row = out.iloc[0]

    assert row["entities"] == "OLD MODEL ENTITY"
    assert row["topics"] == "OLD MODEL TOPIC"

    entity_seeds = json.loads(row["ep24_seed_entities_json"])
    theme_seeds = json.loads(row["ep24_seed_themes_json"])
    sentiments = json.loads(row["ep24_sentiment_targets_json"])
    assert {item["entry_id"] for item in entity_seeds if item["decision"] == "EXISTING"} == {"party-ep", "person-alice"}
    assert {item["entry_id"] for item in theme_seeds if item["decision"] == "EXISTING"} == {"topic-costs"}
    assert {(item["polarity"], item["entry_id"]) for item in sentiments if item["decision"] == "EXISTING"} == {
        ("positive", "party-ep"),
        ("negative", "person-alice"),
    }
    assert "human-informed seed" in row["ep24_human_seed_context"]
