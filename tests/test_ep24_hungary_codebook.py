"""Hungary-specific EP24 codebook regression checks for issue #91."""

from __future__ import annotations

import json
from pathlib import Path

from roihu_codebooks import context_block, identity_key, load_codebook


def _book(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "test-fixture",
                "country_code": "HU",
                "country": "Hungary",
                "language": "hu",
                "entries": [
                    {
                        "kind": "entity",
                        "label": "Kovács Anna",
                        "english_label": "Anna Kovács",
                        "aliases": [],
                        "entity_type": "person",
                        "status": "researcher-grounded",
                    },
                    {
                        "kind": "entity",
                        "label": "Zöld Kör",
                        "english_label": "Green Circle",
                        "aliases": ["Zöld Kör Mozgalom"],
                        "entity_type": "organization",
                        "status": "researcher-grounded",
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def test_hungarian_long_accents_are_not_ascii_folded() -> None:
    assert identity_key("Őrült Űrhajó") == identity_key("ŐRÜLT ŰRHAJÓ")
    assert identity_key("Őrült Űrhajó") != identity_key("Orult Urhajo")


def test_hungarian_and_english_name_order_can_be_forms_of_one_actor(tmp_path: Path) -> None:
    entries, _meta = load_codebook(_book(tmp_path / "ep24_hu_private.json"))
    actor = next(entry for entry in entries if entry.label == "Kovács Anna")
    assert "Anna Kovács" in actor.forms
    assert actor.entry_id


def test_name_order_is_not_globally_normalized_by_identity_key() -> None:
    assert identity_key("Kovács Anna") != identity_key("Anna Kovács")


def test_hungarian_alias_retrieval_is_country_scoped(tmp_path: Path) -> None:
    entries, _meta = load_codebook(_book(tmp_path / "ep24_hu_private.json"))
    block, _ = context_block("A Zöld Kör Mozgalom közleménye", entries, country="HU", language="hu")
    assert block

    foreign, _ = context_block("Zöld Kör Mozgalom", entries, country="PL", language="hu")
    assert not foreign


def test_context_stays_background_not_observed_evidence(tmp_path: Path) -> None:
    entries, _meta = load_codebook(_book(tmp_path / "ep24_hu_private.json"))
    _block, provenance = context_block("Kovács Anna", entries, country="HU", language="hu")
    assert provenance["evidence_role"] == "background_context_not_source_evidence"
