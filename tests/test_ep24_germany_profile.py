#!/usr/bin/env python3
"""Synthetic Germany codebook regressions for EP24 issue #91."""
from __future__ import annotations

import json
from pathlib import Path

from roihu_codebooks import COUNTRY_PROFILES, context_block, identity_key, load_codebook


def _book(tmp_path: Path) -> Path:
    path = tmp_path / "ep24_de_private.json"
    path.write_text(
        json.dumps(
            {
                "schema": "test-fixture",
                "country_code": "DE",
                "country": "Germany",
                "language": "de",
                "entries": [
                    {
                        "kind": "entity",
                        "label": "Christlich Demokratische Union Deutschlands",
                        "english_label": "Christian Democratic Union of Germany",
                        "aliases": ["CDU"],
                        "entity_type": "party",
                    },
                    {
                        "kind": "entity",
                        "label": "Christlich-Soziale Union in Bayern",
                        "english_label": "Christian Social Union in Bavaria",
                        "aliases": ["CSU"],
                        "entity_type": "party",
                    },
                    {
                        "kind": "entity",
                        "label": "BÜNDNIS 90/DIE GRÜNEN",
                        "english_label": "Alliance 90/The Greens",
                        "aliases": ["GRÜNE", "Die Grünen", "Gruene"],
                        "entity_type": "party",
                    },
                    {
                        "kind": "entity",
                        "label": "Ökologisch-Demokratische Partei",
                        "english_label": "Ecological Democratic Party",
                        "aliases": ["ÖDP", "OeDP"],
                        "entity_type": "party",
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def _labels(query: str, entries, country: str = "DE") -> list[str]:
    _block, provenance = context_block(query, entries, country=country, language="de")
    return [row["label"] for row in provenance["selected"]]


def test_de_profile_is_bilingual_and_uses_expected_private_file() -> None:
    profile = COUNTRY_PROFILES["DE"]
    assert profile["languages"] == ["de", "en"]
    assert profile["file"] == "ep24_de_private.json"


def test_identity_key_preserves_german_diacritic_distinctions() -> None:
    assert identity_key("GRÜNE") == "grüne"
    assert identity_key("GRÜNE") != identity_key("GRUNE")
    assert identity_key("ÖDP") != identity_key("ODP")


def test_german_party_acronyms_retrieve_within_de(tmp_path: Path) -> None:
    entries, _meta = load_codebook(_book(tmp_path))
    assert "Christlich Demokratische Union Deutschlands" in _labels("CDU Wahlkampf", entries)
    assert "Christlich-Soziale Union in Bayern" in _labels("CSU Bayern", entries)
    assert "Ökologisch-Demokratische Partei" in _labels("ÖDP Europawahl", entries)


def test_german_short_aliases_do_not_cross_country_boundary(tmp_path: Path) -> None:
    entries, _meta = load_codebook(_book(tmp_path))
    assert _labels("CDU Wahlkampf", entries, country="SE") == []


def test_cdu_and_csu_remain_distinct_canonical_entities(tmp_path: Path) -> None:
    entries, _meta = load_codebook(_book(tmp_path))
    cdu = next(e for e in entries if "CDU" in e.aliases)
    csu = next(e for e in entries if "CSU" in e.aliases)
    assert cdu.entry_id != csu.entry_id
    assert cdu.label != csu.label


def test_german_canonical_label_and_english_context_are_both_preserved(tmp_path: Path) -> None:
    entries, _meta = load_codebook(_book(tmp_path))
    block, _provenance = context_block("Die Grünen", entries, country="DE", language="de")
    assert "BÜNDNIS 90/DIE GRÜNEN / Alliance 90/The Greens" in block
    assert "Gruene" in next(e.aliases for e in entries if e.label == "BÜNDNIS 90/DIE GRÜNEN")
