"""Sweden-specific EP24 codebook regression checks for issue #91."""

from __future__ import annotations

import json
from pathlib import Path

from roihu_codebooks import context_block, identity_key, load_codebook


def _book(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "test-fixture",
                "country_code": "SE",
                "country": "Sweden",
                "language": "sv",
                "entries": [
                    {
                        "kind": "entity",
                        "label": "Sverigedemokraterna",
                        "english_label": "Sweden Democrats",
                        "aliases": ["Sverigedemokrater"],
                        "status": "researcher-grounded",
                    },
                    {
                        "kind": "entity",
                        "label": "Miljöpartiet de gröna",
                        "english_label": "Green Party",
                        "aliases": ["Miljöpartiet"],
                        "status": "researcher-grounded",
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def test_swedish_diacritics_are_not_destructively_ascii_folded() -> None:
    assert identity_key("Miljöpartiet") == identity_key("MILJÖPARTIET")
    assert identity_key("Miljöpartiet") != identity_key("Miljopartiet")


def test_local_and_english_names_are_forms_of_one_entity(tmp_path: Path) -> None:
    entries, _meta = load_codebook(_book(tmp_path / "ep24_se_private.json"))
    sd = next(entry for entry in entries if entry.label == "Sverigedemokraterna")
    assert "Sweden Democrats" in sd.forms
    assert "Sverigedemokrater" in sd.forms


def test_swedish_alias_retrieval_is_country_scoped(tmp_path: Path) -> None:
    entries, _meta = load_codebook(_book(tmp_path / "ep24_se_private.json"))
    block, _ = context_block("Miljöpartiet prioriterar klimatet", entries, country="SE", language="sv")
    assert block

    foreign_block, _ = context_block("Miljöpartiet", entries, country="FI", language="sv")
    assert not foreign_block


def test_short_party_aliases_must_not_be_modelled_as_plain_substrings() -> None:
    # The Sweden audit found that forms such as S/M/V/C/L/KD/MP/SD are useful
    # but unsafe in the existing substring-friendly retrieval scorer. Keep this
    # regression as a design guard until #95 provides boundary-aware short-alias
    # retrieval rather than weakening identity normalization.
    assert identity_key("SD") == "sd"
    assert identity_key("SD") != identity_key("Sverigedemokraterna")
