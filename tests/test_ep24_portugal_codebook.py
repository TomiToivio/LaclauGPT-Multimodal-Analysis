"""Portugal-specific regression coverage for EP24 country QA (#91).

These tests use synthetic entries only.  Private researcher material stays in
LaclauGPT-Private; this suite pins the public runtime contracts that the PT
second-pass audit can verify without publishing private rows.
"""

from __future__ import annotations

import roihu_codebooks as cb


def _entry(label: str, *, aliases=None, english_label="", kind="entity", country="PT"):
    return cb.CodebookEntry(
        entry_id=f"fixture-{label}",
        kind=kind,
        label=label,
        aliases=list(aliases or []),
        english_label=english_label,
        country=country,
        source_languages=["pt"],
    )


def test_pt_profile_is_portuguese_plus_english():
    profile = cb.COUNTRY_PROFILES["PT"]
    assert profile == {
        "country": "Portugal",
        "languages": ["pt", "en"],
        "file": "ep24_pt_private.json",
    }


def test_pt_diacritics_remain_identity_significant():
    for canonical, ascii_variant in (
        ("João", "Joao"),
        ("António", "Antonio"),
        ("Coligação", "Coligacao"),
        ("Cidadãos", "Cidadaos"),
    ):
        assert cb.identity_key(canonical) != cb.identity_key(ascii_variant)


def test_pt_bilingual_label_selects_one_canonical_entry():
    entry = _entry(
        "Partido Socialista",
        english_label="Socialist Party",
        aliases=["PS"],
    )
    selected_pt, _ = cb.select_context("Partido Socialista", [entry], country="PT")
    selected_en, _ = cb.select_context("Socialist Party", [entry], country="PT")
    assert [e.entry_id for e in selected_pt] == [entry.entry_id]
    assert [e.entry_id for e in selected_en] == [entry.entry_id]


def test_pt_short_acronym_requires_a_token_boundary():
    ps = _entry("Partido Socialista", aliases=["PS"])
    assert cb.score_entry("PS apresentou a lista", ps) == 1.0
    assert cb.score_entry("opções para a Europa", ps) < 1.0


def test_pt_coalition_and_constituent_party_are_distinct_identities():
    ad = _entry("AD - Aliança Democrática", aliases=["AD"])
    psd = _entry("Partido Social Democrata", aliases=["PSD"])
    assert cb.identity_key(ad.label) != cb.identity_key(psd.label)
    assert ad.entry_id != psd.entry_id


def test_pt_country_scope_blocks_same_acronym_from_other_country():
    pt_ps = _entry("Partido Socialista", aliases=["PS"], country="PT")
    fr_ps = _entry("Parti socialiste", aliases=["PS"], country="FR")
    selected, provenance = cb.select_context("PS", [pt_ps, fr_ps], country="PT")
    assert [e.country for e in selected] == ["PT"]
    assert provenance["country"] == "PT"


def test_pt_context_is_explicitly_background_not_evidence():
    entry = _entry(
        "CDU - Coligação Democrática Unitária",
        aliases=["CDU"],
        english_label="Unitary Democratic Coalition",
    )
    block, provenance = cb.context_block("CDU", [entry], country="PT", language="pt")
    assert "Background context only, not evidence" in block
    assert provenance["evidence_role"] == "background_context_not_source_evidence"
