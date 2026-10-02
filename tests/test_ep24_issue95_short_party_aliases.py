"""Issue #95: safe short political abbreviations in multilingual retrieval."""
from roihu_codebooks import CodebookEntry, context_block, score_entry


def _entry(entry_id, label, aliases, country, *, kind="entity"):
    return CodebookEntry(
        entry_id=entry_id,
        kind=kind,
        label=label,
        aliases=aliases,
        country=country,
        source_languages=["sv" if country == "SE" else "pl"],
        review_state="CANONICAL",
        origin="researcher",
        locked=True,
        entity_type="party",
    )


SWEDISH = [
    _entry("SE:S", "Socialdemokraterna", ["S"], "SE"),
    _entry("SE:M", "Moderaterna", ["M"], "SE"),
    _entry("SE:V", "Vänsterpartiet", ["V"], "SE"),
    _entry("SE:C", "Centerpartiet", ["C"], "SE"),
    _entry("SE:L", "Liberalerna", ["L"], "SE"),
    _entry("SE:KD", "Kristdemokraterna", ["KD"], "SE"),
    _entry("SE:MP", "Miljöpartiet", ["MP"], "SE"),
    _entry("SE:SD", "Sverigedemokraterna", ["SD"], "SE"),
]


def _labels(query, entries, country):
    _block, prov = context_block(query, entries, country=country)
    return {row["label"] for row in prov["selected"]}


def test_swedish_one_letter_party_aliases_match_only_as_tokens():
    assert "Socialdemokraterna" in _labels("S vill ändra politiken", SWEDISH, "SE")
    assert "Moderaterna" in _labels("M presenterar sitt program", SWEDISH, "SE")
    assert "Socialdemokraterna" not in _labels("Sverige röstar i EU-valet", SWEDISH, "SE")
    assert "Moderaterna" not in _labels("Malmö debatterar", SWEDISH, "SE")


def test_swedish_two_letter_aliases_are_boundary_aware():
    assert "Kristdemokraterna" in _labels("KD går till val", SWEDISH, "SE")
    assert "Miljöpartiet" in _labels("MP presenterar klimatpolitik", SWEDISH, "SE")
    assert "Sverigedemokraterna" in _labels("SD ökar enligt mätningen", SWEDISH, "SE")
    assert "Kristdemokraterna" not in _labels("KDlab är ett syntetiskt ord", SWEDISH, "SE")
    assert "Miljöpartiet" not in _labels("CMP data", SWEDISH, "SE")


def test_three_and_four_character_forms_do_not_use_substring_matching():
    hdz = _entry("HR:HDZ", "Hrvatska demokratska zajednica", ["HDZ"], "HR")
    most = _entry("HR:Most", "Most", [], "HR")
    assert score_entry("HDZ je pobijedio", hdz) == 1.0
    assert score_entry("HDZx", hdz) < 1.0
    assert score_entry("Most je osvojio mandat", most) == 1.0
    assert score_entry("Mostar je grad", most) < 1.0


def test_cross_country_collision_is_scoped_before_short_alias_scoring():
    es_pp = _entry("ES:PP", "Partido Popular", ["PP"], "ES")
    pl_pp = _entry("PL:PP", "Polska Partia", ["PP"], "PL")
    entries = [es_pp, pl_pp]
    assert _labels("PP wydała komunikat", entries, "PL") == {"Polska Partia"}
    assert _labels("El PP publica un comunicado", entries, "ES") == {"Partido Popular"}


def test_same_country_ambiguous_short_alias_returns_all_candidates():
    a = _entry("HR:PIP1", "Pravo i Pravda", ["PiP"], "HR")
    b = _entry("HR:PIP2", "Istarska stranka umirovljenika", ["PiP"], "HR")
    assert _labels("PiP je na izborima", [a, b], "HR") == {
        "Pravo i Pravda",
        "Istarska stranka umirovljenika",
    }


def test_long_inflected_form_still_uses_substring_recall():
    climate = CodebookEntry(
        entry_id="FI:climate",
        kind="topic",
        label="ilmasto",
        country="FI",
        source_languages=["fi"],
        review_state="CANONICAL",
    )
    assert score_entry("Puhumme ilmastosta tänään", climate) == 1.0


def test_background_context_remains_explicitly_non_evidence():
    _block, prov = context_block("SD puhuu", SWEDISH, country="SE", language="sv")
    assert prov["evidence_role"] == "background_context_not_source_evidence"
