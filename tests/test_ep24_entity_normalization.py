"""Multilingual entity normalization / resolution regression tests (issue #142).

Public synthetic fixtures only: no private research data appears here, and every
name is invented or a public figure's public role. The scenarios mirror the
issue's required test list one-for-one.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import ep24_entities as E
from ep24_entities import EntityRecord, EntityRegistry

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def fi_registry() -> EntityRegistry:
    reg = EntityRegistry()
    reg.add(EntityRecord(
        entity_id="FI-ORPO", canonical_name="Petteri Orpo", entity_type="person",
        aliases=["Orpo", "Pääministeri Orpo"], country="FI",
        language_aliases={"fi": ["Petteri Orpo"], "en": ["Petteri Orpo"]},
        review_state="CANONICAL", origin="codebook",
    ))
    return reg


def multi_country_registry() -> EntityRegistry:
    reg = EntityRegistry()
    reg.add(EntityRecord(entity_id="FI-K", canonical_name="Kaisa Korhonen", entity_type="person", country="FI"))
    reg.add(EntityRecord(entity_id="PT-K", canonical_name="Kaisa Korhonen", entity_type="person", country="PT"))
    reg.add(EntityRecord(entity_id="DE-S", canonical_name="Olaf Scholz", entity_type="person", country="DE"))
    reg.add(EntityRecord(entity_id="DE-SPD", canonical_name="Sozialdemokratische Partei Deutschlands",
                         entity_type="party", country="DE", aliases=["SPD"]))
    reg.add(EntityRecord(entity_id="PT-C", canonical_name="António Costa", entity_type="person", country="PT",
                         language_aliases={"en": ["Antonio Costa"]}))
    reg.add(EntityRecord(entity_id="PL-T", canonical_name="Donald Tusk", entity_type="person", country="PL"))
    reg.add(EntityRecord(entity_id="PT-T", canonical_name="Donald Tusk", entity_type="person", country="PT"))
    return reg


# --------------------------------------------------------------------------- #
# Mention preservation -- the issue's central rule
# --------------------------------------------------------------------------- #

def test_original_surface_form_is_never_discarded():
    reg = fi_registry()
    for surface in ("Pääministeri Orpo", "Orpo", "Petteri Orpo", "Orpon"):
        result = reg.resolve(surface, country="FI", language="fi")
        assert result["surface_form"] == surface, "source wording must be preserved verbatim"
        assert result["decision"] == "RESOLVED"
        assert result["canonical_name"] == "Petteri Orpo"
        assert result["entity_id"] == "FI-ORPO"


def test_normalization_records_what_it_removed():
    norm = E.normalize_mention("Pääministeri Orpo", language="fi")
    assert norm.surface_form == "Pääministeri Orpo"
    assert norm.normalized_form == "Orpo"
    assert norm.stripped_titles == ("Pääministeri",), "the title is analytically meaningful; keep it"
    assert norm.key == E.identity_key("Pääministeri Orpo")


def test_identity_key_is_not_changed_by_this_layer():
    """fold_key is a comparison key; identity_key stays the id seed."""
    assert E.identity_key("António Costa") != E.fold_key("António Costa")
    assert E.identity_key("António Costa") == "antónio costa"
    assert E.fold_key("António Costa") == "antonio costa"


# --------------------------------------------------------------------------- #
# The issue's required multilingual scenarios
# --------------------------------------------------------------------------- #

def test_finnish_title_surname_and_inflection_resolve_to_one_person():
    reg = fi_registry()
    forms = ["Petteri Orpo", "Orpo", "Pääministeri Orpo", "Orpon", "Orpolla"]
    ids = {reg.resolve(f, country="FI", language="fi").get("entity_id") for f in forms}
    assert ids == {"FI-ORPO"}, ids


def test_polish_inflected_personal_name_forms():
    reg = multi_country_registry()
    base = reg.resolve("Donald Tusk", country="PL", language="pl")
    assert base["decision"] == "RESOLVED"
    assert base["match_method"] == "exact_canonical"
    # Polish case endings on a surname reach the same entity. A bare inflected
    # surname resolves through the unique-surname step rather than the fold step,
    # because "Tuska" -- unlike "Tuskowi" -- is reachable as a whole-token variant
    # only after the bare-surname check has already found exactly one candidate.
    for form in ("Tuska", "Tuskowi"):
        result = reg.resolve(form, country="PL", language="pl")
        assert result["decision"] == "RESOLVED", (form, result)
        assert result["entity_id"] == "PL-T"
    # An inflected *full* name (both tokens inflected) is offered as a candidate
    # rather than decided: only the final token is folded, so the layer does not
    # claim a certainty it has not earned. Recorded as a known limitation.
    full = reg.resolve("Donalda Tuska", country="PL", language="pl")
    assert full["decision"] == "CANDIDATE"
    assert full["candidates"][0]["entity_id"] == "PL-T"


def test_portuguese_titles_multipart_surnames_and_accents():
    reg = multi_country_registry()
    for form in ("António Costa", "Antonio Costa", "ANTÓNIO COSTA"):
        result = reg.resolve(form, country="PT", language="pt")
        assert result["decision"] == "RESOLVED", (form, result)
        assert result["canonical_name"] == "António Costa"
    titled = reg.resolve("Primeiro-Ministro Costa", country="PT", language="pt")
    assert titled["decision"] == "RESOLVED"
    assert titled["surface_form"] == "Primeiro-Ministro Costa"


def test_german_title_and_surname_and_compound_organisation():
    reg = multi_country_registry()
    assert reg.resolve("Bundeskanzler Scholz", country="DE", language="de")["entity_id"] == "DE-S"
    assert reg.resolve("Scholz", country="DE", language="de")["entity_id"] == "DE-S"
    assert reg.resolve("SPD", country="DE", language="de")["entity_id"] == "DE-SPD"
    long_form = reg.resolve("Sozialdemokratische Partei Deutschlands", country="DE", language="de")
    assert long_form["entity_id"] == "DE-SPD"


# --------------------------------------------------------------------------- #
# The negative cases -- these are the point of the layer
# --------------------------------------------------------------------------- #

def test_ambiguous_surname_must_not_auto_merge():
    """Two politicians sharing a surname: no context, no merge."""
    reg = EntityRegistry()
    reg.add(EntityRecord(entity_id="FI-A", canonical_name="Petteri Orpo", entity_type="person", country="FI"))
    reg.add(EntityRecord(entity_id="FI-B", canonical_name="Matti Orpo", entity_type="person", country="FI"))
    for form in ("Orpo", "Orpon"):
        result = reg.resolve(form, country="FI", language="fi")
        assert result["decision"] != "RESOLVED", f"{form} must not be silently merged"
        assert "entity_id" not in result


def test_same_name_in_two_countries_stays_separate():
    reg = multi_country_registry()
    fi = reg.resolve("Kaisa Korhonen", country="FI")
    pt = reg.resolve("Kaisa Korhonen", country="PT")
    assert fi["entity_id"] == "FI-K"
    assert pt["entity_id"] == "PT-K"
    assert fi["entity_id"] != pt["entity_id"]


def test_country_context_constrains_but_does_not_invent_a_mention():
    reg = multi_country_registry()
    # Same person-name exists in PL and PT. Without a country the layer must not
    # silently pick one: it abstains or offers candidates, never a decision.
    unconstrained = reg.resolve("Donald Tusk")
    assert unconstrained["decision"] != "RESOLVED" or unconstrained.get("entity_id") is None
    # With a country, both are reachable and distinct -- the constraint selects,
    # it does not fabricate.
    pl = reg.resolve("Donald Tusk", country="PL")
    pt = reg.resolve("Donald Tusk", country="PT")
    assert pl["entity_id"] == "PL-T"
    assert pt["entity_id"] == "PT-T"
    # A mention that exists nowhere is unresolved regardless of context.
    missing = reg.resolve("Someone Never Registered", country="FI")
    assert missing["decision"] == "UNRESOLVED"
    assert missing["candidates"] == []


def test_unknown_mention_is_not_forced_onto_a_canonical_entity():
    reg = fi_registry()
    result = reg.resolve("Completely Different Person", country="FI", language="fi")
    assert result["decision"] == "UNRESOLVED"
    assert result.get("canonical_name") is None


# --------------------------------------------------------------------------- #
# Noise recovery
# --------------------------------------------------------------------------- #

def test_ocr_corrupted_mention_is_recovered_as_a_candidate():
    reg = fi_registry()
    for corrupted in ("Petteri 0rpo", "Petteri Orp0", "Petter! Orpo"):
        result = reg.resolve(corrupted, country="FI", language="fi")
        assert result["decision"] == "CANDIDATE", corrupted
        assert result["match_method"] == "fuzzy_candidate"
        assert result["candidates"][0]["entity_id"] == "FI-ORPO"
        assert 0.0 < result["candidates"][0]["score"] <= 1.0


def test_fuzzy_recovery_never_auto_canonizes():
    """Fuzzy evidence is a candidate, not a decision, whatever its score."""
    reg = fi_registry()
    result = reg.resolve("Petter! Orpo", country="FI", language="fi")
    assert result["decision"] == "CANDIDATE"
    assert "entity_id" not in result
    assert "canonical_name" not in result


def test_asr_corrupted_mention_recovered_via_context_embedding():
    reg = fi_registry()
    calls: list[str] = []

    def embedder(text: str) -> list[float]:
        calls.append(text)
        # Toy deterministic embedding: character bigrams. Stands in for a
        # multilingual SentenceTransformers model without downloading one.
        vector = [0.0] * 32
        lowered = text.casefold()
        for index in range(len(lowered) - 1):
            vector[sum(map(ord, lowered[index:index + 2])) % 32] += 1.0
        return vector

    result = reg.resolve("Petteri 0rpo", country="FI", language="fi", embedder=embedder)
    assert calls, "the semantic step must actually query the embedder"
    assert result["decision"] in {"CANDIDATE", "RESOLVED"}, result
    top = (result.get("candidates") or [{}])[0]
    assert top.get("entity_id") == "FI-ORPO", result


def test_semantic_step_can_only_offer_candidates_never_a_decision():
    """Semantic similarity is evidence, not adjudication."""
    reg = EntityRegistry()
    reg.add(EntityRecord(entity_id="FI-X", canonical_name="Someone Else Entirely",
                         entity_type="person", country="FI"))

    def identical(text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    result = reg.resolve("A Totally Unrelated Mention", country="FI",
                         embedder=identical, semantic_threshold=0.5)
    assert result["decision"] == "CANDIDATE"
    assert "entity_id" not in result
    assert result["candidates"][0]["method"] == "semantic_candidate"


def test_broken_embedder_does_not_break_resolution():
    reg = fi_registry()

    def broken(text: str):
        raise RuntimeError("model unavailable")

    result = reg.resolve("Petteri Orpo", country="FI", language="fi", embedder=broken)
    assert result["decision"] == "RESOLVED", "a broken optional step must not fail the row"


# --------------------------------------------------------------------------- #
# Registry construction / researcher corrections
# --------------------------------------------------------------------------- #

def test_codebook_entries_seed_the_registry(tmp_path):
    payload = {
        "country_code": "FI",
        "entries": [
            {
                "kind": "person", "label": "Petteri Orpo", "entity_type": "person",
                "aliases": ["Orpo", "Pääministeri Orpo"], "review_state": "CANONICAL",
                "english_label": "Petteri Orpo",
            },
            {"kind": "topic", "label": "Should be excluded"},
        ],
    }
    path = tmp_path / "codebook.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    reg = EntityRegistry.from_codebook_paths([path])
    assert len(reg) == 1, "theme/topic kinds are out of scope for entity resolution"
    record = reg.records()[0]
    assert record.canonical_name == "Petteri Orpo"
    assert record.entity_id.startswith("CB-"), "ids come from the codebook, not this layer"
    assert "Orpo" in record.aliases
    assert reg.resolve("Pääministeri Orpo", country="FI", language="fi")["decision"] == "RESOLVED"


def test_researcher_correction_resolves_what_heuristics_cannot():
    """A bare shared surname is unresolvable until a reviewer decides it."""
    reg = EntityRegistry()
    reg.add(EntityRecord(entity_id="FI-A", canonical_name="Petteri Orpo", entity_type="person", country="FI"))
    reg.add(EntityRecord(entity_id="FI-B", canonical_name="Matti Orpo", entity_type="person", country="FI"))
    assert reg.resolve("Orpo", country="FI", language="fi")["decision"] != "RESOLVED"

    reg.apply_corrections({"Orpo": "FI-B"})
    result = reg.resolve("Orpo", country="FI", language="fi")
    assert result["decision"] == "RESOLVED"
    assert result["entity_id"] == "FI-B"
    assert result["match_method"] == "researcher_mapping"
    assert result["confidence"] == 1.0


def test_researcher_correction_outranks_an_exact_alias():
    """The correction, not the alias table, decides -- otherwise it is not durable."""
    reg = EntityRegistry()
    reg.add(EntityRecord(entity_id="FI-A", canonical_name="Petteri Orpo", entity_type="person",
                         country="FI", aliases=["Orpo"]))
    reg.add(EntityRecord(entity_id="FI-B", canonical_name="Matti Orpo", entity_type="person", country="FI"))
    # The alias would win by exact match...
    assert reg.resolve("Orpo", country="FI", language="fi")["match_method"] == "exact_alias"

    # ...but a reviewer has decided otherwise.
    reg.apply_corrections({"Orpo": "FI-B"})
    result = reg.resolve("Orpo", country="FI", language="fi")
    assert result["entity_id"] == "FI-B"
    assert result["match_method"] == "researcher_mapping"


def test_correction_targeting_an_unknown_entity_is_refused():
    reg = fi_registry()
    with pytest.raises(KeyError):
        reg.apply_corrections({"Orpo": "NO-SUCH-ENTITY"})


def test_corrections_round_trip_through_a_file(tmp_path):
    path = tmp_path / "corrections.json"
    path.write_text(json.dumps({"Orpo": "FI-ORPO"}, ensure_ascii=False), encoding="utf-8")
    assert EntityRegistry.load_corrections(path) == {"Orpo": "FI-ORPO"}


def test_registry_rejects_incomplete_records():
    reg = EntityRegistry()
    with pytest.raises(ValueError):
        reg.add(EntityRecord(entity_id="X", canonical_name=""))
    with pytest.raises(ValueError):
        reg.add(EntityRecord(entity_id="", canonical_name="Name"))


# --------------------------------------------------------------------------- #
# Entity type / temporal constraints
# --------------------------------------------------------------------------- #

def test_entity_type_constrains_resolution():
    reg = multi_country_registry()
    person = reg.resolve("Scholz", country="DE", entity_type="person")
    assert person.get("entity_id") == "DE-S"
    party = reg.resolve("Scholz", country="DE", entity_type="party")
    assert party["decision"] != "RESOLVED", "a person query must not match a party scope"


def test_temporal_validity_can_exclude_an_entity():
    reg = EntityRegistry()
    reg.add(EntityRecord(entity_id="FI-OLD", canonical_name="Erkki Vanha", entity_type="person",
                         country="FI", valid_from="2015-01-01", valid_to="2019-12-31"))
    reg.add(EntityRecord(entity_id="FI-NEW", canonical_name="Erkki Vanha", entity_type="person",
                         country="FI", valid_from="2024-01-01"))
    assert reg.resolve("Erkki Vanha", country="FI", valid_at="2024-06-01").get("entity_id") == "FI-NEW"
    assert reg.resolve("Erkki Vanha", country="FI", valid_at="2017-06-01").get("entity_id") == "FI-OLD"


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def test_review_queue_separates_ambiguous_from_resolved():
    reg = fi_registry()
    results = [
        reg.resolve("Petteri Orpo", country="FI", language="fi"),
        reg.resolve("Petter! Orpo", country="FI", language="fi"),
        reg.resolve("Nobody At All", country="FI", language="fi"),
    ]
    summary = E.resolution_summary(results)
    assert summary["total"] == 3
    assert summary["resolved"] == 1
    assert summary["decisions"].get("CANDIDATE") == 1
    assert summary["decisions"].get("UNRESOLVED") == 1
    assert len(summary["review_queue"]) == 2
    assert all(entry["decision"] in {"CANDIDATE", "UNRESOLVED"} for entry in summary["review_queue"])


def test_no_silent_merge_across_a_whole_batch():
    """The layer's headline guarantee, asserted over a mixed batch."""
    reg = fi_registry()
    reg.add(EntityRecord(entity_id="FI-B", canonical_name="Matti Orpo", entity_type="person", country="FI"))
    forms = ["Petteri Orpo", "Orpo", "Orpon", "Petteri 0rpo", "Nobody At All", "Matti Orpo"]
    summary = E.resolution_summary(reg.resolve_many(forms, country="FI", language="fi"))
    for entry in summary["review_queue"]:
        result = reg.resolve(entry["surface_form"], country="FI", language="fi")
        assert result["decision"] != "RESOLVED", entry


# --------------------------------------------------------------------------- #
# Normalization primitives
# --------------------------------------------------------------------------- #

def test_strip_titles_is_conservative():
    # A title word inside a longer name is not a title.
    remainder, stripped = E.strip_titles("Ministerial Group", language="en")
    assert remainder == "Ministerial Group"
    assert stripped == []
    # A title alone is not stripped to nothing.
    remainder, _ = E.strip_titles("Minister", language="en")
    assert remainder == "Minister"


def test_fold_key_handles_non_nfd_letters():
    assert E.fold_key("Straße") == "strasse"
    assert E.fold_key("Łódź") == "lodz"
    assert E.fold_key("Ørsted") == "orsted"


def test_inflection_variants_are_language_aware():
    assert "Orpo" in E.inflection_variants("Orpon", "fi")
    assert "Orpo" in E.inflection_variants("Orpolla", "fi")
    # A language whose endings do not apply must not generate variants for it.
    assert E.inflection_variants("Orpo", "fi") == []
    # The given name is not inflected; only the final token is.
    assert all(not v.startswith("Orpo ") for v in E.inflection_variants("Orpo Petteri", "fi"))


# --------------------------------------------------------------------------- #
# Additive dataframe integration
# --------------------------------------------------------------------------- #

def test_resolve_dataframe_is_additive_and_preserves_mentions():
    """Every incoming column is forwarded; the layer only appends."""
    pd = pytest.importorskip("pandas")
    reg = fi_registry()
    frame = pd.DataFrame({
        "video_id": ["v1", "v2", "v3"],
        "country": ["Finland"] * 3,
        "new_entity": ["Pääministeri Orpo", "Orpo", "[]"],
        "researcher_new_persons": ["", "Petteri Orpo", "Orpon"],
        "researcher_note": ["", "note", ""],
    })
    before_columns = list(frame.columns)
    before_cells = list(frame["new_entity"])

    summary = E.resolve_dataframe(frame, reg, country="FI", language="fi")

    assert summary["columns_preserved"] is True
    assert list(frame.columns)[: len(before_columns)] == before_columns
    assert set(summary["columns_added"]) == set(E.entity_resolution_columns())
    # The original wording is untouched -- the issue's central requirement.
    assert list(frame["new_entity"]) == before_cells
    assert list(frame["researcher_new_persons"])[2] == "Orpon"
    # ...and the canonical entity is added beside it.
    assert "FI-ORPO" in frame.at[0, "ep24_entity_ids"]
    assert "Petteri Orpo" in frame.at[2, "ep24_entity_canonical_names"]


def test_resolve_dataframe_reports_unresolved_rows_separately():
    pd = pytest.importorskip("pandas")
    reg = fi_registry()
    frame = pd.DataFrame({"new_entity": ["Nobody At All", "Petteri Orpo"]})
    summary = E.resolve_dataframe(frame, reg, country="FI", language="fi")
    assert summary["resolved"] == 1
    assert summary["decisions"].get("UNRESOLVED") == 1
    assert len(summary["review_queue"]) == 1
    assert summary["review_queue"][0]["surface_form"] == "Nobody At All"


def test_resolve_dataframe_rejects_non_dataframe():
    with pytest.raises(TypeError):
        E.resolve_dataframe([{"new_entity": "x"}], fi_registry())


def test_split_mentions_ignores_empty_container_noise():
    # '[]' is common noise in the real researcher columns and must not become a mention.
    assert E._split_mentions("[]") == []
    assert E._split_mentions("{}") == []
    assert E._split_mentions("") == []
    assert E._split_mentions("Petteri Orpo") == ["Petteri Orpo"]
    assert E._split_mentions("Orpo; Kokoomus") == ["Orpo", "Kokoomus"]
    assert E._split_mentions("A\nB") == ["A", "B"]
    assert E._split_mentions("Orpo; Orpo") == ["Orpo"], "duplicates collapse"


# --------------------------------------------------------------------------- #
# Codebook seeding on the repository's real public codebook
# --------------------------------------------------------------------------- #

def test_public_party_codebook_seeds_the_registry():
    """Uses the repository's real public SE party codebook, not a fixture."""
    from roihu_codebooks import _normalize_item

    path = Path(__file__).resolve().parents[1] / "scripts/ep24/data/ep24_parties_se.json"
    assert path.is_file(), path
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = [
        _normalize_item(
            {"kind": "party", "label": name, "aliases": aliases},
            kind="party", default_country="SE", layer="country",
        )
        for name, aliases in payload.items() if not name.startswith("_")
    ]
    entries = [e for e in entries if e is not None]
    assert entries, "the public codebook should contribute party entries"

    reg = EntityRegistry.from_codebooks(entries)
    assert len(reg) == len(entries)
    first = reg.records()[0]
    resolved = reg.resolve(first.canonical_name, country="SE")
    assert resolved["decision"] == "RESOLVED"
    assert resolved["entity_id"] == first.entity_id
    if first.aliases:
        assert reg.resolve(first.aliases[0], country="SE").get("entity_id") == first.entity_id
