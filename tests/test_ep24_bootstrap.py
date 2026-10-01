"""Bootstrap merge tests for the EP24 pipeline (issue #64).

The four properties the issue makes non-negotiable, tested against the shapes the
real Finland input actually contains (verified from the LFS object):

- `entities` is created from `new_entity` + `researcher_new_persons` before Step 1;
- `themes` is created from `new_theme` + `researcher_new_themes` before Step 1;
- the four source columns remain, unchanged, alongside the merged fields;
- `researcher_note` stays context/provenance and is never turned into a label.

Fixtures are synthetic. The empty-seed shapes (`'[]'`) are copied from the real
file because they are the trap this module exists to avoid -- not research data.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ep24_bootstrap as bs  # noqa: E402


SOURCE_ROW = {
    "video_id": "SYNTH-FI-0001",
    "country": "Finland",
    "author_username": "synthetic-profile-a",
    "account_type": "Synthetic",
    "source_type": "Instagram",
    "source_recording": "screen-synthetic-01",
    "sequence_number": "1",
    "political_preference": "synthetic preference",
    "allas_filename": "https://example.invalid/synthetic.mp4",
    "new_entity": "",
    "new_theme": "ep elections; campaign and campaigning",
    "video_duration": "10.0",
    "researcher_new_persons": "[]",
    "researcher_new_themes": "[]",
    "researcher_note": "",
}


# --- the empty-seed trap ---------------------------------------------------

def test_empty_list_literal_is_treated_as_empty_not_as_a_value():
    """`'[]'` is what the real input carries when there is nothing to merge."""
    assert bs.split_seed_values("[]") == []
    assert bs.merge_seed_fields("Theme A", "[]") == "Theme A"


def test_merging_only_empty_seeds_produces_an_empty_string():
    """No literal `[]` may leak into a canonical field."""
    merged = bs.merge_seed_fields("", "[]", "{}", "nan")
    assert merged == ""
    assert "[]" not in merged


# --- the two named merges --------------------------------------------------

def test_entities_merge_new_entity_and_researcher_new_persons():
    row = dict(SOURCE_ROW, new_entity="Synthetic Person", researcher_new_persons='["Synthetic Other"]')
    assert bs.build_entities(row) == "Synthetic Person; Synthetic Other"


def test_themes_merge_new_theme_and_researcher_new_themes():
    row = dict(SOURCE_ROW, new_theme="Theme A", researcher_new_themes='["Theme B"]')
    assert bs.build_themes(row) == "Theme A; Theme B"


def test_entity_seeds_are_independent_of_theme_seeds():
    """`new_entity` empty while `new_theme` populated is the real shape."""
    row = dict(SOURCE_ROW, new_entity="", researcher_new_persons="[]")
    assert bs.build_entities(row) == ""
    assert bs.build_themes(row) == "ep elections; campaign and campaigning"


def test_merges_are_lossless_and_drop_exact_duplicates_only():
    row = dict(
        SOURCE_ROW,
        new_entity="Party A; Party A",
        researcher_new_persons='["Party A", "Party B"]',
    )
    assert bs.build_entities(row) == "Party A; Party B"


def test_original_order_is_preserved():
    row = dict(SOURCE_ROW, new_entity="Z", researcher_new_persons='["A"]')
    assert bs.build_entities(row) == "Z; A"


# --- bootstrap row: additive, never destructive ----------------------------

def test_bootstrap_row_adds_the_merged_fields_and_a_stable_record_id():
    row = dict(SOURCE_ROW, new_entity="Synthetic Person")
    out = bs.build_bootstrap_row(row, country="Finland", source_row_index=7)
    assert out["entities"] == "Synthetic Person"
    assert out["themes"] == "ep elections; campaign and campaigning"
    assert out["record_id"] == "Finland|SYNTH-FI-0001|7"


def test_every_source_column_is_carried_through_unchanged():
    out = bs.build_bootstrap_row(dict(SOURCE_ROW), country="Finland", source_row_index=0)
    for key, value in SOURCE_ROW.items():
        assert key in out, f"bootstrap dropped {key}"
        assert out[key] == value, f"bootstrap mutated {key}"


def test_the_four_researcher_source_columns_survive_alongside_the_merged_ones():
    row = dict(SOURCE_ROW, new_entity="E", researcher_new_persons='["P"]',
               new_theme="T", researcher_new_themes='["T2"]')
    out = bs.build_bootstrap_row(row, country="Finland", source_row_index=0)
    for column in ("new_entity", "researcher_new_persons", "new_theme", "researcher_new_themes"):
        assert column in out
        assert out[column] == row[column]
    assert out["entities"] == "E; P"
    assert out["themes"] == "T; T2"


def test_bootstrap_does_not_mutate_its_input():
    row = dict(SOURCE_ROW)
    before = dict(row)
    bs.build_bootstrap_row(row, country="Finland", source_row_index=0)
    assert row == before, "bootstrap must not mutate the caller's row"


def test_record_ids_are_unique_within_a_country_and_include_the_row_index():
    rows = [dict(SOURCE_ROW, video_id="SAME") for _ in range(3)]
    out = bs.bootstrap_rows(rows, country="Finland")
    ids = [r["record_id"] for r in out]
    assert len(set(ids)) == 3, "repeated video_id must not collapse to one record"


def test_researcher_note_is_carried_but_never_used_as_a_label():
    row = dict(SOURCE_ROW, researcher_note="check this one, unsure about the party")
    out = bs.build_bootstrap_row(row, country="Finland", source_row_index=0)
    assert out["researcher_note"] == "check this one, unsure about the party"
    assert "check this one" not in out["entities"]
    assert "check this one" not in out["themes"]


# --- codebook normalization preserves originals ----------------------------

def test_codebook_normalization_keeps_unknown_labels_verbatim():
    codebook = {"labels": {"synth party": "Synthetic Party"}}
    assert bs.apply_codebook_labels("Unknown Label", codebook) == "Unknown Label"


def test_codebook_normalization_applies_a_known_spelling_without_dropping_labels():
    codebook = {"labels": {"synth party": "Synthetic Party"}}
    out = bs.apply_codebook_labels("Synth Party; Other", codebook)
    assert out == "Synthetic Party; Other"


def test_codebook_is_optional_and_harmless_when_absent():
    assert bs.apply_codebook_labels("A; B", None) == "A; B"
    assert bs.apply_codebook_labels("", {"labels": {"a": "A"}}) == ""


def test_provenance_note_names_the_merge_sources():
    note = bs.provenance_note()
    assert "new_entity" in note["entities_from"]
    assert "researcher_new_persons" in note["entities_from"]
    assert "new_theme" in note["themes_from"]
    assert "researcher_new_themes" in note["themes_from"]
