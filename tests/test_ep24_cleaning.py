"""Issue #35: non-destructive cleaning of legacy EP24 per-country dataframes.

Every fixture here is synthetic. No private research rows, researcher notes,
codebooks or credentials appear in this file (AGENTS.md public/private boundary).

The tests pin the safety properties the issue demands: cleaning must fail loudly
rather than silently mutate historical research data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import roihu_clean as rc  # noqa: E402

# --- synthetic fixture mirroring the real 95-column shape ------------------
LEGACY_HEADER = [
    "country",
    "author_username",
    "account_type",
    "source_type",
    "source_recording",
    "video_filename",
    "video_file",
    "whisper_transcript",
    "whisper_language",
    "whisper_translated",
    "ocr_1",
    "frame_1",
    "summary_analysis",
    "entities",
    "topics",
    "positive",
    "neutral",
    "negative",
    "recording_date",
    "day_number",
    "political_themes",
    "formula_of_populism_analysis",
    "formula_of_populism_us",
    "formula_of_populism_frontier",
    "formula_of_populism_us_elements",
    "formula_of_populism_frontier_elements",
    "formula_of_populism_us_affects",
    "formula_of_populism_frontier_affects",
    "video_id",
    "sequence_number",
    "recording_datetime",
    "political_preference",
    "allas_filename",
    "lda_topic",
    "lda_minor_topics",
    "lda_topic_words",
    "lda_minor_topic",
    "lda_15_topic",
    "lda_15_minor_topic",
    "lda_15_topic_words",
    "lda_40_topic",
    "lda_40_minor_topic",
    "lda_40_topic_words",
    "lda_country_topic",
    "lda_country_minor_topic",
    "lda_country_topic_words",
    "manifestoberta_predicted_class",
    "manifestoberta_probabilities",
    "corrected_date",
    "original_date",
    "corresponding_date",
    "split_number",
    "new_entity",
    "new_theme",
    "frames",
    "ocr_2",
    "ocr_3",
    "ocr_4",
    "ocr_5",
    "ocr_6",
    "frame_2",
    "frame_3",
    "frame_4",
    "frame_5",
    "frame_6",
    "video_duration",
    "profile_name",
    "daily_file_number",
    "new_id",
    "new_filename",
    "old_id",
    "puhti_filename",
    "spacy_entities",
    "us",
    "them",
    "researcher_video_checked",
    "researcher_delete_video",
    "researcher_dubious_video",
    "researcher_rerun_llm",
    "researcher_rerun_ocr",
    "researcher_rerun_whisper",
    "researcher_cut_video",
    "researcher_cut_at",
    "researcher_cut_keep",
    "researcher_split_video",
    "researcher_split_at",
    "researcher_wrong_language",
    "researcher_new_persons",
    "researcher_new_themes",
    "researcher_note",
    "researcher_note_conflict",
    "researcher_note_sources",
    "researcher_merge_conflicts",
    "researcher_match_routes",
]

# The five obsolete letters map onto these real fields (issue #35 comment [2]).
EXPECTED_LETTER_FIELDS = {
    "Y": "formula_of_populism_us_elements",
    "Z": "formula_of_populism_frontier_elements",
    "AA": "formula_of_populism_us_affects",
    "AB": "formula_of_populism_frontier_affects",
    "AE": "recording_datetime",
}


def _row(video_id: str, **overrides: str) -> dict[str, str]:
    row = {name: "" for name in LEGACY_HEADER}
    row.update(
        {
            "country": "Finland",
            "author_username": "FI1",
            "account_type": "Synthetic",
            "source_type": "Tiktok",
            "video_id": video_id,
            "allas_filename": f"https://example.invalid/{video_id}.mp4",
            "new_entity": "example person a; example person b",
            "new_theme": "ep elections; welfare",
            "video_duration": "12.0",
            **overrides,
        }
    )
    return row


def _write(path: Path, rows: list[dict[str, str]], header: list[str] | None = None) -> Path:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header or LEGACY_HEADER))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


# --- letter resolution -----------------------------------------------------
def test_spreadsheet_letters_resolve_to_real_fields_not_indices():
    resolved = rc.resolve_obsolete_columns(LEGACY_HEADER)
    assert resolved.letters == EXPECTED_LETTER_FIELDS
    assert len(resolved.lda) == 13
    assert all(name.startswith("lda_") for name in resolved.lda)


def test_letter_resolution_fails_loudly_when_header_moves():
    # A shifted header must raise rather than delete the wrong column.
    broken = list(LEGACY_HEADER)
    broken[33] = "some_new_column"
    with pytest.raises(rc.CleaningError):
        rc.resolve_obsolete_columns(broken)


def test_same_letters_map_correctly_for_every_country_schema():
    # One authoritative keep-schema for all countries (comment [6]): the header
    # is identical across countries, so the letter mapping must be too.
    resolved = rc.resolve_obsolete_columns(LEGACY_HEADER)
    assert list(resolved.letters.values()) == [
        EXPECTED_LETTER_FIELDS[letter] for letter in rc.OBSOLETE_LETTERS
    ]


# --- blank is not false ----------------------------------------------------
@pytest.mark.parametrize("blank", ["", "  ", "nan", "NaN", "None", "null", "<NA>"])
def test_blank_tri_state_is_unknown_not_false(blank):
    assert rc.parse_tri_state(blank) is None


@pytest.mark.parametrize("value", ["TRUE", "true", "1", "yes", True])
def test_explicit_true_is_true(value):
    assert rc.parse_tri_state(value) is True


@pytest.mark.parametrize("value", ["FALSE", "false", "0", "no", False])
def test_explicit_false_is_false(value):
    assert rc.parse_tri_state(value) is False


def test_unrecognised_tri_state_raises_instead_of_coercing():
    with pytest.raises(rc.CleaningError):
        rc.parse_tri_state("maybe")


def test_blank_delete_never_excludes_a_row(tmp_path):
    source = _write(tmp_path / "ep24_finland_with_researcher_notes.csv", [_row("V-1")])
    outcome = rc.clean_dataframe(source)
    assert outcome.decisions[0].decision == rc.DECISION_KEEP
    assert outcome.decisions[0].include_in_reprocess is True


# --- explicit human decisions ---------------------------------------------
def test_explicit_delete_excludes_but_source_row_survives(tmp_path):
    rows = [_row("V-KEEP"), _row("V-DELETE", researcher_delete_video="TRUE")]
    source = _write(tmp_path / "ep24_finland_with_researcher_notes.csv", rows)
    before = source.read_text(encoding="utf-8")
    outcome = rc.clean_dataframe(source)

    # the source file on disk is byte-identical after cleaning
    assert source.read_text(encoding="utf-8") == before
    # and the deleted row is still present in the derivative, only flagged
    assert len(outcome.rows) == 2
    decisions = {d.video_id: d for d in outcome.decisions}
    assert decisions["V-DELETE"].decision == rc.DECISION_EXCLUDE
    assert decisions["V-DELETE"].include_in_reprocess is False
    assert [r["video_id"] for r in outcome.reprocess_rows()] == ["V-KEEP"]


def test_dubious_videos_are_excluded_and_flagged_for_review(tmp_path):
    rows = [_row("V-1", researcher_dubious_video="TRUE")]
    outcome = rc.clean_dataframe(_write(tmp_path / "ep24_finland_with_researcher_notes.csv", rows))
    decision = outcome.decisions[0]
    assert decision.decision == rc.DECISION_DUBIOUS
    assert decision.needs_human_review is True
    assert decision.include_in_reprocess is False


def test_cut_and_split_go_to_the_recut_worklist(tmp_path):
    rows = [
        _row("V-CUT", researcher_cut_video="TRUE", researcher_cut_at="0:04"),
        _row("V-SPLIT", researcher_split_video="TRUE", researcher_split_at="0:10"),
    ]
    outcome = rc.clean_dataframe(_write(tmp_path / "ep24_finland_with_researcher_notes.csv", rows))
    statuses = {row["video_id"]: row["cleaning_status"] for row in outcome.rows}
    assert statuses == {"V-CUT": rc.DECISION_CUT, "V-SPLIT": rc.DECISION_SPLIT}
    assert {row["video_id"] for row in outcome.recut_rows()} == {"V-CUT", "V-SPLIT"}


def test_recut_worklist_aggregates_across_countries(tmp_path):
    out_dir = tmp_path / "by_country"
    reprocess = tmp_path / "to_reprocess"
    worklist = tmp_path / "ep24_videos_to_recut.csv"

    fi = rc.clean_dataframe(
        _write(
            out_dir / "ep24_finland_with_researcher_notes.csv",
            [_row("V-FI", researcher_cut_video="TRUE")],
        ),
        country="Finland",
    )
    pl = rc.clean_dataframe(
        _write(
            out_dir / "ep24_poland_with_researcher_notes.csv",
            [_row("V-PL", researcher_split_video="TRUE", country="Poland")],
        ),
        country="Poland",
    )
    rc.write_outputs(fi, output_dir=out_dir, to_reprocess_dir=reprocess, recut_worklist=worklist)
    rc.write_outputs(pl, output_dir=out_dir, to_reprocess_dir=reprocess, recut_worklist=worklist)

    import csv

    with worklist.open(encoding="utf-8", newline="") as handle:
        entries = list(csv.DictReader(handle))
    assert {entry["country"] for entry in entries} == {"Finland", "Poland"}
    assert worklist.read_text(encoding="utf-8").count("video_id") == 1  # header written once


# --- synthetic-only and note triage ---------------------------------------
def test_organic_profiles_are_excluded_from_reprocessing(tmp_path):
    rows = [_row("V-SYN"), _row("V-ORG", account_type="Organic")]
    outcome = rc.clean_dataframe(_write(tmp_path / "ep24_finland_with_researcher_notes.csv", rows))
    decisions = {d.video_id: d for d in outcome.decisions}
    assert decisions["V-ORG"].decision == rc.DECISION_EXCLUDE
    assert decisions["V-SYN"].include_in_reprocess is True


def test_trivial_test_note_is_normalised_to_empty(tmp_path):
    rows = [_row("V-1", researcher_note="Just a test note!")]
    outcome = rc.clean_dataframe(_write(tmp_path / "ep24_finland_with_researcher_notes.csv", rows))
    decision = outcome.decisions[0]
    assert decision.decision == rc.DECISION_KEEP_WITH_CORRECTION
    assert decision.cleaned_value == ""


def test_informative_note_is_preserved_with_human_provenance(tmp_path):
    note = "This video expresses highly polarized far right discourse"
    rows = [_row("V-1", researcher_note=note)]
    outcome = rc.clean_dataframe(_write(tmp_path / "ep24_finland_with_researcher_notes.csv", rows))
    decision = outcome.decisions[0]
    assert decision.cleaned_value == note
    assert decision.origin == "human"
    assert decision.include_in_reprocess is True


def test_ambiguous_note_is_not_auto_deleted(tmp_path):
    rows = [_row("V-1", researcher_note="not sure about this one")]
    outcome = rc.clean_dataframe(_write(tmp_path / "ep24_finland_with_researcher_notes.csv", rows))
    assert outcome.decisions[0].include_in_reprocess is True


# --- seed construction -----------------------------------------------------
def test_seeds_merge_human_fields_and_deduplicate_after_matching(tmp_path):
    rows = [
        _row(
            "V-1",
            new_entity="example person a; example person b",
            researcher_new_persons="Example Person A; example person c",
            new_theme="welfare",
            researcher_new_themes="welfare; migration",
        )
    ]
    outcome = rc.clean_dataframe(_write(tmp_path / "ep24_finland_with_researcher_notes.csv", rows))
    row = outcome.rows[0]
    assert row["seed_entities"] == "example person a; example person b; example person c"
    assert row["seed_themes"] == "migration; welfare"
    # raw source fields are untouched
    assert row["new_entity"] == "example person a; example person b"
    assert row["researcher_new_persons"] == "Example Person A; example person c"


def test_no_researcher_data_is_invented_when_seed_fields_are_blank():
    assert rc.merge_seed_values("", None, "[]") == []


# --- tri-state note conflicts survive -------------------------------------
def test_note_conflict_is_preserved_not_flattened(tmp_path):
    conflict = json.dumps(
        {
            "delete_video": [False, True],
            "researcher_note": ["", "Transcript does not match video."],
        },
        ensure_ascii=False,
    )
    rows = [_row("V-1", researcher_merge_conflicts=conflict, researcher_note_conflict="true")]
    outcome = rc.clean_dataframe(_write(tmp_path / "ep24_finland_with_researcher_notes.csv", rows))
    assert outcome.rows[0]["researcher_merge_conflicts"] == conflict


# --- duplicates are flagged, never silently dropped -----------------------
def test_duplicate_video_ids_are_flagged_not_deduplicated(tmp_path):
    rows = [_row("V-DUP"), _row("V-DUP"), _row("V-OTHER")]
    outcome = rc.clean_dataframe(_write(tmp_path / "ep24_finland_with_researcher_notes.csv", rows))
    assert len(outcome.rows) == 3  # nothing dropped
    duplicates = [d for d in outcome.decisions if d.decision == rc.DECISION_DUPLICATE]
    assert len(duplicates) == 2
    assert all(d.needs_human_review for d in duplicates)


# --- column handling ------------------------------------------------------
def test_obsolete_and_always_empty_columns_removed_from_derivative_only(tmp_path):
    source = _write(tmp_path / "ep24_finland_with_researcher_notes.csv", [_row("V-1")])
    outcome = rc.clean_dataframe(source)
    for name in ("recording_datetime", "lda_topic", "lda_country_topic_words"):
        assert name in outcome.source_columns
        assert name not in outcome.cleaned_columns
    # genuinely dead legacy columns are reported
    assert "corresponding_date" in outcome.empty_columns


def test_correction_retains_both_original_and_cleaned_value(tmp_path):
    rows = [_row("V-1", researcher_note="Test")]
    outcome = rc.clean_dataframe(_write(tmp_path / "ep24_finland_with_researcher_notes.csv", rows))
    row = outcome.rows[0]
    assert row["cleaning_original_value"] == "Test"
    assert row["cleaning_corrected_value"] == ""
    assert row["researcher_note"] == "Test"  # legacy column untouched


# --- pseudo-country rejection --------------------------------------------
def test_pseudo_country_is_rejected_as_a_country_dataframe(tmp_path):
    rows = [_row("V-1", country="MobileBackup")]
    with pytest.raises(rc.CleaningError):
        rc.clean_dataframe(_write(tmp_path / "ep24_mobilebackup_with_researcher_notes.csv", rows))


@pytest.mark.parametrize("name", ["MobileBackup", "mobilebackup", "backup", "SCRATCH"])
def test_pseudo_country_detection(name):
    assert rc.is_pseudo_country(name) is True


@pytest.mark.parametrize("name", ["Finland", "Poland", "Germany", "Bulgaria"])
def test_real_countries_are_not_pseudo_countries(name):
    assert rc.is_pseudo_country(name) is False


# --- reprocess export contract -------------------------------------------
def test_reprocess_export_is_keep_schema_only_with_video_id_first(tmp_path):
    import csv

    out_dir = tmp_path / "by_country"
    rows = [_row("V-1"), _row("V-2", researcher_delete_video="TRUE")]
    outcome = rc.clean_dataframe(_write(out_dir / "ep24_finland_with_researcher_notes.csv", rows))
    files = rc.write_outputs(
        outcome,
        output_dir=out_dir,
        to_reprocess_dir=tmp_path / "to_reprocess",
        recut_worklist=tmp_path / "recut.csv",
    )
    with files["reprocess"].open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        data = list(reader)

    assert header == rc.reprocess_columns()
    assert header[0] == "video_id"
    assert set(header) <= set(rc.KEEP_SCHEMA_ORDER)
    # excluded rows do not reach the reprocessing input
    assert [row[0] for row in data] == ["V-1"]
    assert files["reprocess"].name == "ep24_finland.csv"


def test_keep_schema_is_the_single_authoritative_15_column_schema():
    assert rc.KEEP_SCHEMA_ORDER == (
        "country",
        "author_username",
        "account_type",
        "source_type",
        "source_recording",
        "video_id",
        "sequence_number",
        "political_preference",
        "allas_filename",
        "new_entity",
        "new_theme",
        "video_duration",
        "researcher_new_persons",
        "researcher_new_themes",
        "researcher_note",
    )


def test_cleaned_derivative_keeps_every_source_row_and_column(tmp_path):
    import csv

    out_dir = tmp_path / "by_country"
    rows = [_row("V-1"), _row("V-2", researcher_delete_video="TRUE"), _row("V-3")]
    outcome = rc.clean_dataframe(_write(out_dir / "ep24_finland_with_researcher_notes.csv", rows))
    files = rc.write_outputs(
        outcome,
        output_dir=out_dir,
        to_reprocess_dir=tmp_path / "to_reprocess",
        recut_worklist=tmp_path / "recut.csv",
    )
    with files["cleaned"].open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        data = list(reader)
    assert len(data) == 3
    removed = set(outcome.removed_columns["by_spreadsheet_letters"].values())
    removed |= set(outcome.removed_columns["lda_family"])
    removed |= set(outcome.removed_columns["rerun_flags"])
    # every legacy column survives except the explicitly authorized removals
    for name in LEGACY_HEADER:
        if name in removed:
            assert name not in header, name
        else:
            assert name in header, name
    for name in rc.CLEANING_FIELDS:
        assert name in header


def test_artifacts_are_written_and_report_is_human_readable(tmp_path):
    out_dir = tmp_path / "by_country"
    outcome = rc.clean_dataframe(
        _write(out_dir / "ep24_finland_with_researcher_notes.csv", [_row("V-1")])
    )
    files = rc.write_outputs(
        outcome,
        output_dir=out_dir,
        to_reprocess_dir=tmp_path / "to_reprocess",
        recut_worklist=tmp_path / "recut.csv",
    )
    for key in ("cleaned", "decisions", "provenance", "report", "reprocess"):
        assert files[key].exists(), key
    report = files["report"].read_text(encoding="utf-8")
    assert "source rows" in report and "output rows" in report
    provenance = json.loads(files["provenance"].read_text(encoding="utf-8"))
    assert provenance["destructive"] is False
    assert provenance["blank_is_not_false"] is True


def test_media_skip_seconds_comes_from_the_canonical_rule():
    assert rc.VIDEO_INITIAL_SKIP_SECONDS == 1.0


def test_country_is_taken_from_data_not_from_the_filename(tmp_path):
    # comment [1]: country assignment must come from metadata, not a path component.
    rows = [_row("V-1", country="Poland")]
    outcome = rc.clean_dataframe(_write(tmp_path / "ep24_finland_with_researcher_notes.csv", rows))
    assert outcome.country == "Poland"
