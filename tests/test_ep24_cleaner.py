from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# `pytest -q tests/` (CI) does not put the repository root on sys.path, only
# `python -m pytest` does. Sibling test modules already handle this the same
# way; without it this module is a collection error that aborts the whole job.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ep24_cleaner import (  # noqa: E402
    clean_dataframe,
    explicit_bool,
    merge_human_labels,
    resolve_legacy_drop_columns,
    spreadsheet_column_name,
    validate_country,
)


def _source() -> pd.DataFrame:
    useful = [
        "video_id",
        "allas_file",
        "country",
        "account_username",
        "account_type",
        "political_preference",
        "new_entity",
        "new_theme",
        "researcher_new_persons",
        "researcher_new_themes",
        "researcher_delete_video",
        "researcher_dubious_video",
        "researcher_cut_start",
        "researcher_split_video",
        "researcher_note",
        "source_type",
        "source_recording",
        "sequence_number",
        "recording_date",
        "summary_analysis",
        "topics",
        "entities",
        "positive",
        "frame_1",
    ]
    columns = useful + [f"legacy_{i}" for i in range(25, 55)]
    rows = [
        {
            "video_id": "v1",
            "allas_file": "allas://v1.mp4",
            "country": "Finland",
            "account_username": "synthetic-a",
            "account_type": "Synthetic",
            "political_preference": "Red-green",
            "new_entity": "Sanna Marin, EU",
            "researcher_new_persons": "European Union; Marin",
            "new_theme": "Climate",
            "researcher_new_themes": "climate change",
            "researcher_delete_video": pd.NA,
            "researcher_dubious_video": False,
            "researcher_cut_start": pd.NA,
            "researcher_split_video": pd.NA,
            "researcher_note": "Useful contextual note",
            "source_type": "TikTok",
            "source_recording": "session-1",
            "sequence_number": 1,
            "recording_date": "2024-05-01",
            "summary_analysis": "old model output",
            "topics": "old topic",
            "entities": "old model entity",
            "positive": 0.4,
            "frame_1": "old frame",
        },
        {
            "video_id": "v2",
            "allas_file": "allas://v2.mp4",
            "country": "Finland",
            "account_username": "synthetic-b",
            "account_type": "Synthetic",
            "researcher_delete_video": True,
        },
        {
            "video_id": "v3",
            "allas_file": "allas://v3.mp4",
            "country": "Finland",
            "account_username": "synthetic-c",
            "account_type": "Synthetic",
            "researcher_dubious_video": "TRUE",
        },
        {
            "video_id": "v4",
            "allas_file": "allas://v4.mp4",
            "country": "Finland",
            "account_username": "synthetic-d",
            "account_type": "Synthetic",
            "researcher_cut_start": True,
        },
        {
            "video_id": "v5",
            "allas_file": "allas://v5.mp4",
            "country": "Finland",
            "account_username": "organic-a",
            "account_type": "Organic",
        },
        {
            "video_id": "v6",
            "allas_file": "allas://v6.mp4",
            "country": "Finland",
            "account_username": "synthetic-e",
            "account_type": "Synthetic",
            "researcher_note": "Test",
        },
        {
            "video_id": "v7",
            "allas_file": "allas://v7.mp4",
            "country": "Finland",
            "account_username": "synthetic-f",
            "account_type": "Synthetic",
            "researcher_note": "This is filthy garbage",
        },
    ]
    return pd.DataFrame(rows, columns=columns)


def _keep_schema() -> list[str]:
    return [
        "video_id",
        "allas_file",
        "country",
        "account_username",
        "account_type",
        "political_preference",
        "new_entity",
        "new_theme",
        "researcher_new_persons",
        "researcher_new_themes",
        "researcher_note",
        "source_type",
        "source_recording",
        "sequence_number",
        "recording_date",
        "entities",
        "themes",
    ]


def test_blank_researcher_state_is_not_false():
    assert explicit_bool(pd.NA) is None
    assert explicit_bool("") is None
    assert explicit_bool(False) is False
    assert explicit_bool("FALSE") is False
    assert explicit_bool("TRUE") is True


def test_spreadsheet_letter_mapping_includes_full_ah_at_range():
    assert spreadsheet_column_name(34) == "AH"
    assert spreadsheet_column_name(46) == "AT"
    mapping = resolve_legacy_drop_columns([f"c{i}" for i in range(1, 55)])
    assert set(["Y", "Z", "AA", "AB", "AE", "AH", "AT"]).issubset(mapping)
    assert len(mapping) == 18


def test_mobilebackup_is_rejected():
    with pytest.raises(ValueError, match="pseudo-country"):
        validate_country("MobileBackup")


def test_human_labels_merge_by_canonical_identity():
    row = pd.Series(
        {
            "new_entity": "Sanna Marin, EU",
            "researcher_new_persons": "Marin; European Union",
        }
    )
    values, provenance = merge_human_labels(
        row,
        ("new_entity", "researcher_new_persons"),
        {"Marin": "Sanna Marin", "EU": "European Union"},
    )
    assert values == "Sanna Marin | European Union"
    assert '"source": "researcher_new_persons"' in provenance


def test_cleaner_filters_and_preserves_source():
    source = _source()
    original = source.copy(deep=True)

    export, decisions, recut, summary = clean_dataframe(
        source,
        country="Finland",
        keep_schema=_keep_schema(),
        entity_aliases={"Marin": "Sanna Marin", "EU": "European Union"},
        theme_aliases={"climate change": "Climate"},
    )

    pd.testing.assert_frame_equal(source, original)
    assert export["video_id"].tolist() == ["v1", "v6"]
    assert export.columns[0] == "video_id"
    assert export.loc[export.video_id == "v1", "entities"].item() == "Sanna Marin | European Union"
    assert export.loc[export.video_id == "v1", "themes"].item() == "Climate"
    assert export.loc[export.video_id == "v6", "researcher_note"].item() == ""
    assert "summary_analysis" not in export.columns
    assert "frame_1" not in export.columns

    assert set(recut["video_id"]) == {"v4"}
    by_id = decisions.set_index("row_id")
    assert by_id.loc["v2", "include_in_reprocess"] == False  # noqa: E712
    assert by_id.loc["v3", "include_in_reprocess"] == False  # noqa: E712
    assert by_id.loc["v7", "include_in_reprocess"] == False  # noqa: E712
    assert by_id.loc["v1", "needs_human_review"] == True  # noqa: E712

    assert summary.source_rows == 7
    assert summary.reprocess_rows == 2
    assert summary.recut_rows == 1
    assert summary.excluded_delete == 1
    assert summary.excluded_dubious == 1
    assert summary.excluded_non_synthetic == 1


def test_same_keep_schema_applies_to_other_country():
    source = _source()
    source["country"] = "Poland"
    export, _, _, summary = clean_dataframe(
        source,
        country="Poland",
        keep_schema=_keep_schema(),
    )
    assert summary.country == "Poland"
    assert export.columns.tolist()[0] == "video_id"
