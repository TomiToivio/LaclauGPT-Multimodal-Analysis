from pathlib import Path

import pandas as pd
import pytest

from ep24_pipeline import (
    assert_source_metadata_preserved,
    load_cumulative_csv,
    metadata_context,
    write_cumulative_csv,
)
from ep24_schema import EP24_REPROCESS_COLUMNS

ROW = {
    "country": "Finland",
    "author_username": "researcher-feed-a",
    "account_type": "research profile",
    "source_type": "TikTok",
    "source_recording": "session-01.mp4",
    "video_id": "00012345678901234567890",
    "sequence_number": "7",
    "political_preference": "researcher annotation",
    "allas_filename": "ep24/finland/session-01/clip-007.mp4",
    "entities": '["Person A", "Person B"]',
    "themes": '["Theme A", "Theme B"]',
    "video_duration": "12.25",
    "researcher_note": "human note",
}


def test_canonical_schema_is_exactly_the_researcher_feed_contract():
    assert tuple(ROW) == EP24_REPROCESS_COLUMNS


def test_load_preserves_video_id_byte_for_byte_and_all_source_fields(tmp_path):
    path = tmp_path / "ep24_finland.csv"
    pd.DataFrame([ROW]).to_csv(path, index=False)
    df = load_cumulative_csv(path)
    assert df.loc[0, "video_id"] == ROW["video_id"]
    assert df.loc[0, "allas_filename"] == ROW["allas_filename"]
    assert df.loc[0, "source_recording"] == ROW["source_recording"]
    assert df.loc[0, "researcher_note"] == ROW["researcher_note"]


def test_cumulative_write_preserves_source_and_upstream_fields(tmp_path):
    before = pd.DataFrame([ROW])
    after = before.copy()
    after["asr_transcript"] = ["hei"]
    after["frame_analysis_1"] = ["visible text"]
    after["summary_analysis"] = ["markdown"]
    after["dna_statements_json"] = ['[]']
    path = tmp_path / "stage.csv"
    write_cumulative_csv(before, after, path)
    reloaded = pd.read_csv(path, dtype=str, keep_default_na=False)
    assert list(reloaded.columns[:len(EP24_REPROCESS_COLUMNS)]) == list(EP24_REPROCESS_COLUMNS)
    for column, expected in ROW.items():
        assert reloaded.loc[0, column] == expected
    # Every field the stage added must survive the round-trip. This list must
    # name the columns the test actually sets above: it previously asserted
    # `whisper_transcript`, which the stage no longer produces after the
    # generic ASR rename (#128), so the test failed on an assertion about a
    # column it never created.
    for column in ("asr_transcript", "frame_analysis_1", "summary_analysis", "dna_statements_json"):
        assert column in reloaded.columns


def test_source_metadata_mutation_is_rejected():
    before = pd.DataFrame([ROW])
    after = before.copy()
    after.loc[0, "source_recording"] = "different.mp4"
    with pytest.raises(AssertionError, match="source_recording"):
        assert_source_metadata_preserved(before, after)


def test_prompt_context_separates_provenance_classes():
    row = pd.Series({
        **ROW,
        "whisper_transcript": "model transcript",
        "summary_analysis": "model summary",
        "codebook_matches": "derived codebook context",
    })
    context = metadata_context(row)
    assert "EP24 SOURCE METADATA" in context
    assert "RESEARCHER ANNOTATION" in context
    assert "UPSTREAM MODEL / ENRICHMENT CONTEXT" in context
    assert "- researcher_note: human note" in context
    assert "- summary_analysis: model summary" in context


def test_active_numbered_pipeline_does_not_require_scraper_identity_names():
    root = Path(__file__).resolve().parents[1]
    active = [
        "roihu_preprocess.py",
        "roihu_frame.py",
        "roihu_summary.py",
        "roihu_postprocess.py",
        "roihu_populism.py",
        "step_7_roihu_discourse_network_analysis.py",
        "step_8_roihu_social_network_analysis.py",
    ]
    forbidden_direct = ("row['authorUniqueId']", 'row["authorUniqueId"]',
                        "row['videoId']", 'row["videoId"]',
                        "row['scrapedCountry']", 'row["scrapedCountry"]')
    for relative in active:
        text = (root / relative).read_text(encoding="utf-8")
        for token in forbidden_direct:
            assert token not in text, f"{relative} still directly requires {token}"
