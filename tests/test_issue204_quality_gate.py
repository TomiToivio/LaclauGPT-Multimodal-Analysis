from pathlib import Path

import pandas as pd

from laclaugpt_quality import (
    merge_status,
    partition_by_status,
    quality_decision_from_analysis,
    reprocess_output_path,
    summary_quality_decision,
)


def test_quality_precedence_delete_cannot_be_downgraded():
    status, reason = merge_status(
        "DELETE", "OK", current_reason="garbage source", new_reason="later model says ok"
    )
    assert status == "DELETE"
    assert reason == "garbage source"


def test_frame_extraction_failure_is_reprocess():
    status, reason = quality_decision_from_analysis(
        "", failure=True, failure_reason="decoder failed"
    )
    assert status == "REPROCESS"
    assert "decoder failed" in reason


def test_obvious_garbage_is_delete():
    status, _ = quality_decision_from_analysis(
        "Video problems: meaningless content, garbage. DELETE."
    )
    assert status == "DELETE"


def test_empty_transcript_valid_visuals_is_reprocess():
    status, reason = summary_quality_decision(
        transcript="",
        frame_analysis="A person speaks beside campaign text and a flag. OK.",
        video_analysis="Valid rally scene with people and captions. OK.",
    )
    assert status == "REPROCESS"
    assert "transcription likely failed" in reason


def test_empty_transcript_meaningless_multimodal_is_delete():
    status, reason = summary_quality_decision(
        transcript="",
        frame_analysis="Blank frame, no meaningful content. DELETE.",
        video_analysis="No useful content, garbage. DELETE.",
    )
    assert status == "DELETE"
    assert "empty transcript" in reason


def test_meaningful_multimodal_with_transcript_is_ok():
    status, _ = summary_quality_decision(
        transcript="Vote in the European elections",
        frame_analysis="Person and campaign poster. OK.",
        video_analysis="Campaign event with speech. OK.",
    )
    assert status == "OK"


def test_reprocess_filename_is_sibling_csv():
    assert reprocess_output_path(Path("/tmp/step_05_postprocess.csv")).name == "step_05_reprocess_postprocess.csv"



def test_step5_partition_routes_only_ok_to_normal_output():
    frame = pd.DataFrame(
        [
            {"video_id": "ok", "processing_status": "OK"},
            {"video_id": "retry", "processing_status": "REPROCESS"},
            {"video_id": "trash", "processing_status": "DELETE"},
        ]
    )
    ok, retry, delete = partition_by_status(frame)
    assert ok["video_id"].tolist() == ["ok"]
    assert retry["video_id"].tolist() == ["retry"]
    assert delete["video_id"].tolist() == ["trash"]


def test_step5_partition_preserves_schema_in_all_outputs():
    frame = pd.DataFrame(
        [
            {
                "video_id": "ok",
                "country": "Finland",
                "processing_status": "OK",
                "processing_status_reason": "usable",
                "frame_quality_status": "OK",
                "video_quality_status": "OK",
                "summary_quality_status": "OK",
            },
            {
                "video_id": "retry",
                "country": "Poland",
                "processing_status": "REPROCESS",
                "processing_status_reason": "transcription likely failed",
                "frame_quality_status": "OK",
                "video_quality_status": "OK",
                "summary_quality_status": "REPROCESS",
            },
        ]
    )
    ok, retry, delete = partition_by_status(frame)
    assert list(ok.columns) == list(frame.columns)
    assert list(retry.columns) == list(frame.columns)
    assert list(delete.columns) == list(frame.columns)
