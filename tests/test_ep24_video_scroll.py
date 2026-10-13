"""Contract tests for EP24 initial-scroll handling (issue #22)."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ep24_video as video


def test_canonical_skip_is_one_second():
    assert video.VIDEO_INITIAL_SKIP_SECONDS == 1.0
    assert video.analysis_start_seconds() == 1.0


def test_frame_sampling_is_exactly_one_frame_at_analysis_boundary():
    assert video.analysis_frame_times(100) == [1.0]
    assert video.analysis_frame_times(180) == [1.0]


def test_short_clips_are_rejected_gracefully():
    assert video.is_too_short(0.5)
    assert video.is_too_short(1.0)
    assert not video.is_too_short(1.001)
    assert video.analysis_frame_times(1.0) == []


def test_asr_trim_command_skips_first_second():
    command = video.build_trim_command("source.mp4", "analysis.mp4")
    assert command[command.index("-ss") + 1] == "1"
    assert command[-1] == "analysis.mp4"


def test_analysis_clip_is_deterministic_and_non_destructive():
    path = video.analysis_clip_path("research/source.mp4", "derived")
    assert path == Path("derived/source.analysis-from-1s.mp4")
    assert str(path) != "research/source.mp4"


def test_scroll_schema_false_has_empty_timestamps():
    assert video.normalize_scroll_metadata(False, [8.4]) == {
        "SCROLL": False,
        "SCROLL_SECONDS": [],
    }
    parsed = video.parse_scroll_metadata(
        'description\\n{"SCROLL": false, "SCROLL_SECONDS": []}'
    )
    assert parsed == {"SCROLL": False, "SCROLL_SECONDS": []}


def test_known_initial_scroll_never_triggers_scroll():
    parsed = video.normalize_scroll_metadata(True, [0.2, 1.0])
    assert parsed == {"SCROLL": False, "SCROLL_SECONDS": []}


def test_additional_scroll_triggers_needs_resplit():
    parsed = video.parse_scroll_metadata(
        'text\\n{"SCROLL": true, "SCROLL_SECONDS": [8.4, 17.9]}'
    )
    assert parsed == {"SCROLL": True, "SCROLL_SECONDS": [8.4, 17.9]}
    assert video.needs_resplit(parsed)


def test_resplit_plan_is_deterministic_and_depth_limited():
    expected = [(0.0, 8.4), (8.4, 17.9), (17.9, 30.0)]
    assert video.split_plan(30, [17.9, 8.4, 8.4]) == expected
    assert video.split_plan(30, [8.4], resplit_depth=video.MAX_RESPLIT_DEPTH) == []
    assert not video.needs_resplit(
        {"SCROLL": True, "SCROLL_SECONDS": [8.4]},
        resplit_depth=video.MAX_RESPLIT_DEPTH,
    )


def test_derived_ids_preserve_parent_provenance():
    child = video.derived_segment_id("alice/123", 2, 8.4, 17.9)
    assert child.startswith("alice_123__resplit-02__")
    assert "8.400-17.900s" in child


def test_preprocess_uses_exact_one_second_frame_and_generic_status():
    source = Path("roihu_preprocess.py").read_text(encoding="utf-8")
    assert "FRAME_TIMESTAMP_SECONDS = 1.0" in source
    assert "frame_file" in source
    assert "frame_timestamp_seconds" in source
    assert "preprocess_status" in source
    assert "too_short" in source
    # The repository-wide EP24 invariant excludes the known 0-1 s scroll artifact
    # from ASR while keeping the t=1.0 s source frame for OCR.
    assert "prepare_analysis_clip(local_path" in source
    assert "asr.transcribe(str(analysis_clip)" in source
    assert "asr.transcribe(local_path" not in source


def test_vllm_prompt_and_output_expose_scroll_contract():
    source = Path("experiments/vllm_video_test.py").read_text(encoding="utf-8")
    assert "prepare_analysis_clip(" in source
    assert "local_path" in source
    assert '"SCROLL"' in source
    assert '"SCROLL_SECONDS"' in source
    assert '"needs_resplit"' in source
    assert "known initial" in source
    assert "feed-scroll" in source


def test_resplit_rows_copy_source_url_and_legacy_metadata():
    source = {
        "source_url": "https://example.invalid/video/123",
        "whisperResult": "legacy transcript",
        "videoId": "123",
    }
    children = video.derived_rows(
        source,
        30,
        [8.4],
        parent_id="alice/123",
    )
    assert len(children) == 2
    assert all(row["source_url"] == source["source_url"] for row in children)
    assert all(row["whisperResult"] == "legacy transcript" for row in children)
    assert all(row["resplit_parent_id"] == "alice/123" for row in children)


def test_active_preprocess_does_not_regenerate_superseded_six_frame_whisper_schema():
    source = Path("roihu_preprocess.py").read_text(encoding="utf-8")
    assert "ocr_1" in source
    for field in (
        "frame_files",
        "ocr_2",
        "ocr_3",
        "ocr_4",
        "ocr_5",
        "ocr_6",
        "whisperResult",
        "whisper_transcript",
        "whisper_language",
        "whisper_translated",
    ):
        assert field not in source


def test_numbered_steps_document_cumulative_one_frame_then_video_contract():
    step1 = Path("step_1_roihu_preprocess.py").read_text(encoding="utf-8")
    step2 = Path("step_2_roihu_frame.py").read_text(encoding="utf-8")
    step3 = Path("step_3_roihu_video.py").read_text(encoding="utf-8")
    assert "exactly one original-video keyframe" in step1
    assert "t=1.0s" in step2
    assert "complete Step 2 dataframe" in step3
    assert "LACLAUGPT_INPUT_CSV" in step3
    assert "LACLAUGPT_OUTPUT_CSV" in step3


def test_step2_is_strictly_one_frame_at_original_t1():
    source = Path("roihu_frame.py").read_text(encoding="utf-8")
    assert "frame_file = str(row.get('frame_file', '')).strip()" in source
    assert "for i, frame_file in enumerate" not in source
    assert "frame_analysis_timestamp_seconds" in source
    assert "VIDEO_INITIAL_SKIP_SECONDS" in source
    assert "frame_timestamp_seconds" in source


def test_readme_documents_frame_video_asr_division_of_labor():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "exactly one keyframe extracted at original source t=1.0s" in readme
    assert "native whole-video Qwen3-VL/vLLM analysis" in readme
    assert "backend-neutral ASR transcript/translation" in readme
    assert "activate_vllm_video.sh && roihu_vllm_submit" in readme
