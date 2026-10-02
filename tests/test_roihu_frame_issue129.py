from __future__ import annotations

import importlib
import sys
import types

import pandas as pd


def _load_frame_module(monkeypatch):
    # Keep this contract test independent of GPU/Ollama/OpenCV availability in CI.
    fake_cv2 = types.SimpleNamespace(imread=lambda _path: None)
    fake_ollama = types.SimpleNamespace(chat=lambda **_kwargs: {"message": {"content": "unused"}})
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setitem(sys.modules, "ollama", fake_ollama)
    sys.modules.pop("roihu_frame", None)
    return importlib.import_module("roihu_frame")


def test_platform_detection_handles_tiktok_and_instagram(monkeypatch):
    frame = _load_frame_module(monkeypatch)
    assert frame._detect_platform(pd.Series({"source_type": "TikTok"})) == "tiktok"
    assert frame._detect_platform(pd.Series({"source_type": "Instagram"})) == "instagram"


def test_step2_preserves_all_fields_and_passes_full_context(tmp_path, monkeypatch):
    frame = _load_frame_module(monkeypatch)

    input_csv = tmp_path / "step1.csv"
    output_csv = tmp_path / "step2.csv"
    rows = [
        {
            "video_id": "tt-1",
            "allas_filename": "Scraper/TikTok/Videos/finland/a/tt-1.mp4",
            "author_username": "alice",
            "source_type": "TikTok",
            "country": "Finland",
            "frame_file": str(tmp_path / "tt.jpg"),
            "frame_timestamp_seconds": "1.0",
            "ocr_1": "VISIBLE TIKTOK TEXT",
            "asr_transcript": "spoken TikTok words",
            "asr_translated": "translated TikTok words",
            "researcher_note": "human TikTok note",
            "arbitrary_future_field": "must survive",
        },
        {
            "video_id": "ig-1",
            "allas_filename": "Scraper/Instagram/Videos/finland/b/ig-1.mp4",
            "author_username": "bob",
            "source_type": "Instagram",
            "country": "Finland",
            "frame_file": str(tmp_path / "ig.jpg"),
            "frame_timestamp_seconds": "1.0",
            "ocr_1": "VISIBLE INSTAGRAM TEXT",
            "asr_transcript": "spoken Instagram words",
            "asr_translated": "translated Instagram words",
            "researcher_note": "human Instagram note",
            "arbitrary_future_field": "also survives",
        },
    ]
    pd.DataFrame(rows).to_csv(input_csv, index=False)

    captured_contexts = []

    monkeypatch.setenv("LACLAUGPT_INPUT_CSV", str(input_csv))
    monkeypatch.setenv("LACLAUGPT_OUTPUT_CSV", str(output_csv))
    monkeypatch.setenv("LACLAUGPT_FRAME_SQLITE", str(tmp_path / "frame.db"))
    monkeypatch.setattr(frame, "DB_PATH", tmp_path / "frame.db")
    monkeypatch.setattr(
        frame,
        "_validate_keyframe",
        lambda path, timestamp: (tmp_path / path, (1920, 1080)),
    )

    def fake_analysis(_frame_file, row_context=""):
        captured_contexts.append(row_context)
        return "synthetic frame analysis"

    monkeypatch.setattr(frame, "get_analysis", fake_analysis)
    frame.analyze_videos(None)

    out = pd.read_csv(output_csv, dtype=str, keep_default_na=False)
    assert list(out["video_id"]) == ["tt-1", "ig-1"]
    for column in rows[0]:
        assert column in out.columns
        assert list(out[column]) == [str(row[column]) for row in rows]

    assert list(out["frame_analysis_timestamp_seconds"]) == ["1.0", "1.0"]
    assert list(out["frame_analysis_status"]) == ["ok", "ok"]
    assert all("synthetic frame analysis" in value for value in out["frame_analysis_1"])

    combined = "\n".join(captured_contexts)
    for expected in (
        "spoken TikTok words",
        "spoken Instagram words",
        "VISIBLE TIKTOK TEXT",
        "VISIBLE INSTAGRAM TEXT",
        "translated TikTok words",
        "translated Instagram words",
        "human TikTok note",
        "human Instagram note",
        "arbitrary_future_field",
        "must survive",
        "also survives",
    ):
        assert expected in combined


def test_step2_rejects_non_one_second_keyframe(monkeypatch, tmp_path):
    frame = _load_frame_module(monkeypatch)
    fake = tmp_path / "frame.jpg"
    fake.write_bytes(b"not-an-image")
    try:
        frame._validate_keyframe(str(fake), 0.5)
    except ValueError as exc:
        assert "exactly 1.0s" in str(exc)
    else:
        raise AssertionError("Step 2 accepted a non-1.0s keyframe")


def test_researcher_system_prompt_content_is_preserved():
    source = open("roihu_frame.py", encoding="utf-8").read()
    assert "multimodal social-semiotic pre-analysis" in source
    assert "Halliday/SFL, Kress & van Leeuwen" in source
    assert "These videos are from TikTok and Instagram feeds" in source
