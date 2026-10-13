"""Executable regression proof for the active single-frame preprocess contract (#128).

Synthetic/mocked only: no private EP24 data, model download, GPU, or CSC access.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import roihu_preprocess as preprocess


def test_active_contract_is_one_original_video_frame_at_exactly_one_second():
    assert preprocess.FRAME_TIMESTAMP_SECONDS == 1.0
    assert "frame_file" in preprocess.PREPROCESS_COLUMNS
    assert "ocr_1" in preprocess.PREPROCESS_COLUMNS
    assert "frame_files" not in preprocess.PREPROCESS_COLUMNS
    for index in range(2, 7):
        assert f"ocr_{index}" not in preprocess.PREPROCESS_COLUMNS


def test_save_single_keyframe_reads_and_writes_exactly_once(monkeypatch, tmp_path: Path):
    events = []

    class Capture:
        def __init__(self, path):
            events.append(("open", path))

        def isOpened(self):
            return True

        def set(self, prop, value):
            events.append(("set", prop, value))

        def read(self):
            events.append(("read",))
            return True, object()

        def release(self):
            events.append(("release",))

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(preprocess.cv2, "VideoCapture", Capture)
    monkeypatch.setattr(
        preprocess.cv2,
        "imwrite",
        lambda path, image: events.append(("write", path)) or True,
    )

    result = preprocess.save_single_keyframe(
        "source.mp4",
        video_id="synthetic-video",
        author_username="synthetic-author",
        source_type="TikTok",
    )

    assert [e for e in events if e[0] == "set"] == [
        ("set", preprocess.cv2.CAP_PROP_POS_MSEC, 1000.0)
    ]
    assert len([e for e in events if e[0] == "read"]) == 1
    assert len([e for e in events if e[0] == "write"]) == 1
    assert result.endswith(
        "Keyframes/tiktok/synthetic-author/synthetic-video/frame_t1.0s.jpg"
    )


def test_save_single_keyframe_fails_instead_of_sampling_another_timestamp(
    monkeypatch, tmp_path: Path
):
    class Capture:
        def __init__(self, path):
            pass

        def isOpened(self):
            return True

        def set(self, prop, value):
            assert prop == preprocess.cv2.CAP_PROP_POS_MSEC
            assert value == 1000.0

        def read(self):
            return False, None

        def release(self):
            pass

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(preprocess.cv2, "VideoCapture", Capture)

    with pytest.raises(ValueError, match="t=1.0s"):
        preprocess.save_single_keyframe(
            "short-or-invalid.mp4",
            video_id="synthetic-video",
            author_username="synthetic-author",
            source_type="Instagram",
        )


def test_preprocess_source_calls_ocr_once_on_the_single_frame():
    source = (Path(__file__).resolve().parents[1] / "roihu_preprocess.py").read_text(
        encoding="utf-8"
    )
    # The active OCR adapter exposes one read() call and receives frame_file,
    # not a list/loop of sampled frames.
    assert source.count("ocr.read(frame_file)") == 1
    assert "reader.readtext(frame_files[0])" not in source
    assert "for i, frame_file in enumerate(frame_files)" not in source


def test_active_source_has_no_six_frame_or_whisper_output_schema():
    source = (Path(__file__).resolve().parents[1] / "roihu_preprocess.py").read_text(
        encoding="utf-8"
    )
    for old in (
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
        assert old not in source


def test_country_order_starts_with_smoke_countries_and_is_fully_pinned():
    assert preprocess.COUNTRY_ORDER == (
        "finland",
        "poland",
        "portugal",
        "germany",
        "spain",
        "hungary",
        "croatia",
        "france",
        "bulgaria",
        "sweden",
    )
