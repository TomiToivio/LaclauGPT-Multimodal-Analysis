"""Regression guards for two preprocess defects found while repairing #128.

Both are production-code bugs, not test-shape issues, and neither was covered
by the existing #128 tests.

1. ``connect_cache()`` opened ``./database/preprocess_v2.db`` without creating
   ``./database``. The module-level ``mkdir`` only runs for whatever CWD was
   current at *import* time, so any caller that changed directory afterwards
   (or an orchestrator importing the stage from elsewhere) failed with an
   opaque ``sqlite3.OperationalError: unable to open database file``.

2. The ``too_short`` guard used a local ``duration < FRAME_TIMESTAMP_SECONDS``
   comparison, while the authoritative ``ep24_video.is_too_short`` rule is
   ``duration <= VIDEO_INITIAL_SKIP_SECONDS``. A clip of exactly 1.0 s
   therefore passed the local guard, failed to yield a frame at t=1.0 s, and
   was recorded as ``preprocess_status="error"`` instead of ``"too_short"`` —
   a silent, misleading failure label.

Synthetic only: no private EP24 data, model download, GPU or CSC access.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import ep24_video as video  # noqa: E402
import roihu_preprocess as preprocess  # noqa: E402


def _write_video(path: Path, *, seconds: float, fps: int = 10) -> None:
    # Codec 0 (rawvideo) is rejected for .mp4 by the bundled ffmpeg, so these
    # fixtures use .avi + FFV1: a codec/container pair that always opens, so the
    # test cannot fail for a reason unrelated to the behaviour under test.
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (32, 32))
    assert writer.isOpened(), "could not open a video writer"
    try:
        for _ in range(max(int(seconds * fps), 1)):
            writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
    finally:
        writer.release()


def test_stage_constant_matches_the_shared_frame_rule():
    """The stage may not carry its own drifting copy of the 1.0 s rule."""
    assert preprocess.FRAME_TIMESTAMP_SECONDS == video.analysis_start_seconds()


def test_connect_cache_creates_its_directory(tmp_path, monkeypatch):
    """A caller whose CWD has no ./database must still be able to open the cache."""
    monkeypatch.chdir(tmp_path)
    assert not Path("database").exists()

    conn = preprocess.connect_cache()
    try:
        assert isinstance(conn, sqlite3.Connection)
        assert Path("database/preprocess_v2.db").exists()
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert "preprocess_cache" in tables
    finally:
        conn.close()


@pytest.mark.parametrize("seconds", [0.5, 1.0])
def test_clips_at_or_below_the_boundary_are_too_short_not_error(
    tmp_path, monkeypatch, seconds
):
    """A 1.0 s clip is too short, not a bogus error."""
    monkeypatch.chdir(tmp_path)
    video_path = tmp_path / "clip.avi"
    _write_video(video_path, seconds=seconds, fps=10)

    ocr_calls: list[str] = []

    class StubOCR:
        engine = "stub"
        model = "stub-v1"

        def read(self, path: str) -> tuple[str, int]:
            ocr_calls.append(path)
            return "should not run", 1

    class StubASR:
        engine = "stub"
        model = "stub-v1"

        def transcribe(self, path, language=None):
            from asr_backend import ASRResult

            return ASRResult("", "", "")

    monkeypatch.setattr(preprocess, "load_ocr_backend", lambda: StubOCR())
    monkeypatch.setattr(preprocess, "load_asr_model", lambda: StubASR())
    monkeypatch.setattr(preprocess, "prepare_analysis_clip", lambda path, output_dir: video_path)
    monkeypatch.setattr(
        preprocess, "local_media_path", lambda row, root=None: video_path
    )

    df = pd.DataFrame(
        [
            {
                "country": "finland",
                "video_id": "short",
                "author_username": "author",
                "allas_filename": "clip.avi",
                "source_type": "tiktok",
            }
        ]
    )

    out = preprocess.preprocess_dataframe(df)
    row = out.iloc[0]
    # this is the regression: it used to be "error"
    assert row["preprocess_status"] == "too_short"
    assert ocr_calls == [], "OCR must not run below the boundary"
    assert not list(Path("Keyframes").rglob("*.jpg"))


def test_just_above_the_boundary_still_produces_one_frame(tmp_path, monkeypatch):
    """The boundary must not over-reach: 1.001 s is analysable."""
    monkeypatch.chdir(tmp_path)
    video_path = tmp_path / "clip.avi"
    _write_video(video_path, seconds=2.0, fps=10)

    calls: list[str] = []

    class StubOCR:
        engine = "stub"
        model = "stub-v1"

        def read(self, path: str) -> tuple[str, int]:
            calls.append(path)
            return "synthetic text", 1

    class StubASR:
        engine = "stub"
        model = "stub-v1"

        def transcribe(self, path, language=None):
            from asr_backend import ASRResult

            return ASRResult("hello", "fi", "hello")

    monkeypatch.setattr(preprocess, "load_ocr_backend", lambda: StubOCR())
    monkeypatch.setattr(preprocess, "load_asr_model", lambda: StubASR())
    monkeypatch.setattr(
        preprocess,
        "prepare_analysis_clip",
        lambda path, output_dir: video_path,
    )
    monkeypatch.setattr(
        preprocess, "local_media_path", lambda row, root=None: video_path
    )

    df = pd.DataFrame(
        [
            {
                "country": "finland",
                "video_id": "ok",
                "author_username": "author",
                "allas_filename": "clip.avi",
                "source_type": "tiktok",
            }
        ]
    )

    out = preprocess.preprocess_dataframe(df)
    row = out.iloc[0]
    assert row["preprocess_status"] == "ok"
    assert row["frame_timestamp_seconds"] == "1.0"
    assert len(calls) == 1, f"expected exactly one OCR call, got {len(calls)}"
    files = [p for p in Path("Keyframes").rglob("*") if p.is_file()]
    assert len(files) == 1, f"expected exactly one saved frame, got {files}"
