from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import roihu_preprocess as rp


def test_active_schema_has_exactly_one_frame_and_one_ocr():
    assert "frame_file" in rp.PREPROCESS_COLUMNS
    assert "ocr_1" in rp.PREPROCESS_COLUMNS
    assert "frame_files" not in rp.PREPROCESS_COLUMNS
    for i in range(2, 7):
        assert f"ocr_{i}" not in rp.PREPROCESS_COLUMNS
    for legacy in ("whisperResult", "whisper_transcript", "whisper_language", "whisper_translated"):
        assert legacy not in rp.PREPROCESS_COLUMNS


def test_country_order_is_explicit_and_deterministic():
    assert rp.COUNTRY_ORDER == (
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


def test_lfs_pointer_is_rejected(tmp_path):
    p = tmp_path / "ep24_finland.csv"
    p.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:" + "a" * 64 + "\nsize 10\n",
        encoding="utf-8",
    )
    assert rp.is_lfs_pointer(p)
    with pytest.raises(RuntimeError, match="Git LFS pointer"):
        rp.read_materialized_csv(p)


def test_real_csv_schema_is_preserved_by_reader(tmp_path):
    p = tmp_path / "ep24_finland.csv"
    columns = [
        "video_id", "country", "author_username", "account_type", "source_type",
        "source_recording", "sequence_number", "political_preference", "allas_filename",
        "entities", "themes", "video_duration", "researcher_note",
    ]
    pd.DataFrame([{c: f"x-{c}" for c in columns}]).to_csv(p, index=False)
    df = rp.read_materialized_csv(p)
    assert list(df.columns) == columns
    for legacy in ("new_entity", "researcher_new_persons", "new_theme", "researcher_new_themes"):
        assert legacy not in df.columns


def test_single_keyframe_uses_exact_original_one_second(monkeypatch, tmp_path):
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
    monkeypatch.setattr(rp.cv2, "VideoCapture", Capture)
    monkeypatch.setattr(rp.cv2, "imwrite", lambda path, image: events.append(("write", path)) or True)

    result = rp.save_single_keyframe(
        "video.mp4",
        video_id="v1",
        author_username="alice",
        source_type="instagram",
    )

    sets = [e for e in events if e[0] == "set"]
    reads = [e for e in events if e[0] == "read"]
    writes = [e for e in events if e[0] == "write"]
    assert sets == [("set", rp.cv2.CAP_PROP_POS_MSEC, 1000.0)]
    assert len(reads) == 1
    assert len(writes) == 1
    assert result.endswith("Keyframes/instagram/alice/v1/frame_t1.0s.jpg")


def test_backup_and_write_always_makes_versioned_copy(tmp_path):
    out = tmp_path / "step_01_preprocess.csv"
    df = pd.DataFrame([{"video_id": "v1", "ocr_1": "text"}])
    backup = rp.backup_and_write(df, out)
    assert out.exists()
    assert backup.exists()
    assert pd.read_csv(backup).to_dict("records") == pd.read_csv(out).to_dict("records")


def test_source_contains_no_active_six_ocr_or_whisper_schema():
    text = (Path(__file__).resolve().parents[1] / "roihu_preprocess.py").read_text(encoding="utf-8")
    for name in ("ocr_2", "ocr_3", "ocr_4", "ocr_5", "ocr_6", "whisperResult", "whisper_transcript", "whisper_language", "whisper_translated"):
        assert name not in text
