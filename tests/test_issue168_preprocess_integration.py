"""Integration-level guards for Step 1 shared infrastructure (issue #168)."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import roihu_preprocess as rp
from asr_backend import ASRResult
from ep24_rag import rag_text


def canonical_row(media: str, *, source_type: str = "tiktok") -> dict[str, str]:
    return {
        "country": "finland",
        "video_id": "v1",
        "author_username": "alice",
        "allas_filename": media,
        "source_type": source_type,
        "entities": '["Petteri Orpo"]',
        "themes": '["economy"]',
        "researcher_note": "human note",
    }


def test_rag_prefers_canonical_asr_ocr_and_video_fields():
    row = pd.Series({
        "asr_translated": "canonical translation",
        "whisper_translated": "legacy translation",
        "asr_transcript": "canonical transcript",
        "whisper_transcript": "legacy transcript",
        "ocr_1": "visible words",
        "vllm_video_analysis": "canonical video",
        "video_analysis": "legacy video",
    })
    text = rag_text(row)
    assert "canonical translation" in text
    assert "canonical transcript" in text
    assert "visible words" in text
    assert "canonical video" in text
    assert "legacy translation" not in text
    assert "legacy transcript" not in text
    assert "legacy video" not in text


class FakeStorage:
    def __init__(self, country: str):
        self.config = SimpleNamespace(dataset="ep24", country=country)
        self.patched = []
        self.upserts: dict[str, list[dict]] = {}

    def patch_documents(self, purpose, docs):
        docs = [dict(doc) for doc in docs]
        self.patched.extend((purpose, doc) for doc in docs)
        return len(docs)

    def upsert_documents(self, purpose, docs, **kwargs):
        docs = [dict(doc) for doc in docs]
        self.upserts.setdefault(purpose, []).extend(docs)
        return len(docs)


def test_shared_persistence_patches_analysis_and_seeds_memory_and_rag(monkeypatch):
    storage = FakeStorage("finland")

    @contextmanager
    def fake_country_storage(country):
        assert country == "finland"
        yield storage

    monkeypatch.setenv("LACLAUGPT_MONGO_ENABLED", "1")
    monkeypatch.setenv("LACLAUGPT_STAGE_RUN_ID", "test-run")
    monkeypatch.setattr(rp, "country_storage", fake_country_storage)

    frame = pd.DataFrame([{
        **canonical_row("clip.mp4"),
        "ocr_1": "OCR evidence",
        "asr_transcript": "Finnish evidence",
        "asr_translated": "English evidence",
        "preprocess_status": "ok",
    }])
    before_entities = frame.loc[0, "entities"]
    before_themes = frame.loc[0, "themes"]

    counts = rp.persist_shared_state(frame, source_csv=Path("input.csv"))

    assert counts["analysis"] == 1
    assert counts["memory"] == 2
    assert counts["rag"] == 1
    assert frame.loc[0, "entities"] == before_entities
    assert frame.loc[0, "themes"] == before_themes
    purpose, analysis_doc = storage.patched[0]
    assert purpose == "analysis"
    assert analysis_doc["asr_transcript"] == "Finnish evidence"
    assert analysis_doc["_provenance"]["pipeline_stage"] == "step_01_preprocess"
    assert {doc["evidence_role"] for doc in storage.upserts["memory"]} == {
        "normalization_context_not_source_evidence"
    }
    rag_doc = storage.upserts["rag"][0]
    assert "OCR evidence" in rag_doc["text"]
    assert "Finnish evidence" in rag_doc["text"]
    assert rag_doc["evidence_role"] == "prior_analysis_context_not_source_evidence"


def test_preprocess_preserves_row_and_uses_redis_statuses(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"synthetic-media")

    events: list[tuple[str, str]] = []

    class Coordinator:
        def __init__(self, country, step):
            assert country == "finland"
            assert step == 1

        @contextmanager
        def lock(self, record_id, timeout=3600):
            events.append(("lock", record_id))
            yield True

        def mark(self, record_id, status):
            events.append((status, record_id))

    class OCR:
        engine = "stub-ocr"
        model = "stub-ocr-v1"

        def read(self, frame_file):
            return "visible text", 1

    class ASR:
        engine = "stub-asr"
        model = "stub-asr-v1"

        def transcribe(self, clip, language=None):
            return ASRResult("spoken words", "fi", "spoken words")

    monkeypatch.setattr(rp, "RedisCoordinator", Coordinator)
    monkeypatch.setattr(rp, "load_ocr_backend", lambda: OCR())
    monkeypatch.setattr(rp, "load_asr_model", lambda: ASR())
    monkeypatch.setattr(rp, "describe_ocr_backend", lambda: {"engine": "stub-ocr", "model": "stub-ocr-v1"})
    monkeypatch.setattr(rp, "describe_backend", lambda: {"engine": "stub-asr", "model": "stub-asr-v1"})
    monkeypatch.setattr(rp, "get_video_duration", lambda path: 2.0)
    monkeypatch.setattr(rp, "save_single_keyframe", lambda *args, **kwargs: "frame.jpg")
    monkeypatch.setattr(rp, "prepare_analysis_clip", lambda path, output_dir: media)
    monkeypatch.setattr(rp, "local_media_path", lambda row, root=None: media)

    source = pd.DataFrame([canonical_row("clip.mp4")])
    out = rp.preprocess_dataframe(source)

    for column in source.columns:
        assert out.loc[0, column] == source.loc[0, column]
    assert out.loc[0, "preprocess_status"] == "ok"
    assert out.loc[0, "ocr_1"] == "visible text"
    assert out.loc[0, "asr_transcript"] == "spoken words"
    statuses = [event[0] for event in events]
    assert "lock" in statuses
    assert "running" in statuses
    assert "completed" in statuses


def test_tiktok_and_instagram_frame_paths_remain_distinct(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    class Capture:
        def __init__(self, path):
            pass
        def isOpened(self):
            return True
        def set(self, prop, value):
            pass
        def read(self):
            return True, object()
        def release(self):
            pass

    monkeypatch.setattr(rp.cv2, "VideoCapture", Capture)
    monkeypatch.setattr(rp.cv2, "imwrite", lambda path, image: True)

    tiktok = rp.save_single_keyframe(
        "x.mp4", video_id="v1", author_username="alice", source_type="TikTok"
    )
    instagram = rp.save_single_keyframe(
        "x.mp4", video_id="v2", author_username="alice", source_type="Instagram"
    )
    assert "Keyframes/tiktok/" in tiktok
    assert "Keyframes/instagram/" in instagram
