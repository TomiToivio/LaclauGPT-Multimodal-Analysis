from __future__ import annotations

import sqlite3

import pandas as pd

import roihu_summary as summary


def test_evidence_blocks_are_distinct_and_metadata_keeps_other_upstream_fields():
    row = pd.Series(
        {
            "researcher_note": "human note",
            "frame_analysis_1": "frame evidence",
            "ocr_1": "VISIBLE TEXT",
            "asr_transcript": "spoken words",
            "vllm_video_analysis": "whole video evidence",
            "vllm_video_structured_json": '{"scene":"rally"}',
            "vllm_video_status": "ok",
        }
    )
    metadata, transcript, frame, video = summary._evidence_from_row(row)
    assert "human note" in metadata
    assert "vllm_video_structured_json" in metadata
    assert "vllm_video_status" in metadata
    assert "frame_analysis_1" not in metadata
    assert "vllm_video_analysis" not in metadata
    assert transcript == "spoken words"
    assert "frame evidence" in frame
    assert "VISIBLE TEXT" in frame
    assert video == "whole video evidence"

    prompt = summary.get_llama_summary_user_prompt(metadata, transcript, frame, video)
    assert "whole video evidence" in prompt
    assert "frame evidence" in prompt
    assert "spoken words" in prompt


def test_prompt_hash_invalidates_on_model_or_context_change():
    a = summary._prompt_sha256("system", "user", "model-a")
    b = summary._prompt_sha256("system", "user changed", "model-a")
    c = summary._prompt_sha256("system", "user", "model-b")
    assert len({a, b, c}) == 3


def test_summary_cache_is_model_and_context_sensitive(tmp_path, monkeypatch):
    monkeypatch.setattr(summary, "DB_PATH", tmp_path / "summary.sqlite")
    conn = summary._open_cache()
    try:
        summary._cache_store(
            conn,
            source_id="FI|1",
            model="model-a",
            context_sha256="ctx-a",
            summary_analysis="cached",
        )
        assert summary._cache_lookup(conn, "FI|1", "model-a", "ctx-a") == "cached"
        assert summary._cache_lookup(conn, "FI|1", "model-b", "ctx-a") is None
        assert summary._cache_lookup(conn, "FI|1", "model-a", "ctx-b") is None
    finally:
        conn.close()


def test_step4_preserves_every_incoming_field_and_needs_no_legacy_whisper(
    tmp_path, monkeypatch
):
    source = tmp_path / "input.csv"
    output = tmp_path / "output.csv"
    incoming = pd.DataFrame(
        [
            {
                "country": "FI",
                "author_username": "research-account",
                "video_id": "video-1",
                "allas_filename": "TikTok/video-1.mp4",
                "researcher_note": "keep me exactly",
                "entities": "seed entity",
                "themes": "seed theme",
                "frame_analysis_1": "frame description",
                "ocr_1": "OCR words",
                "asr_transcript": "transcript",
                "vllm_video_analysis": "video description",
                "vllm_video_structured_json": '{"participants":[]}',
                "vllm_video_status": "ok",
                "arbitrary_future_field": "must survive",
            }
        ]
    )
    incoming.to_csv(source, index=False)

    monkeypatch.setenv("LACLAUGPT_INPUT_CSV", str(source))
    monkeypatch.setenv("LACLAUGPT_OUTPUT_CSV", str(output))
    monkeypatch.setenv("LACLAUGPT_MONGO_ENABLED", "0")
    monkeypatch.setenv("LACLAUGPT_COUNTRY", "FI")
    monkeypatch.setattr(summary, "DB_PATH", tmp_path / "cache.sqlite")
    monkeypatch.setattr(
        summary,
        "get_llama_summary_response",
        lambda system_prompt, user_prompt, model=None: "synthetic summary",
    )

    summary.analyze_videos(None)

    result = pd.read_csv(output, dtype=str, keep_default_na=False)
    assert len(result) == 1
    for column in incoming.columns:
        assert result.loc[0, column] == str(incoming.loc[0, column]), column
    assert result.loc[0, "summary_analysis"] == "synthetic summary"
    assert result.loc[0, "summary_summary_md"] == "synthetic summary"
    assert "researcher_note" in result.loc[0, "metadata"]
    assert "whisperResult" not in incoming.columns


def test_mongo_patch_preserves_complete_cumulative_row():
    captured = {}

    class FakeStorage:
        def patch_documents(self, purpose, documents):
            captured["purpose"] = purpose
            captured["documents"] = list(documents)
            return 1

    row = pd.Series(
        {
            "_storage_id": "FI|video-1",
            "metadata": "meta",
            "summary_analysis": "summary",
            "summary_summary_md": "summary",
            "upstream_field": "must survive in Mongo",
            "entities": "Petteri Orpo",
            "themes": "EU politics",
        }
    )
    count = summary._persist_mongo_row(
        FakeStorage(),
        row,
        source_id="FI|video-1",
        model="qwen3.8:27b",
        context_sha256="abc",
    )
    assert count == 1
    assert captured["purpose"] == "dataframe"
    doc = captured["documents"][0]
    assert doc["_storage_id"] == "FI|video-1"
    assert doc["upstream_field"] == "must survive in Mongo"
    assert doc["entities"] == "Petteri Orpo"
    assert doc["themes"] == "EU politics"
    assert doc["step4_summary_provenance"]["context_sha256"] == "abc"


def test_mongo_resume_requires_matching_model_and_context():
    class FakeStorage:
        def __init__(self, document):
            self.document = document

        def find(self, purpose, query, limit=0):
            assert purpose == "dataframe"
            assert query == {"_storage_id": "FI|video-1"}
            return [self.document]

    doc = {
        "_storage_id": "FI|video-1",
        "summary_analysis": "durable summary",
        "step4_summary_provenance": {
            "model": "model-a",
            "context_sha256": "ctx-a",
        },
    }
    storage = FakeStorage(doc)
    assert summary._mongo_resume_summary(
        storage, "FI|video-1", model="model-a", context_sha256="ctx-a"
    ) == "durable summary"
    assert summary._mongo_resume_summary(
        storage, "FI|video-1", model="model-b", context_sha256="ctx-a"
    ) is None
    assert summary._mongo_resume_summary(
        storage, "FI|video-1", model="model-a", context_sha256="ctx-b"
    ) is None


def test_prompt_keeps_memory_and_rag_separate_from_source_evidence():
    prompt = summary.get_llama_summary_user_prompt(
        "source metadata",
        "spoken words",
        "frame evidence",
        "video evidence",
        "entity: Petteri Orpo [role=normalization_context_not_source_evidence]",
        "prior summary [prior_analysis_context_not_source_evidence]",
    )
    assert "Researcher memory / normalization context (NOT source evidence)" in prompt
    assert "Retrieved prior-corpus context (NOT source evidence)" in prompt
    assert "normalization_context_not_source_evidence" in prompt
    assert "prior_analysis_context_not_source_evidence" in prompt


def test_memory_and_rag_formatters_preserve_evidence_roles():
    memory = summary._format_memory_context([
        {
            "kind": "entity",
            "label": "Petteri Orpo",
            "evidence_role": "normalization_context_not_source_evidence",
        }
    ])
    rag = summary._format_rag_context([
        {
            "stage": "frame",
            "source_record_id": "other",
            "text": "Prior derived analysis",
        }
    ])
    assert "Petteri Orpo" in memory
    assert "normalization_context_not_source_evidence" in memory
    assert "Do not treat them as direct evidence" in rag


def test_storage_id_prefers_pipeline_storage_id():
    row = pd.Series({
        "_storage_id": "stable-mongo-id",
        "video_id": "video-1",
        "allas_filename": "TikTok/video-1.mp4",
    })
    assert summary._row_storage_id(row) == "stable-mongo-id"