from __future__ import annotations

import pandas as pd

from ep24_memory import retrieve_researcher_memory
from ep24_rag import RAG_TEXT_FIELDS, retrieve_stage_rag, upsert_stage_rag


class FakeStorage:
    def __init__(self, memory=None, rag=None):
        self.memory = list(memory or [])
        self.rag = list(rag or [])
        self.upserts = []

    def find(self, purpose, query=None, *, limit=0):
        values = self.memory if purpose == "memory" else self.rag
        if purpose == "memory" and query:
            values = [
                item for item in values
                if all(item.get(key) == value for key, value in query.items())
            ]
        return list(values[:limit or None])

    def upsert_documents(self, purpose, documents):
        docs = list(documents)
        self.upserts.append((purpose, docs))
        return len(docs)


def test_researcher_memory_retrieval_is_seed_only_and_relevant():
    storage = FakeStorage(memory=[
        {
            "_storage_id": "1",
            "kind": "entity",
            "label": "Petteri Orpo",
            "review_state": "RESEARCHER_SEED",
            "evidence_role": "normalization_context_not_source_evidence",
        },
        {
            "_storage_id": "2",
            "kind": "entity",
            "label": "Unrelated Person",
            "review_state": "RESEARCHER_SEED",
            "evidence_role": "normalization_context_not_source_evidence",
        },
        {
            "_storage_id": "3",
            "kind": "entity",
            "label": "Petteri Orpo",
            "review_state": "MODEL_GUESS",
        },
    ])
    result = retrieve_researcher_memory(storage, "Orpo discusses EU politics", limit=5)
    assert [item["_storage_id"] for item in result] == ["1"]
    assert result[0]["evidence_role"] == "normalization_context_not_source_evidence"


def test_rag_retrieval_excludes_current_record_and_keeps_prior_context():
    storage = FakeStorage(rag=[
        {
            "_storage_id": "rag-current",
            "source_record_id": "current",
            "stage": "frame",
            "text": "Orpo campaign",
        },
        {
            "_storage_id": "rag-other",
            "source_record_id": "other",
            "stage": "summary",
            "text": "Orpo and EU campaign discussion",
            "evidence_role": "prior_analysis_context_not_source_evidence",
        },
    ])
    result = retrieve_stage_rag(
        storage,
        "Orpo EU",
        exclude_source_record_id="current",
        limit=5,
    )
    assert [item["_storage_id"] for item in result] == ["rag-other"]


def test_rag_persistence_uses_current_asr_and_vllm_fields():
    assert "asr_transcript" in RAG_TEXT_FIELDS
    assert "asr_translated" in RAG_TEXT_FIELDS
    assert "vllm_video_analysis" in RAG_TEXT_FIELDS

    storage = FakeStorage()
    df = pd.DataFrame([{
        "_storage_id": "record-1",
        "asr_transcript": "spoken source",
        "vllm_video_analysis": "whole-video derived analysis",
        "summary_analysis": "summary",
        "entities": "Petteri Orpo",
        "themes": "EU politics",
    }])
    count = upsert_stage_rag(storage, df, stage="summary")
    assert count == 1
    purpose, docs = storage.upserts[0]
    assert purpose == "rag"
    assert "spoken source" in docs[0]["text"]
    assert "whole-video derived analysis" in docs[0]["text"]
    assert docs[0]["evidence_role"] == "prior_analysis_context_not_source_evidence"