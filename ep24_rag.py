"""MongoDB EP24 RAG persistence helpers.

Canonical current-pipeline fields are preferred. Legacy names remain read-only
fallbacks so old persisted EP24 rows stay retrievable without being regenerated.
"""
from __future__ import annotations

import hashlib
from typing import Iterable

import pandas as pd


def _first_nonempty(row: pd.Series, names: Iterable[str]) -> str:
    for name in names:
        value = str(row.get(name, "") or "").strip()
        if value and value.casefold() != "nan":
            return value
    return ""


def rag_text(row: pd.Series) -> str:
    """Build retrieval text without duplicating canonical + legacy aliases."""
    parts = [
        _first_nonempty(row, ("asr_translated", "whisper_translated")),
        _first_nonempty(row, ("asr_transcript", "whisper_transcript")),
        _first_nonempty(row, ("ocr_1",)),
        _first_nonempty(row, ("frame_analysis_1",)),
        _first_nonempty(row, ("vllm_video_analysis", "video_analysis")),
        _first_nonempty(row, ("summary_analysis",)),
        _first_nonempty(row, ("formula_of_populism_analysis",)),
        _first_nonempty(row, ("dna_analysis_markdown",)),
        _first_nonempty(row, ("sna_analysis_markdown",)),
    ]
    return "\n".join(part for part in parts if part)


def upsert_stage_rag(storage, df: pd.DataFrame, *, stage: str) -> int:
    docs = []
    for _, row in df.iterrows():
        rid = str(row.get("_storage_id", "") or "").strip()
        text = rag_text(row)
        if not rid or not text:
            continue
        docs.append({
            "_storage_id": hashlib.sha256(f"{rid}|{stage}".encode()).hexdigest(),
            "source_record_id": rid,
            "stage": stage,
            "text": text,
            "entities": str(row.get("entities", "")),
            "themes": str(row.get("themes", "")),
            "evidence_role": "prior_analysis_context_not_source_evidence",
        })
    return storage.upsert_documents("rag", docs)
