"""MongoDB EP24 RAG persistence helpers."""
from __future__ import annotations

import hashlib

import pandas as pd


def upsert_stage_rag(storage, df: pd.DataFrame, *, stage: str) -> int:
    docs = []
    for _, row in df.iterrows():
        rid = str(row.get("_storage_id", ""))
        text = "\n".join(
            str(row.get(key, ""))
            for key in (
                "whisper_translated", "whisper_transcript", "summary_analysis",
                "frame_analysis_1", "video_analysis", "formula_of_populism_analysis",
                "dna_analysis_markdown", "sna_analysis_markdown",
            )
            if str(row.get(key, "")).strip()
        )
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
