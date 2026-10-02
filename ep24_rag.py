"""MongoDB EP24 RAG persistence and retrieval helpers."""
from __future__ import annotations

import hashlib
import re

import pandas as pd

RAG_TEXT_FIELDS = (
    "asr_translated",
    "asr_transcript",
    "whisper_translated",
    "whisper_transcript",
    "summary_analysis",
    "frame_analysis_1",
    "vllm_video_analysis",
    "vllm_video_markdown_analysis",
    "video_analysis",
    "formula_of_populism_analysis",
    "dna_analysis_markdown",
    "sna_analysis_markdown",
)


def upsert_stage_rag(storage, df: pd.DataFrame, *, stage: str) -> int:
    docs = []
    for _, row in df.iterrows():
        rid = str(row.get("_storage_id", ""))
        text = "\n".join(
            str(row.get(key, ""))
            for key in RAG_TEXT_FIELDS
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


def retrieve_stage_rag(
    storage,
    text: str,
    *,
    exclude_source_record_id: str | None = None,
    limit: int = 6,
) -> list[dict]:
    """Retrieve lexical RAG context from the same country-scoped Mongo store."""
    terms = {t.casefold() for t in re.findall(r"\w+", str(text)) if len(t) > 2}
    candidates = storage.find("rag", limit=500)
    scored: list[tuple[int, dict]] = []
    for item in candidates:
        if exclude_source_record_id and str(item.get("source_record_id", "")) == exclude_source_record_id:
            continue
        haystack = " ".join(
            str(item.get(k, "")) for k in ("text", "entities", "themes", "stage")
        ).casefold()
        score = sum(term in haystack for term in terms)
        if score:
            scored.append((score, item))
    scored.sort(
        key=lambda pair: (
            -pair[0],
            str(pair[1].get("stage", "")),
            str(pair[1].get("_storage_id", "")),
        )
    )
    return [item for _, item in scored[:limit]]