"""Mongo-backed EP24 codebook, memory, RAG and normalization context.

Context is explicitly an aid for continuity/normalization. It is never promoted to
source evidence. Current records are excluded from their own RAG retrieval.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from ep24_memory import seed_researcher_memory
from ep24_rag import upsert_stage_rag
from roihu_codebooks import COUNTRY_PROFILES, load_profile

COUNTRY_CODES = {meta["country"].casefold(): code for code, meta in COUNTRY_PROFILES.items()}
TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> set[str]:
    return {t.casefold() for t in TOKEN_RE.findall(text or "") if len(t) > 2}


def _json_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    try:
        parsed = json.loads(str(value))
        if isinstance(parsed, list):
            return [str(x) for x in parsed if str(x).strip()]
    except (json.JSONDecodeError, TypeError):
        pass
    return [str(value)]


def _score(query: str, item: dict) -> int:
    q = _tokens(query)
    if not q:
        return 0
    text = " ".join(
        str(item.get(k, ""))
        for k in ("label", "english_label", "text", "aliases", "definition", "english_definition")
    )
    return len(q & _tokens(text))


def _top(query: str, items: list[dict], *, limit: int = 8, exclude_id: str = "") -> list[dict]:
    ranked = []
    for item in items:
        if exclude_id and str(item.get("source_record_id", "")) == exclude_id:
            continue
        score = _score(query, item)
        if score:
            ranked.append((score, item))
    ranked.sort(key=lambda pair: (-pair[0], str(pair[1].get("_storage_id", ""))))
    return [{**item, "_retrieval_score": score} for score, item in ranked[:limit]]


def bootstrap_context(storage, df: pd.DataFrame, *, private_root: Path, country: str) -> dict:
    """Load bilingual codebooks and researcher-seeded memory into MongoDB."""
    code = COUNTRY_CODES.get(country.casefold())
    codebook_count = 0
    fingerprint = ""
    if code:
        try:
            entries, meta = load_profile(private_root, code)
            fingerprint = str(meta.get("fingerprint", ""))
            docs = []
            entity_docs = []
            theme_docs = []
            for entry in entries:
                doc = asdict(entry)
                doc["_storage_id"] = entry.entry_id
                doc["evidence_role"] = "background_context_not_source_evidence"
                doc["codebook_fingerprint"] = fingerprint
                docs.append(doc)
                if entry.kind in {"entity", "actor"}:
                    entity_docs.append(doc)
                if entry.kind in {"topic", "theme", "signifier"}:
                    theme_docs.append(doc)
            codebook_count = storage.upsert_documents("codebooks", docs)
            storage.upsert_documents("entities", entity_docs)
            storage.upsert_documents("themes", theme_docs)
        except FileNotFoundError:
            pass

    memory_seed_count = seed_researcher_memory(storage, df, country=country)
    return {"codebook_count": codebook_count, "codebook_fingerprint": fingerprint,
            "memory_seed_count": memory_seed_count}


def enrich_dataframe(storage, df: pd.DataFrame) -> pd.DataFrame:
    """Append bounded codebook/memory/RAG context and exact normalization mappings."""
    out = df.copy()
    codebooks = storage.find("codebooks", limit=10000)
    memory = storage.find("memory", limit=10000)
    rag = storage.find("rag", limit=10000)

    exact_entities: dict[str, dict] = {}
    exact_themes: dict[str, dict] = {}
    for item in codebooks:
        forms = [item.get("label", ""), item.get("english_label", ""), *(item.get("aliases") or [])]
        kind = item.get("kind")
        if kind in {"entity", "actor"}:
            target = exact_entities
        elif kind in {"topic", "theme", "signifier"}:
            target = exact_themes
        else:
            continue
        for form in forms:
            if str(form).strip():
                target[str(form).strip().casefold()] = item

    codebook_ctx, memory_ctx, rag_ctx, entity_norm, theme_norm = [], [], [], [], []
    for _, row in out.iterrows():
        query = " ".join(
            str(row.get(k, ""))
            for k in (
                "asr_translated", "asr_transcript",
                "whisper_translated", "whisper_transcript", "whisperResult",
                "summary_analysis", "frame_analysis_1", "video_analysis",
                "ocr_1", "entities", "themes",
            )
            if str(row.get(k, "")).strip()
        )
        rid = str(row.get("_storage_id", ""))
        codebook_ctx.append(json.dumps(_top(query, codebooks), ensure_ascii=False, default=str))
        memory_ctx.append(json.dumps(_top(query, memory), ensure_ascii=False, default=str))
        rag_ctx.append(json.dumps(_top(query, rag, exclude_id=rid), ensure_ascii=False, default=str))

        emap = []
        for label in _json_list(row.get("entities", "")):
            match = exact_entities.get(label.casefold())
            emap.append({"input": label, "canonical_id": match.get("_storage_id") if match else None,
                         "canonical_label": match.get("label") if match else label})
        tmap = []
        for label in _json_list(row.get("themes", "")):
            match = exact_themes.get(label.casefold())
            tmap.append({"input": label, "canonical_id": match.get("_storage_id") if match else None,
                         "canonical_label": match.get("label") if match else label})
        entity_norm.append(json.dumps(emap, ensure_ascii=False))
        theme_norm.append(json.dumps(tmap, ensure_ascii=False))

    out["codebook_context_json"] = codebook_ctx
    out["memory_context_json"] = memory_ctx
    out["rag_context_json"] = rag_ctx
    out["entity_normalization_json"] = entity_norm
    out["theme_normalization_json"] = theme_norm
    out["context_evidence_role"] = "normalization_and_retrieval_context_not_source_evidence"
    return out


def update_retrieval(storage, df: pd.DataFrame, *, stage: str) -> None:
    """Persist cumulative row representations for later RAG."""
    upsert_stage_rag(storage, df, stage=stage)
