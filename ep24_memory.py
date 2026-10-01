"""MongoDB-backed EP24 normalization memory helpers."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import pandas as pd


def _values(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    try:
        parsed = json.loads(str(value))
    except (json.JSONDecodeError, TypeError):
        parsed = None
    if isinstance(parsed, list):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def seed_researcher_memory(storage, df: pd.DataFrame, *, country: str) -> int:
    docs: dict[str, dict] = {}
    for _, row in df.iterrows():
        for kind, field in (("entity", "entities"), ("theme", "themes")):
            for label in _values(row.get(field, "")):
                sid = hashlib.sha256(f"{country}|{kind}|{label.casefold()}".encode()).hexdigest()
                docs[sid] = {
                    "_storage_id": sid,
                    "kind": kind,
                    "label": label,
                    "country": country,
                    "review_state": "RESEARCHER_SEED",
                    "origin": "pre_step_1_researcher_merge",
                    "evidence_role": "normalization_context_not_source_evidence",
                }
    storage.upsert_documents("memory", docs.values())
    return len(docs)
