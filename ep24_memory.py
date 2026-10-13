"""MongoDB-backed EP24 normalization memory helpers."""
from __future__ import annotations

import hashlib
import json
import re
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
                    "origin": "canonical_private_input",
                    "evidence_role": "normalization_context_not_source_evidence",
                }
    storage.upsert_documents("memory", docs.values())
    return len(docs)


def retrieve_researcher_memory(storage, text: str, *, limit: int = 12) -> list[dict]:
    """Retrieve relevant researcher-seeded memory as normalization context only.

    This intentionally uses a portable lexical fallback so Roihu does not require
    a vector index. Returned records are context, never evidence for the current
    document.
    """
    terms = {t.casefold() for t in re.findall(r"\w+", str(text)) if len(t) > 2}
    candidates = storage.find("memory", {"review_state": "RESEARCHER_SEED"}, limit=500)
    scored: list[tuple[int, dict]] = []
    for item in candidates:
        haystack = " ".join(str(item.get(k, "")) for k in ("label", "kind")).casefold()
        score = sum(term in haystack for term in terms)
        if score:
            scored.append((score, item))
    scored.sort(key=lambda pair: (-pair[0], str(pair[1].get("label", ""))))
    return [item for _, item in scored[:limit]]