"""Leifeld-compatible Discourse Network Analysis helpers for EP24.

The internal representation is richer than the strict DNA/rDNA event-list
projection.  Ambiguous statements remain in JSON/review output but are excluded
from the binary agreement event list.

Methodology:
Philip Leifeld (2017), "Discourse Network Analysis: Policy Debates as Dynamic
Networks", Oxford Handbook of Political Networks, chapter 25.
https://eprints.gla.ac.uk/121525/

Compatibility target:
https://github.com/leifeld-lab/dna
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from ep24_entities import fold_key

PROMPT_VERSION = "ep24-leifeld-dna-v1"
NETWORK_LAYER = "discourse"

DNA_COLUMNS: tuple[str, ...] = (
    "dna_analysis_markdown",
    "dna_statements_json",
    "dna_exportable_count",
    "dna_review_count",
    "dna_eventlist_path",
    "dna_prompt_version",
    "dna_model_metadata_json",
    "dna_generated_at",
    "dna_context_sha256",
    "dna_memory_context_json",
    "dna_rag_context_json",
    "dna_status",
    "dna_error",
    "dna_persistence_status",
)

DATE_FIELDS: tuple[str, ...] = (
    "published_at",
    "publication_date",
    "create_time",
    "created_at",
    "date",
    "timestamp",
    "video_date",
)

DOCUMENT_FIELDS: tuple[str, ...] = (
    "document_id",
    "video_id",
    "post_id",
    "id",
    "url",
    "webVideoUrl",
)

SOURCE_EVIDENCE_FIELDS: tuple[str, ...] = (
    "caption",
    "description",
    "text",
    "asr_translated",
    "asr_transcript",
    "whisper_translated",
    "whisper_transcript",
    "ocr_1",
    "frame_analysis_1",
    "vllm_video_analysis",
    "vllm_video_markdown_analysis",
    "summary_analysis",
    "summary_summary_md",
)

STEP6_FIELDS: tuple[str, ...] = (
    "formula_of_populism_analysis",
    "formula_of_populism_us",
    "formula_of_populism_frontier",
    "laclau_summary_md",
    "laclau_structured_json",
    "laclau_formula_conditions_met",
    "laclau_abstention_reason",
)

NORMALIZATION_FIELDS: tuple[str, ...] = (
    "entities",
    "themes",
    "topics",
    "ep24_entity_resolution_json",
    "ep24_entity_ids",
    "ep24_entity_canonical_names",
    "ep24_theme_resolution_json",
    "ep24_theme_canonical_names",
    "ep24_codebook_fingerprint",
    "ep24_codebook_context_json",
    "ep24_human_seed_context",
)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.casefold() == "nan" else text


def first_value(row: Mapping[str, Any], fields: Iterable[str]) -> str:
    for field in fields:
        value = clean_text(row.get(field))
        if value:
            return value
    return ""


def statement_time(row: Mapping[str, Any]) -> str:
    """Return the best available statement timestamp without inventing one."""
    return first_value(row, DATE_FIELDS)


def document_id(row: Mapping[str, Any], source_record_id: str) -> str:
    return first_value(row, DOCUMENT_FIELDS) or source_record_id


def stable_concept_id(country: str, canonical_label: str) -> str:
    raw = f"{country.casefold()}|concept|{fold_key(canonical_label)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def stable_statement_id(
    *,
    source_record_id: str,
    actor_id_or_name: str,
    concept_id_or_label: str,
    agreement: bool | None,
    proposition: str,
) -> str:
    qualifier = "unknown" if agreement is None else ("support" if agreement else "oppose")
    payload = "|".join(
        (
            source_record_id,
            fold_key(actor_id_or_name),
            fold_key(concept_id_or_label),
            qualifier,
            " ".join(proposition.split()).casefold(),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _context_block(title: str, row: Mapping[str, Any], fields: Iterable[str]) -> str:
    lines: list[str] = []
    for name in fields:
        value = clean_text(row.get(name))
        if value:
            lines.append(f"- {name}: {value}")
    return f"## {title}\n" + ("\n".join(lines) if lines else "- <none>")


def build_prompt_context(
    row: Mapping[str, Any],
    *,
    memory_items: list[dict[str, Any]] | None = None,
    rag_items: list[dict[str, Any]] | None = None,
    max_chars: int = 30000,
) -> tuple[str, bool]:
    """Build deterministic evidence/context blocks with strict provenance boundaries."""
    memory_items = memory_items or []
    rag_items = rag_items or []

    known = set(SOURCE_EVIDENCE_FIELDS) | set(STEP6_FIELDS) | set(NORMALIZATION_FIELDS)
    metadata_fields = sorted(
        name
        for name in row.keys()
        if name not in known and not str(name).startswith("dna_") and clean_text(row.get(name))
    )

    memory_payload = [
        {
            "kind": item.get("kind"),
            "label": item.get("label"),
            "country": item.get("country"),
            "evidence_role": item.get(
                "evidence_role", "normalization_context_not_source_evidence"
            ),
        }
        for item in memory_items
    ]
    rag_payload = [
        {
            "source_record_id": item.get("source_record_id"),
            "stage": item.get("stage"),
            "text": clean_text(item.get("text"))[:1200],
            "entities": item.get("entities"),
            "themes": item.get("themes"),
            "evidence_role": item.get(
                "evidence_role", "prior_analysis_context_not_source_evidence"
            ),
        }
        for item in rag_items
    ]

    blocks = [
        _context_block("CURRENT SOURCE EVIDENCE", row, SOURCE_EVIDENCE_FIELDS),
        _context_block("STEP 6 LACLAU/PALONEN ANALYSIS CONTEXT", row, STEP6_FIELDS),
        _context_block("ENTITY/THEME/CODEBOOK NORMALIZATION CONTEXT", row, NORMALIZATION_FIELDS),
        _context_block("SOURCE AND PIPELINE METADATA", row, metadata_fields),
        "## RESEARCHER MEMORY (NORMALIZATION ONLY, NOT SOURCE EVIDENCE)\n"
        + json.dumps(memory_payload, ensure_ascii=False, sort_keys=True),
        "## RETRIEVED RAG CONTEXT (NOT SOURCE EVIDENCE)\n"
        + json.dumps(rag_payload, ensure_ascii=False, sort_keys=True),
    ]
    text = "\n\n".join(blocks)
    if len(text) <= max_chars:
        return text, False

    # Deterministic truncation: preserve source evidence and Step 6 first, then
    # progressively clip lower-priority context rather than dropping blocks.
    source = blocks[0]
    step6 = blocks[1]
    reserved = len(source) + len(step6) + 8
    if reserved >= max_chars:
        return (source + "\n\n" + step6)[:max_chars], True

    remainder = max_chars - reserved
    tails = "\n\n".join(blocks[2:])
    return source + "\n\n" + step6 + "\n\n" + tails[:remainder], True


def canonicalize_concept(
    raw_label: str,
    *,
    country: str,
    memory_items: list[dict[str, Any]] | None = None,
    rag_items: list[dict[str, Any]] | None = None,
) -> tuple[str, str, str]:
    """Return canonical label, stable ID and provenance.

    Memory/RAG may normalize only exact folded label matches. Fuzzy semantic
    replacement belongs in a reviewed normalization layer, not in extraction.
    """
    raw = clean_text(raw_label)
    folded = fold_key(raw)
    for item in memory_items or []:
        label = clean_text(item.get("label"))
        if label and fold_key(label) == folded:
            return label, stable_concept_id(country, label), "researcher_memory_exact"
    for item in rag_items or []:
        for field in ("concept_label", "concept", "canonical_concept_label"):
            label = clean_text(item.get(field))
            if label and fold_key(label) == folded:
                return label, stable_concept_id(country, label), "rag_exact"
    return raw, stable_concept_id(country, raw), "source_label"


def enrich_statement(
    statement: Mapping[str, Any],
    *,
    row: Mapping[str, Any],
    source_record_id: str,
    country: str,
    actor_lookup: Mapping[str, Mapping[str, str]] | None = None,
    memory_items: list[dict[str, Any]] | None = None,
    rag_items: list[dict[str, Any]] | None = None,
    model: str = "",
) -> dict[str, Any]:
    actor_lookup = actor_lookup or {}
    item = dict(statement)
    raw_actor = clean_text(item.get("actor_name"))
    actor_hit = actor_lookup.get(fold_key(raw_actor))
    if actor_hit:
        actor_id = clean_text(actor_hit.get("entity_id"))
        actor_canonical = clean_text(actor_hit.get("canonical_name")) or raw_actor
        actor_provenance = "entity_resolution"
    else:
        actor_id = ""
        actor_canonical = raw_actor
        actor_provenance = "unresolved_source_label"

    raw_concept = clean_text(item.get("concept_label"))
    concept_canonical, concept_id, concept_provenance = canonicalize_concept(
        raw_concept,
        country=country,
        memory_items=memory_items,
        rag_items=rag_items,
    )
    agreement = item.get("agreement")
    if agreement not in (True, False, None):
        agreement = None

    item.update(
        {
            "network_layer": NETWORK_LAYER,
            "source_record_id": source_record_id,
            "document_id": document_id(row, source_record_id),
            "statement_time": statement_time(row),
            "actor_raw_name": raw_actor,
            "actor_id": actor_id or None,
            "actor_canonical_name": actor_canonical,
            "actor_normalization_provenance": actor_provenance,
            "concept_raw_label": raw_concept,
            "concept_canonical_label": concept_canonical,
            "concept_id": concept_id,
            "concept_normalization_provenance": concept_provenance,
            "agreement": agreement,
            "qualifier": "positive" if agreement is True else "negative" if agreement is False else None,
            "exportable_to_dna": agreement in (True, False),
            "provenance": {
                "method": "Leifeld_DNA_statement_coding",
                "prompt_version": PROMPT_VERSION,
                "model": model,
                "current_source_evidence_required": True,
            },
        }
    )
    item["statement_id"] = stable_statement_id(
        source_record_id=source_record_id,
        actor_id_or_name=actor_id or actor_canonical,
        concept_id_or_label=concept_id or concept_canonical,
        agreement=agreement,
        proposition=clean_text(item.get("proposition")),
    )
    return item


EVENTLIST_COLUMNS: tuple[str, ...] = (
    "statement_id",
    "time",
    "organization",
    "concept",
    "agreement",
    "document",
    "source_record_id",
    "network_layer",
    "proposition",
    "evidence_quote",
    "confidence",
    "actor_id",
    "concept_id",
)


def statements_to_event_rows(statements: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Project rich statements into a strict binary DNA/rDNA-compatible event list."""
    rows: list[dict[str, Any]] = []
    for item in statements:
        agreement = item.get("agreement")
        if agreement not in (True, False):
            continue
        rows.append(
            {
                "statement_id": clean_text(item.get("statement_id")),
                "time": clean_text(item.get("statement_time")),
                "organization": clean_text(item.get("actor_canonical_name"))
                or clean_text(item.get("actor_name")),
                "concept": clean_text(item.get("concept_canonical_label"))
                or clean_text(item.get("concept_label")),
                "agreement": bool(agreement),
                "document": clean_text(item.get("document_id")),
                "source_record_id": clean_text(item.get("source_record_id")),
                "network_layer": NETWORK_LAYER,
                "proposition": clean_text(item.get("proposition")),
                "evidence_quote": clean_text(item.get("evidence_quote")),
                "confidence": item.get("confidence"),
                "actor_id": clean_text(item.get("actor_id")),
                "concept_id": clean_text(item.get("concept_id")),
            }
        )
    return rows


def write_eventlist(statements: Iterable[Mapping[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(statements_to_event_rows(statements), columns=EVENTLIST_COLUMNS)
    frame.to_csv(path, index=False)
    return path


def deduplicate_event_rows(
    rows: Iterable[Mapping[str, Any]], *, policy: str = "include"
) -> list[dict[str, Any]]:
    """Apply DNA-style deterministic duplicate policies to event rows.

    Supported: include, document, week, month, year, acrossrange.
    """
    items = [dict(row) for row in rows]
    if policy == "include":
        return items
    if policy not in {"document", "week", "month", "year", "acrossrange"}:
        raise ValueError(f"unsupported DNA duplicate policy: {policy}")

    seen: set[tuple[Any, ...]] = set()
    result: list[dict[str, Any]] = []
    for row in items:
        stamp = pd.to_datetime(row.get("time"), errors="coerce", utc=True)
        if policy == "document":
            bucket = clean_text(row.get("document"))
        elif policy == "week":
            bucket = "" if pd.isna(stamp) else f"{stamp.isocalendar().year}-W{stamp.isocalendar().week:02d}"
        elif policy == "month":
            bucket = "" if pd.isna(stamp) else stamp.strftime("%Y-%m")
        elif policy == "year":
            bucket = "" if pd.isna(stamp) else stamp.strftime("%Y")
        else:
            bucket = "all"
        key = (
            fold_key(row.get("organization")),
            fold_key(row.get("concept")),
            bool(row.get("agreement")),
            bucket,
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def actor_projection(
    rows: Iterable[Mapping[str, Any]],
    *,
    mode: str = "congruence",
    normalization: str = "no",
) -> pd.DataFrame:
    """Small deterministic reference implementation for tests/export validation.

    DNA/rDNA remains the canonical network-analysis engine. This helper exists to
    validate that our event semantics match congruence/conflict/subtract behavior.
    """
    events = [dict(r) for r in rows if r.get("agreement") in (True, False)]
    actors = sorted({clean_text(r.get("organization")) for r in events if clean_text(r.get("organization"))})
    profiles: dict[str, set[tuple[str, bool]]] = {a: set() for a in actors}
    concepts: dict[str, set[str]] = {a: set() for a in actors}
    for r in events:
        actor = clean_text(r.get("organization"))
        concept = clean_text(r.get("concept"))
        if actor and concept:
            profiles[actor].add((concept, bool(r.get("agreement"))))
            concepts[actor].add(concept)

    matrix = pd.DataFrame(0.0, index=actors, columns=actors)
    for i, a in enumerate(actors):
        for b in actors[i + 1 :]:
            congruence = len(profiles[a] & profiles[b])
            conflict = sum(
                1
                for concept in concepts[a] & concepts[b]
                if ((concept, True) in profiles[a] and (concept, False) in profiles[b])
                or ((concept, False) in profiles[a] and (concept, True) in profiles[b])
            )
            if mode == "congruence":
                weight = float(congruence)
            elif mode == "conflict":
                weight = float(conflict)
            elif mode == "subtract":
                weight = float(congruence - conflict)
            else:
                raise ValueError(f"unsupported projection mode: {mode}")

            if normalization != "no" and mode == "congruence":
                union = len(profiles[a] | profiles[b])
                if normalization == "jaccard":
                    weight = congruence / union if union else 0.0
                elif normalization == "cosine":
                    denom = math.sqrt(len(profiles[a]) * len(profiles[b]))
                    weight = congruence / denom if denom else 0.0
                elif normalization == "average":
                    denom = (len(concepts[a]) + len(concepts[b])) / 2
                    weight = congruence / denom if denom else 0.0
                else:
                    raise ValueError(f"unsupported normalization: {normalization}")
            matrix.at[a, b] = matrix.at[b, a] = weight
    return matrix


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
