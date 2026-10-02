#!/usr/bin/env python3
"""Step 7: Discourse Network Analysis (DNA) statement extraction for EP24.

Coding follows Philip Leifeld's Discourse Network Analysis:

    Leifeld, Philip (2017). Discourse Network Analysis: Policy Debates as Dynamic
    Networks. In: Victor, Lubell & Montgomery (eds.), The Oxford Handbook of
    Political Networks, ch. 25. Oxford University Press.
    Preprint: https://eprints.gla.ac.uk/121525/

The unit of analysis is the statement: one actor making one claim about one
concept at one time, with an explicit positive/support or negative/oppose
qualifier where the source supports one. This stage turns accumulated EP24
evidence into those statements and exports them for the Leifeld DNA/rDNA
ecosystem; the compatibility layer lives in ``ep24_dna``.

This stage complements Step 6 rather than replacing it (Step 6 interprets
discursive formations, this stage codes observable actor-concept claims) and
complements Step 8 rather than duplicating it (DNA ties are actor<->concept and
derived congruence/conflict, SNA ties are evidenced social/communication
relations). Records carry ``network_layer`` so downstream code can tell them
apart.

Memory, RAG and codebooks are normalization/disambiguation context only. They are
never evidence that the current actor made the current claim.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ep24_cli import configure_step_cli
from ep24_dna import (
    DNA_COLUMNS,
    canonical_agreement,
    iter_statements,
    parse_date_time,
    statement_id,
    summarize,
)
from ep24_models import ollama_model, ollama_model_source
from ep24_pipeline import ensure_columns, load_cumulative_csv, write_cumulative_csv
from ep24_schema import stable_source_id

LOG = logging.getLogger("ep24.step7")

PROMPT_VERSION = "ep24-dna-leifeld-v1"
DEFAULT_MAX_CONTEXT_CHARS = 28000
DEFAULT_NUM_CTX = 16384
DEFAULT_NUM_PREDICT = 4096

#: Every statement record records which network layer it belongs to (issue #182 s13).
NETWORK_LAYER = "discourse"
#: Evidence role for statements coded from the current record.
STATEMENT_EVIDENCE_ROLE = "current_source_evidence"
#: Concept provenance when a codebook match exists vs. a genuinely new concept.
CONCEPT_KNOWN = "codebook"
CONCEPT_NOVEL = "novel"

#: Prompt section -> provenance role. Mirrors Step 6 so the two stages read alike.
EXTERNAL_CONTEXT_COLUMNS = (
    "codebook_context_json",
    "memory_context_json",
    "rag_context_json",
    "entity_normalization_json",
    "theme_normalization_json",
    "context_evidence_role",
)

_COUNTRY_ALIASES = {
    "finland": "finland", "fi": "finland",
    "poland": "poland", "pl": "poland",
    "portugal": "portugal", "pt": "portugal",
    "germany": "germany", "de": "germany",
    "spain": "spain", "es": "spain",
    "hungary": "hungary", "hu": "hungary",
    "croatia": "croatia", "hr": "croatia",
    "france": "france", "fr": "france",
    "bulgaria": "bulgaria", "bg": "bulgaria",
    "sweden": "sweden", "sv": "sweden",
}


class Step7ParseError(ValueError):
    def __init__(self, message: str, *, raw_response: str):
        super().__init__(message)
        self.raw_response = raw_response


# The system prompt implements the coding logic rather than gesturing at it.
SYSTEM_PROMPT = """You are coding statements for Discourse Network Analysis (DNA) following
Philip Leifeld's methodology (Leifeld 2017, Discourse Network Analysis: Policy
Debates as Dynamic Networks).

The unit of analysis is a statement linking:
(1) one identifiable actor,
(2) one discourse concept/claim/justification/policy position,
(3) a positive/support or negative/oppose qualifier where explicit,
(4) a time/source context.

Return JSON only and conform exactly to the supplied schema.

EVIDENCE DISCIPLINE
- Extract only actor-concept claims supported by the CURRENT SOURCE EVIDENCE.
- One statement = one explicit actor-concept claim. Do not infer positions from
  topic co-occurrence, and do not split one claim into several statements.
- Empty statements are valid and preferred when evidence is absent.
- Memory, RAG, codebooks and prior records are normalization/disambiguation
  context only. They are NEVER evidence that the current actor made the current
  claim.

QUALIFIER
- agreement is "support" only for explicit support, "oppose" only for explicit
  opposition, otherwise "uncertain".
- Two actors mentioning the same concept with opposite positions must remain
  distinguishable. Never collapse this distinction.
- Do not force a binary qualifier. An ambiguous statement takes "uncertain" and
  must say why in uncertainty_notes.

ACTOR VS CONCEPT
- Do not confuse actor extraction with concept extraction.
- A concept is an analysable claim, policy position, belief, justification,
  narrative or frame -- not an arbitrary named entity.
- Use the canonical actor/concept label when trusted normalization context
  supplies one, and always preserve the raw label you observed.

TIME
- Give the best available timestamp for the statement (the source record's date).
- Do not reduce the discourse to a timeless graph.

PROVENANCE
- Every statement needs an evidence_quote grounded in the CURRENT SOURCE EVIDENCE.
  Never cite memory/RAG/codebook text as the quote.
- Record uncertainty and counter-evidence rather than guessing.
"""


def _models() -> type:
    """Import runtime-only dependencies after CLI parsing (see #152)."""
    from pydantic import BaseModel, Field

    class DNAStatement(BaseModel):
        actor_name: str = Field(description="The actor as observed in the source")
        actor_canonical: str = Field(
            default="", description="Canonical actor label if supplied by context, else empty"
        )
        concept_label: str = Field(description="The claim/policy position/justification")
        concept_canonical: str = Field(
            default="", description="Canonical concept label if supplied by context, else empty"
        )
        proposition: str = Field(default="", description="The claim in one sentence")
        stance: str = Field(
            default="", description="observed qualifier wording; support/oppose/uncertain"
        )
        agreement: str = Field(
            default="uncertain", description="support, oppose, or uncertain"
        )
        date_time: str = Field(default="", description="Best available ISO date/time for the claim")
        evidence_quote: str = Field(default="", description="Verbatim span from current source evidence")
        evidence_fields: list[str] = Field(
            default_factory=list, description="Source fields the evidence came from"
        )
        confidence: float = Field(default=0.0, ge=0.0, le=1.0)
        uncertainty_notes: list[str] = Field(default_factory=list)
        counter_evidence: list[str] = Field(default_factory=list)

    class DNAResult(BaseModel):
        analysis_markdown: str
        statements: list[DNAStatement] = Field(default_factory=list)
        corpus_level_cautions: list[str] = Field(default_factory=list)

    return DNAResult


def _text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.casefold() == "nan" else text


def _bounded(value: str, limit: int) -> tuple[str, bool]:
    if len(value) <= limit:
        return value, False
    return value[:limit], True


def _render_fields(row: pd.Series, names: list[str]) -> str:
    lines = [f"- {name}: {text}" for name in names if (text := _text(row.get(name, "")))]
    return "\n".join(lines) if lines else "- <none>"


def _external_context(row: pd.Series, column: str, heading: str, role: str) -> str:
    value = _text(row.get(column, ""))
    return f"{heading}\nROLE: {role}\n{value or '[]'}"


def build_prompt_context(row: pd.Series, *, max_chars: int | None = None) -> tuple[str, bool]:
    """Build deterministic context, preserving the theoretically important fields.

    Unlike the previous stub this does not hand the model a five-field subset: it
    classifies the whole cumulative row by provenance and keeps every non-empty
    field, truncating only once, at a documented budget (issue #182 s7).
    """
    max_chars = max_chars or int(
        os.getenv("LACLAUGPT_STEP7_MAX_CONTEXT_CHARS", DEFAULT_MAX_CONTEXT_CHARS)
    )
    researcher = {"entities", "themes", "political_preference", "researcher_note"}
    source_representation_prefixes = ("asr_", "ocr_", "preprocess_")
    prior_analysis_prefixes = (
        "frame_analysis_", "vllm_", "summary_", "postprocess_", "ep24_entity_",
        "ep24_theme_", "ep24_memory_", "ep24_seed_", "ep24_sentiment_",
        "formula_of_populism_", "laclau_", "sna_",
    )
    source_representation_exact = {
        "frame_file", "frame_timestamp_seconds", "video_duration_seconds",
        "whisper_transcript", "whisper_translated", "whisperResult",
    }
    prior_analysis_exact = {
        "metadata", "positive", "neutral", "negative", "video_analysis",
        "frame_analysis_1", "summary_analysis",
    }

    step7_owned = set(DNA_COLUMNS)
    source_meta, source_repr, researcher_fields, prior_analysis = [], [], [], []
    for name in row.index:
        if name in step7_owned or name in EXTERNAL_CONTEXT_COLUMNS:
            # Never feed Step 7's own prior output back in: keeps the context hash
            # stable and prevents circular reinforcement on a rerun.
            continue
        if name in researcher:
            researcher_fields.append(name)
        elif name in source_representation_exact or name.startswith(source_representation_prefixes):
            source_repr.append(name)
        elif name in prior_analysis_exact or name.startswith(prior_analysis_prefixes):
            prior_analysis.append(name)
        else:
            source_meta.append(name)

    sections = [
        "=== CURRENT SOURCE METADATA ===\nROLE: recorded source context\n"
        + _render_fields(row, source_meta),
        "=== CURRENT SOURCE-DERIVED REPRESENTATIONS ===\n"
        "ROLE: ASR/OCR/media-derived cues; usable as current-document evidence with "
        "normal model-error caution\n" + _render_fields(row, source_repr),
        "=== HUMAN RESEARCHER ANNOTATION ===\n"
        "ROLE: authoritative canonical seeds for normalization; not proof of a claim\n"
        + _render_fields(row, researcher_fields),
        "=== DERIVED PRIOR-STAGE ANALYSIS ===\n"
        "ROLE: derived_prior_stage_analysis_not_source_evidence\n"
        + _render_fields(row, prior_analysis),
        _external_context(
            row, "codebook_context_json", "=== RESEARCHER/CODEBOOK CONTEXT ===",
            "background_context_not_source_evidence",
        ),
        _external_context(
            row, "memory_context_json", "=== RESEARCHER MEMORY ===",
            "normalization_context_not_source_evidence",
        ),
        _external_context(
            row, "rag_context_json", "=== RETRIEVED CORPUS CONTEXT ===",
            "prior_analysis_context_not_source_evidence",
        ),
    ]
    return _bounded("\n\n".join(sections), max_chars)


def _prompt_hash(context: str) -> str:
    payload = f"{PROMPT_VERSION}\n{ollama_model()}\n{SYSTEM_PROMPT}\n{context}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _model_metadata() -> dict[str, Any]:
    return {
        "provider": "ollama",
        "model": ollama_model(),
        "model_source": ollama_model_source(),
        "num_ctx": int(os.getenv("LACLAUGPT_STEP7_NUM_CTX", DEFAULT_NUM_CTX)),
        "num_predict": int(os.getenv("LACLAUGPT_STEP7_NUM_PREDICT", DEFAULT_NUM_PREDICT)),
        "temperature": 0.0,
    }


def analyze_context(context: str) -> tuple[str, Any]:
    import ollama

    DNAResult = _models()
    metadata = _model_metadata()
    response = ollama.chat(
        model=metadata["model"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ],
        format=DNAResult.model_json_schema(),
        options={
            "temperature": 0.0,
            "num_ctx": metadata["num_ctx"],
            "num_predict": metadata["num_predict"],
        },
    )
    raw = response["message"]["content"]
    try:
        parsed = DNAResult.model_validate_json(raw)
    except Exception as exc:
        raise Step7ParseError(
            f"Step 7 structured JSON validation failed: {exc}", raw_response=raw
        ) from exc
    return raw, parsed


def _normalization_lookups(row: pd.Series) -> tuple[dict[str, str], dict[str, str]]:
    """Build canonical-label lookups from the shared normalization columns.

    These are labels to *prefer* for display; the stable IDs still come from the
    entity resolver, so the LLM never mints a canonical ID.
    """
    actors: dict[str, str] = {}
    concepts: dict[str, str] = {}
    for column, target in (("entity_normalization_json", actors), ("theme_normalization_json", concepts)):
        try:
            parsed = json.loads(_text(row.get(column, "")) or "[]")
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(parsed, list):
            continue
        for item in parsed:
            if not isinstance(item, dict):
                continue
            label = _text(item.get("input", "")).casefold()
            canonical = _text(item.get("canonical_label", ""))
            canonical_id = _text(item.get("canonical_id", ""))
            if label and canonical:
                target[label] = canonical
            if label and canonical_id:
                target[f"{label}|id"] = canonical_id
    return actors, concepts


def _statement_records(
    parsed: Any,
    row: pd.Series,
    *,
    context_hash: str,
    entity_lookup: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    """Turn model statements into deterministic records carrying DNA provenance."""
    actors, concepts = _normalization_lookups(row)
    record_id = str(row.get("_storage_id") or stable_source_id(row))
    document_id = _text(row.get("video_id")) or record_id
    source_date = _text(
        row.get("date_time")
        or row.get("published_at")
        or row.get("created_at")
        or row.get("preprocess_completed_at")
    )
    country = _text(row.get("country"))

    records: list[dict[str, Any]] = []
    for item in parsed.statements:
        raw_actor = _text(item.actor_name)
        if not raw_actor:
            continue
        concept = _text(item.concept_label)
        if not concept:
            continue

        from ep24_entities import fold_key

        resolved = entity_lookup.get(fold_key(raw_actor)) or {}
        actor_id = _text(resolved.get("entity_id", "")) or _text(
            actors.get(f"{raw_actor.casefold()}|id", "")
        )
        actor_canonical = (
            _text(resolved.get("canonical_name", ""))
            or _text(item.actor_canonical)
            or actors.get(raw_actor.casefold(), "")
            or raw_actor
        )
        concept_canonical = _text(item.concept_canonical) or concepts.get(concept.casefold(), "") or concept
        concept_id = _text(concepts.get(f"{concept.casefold()}|id", ""))
        if not concept_id:
            concept_id = "concept:" + hashlib.sha256(
                f"{country}|{concept_canonical.casefold()}".encode()
            ).hexdigest()[:16]
        concept_provenance = CONCEPT_KNOWN if concepts.get(f"{concept.casefold()}|id") else CONCEPT_NOVEL

        date_time = parse_date_time(item.date_time) or parse_date_time(source_date)
        # The model's stance wording is kept raw; only the documented markers map
        # onto a binary qualifier. Anything else stays uncertain.
        agreement = canonical_agreement(item.agreement) 
        if agreement is None:
            agreement = canonical_agreement(item.stance)

        records.append({
            "statement_id": statement_id(
                source_record_id=record_id,
                actor=actor_canonical,
                concept=concept_canonical,
                agreement=agreement,
                date_time=date_time,
                proposition=_text(item.proposition),
            ),
            "source_record_id": record_id,
            "document_id": document_id,
            "network_layer": NETWORK_LAYER,
            "organization": actor_canonical,
            "actor_name_raw": raw_actor,
            "actor_id": actor_id,
            "actor_type": _text(row.get("account_type")),
            "concept": concept_canonical,
            "concept_label_raw": concept,
            "concept_id": concept_id,
            "concept_provenance": concept_provenance,
            "proposition": _text(item.proposition),
            "stance": _text(item.stance),
            "agreement": agreement,
            "date_time": date_time,
            "evidence_quote": _text(item.evidence_quote),
            "evidence_fields": list(item.evidence_fields),
            "confidence": float(item.confidence),
            "uncertainty_notes": list(item.uncertainty_notes),
            "counter_evidence": list(item.counter_evidence),
            "evidence_role": STATEMENT_EVIDENCE_ROLE,
            "provenance": "model_derived",
            "extraction_prompt_version": PROMPT_VERSION,
            "context_sha256": context_hash,
            "source_country": country,
        })
    return records


def _clean_document(row: pd.Series) -> dict[str, Any]:
    doc: dict[str, Any] = {}
    for key, value in row.to_dict().items():
        try:
            if pd.isna(value):
                value = None
        except (TypeError, ValueError):
            pass
        doc[str(key)] = value
    return doc


def _country(value: str | None) -> str:
    candidate = value or os.getenv("LACLAUGPT_COUNTRY") or ""
    key = str(candidate).strip().casefold()
    return _COUNTRY_ALIASES.get(key, key or "unknown")


def _prepare_context(df: pd.DataFrame, country: str) -> tuple[pd.DataFrame, Any | None]:
    """Attach bounded Mongo-backed memory/RAG/codebook context when configured."""
    if os.getenv("LACLAUGPT_MONGO_ENABLED", "0").casefold() not in {"1", "true", "yes", "on"}:
        LOG.warning("MongoDB disabled; Step 7 will run without durable memory/RAG persistence")
        return df.copy(), None

    from ep24_context import bootstrap_context, enrich_dataframe
    from ep24_db import country_storage

    cm = country_storage(country)
    storage = cm.__enter__()
    try:
        private_root = Path(os.getenv("LACLAUGPT_MULTIMODAL_PRIVATE_ROOT", "."))
        bootstrap = bootstrap_context(storage, df, private_root=private_root, country=country)
        LOG.info(
            "context bootstrap country=%s codebooks=%s memory=%s fingerprint=%s",
            country,
            bootstrap.get("codebook_count"),
            bootstrap.get("memory_seed_count"),
            bootstrap.get("codebook_fingerprint"),
        )
        return enrich_dataframe(storage, df), (cm, storage, bootstrap)
    except Exception:
        cm.__exit__(*sys.exc_info())
        raise


def _persist_row(storage, row: pd.Series, *, country: str) -> int:
    """Patch Step 7 fields into the durable cumulative document (never replace)."""
    doc = _clean_document(row)
    record_id = str(doc.get("_storage_id") or stable_source_id(row))
    doc["_storage_id"] = record_id
    doc["_pipeline_step_7"] = {
        "status": doc.get("dna_status"),
        "prompt_version": PROMPT_VERSION,
        "generated_at": doc.get("dna_generated_at"),
        "context_sha256": doc.get("dna_context_sha256"),
        "country": country,
        "network_layer": NETWORK_LAYER,
    }
    return storage.patch_documents("dataframe", [doc])


def _persist_statements(storage, statements: list[dict[str, Any]]) -> int:
    """Persist statements in their own collection for relational/dedup queries."""
    if not statements:
        return 0
    docs = []
    for item in statements:
        docs.append({**item, "_storage_id": f"{item.get('statement_id')}"})
    return storage.upsert_documents("dna_statements", docs)


def _resume_from_mongo(storage, record_id: str, *, context_hash: str) -> dict[str, Any] | None:
    """Return Step 7 fields only when durable provenance exactly matches."""
    docs = storage.find("dataframe", {"_storage_id": record_id}, limit=1)
    if not docs:
        return None
    doc = docs[0]
    if str(doc.get("dna_status", "")) != "ok":
        return None
    if str(doc.get("dna_prompt_version", "")) != PROMPT_VERSION:
        return None
    if str(doc.get("dna_context_sha256", "")) != context_hash:
        return None
    try:
        model_meta = json.loads(str(doc.get("dna_model_metadata_json", "") or "{}"))
    except json.JSONDecodeError:
        return None
    if str(model_meta.get("model", "")) != ollama_model():
        return None
    return {column: doc.get(column, "") for column in DNA_COLUMNS}


def _finalize_context_handle(handle: Any | None, df: pd.DataFrame) -> None:
    if not handle:
        return
    cm, storage, _ = handle
    try:
        from ep24_context import update_retrieval

        update_retrieval(storage, df, stage="step_07_discourse_network_analysis")
    finally:
        cm.__exit__(None, None, None)


def _export_dir(output_path: Path, country: str) -> Path:
    override = os.getenv("LACLAUGPT_DNA_EXPORT_DIR")
    if override:
        return Path(override)
    return output_path.parent / "dna" / country


def write_exports(df: pd.DataFrame, country: str, output_path: Path) -> dict[str, str]:
    """Write the rDNA-compatible artefacts for one country (issue #182 s9).

    Deterministic and offline: the event list, the network-ready CSVs, a GraphML
    congruence graph, and the generated rDNA import script. No R required.
    """
    from ep24_dna import (
        actor_conflict_network,
        actor_congruence_network,
        apply_normalization,
        build_event_list,
        read_event_list_csv,
        two_mode_network,
        write_edge_csv,
        write_event_list_csv,
        write_graphml,
        write_network_csv,
        write_rdna_import_script,
    )

    directory = _export_dir(output_path, country)
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    events: list[dict] = []
    all_statements: list[dict] = []
    for _, row in df.iterrows():
        statements = list(iter_statements(row.get("dna_statements_json", "")))
        if not statements:
            continue
        all_statements.extend(statements)
        events.extend(
            build_event_list(
                statements,
                document_id=_text(row.get("video_id")) or str(row.get("_storage_id", "")),
                document_title=_text(row.get("video_id")),
                document_source=_text(row.get("allas_filename")),
            )
        )

    event_path = directory / f"dna_events_{country}.csv"
    write_event_list_csv(events, event_path)
    written["event_list"] = str(event_path)

    if not all_statements:
        return written

    two_mode = two_mode_network(all_statements, qualifier_aggregation="subtract")
    path = directory / f"dna_twomode_{country}.csv"
    write_network_csv(two_mode, path)
    written["two_mode"] = str(path)

    congruence = actor_congruence_network(all_statements)
    normalized = apply_normalization(congruence, statements=all_statements, normalization="jaccard")
    path = directory / f"dna_actor_congruence_{country}.csv"
    write_edge_csv(normalized, path, label="jaccard")
    written["actor_congruence"] = str(path)
    written["actor_congruence_graphml"] = str(
        write_graphml(normalized, directory / f"dna_actor_congruence_{country}.graphml", label="jaccard")
    )

    conflict = actor_conflict_network(all_statements)
    path = directory / f"dna_actor_conflict_{country}.csv"
    write_edge_csv(conflict, path, label="conflict")
    written["actor_conflict"] = str(path)

    # Round-trip through the interchange file so what we hand rDNA is what we tested.
    reread = read_event_list_csv(event_path)
    written["importer"] = str(
        write_rdna_import_script(event_path, directory / f"dna_import_{country}.R")
    )
    LOG.info(
        "Step 7 export country=%s events=%s reread=%s dir=%s",
        country, len(events), len(reread), directory,
    )
    return written


def process_country(country: str | None = None) -> Path:
    normalized_country = _country(country)
    input_path = Path(os.getenv("LACLAUGPT_INPUT_CSV") or f"ep24_{normalized_country}.csv")
    output_path = Path(os.getenv("LACLAUGPT_OUTPUT_CSV") or input_path)

    original = load_cumulative_csv(input_path, require_canonical=bool(os.getenv("LACLAUGPT_INPUT_CSV")))
    max_rows = int(os.getenv("LACLAUGPT_MAX_ROWS", "0") or 0)
    if max_rows > 0:
        if output_path.resolve() == input_path.resolve() and max_rows < len(original):
            raise ValueError(
                "Refusing to truncate the input CSV: when LACLAUGPT_MAX_ROWS is set, "
                "LACLAUGPT_OUTPUT_CSV must be a different path."
            )
        original = original.head(max_rows).copy()

    out = original.copy()
    ensure_columns(out, DNA_COLUMNS)

    enriched, context_handle = _prepare_context(out, normalized_country)
    for column in (*EXTERNAL_CONTEXT_COLUMNS, "context_evidence_role"):
        if column in enriched.columns:
            out[column] = enriched[column]

    from ep24_entities import resolution_lookup
    from ep24_redis import RedisCoordinator

    redis = RedisCoordinator(normalized_country, 7)
    checkpoint_every = max(1, int(os.getenv("LACLAUGPT_STEP7_CHECKPOINT_EVERY", "10") or 10))
    processed_since_checkpoint = 0

    try:
        storage = context_handle[1] if context_handle else None
        for index, row in out.iterrows():
            record_id = str(row.get("_storage_id") or stable_source_id(row))
            with redis.lock(record_id) as acquired:
                if not acquired:
                    LOG.info("record lock busy id=%s; skipped", record_id)
                    continue
                redis.mark(record_id, "running")
                started = time.monotonic()
                context, truncated = build_prompt_context(row)
                context_hash = _prompt_hash(context)
                LOG.info(
                    "step7 country=%s row=%s id=%s model=%s prompt=%s context_sha256=%s truncated=%s",
                    normalized_country, index, record_id, ollama_model(), PROMPT_VERSION,
                    context_hash, truncated,
                )
                try:
                    resumed = (
                        _resume_from_mongo(storage, record_id, context_hash=context_hash)
                        if storage is not None
                        else None
                    )
                    if resumed is not None:
                        for column, value in resumed.items():
                            out.at[index, column] = value
                        out.at[index, "dna_persistence_status"] = "mongo_resume"
                        redis.mark(record_id, "complete")
                        LOG.info("Step 7 Mongo resume hit id=%s context_sha256=%s", record_id, context_hash)
                        continue

                    raw, parsed = analyze_context(context)
                    records = _statement_records(
                        parsed, row, context_hash=context_hash,
                        entity_lookup=resolution_lookup(row.get("ep24_entity_resolution_json")),
                    )
                    counts = summarize(records)
                    generated_at = datetime.now(timezone.utc).isoformat()

                    out.at[index, "dna_analysis_markdown"] = parsed.analysis_markdown
                    out.at[index, "dna_statements_json"] = json.dumps(records, ensure_ascii=False)
                    out.at[index, "dna_prompt_version"] = PROMPT_VERSION
                    out.at[index, "dna_model_metadata_json"] = json.dumps(
                        _model_metadata(), ensure_ascii=False, sort_keys=True
                    )
                    out.at[index, "dna_generated_at"] = generated_at
                    out.at[index, "dna_context_sha256"] = context_hash
                    out.at[index, "dna_context_truncated"] = json.dumps(truncated)
                    out.at[index, "dna_statement_count"] = counts["statement_count"]
                    out.at[index, "dna_binary_count"] = counts["binary_count"]
                    out.at[index, "dna_uncertain_count"] = counts["uncertain_count"]
                    out.at[index, "dna_actor_unresolved_count"] = counts["actor_unresolved_count"]
                    out.at[index, "dna_concept_novel_count"] = counts["concept_novel_count"]
                    out.at[index, "dna_codebook_fingerprint"] = _text(
                        (context_handle[2].get("codebook_fingerprint", "") if context_handle else "")
                        or row.get("ep24_codebook_fingerprint", "")
                    )
                    out.at[index, "dna_memory_context_json"] = _text(row.get("memory_context_json", ""))
                    out.at[index, "dna_rag_context_json"] = _text(row.get("rag_context_json", ""))
                    out.at[index, "dna_raw_response"] = raw
                    out.at[index, "dna_status"] = "ok"
                    out.at[index, "dna_error"] = ""
                    out.at[index, "dna_runtime_seconds"] = f"{time.monotonic() - started:.3f}"

                    LOG.info(
                        "step7 coded country=%s id=%s statements=%s binary=%s uncertain=%s "
                        "actor_unresolved=%s concept_novel=%s",
                        normalized_country, record_id, counts["statement_count"],
                        counts["binary_count"], counts["uncertain_count"],
                        counts["actor_unresolved_count"], counts["concept_novel_count"],
                    )

                    if storage is not None:
                        out.at[index, "dna_persistence_status"] = "mongo_patch:pending"
                        persisted = _persist_row(storage, out.loc[index], country=normalized_country)
                        statement_count = _persist_statements(storage, records)
                        out.at[index, "dna_persistence_status"] = f"mongo_patch:{persisted}"
                        LOG.info(
                            "step7 mongo id=%s cumulative=%s dna_statements=%s",
                            record_id, persisted, statement_count,
                        )
                    else:
                        out.at[index, "dna_persistence_status"] = "mongo_disabled"
                    redis.mark(record_id, "completed")
                except Exception as exc:
                    out.at[index, "dna_status"] = "error"
                    out.at[index, "dna_error"] = f"{type(exc).__name__}: {exc}"
                    if isinstance(exc, Step7ParseError):
                        out.at[index, "dna_raw_response"] = exc.raw_response
                    out.at[index, "dna_runtime_seconds"] = f"{time.monotonic() - started:.3f}"
                    if storage is not None:
                        out.at[index, "dna_persistence_status"] = "mongo_patch:error_record"
                        try:
                            _persist_row(storage, out.loc[index], country=normalized_country)
                        except Exception:
                            LOG.exception("Could not persist Step 7 error state id=%s", record_id)
                    redis.mark(record_id, "failed")
                    LOG.exception("Step 7 failed country=%s row=%s id=%s", normalized_country, index, record_id)

                processed_since_checkpoint += 1
                if processed_since_checkpoint >= checkpoint_every:
                    write_cumulative_csv(original, out, output_path)
                    LOG.info("CSV checkpoint path=%s row=%s", output_path, index)
                    processed_since_checkpoint = 0

        write_cumulative_csv(original, out, output_path)
        LOG.info("Step 7 complete country=%s output=%s rows=%s", normalized_country, output_path, len(out))
        try:
            exports = write_exports(out, normalized_country, output_path)
            LOG.info("Step 7 exports country=%s %s", normalized_country, exports)
        except Exception:
            LOG.exception("Step 7 export layer failed country=%s (CSV is still written)", normalized_country)
        return output_path
    finally:
        _finalize_context_handle(context_handle, out)


# Historical module-level country set kept for documentation/contract discovery.
countries = ["fi", "pl", "pt", "de", "es", "hu", "hr", "fr", "bg", "sv"]


if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, os.getenv("LACLAUGPT_LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    selection = configure_step_cli(7, sys.argv[1:])
    if selection.remaining_argv:
        raise SystemExit(f"unrecognized arguments: {' '.join(selection.remaining_argv)}")
    if os.getenv("LACLAUGPT_INPUT_CSV"):
        process_country(os.getenv("LACLAUGPT_COUNTRY"))
    else:
        for item in countries:
            process_country(item)
