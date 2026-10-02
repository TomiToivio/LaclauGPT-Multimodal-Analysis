"""Step 6: evidence-linked Laclau/Mouffe/Palonen discourse analysis for EP24.

The rich structured JSON is the source of truth. Historical
formula_of_populism_* text columns are compatibility projections for downstream
RDF/legacy consumers and are derived deterministically from evidenced affects.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from ep24_db import country_storage
from ep24_memory import retrieve_researcher_memory
from ep24_models import ollama_model, ollama_model_source
from ep24_pipeline import load_cumulative_csv, metadata_context, write_cumulative_csv
from ep24_rag import retrieve_stage_rag, upsert_stage_rag
from ep24_redis import RedisCoordinator
from ep24_schema import stable_source_id
from roihu_storage import StorageConfig

logger = logging.getLogger(__name__)

PROMPT_VERSION = "ep24-laclau-palonen-v3"
DEFAULT_NUM_CTX = 32768
DEFAULT_NUM_PREDICT = 4096
DEFAULT_MAX_CONTEXT_CHARS = 48000
CACHE_PATH = Path(os.getenv("LACLAUGPT_POPULISM_SQLITE", "./database/formula_of_populism.db"))

COUNTRY_CODEBOOK = {
    "finland": ("FI", "fi"),
    "sweden": ("SE", "sv"),
    "poland": ("PL", "pl"),
    "portugal": ("PT", "pt"),
    "germany": ("DE", "de"),
    "spain": ("ES", "es"),
    "hungary": ("HU", "hu"),
    "croatia": ("HR", "hr"),
    "france": ("FR", "fr"),
    "bulgaria": ("BG", "bg"),
    "fi": ("FI", "fi"),
    "sv": ("SE", "sv"),
    "pl": ("PL", "pl"),
    "pt": ("PT", "pt"),
    "de": ("DE", "de"),
    "es": ("ES", "es"),
    "hu": ("HU", "hu"),
    "hr": ("HR", "hr"),
    "fr": ("FR", "fr"),
    "bg": ("BG", "bg"),
}

OUTPUT_COLUMNS = (
    "formula_of_populism_analysis",
    "formula_of_populism_us",
    "formula_of_populism_frontier",
    "laclau_summary_md",
    "laclau_structured_json",
    "laclau_raw_response",
    "laclau_status",
    "laclau_error",
    "laclau_prompt_version",
    "laclau_model",
    "laclau_context_sha256",
    "laclau_generated_at",
    "formula_of_populism_codebook_context_json",
    "formula_of_populism_codebook_fingerprint",
)


class DiscourseParseError(ValueError):
    def __init__(self, message: str, *, raw_response: str, metadata: dict[str, Any] | None = None):
        super().__init__(message)
        self.raw_response = raw_response
        self.metadata = dict(metadata or {})


class UsConstruct(BaseModel):
    label: str
    demands: list[str] = Field(default_factory=list)
    text_span: str
    confidence: float = Field(ge=0.0, le=1.0)
    provenance: str = "current_source"


class FrontierConstruct(BaseModel):
    us_side: str | None = None
    them_side: str
    relation: Literal[
        "opposition",
        "criticism",
        "blame",
        "threat_construction",
        "exclusion",
        "boundary_construction",
        "antagonistic_boundary",
    ]
    text_span: str
    confidence: float = Field(ge=0.0, le=1.0)
    provenance: str = "current_source"


class AffectObservation(BaseModel):
    affect: str
    target: str | None = None
    text_span: str
    confidence: float = Field(ge=0.0, le=1.0)
    provenance: str = "current_source"


class RelationCandidate(BaseModel):
    relation: Literal["equivalence", "difference", "articulation"]
    left: str
    right: str
    text_span: str
    confidence: float = Field(ge=0.0, le=1.0)


class SignifierCandidate(BaseModel):
    label: str
    role: Literal["nodal", "floating", "empty"]
    text_span: str
    confidence: float = Field(ge=0.0, le=1.0)
    caveat: str = ""


class PopulistDynamicEvidence(BaseModel):
    dynamic: Literal["fringe", "mainstream", "competing"]
    text_span: str
    confidence: float = Field(ge=0.0, le=1.0)
    caveat: str = ""


class EP24DiscourseResult(BaseModel):
    analysis_markdown: str
    us_constructs: list[UsConstruct] = Field(default_factory=list)
    frontier_constructs: list[FrontierConstruct] = Field(default_factory=list)
    affects: list[AffectObservation] = Field(default_factory=list)
    relations: list[RelationCandidate] = Field(default_factory=list)
    signifier_candidates: list[SignifierCandidate] = Field(default_factory=list)
    populist_dynamic_evidence: list[PopulistDynamicEvidence] = Field(default_factory=list)
    formula_minimum_conditions_met: bool = False
    hegemonic_evidence_candidates: list[str] = Field(default_factory=list)
    rhetoric_performative_observations: list[str] = Field(default_factory=list)
    counter_evidence: list[str] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)
    prompt_version: str = PROMPT_VERSION
    model_metadata: dict[str, Any] = Field(default_factory=dict)
    generated_at: str | None = None


SYSTEM_PROMPT = """You are LaclauGPT, a University of Helsinki social-science research assistant.
Analyze one EP24 TikTok/Instagram record from the 2024 European Parliament elections using the
generic discourse-theoretical framework of Ernesto Laclau, Laclau & Mouffe, and Emilia Palonen.
Return JSON only and conform exactly to the supplied schema.

This is evidence-first document-level coding. Candidate interpretations are provisional and
human-reviewable, never final theoretical facts.

THEORETICAL RULES
- Populism is a political logic, not a permanent party or actor label.
- Do not force an Us, Frontier, affect, chain, signifier role, or populist dynamic.
- A plural pronoun alone is not a meaningful collective Us.
- A disliked entity, criticism, blame, or negative sentiment is not automatically an antagonistic Frontier.
  Use the relation field to distinguish ordinary opposition/criticism/blame/threat/exclusion/boundary
  construction from an antagonistic boundary.
- Affect is affective investment, not detachable sentiment. Never map Us -> positive or Frontier -> negative.
  Anger can invest an Us; admiration can concern an opponent; ambivalence can matter.
  If affect is not evidenced, omit it.
- Co-occurrence is not a chain of equivalence. Only code equivalence/difference/articulation when the
  current source actually constructs the relation.
- Frequency/prominence is not hegemony. A single document cannot establish hegemony.
- Polysemy alone is not floating signification. One heterogeneous use is not enough to establish an
  empty signifier. Nodal/floating/empty labels are document-level candidates only and require evidence.
- A two-sided disagreement is not automatically political polarisation. Persistent bipolar hegemony
  is a corpus-level claim.
- Palonen's fringe/mainstream/competing populist dynamics are relational heuristics, not party labels.
  Only emit provisional evidence when the current document supports it.
- Analyze rhetoric performatively where evidenced: what does naming, metaphor, contrast, repetition,
  or other rhetoric connect, constitute, exclude, or make represent a wider chain?
- Every substantive candidate must be grounded in a short span from CURRENT SOURCE EVIDENCE.
- Researcher memory, codebooks, upstream model analyses, and RAG context can assist interpretation and
  normalization but are NOT direct evidence that a phenomenon occurs in the current record.
- Human entities/themes are authoritative canonical spellings for normalization, but do not force a
  theoretical interpretation.
- Step-5 sentiment fields are auxiliary context only and are never sufficient evidence of affective investment.
- Empty lists and formula_minimum_conditions_met=false are valid and expected for non-populist or weakly
  evidenced material.
- formula_minimum_conditions_met should be true only when the current source supports both a meaningful
  collective Us and a genuine antagonistic/boundary Frontier. It is a document-level evidentiary condition,
  not a permanent label or score.
"""


def _clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {"nan", "none", "null"} else text


def _source_date(row: pd.Series) -> str | None:
    for key in ("recording_date", "recording_datetime", "create_time", "date"):
        value = _clean(row.get(key, ""))
        if value:
            return value
    return None


def _row_storage_id(row: pd.Series) -> str:
    return _clean(row.get("_storage_id", "")) or stable_source_id(row)


def _bounded(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def _current_source_evidence(row: pd.Series) -> str:
    fields = (
        "asr_translated",
        "asr_transcript",
        "ocr_1",
        "frame_analysis_1",
        "vllm_video_analysis",
        "vllm_video_markdown_analysis",
        "vllm_video_structured_json",
    )
    blocks = []
    for key in fields:
        value = _clean(row.get(key, ""))
        if value:
            blocks.append(f"### {key}\n{value}")
    # Source/platform metadata excluding model-derived fields remains factual context.
    source_meta = metadata_context(row, include_model_fields=False)
    if source_meta.strip():
        blocks.append("### source_and_researcher_metadata\n" + source_meta)
    return "\n\n".join(blocks)


def _derived_prior_analysis(row: pd.Series) -> str:
    fields = (
        "summary_analysis",
        "postprocess_summary_md",
        "postprocess_entities",
        "postprocess_themes",
        "positive",
        "neutral",
        "negative",
        "ep24_entity_canonical_names",
        "ep24_theme_canonical_names",
    )
    blocks = []
    for key in fields:
        value = _clean(row.get(key, ""))
        if value:
            blocks.append(f"- {key}: {value}")
    return "\n".join(blocks)


def _format_memory(items: list[dict]) -> str:
    if not items:
        return "<none>"
    lines = ["Normalization context only; not source evidence."]
    for item in items:
        lines.append(
            f"- {item.get('kind', 'item')}: {item.get('label', '')} "
            f"[role={item.get('evidence_role', 'normalization_context_not_source_evidence')}]"
        )
    return "\n".join(lines)


def _format_rag(items: list[dict]) -> str:
    if not items:
        return "<none>"
    lines = ["Prior-corpus/model context only; not source evidence."]
    for item in items:
        excerpt = _clean(item.get("text", "")).replace("\n", " ")[:1000]
        lines.append(
            f"- stage={item.get('stage', '')} source_record_id={item.get('source_record_id', '')}: {excerpt}"
        )
    return "\n".join(lines)


_CODEBOOK_CACHE: dict[tuple[str, str, str], tuple[Any, Any]] = {}


def _codebook_context(country: str, query: str) -> tuple[str, str, str]:
    code, language = COUNTRY_CODEBOOK.get(country.casefold(), ("", ""))
    if not code:
        return "<none>", "", ""
    try:
        from roihu_codebooks import context_block, load_profile

        private_root = os.getenv("LACLAUGPT_MULTIMODAL_PRIVATE_ROOT", ".")
        key = (private_root, code, language)
        if key not in _CODEBOOK_CACHE:
            _CODEBOOK_CACHE[key] = load_profile(private_root, code, language=language)
        entries, profile = _CODEBOOK_CACHE[key]
        block, selection = context_block(query, entries, country=code, language=language)
        return block or "<none>", json.dumps(selection, ensure_ascii=False, sort_keys=True), str(profile.get("fingerprint", ""))
    except Exception as exc:
        logger.warning("codebook_context_unavailable country=%s error=%s", country, exc)
        return "<none>", "", ""


def build_step6_context(
    row: pd.Series,
    *,
    memory_items: list[dict] | None = None,
    rag_items: list[dict] | None = None,
    codebook_block: str = "<none>",
    max_chars: int | None = None,
) -> tuple[str, dict[str, Any]]:
    max_chars = max_chars or int(os.getenv("LACLAUGPT_DISCOURSE_MAX_CHARS", str(DEFAULT_MAX_CONTEXT_CHARS)))
    source = _current_source_evidence(row)
    prior = _derived_prior_analysis(row)
    memory = _format_memory(memory_items or [])
    rag = _format_rag(rag_items or [])
    sections = [
        ("CURRENT SOURCE EVIDENCE — the only direct evidence for document-level coding", source or "<none>"),
        ("DERIVED PRIOR-STAGE ANALYSIS — context only, not direct source evidence", prior or "<none>"),
        ("RESEARCHER/CODEBOOK MEMORY — normalization/background only, not source evidence", memory + "\n\n" + codebook_block),
        ("RETRIEVED CORPUS CONTEXT — comparison only, not source evidence", rag),
    ]
    full = "\n\n".join(f"## {title}\n{body}" for title, body in sections)
    bounded, truncated = _bounded(full, max_chars)
    metadata = {
        "original_chars": len(full),
        "sent_chars": len(bounded),
        "truncated": truncated,
        "max_chars": max_chars,
        "source_date": _source_date(row),
    }
    return bounded, metadata


def _context_hash(system_prompt: str, context: str, model: str) -> str:
    payload = f"{PROMPT_VERSION}\n{model}\n{system_prompt}\n{context}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def analyze_context(context: str, *, model: str | None = None) -> tuple[str, EP24DiscourseResult]:
    selected_model = model or ollama_model()
    num_ctx = int(os.getenv("LACLAUGPT_DISCOURSE_NUM_CTX", str(DEFAULT_NUM_CTX)))
    num_predict = int(os.getenv("LACLAUGPT_DISCOURSE_NUM_PREDICT", str(DEFAULT_NUM_PREDICT)))
    import ollama

    response = ollama.chat(
        model=selected_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ],
        format=EP24DiscourseResult.model_json_schema(),
        options={
            "temperature": 0.0,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
    )
    raw = str(response["message"]["content"])
    request_metadata = {
        "model": selected_model,
        "prompt_version": PROMPT_VERSION,
        "num_ctx": num_ctx,
        "num_predict": num_predict,
    }
    try:
        parsed = EP24DiscourseResult.model_validate_json(raw)
    except Exception as exc:
        raise DiscourseParseError(
            f"Step 6 structured output validation failed: {exc}",
            raw_response=raw,
            metadata=request_metadata,
        ) from exc
    parsed.prompt_version = PROMPT_VERSION
    parsed.model_metadata = {
        "provider": "ollama",
        "model": selected_model,
        "model_source": ollama_model_source(),
        "num_ctx": num_ctx,
        "num_predict": num_predict,
    }
    parsed.generated_at = datetime.now(timezone.utc).isoformat()
    return raw, parsed


def _affects_for_target(result: EP24DiscourseResult, target: str) -> list[AffectObservation]:
    key = target.casefold().strip()
    return [
        item for item in result.affects
        if _clean(item.target).casefold() == key and _clean(item.affect)
    ]


def legacy_formula_projection(result: EP24DiscourseResult) -> tuple[str, str]:
    """Project rich candidates to historical element^affect lines without fabricating affect."""
    us_lines: list[str] = []
    frontier_lines: list[str] = []
    for item in result.us_constructs:
        for affect in _affects_for_target(result, item.label):
            line = f"{item.label}^{affect.affect}"
            if line not in us_lines:
                us_lines.append(line)
    for item in result.frontier_constructs:
        for affect in _affects_for_target(result, item.them_side):
            line = f"{item.them_side}^{affect.affect}"
            if line not in frontier_lines:
                frontier_lines.append(line)
    return (
        "\n".join(us_lines) + ("\n" if us_lines else ""),
        "\n".join(frontier_lines) + ("\n" if frontier_lines else ""),
    )


def _open_cache() -> sqlite3.Connection:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CACHE_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS discourse_cache (
            source_id TEXT NOT NULL,
            model TEXT NOT NULL,
            context_sha256 TEXT NOT NULL,
            raw_response TEXT NOT NULL,
            structured_json TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (source_id, model, context_sha256)
        )"""
    )
    conn.commit()
    return conn


def _cache_lookup(conn: sqlite3.Connection, source_id: str, model: str, context_sha256: str) -> tuple[str, EP24DiscourseResult] | None:
    row = conn.execute(
        """SELECT raw_response, structured_json FROM discourse_cache
           WHERE source_id=? AND model=? AND context_sha256=?""",
        (source_id, model, context_sha256),
    ).fetchone()
    if row is None:
        return None
    return str(row[0]), EP24DiscourseResult.model_validate_json(str(row[1]))


def _cache_store(
    conn: sqlite3.Connection,
    *,
    source_id: str,
    model: str,
    context_sha256: str,
    raw_response: str,
    result: EP24DiscourseResult,
) -> None:
    conn.execute(
        """INSERT INTO discourse_cache
           (source_id, model, context_sha256, raw_response, structured_json, updated_at)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(source_id, model, context_sha256) DO UPDATE SET
             raw_response=excluded.raw_response,
             structured_json=excluded.structured_json,
             updated_at=excluded.updated_at""",
        (
            source_id,
            model,
            context_sha256,
            raw_response,
            result.model_dump_json(),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()


def _mongo_resume(storage, source_id: str, *, model: str, context_sha256: str) -> tuple[str, EP24DiscourseResult] | None:
    docs = storage.find("dataframe", {"_storage_id": source_id}, limit=1)
    if not docs:
        return None
    doc = docs[0]
    if _clean(doc.get("laclau_model")) != model or _clean(doc.get("laclau_context_sha256")) != context_sha256:
        return None
    structured = _clean(doc.get("laclau_structured_json"))
    if not structured or _clean(doc.get("laclau_status")) != "ok":
        return None
    return _clean(doc.get("laclau_raw_response")), EP24DiscourseResult.model_validate_json(structured)


def _apply_result(
    df: pd.DataFrame,
    index: Any,
    *,
    result: EP24DiscourseResult,
    raw_response: str,
    model: str,
    context_sha256: str,
    codebook_context_json: str,
    codebook_fingerprint: str,
) -> None:
    us_text, frontier_text = legacy_formula_projection(result)
    df.at[index, "formula_of_populism_analysis"] = result.analysis_markdown
    df.at[index, "formula_of_populism_us"] = us_text
    df.at[index, "formula_of_populism_frontier"] = frontier_text
    df.at[index, "laclau_summary_md"] = result.analysis_markdown
    df.at[index, "laclau_structured_json"] = result.model_dump_json()
    df.at[index, "laclau_raw_response"] = raw_response
    df.at[index, "laclau_status"] = "ok"
    df.at[index, "laclau_error"] = ""
    df.at[index, "laclau_prompt_version"] = PROMPT_VERSION
    df.at[index, "laclau_model"] = model
    df.at[index, "laclau_context_sha256"] = context_sha256
    df.at[index, "laclau_generated_at"] = result.generated_at or datetime.now(timezone.utc).isoformat()
    df.at[index, "formula_of_populism_codebook_context_json"] = codebook_context_json
    df.at[index, "formula_of_populism_codebook_fingerprint"] = codebook_fingerprint


def _apply_error(
    df: pd.DataFrame,
    index: Any,
    *,
    error: Exception,
    raw_response: str,
    model: str,
    context_sha256: str,
) -> None:
    df.at[index, "laclau_status"] = "error"
    df.at[index, "laclau_error"] = str(error)
    df.at[index, "laclau_raw_response"] = raw_response
    df.at[index, "laclau_prompt_version"] = PROMPT_VERSION
    df.at[index, "laclau_model"] = model
    df.at[index, "laclau_context_sha256"] = context_sha256
    df.at[index, "laclau_generated_at"] = datetime.now(timezone.utc).isoformat()


def _persist_mongo_row(storage, row: pd.Series, source_id: str) -> int:
    document = {str(k): v for k, v in row.to_dict().items() if str(k) != "_storage_id"}
    document["_storage_id"] = source_id
    document["step6_discourse_provenance"] = {
        "pipeline_stage": "step_6_discourse_analysis",
        "prompt_version": PROMPT_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return storage.patch_documents("dataframe", [document])


def run_step6(country: str | None = None) -> None:
    started = time.monotonic()
    filename = os.getenv("LACLAUGPT_INPUT_CSV") or f"ep24_{country}.csv"
    output = os.getenv("LACLAUGPT_OUTPUT_CSV") or filename
    max_rows = int(os.getenv("LACLAUGPT_MAX_ROWS", "0") or 0)
    checkpoint_every = max(1, int(os.getenv("LACLAUGPT_CHECKPOINT_EVERY", "10")))
    model = ollama_model()

    df = load_cumulative_csv(filename, require_canonical=bool(os.getenv("LACLAUGPT_INPUT_CSV")))
    if max_rows > 0:
        if Path(filename).resolve() == Path(output).resolve() and max_rows < len(df):
            raise ValueError("Refusing to truncate the input CSV: with LACLAUGPT_MAX_ROWS set, LACLAUGPT_OUTPUT_CSV must be a different path")
        df = df.head(max_rows).copy()

    before = df.drop(columns=[c for c in OUTPUT_COLUMNS if c in df.columns], errors="ignore").copy(deep=True)
    for column in OUTPUT_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    config = StorageConfig.from_env()
    active_country = (country or config.country or "").casefold()
    redis = RedisCoordinator(active_country, 6)
    connection = _open_cache()
    storage_cm = country_storage(active_country) if config.mongo_enabled else None
    storage = storage_cm.__enter__() if storage_cm is not None else None

    stats = {
        "processed": 0,
        "failed": 0,
        "cache_hits": 0,
        "mongo_resume_hits": 0,
        "mongo_writes": 0,
        "rag_writes": 0,
        "lock_skips": 0,
    }
    logger.info(
        "step6_start input=%s output=%s country=%s rows=%d columns=%d model=%s model_source=%s mongo=%s redis=%s",
        filename, output, active_country, len(df), len(df.columns), model, ollama_model_source(),
        config.mongo_enabled, bool(redis.client),
    )

    try:
        for ordinal, (index, row) in enumerate(df.iterrows(), start=1):
            source_id = _row_storage_id(row)
            retrieval_seed = "\n".join(
                _clean(row.get(key, ""))
                for key in ("entities", "themes", "summary_analysis", "asr_translated", "asr_transcript")
                if _clean(row.get(key, ""))
            )
            memory_items: list[dict] = []
            rag_items: list[dict] = []
            if storage is not None:
                try:
                    memory_items = retrieve_researcher_memory(storage, retrieval_seed, limit=16)
                    rag_items = retrieve_stage_rag(
                        storage,
                        retrieval_seed,
                        exclude_source_record_id=source_id,
                        limit=8,
                    )
                except Exception:
                    logger.exception("step6_context_retrieval_failed source_id=%s", source_id)

            codebook_block, codebook_json, codebook_fingerprint = _codebook_context(active_country, retrieval_seed)
            context, context_meta = build_step6_context(
                row,
                memory_items=memory_items,
                rag_items=rag_items,
                codebook_block=codebook_block,
            )
            context_sha256 = _context_hash(SYSTEM_PROMPT, context, model)
            logger.info(
                "step6_row ordinal=%d/%d source_id=%s context_chars=%d truncated=%s memory=%d rag=%d codebook=%s hash=%s",
                ordinal, len(df), source_id, context_meta["sent_chars"], context_meta["truncated"],
                len(memory_items), len(rag_items), bool(codebook_json), context_sha256,
            )

            raw_response = ""
            try:
                with redis.lock(source_id) as acquired:
                    if not acquired:
                        stats["lock_skips"] += 1
                        redis.mark(source_id, "skipped_locked")
                        continue
                    redis.mark(source_id, "running")

                    cached = None
                    if storage is not None:
                        cached = _mongo_resume(storage, source_id, model=model, context_sha256=context_sha256)
                        if cached is not None:
                            stats["mongo_resume_hits"] += 1
                    if cached is None:
                        cached = _cache_lookup(connection, source_id, model, context_sha256)
                    if cached is not None:
                        raw_response, result = cached
                        stats["cache_hits"] += 1
                    else:
                        raw_response, result = analyze_context(context, model=model)
                        _cache_store(
                            connection,
                            source_id=source_id,
                            model=model,
                            context_sha256=context_sha256,
                            raw_response=raw_response,
                            result=result,
                        )

                    _apply_result(
                        df,
                        index,
                        result=result,
                        raw_response=raw_response,
                        model=model,
                        context_sha256=context_sha256,
                        codebook_context_json=codebook_json,
                        codebook_fingerprint=codebook_fingerprint,
                    )
                    stats["processed"] += 1

                    if storage is not None:
                        stats["mongo_writes"] += _persist_mongo_row(storage, df.loc[index], source_id)
                        rag_df = df.loc[[index]].copy()
                        rag_df["_storage_id"] = source_id
                        stats["rag_writes"] += upsert_stage_rag(storage, rag_df, stage="discourse_analysis")
                    redis.mark(source_id, "completed")

                    if ordinal % checkpoint_every == 0:
                        write_cumulative_csv(before, df, output)
                        logger.info("step6_checkpoint ordinal=%d output=%s", ordinal, output)

            except Exception as exc:
                stats["failed"] += 1
                if isinstance(exc, DiscourseParseError):
                    raw_response = exc.raw_response
                _apply_error(
                    df,
                    index,
                    error=exc,
                    raw_response=raw_response,
                    model=model,
                    context_sha256=context_sha256,
                )
                redis.mark(source_id, "failed")
                logger.exception("step6_row_failed source_id=%s error=%s", source_id, exc)

        write_cumulative_csv(before, df, output)
    finally:
        connection.close()
        if storage_cm is not None:
            storage_cm.__exit__(None, None, None)

    logger.info(
        "step6_complete processed=%d failed=%d cache_hits=%d mongo_resume_hits=%d mongo_writes=%d rag_writes=%d lock_skips=%d elapsed=%.3f output=%s",
        stats["processed"], stats["failed"], stats["cache_hits"], stats["mongo_resume_hits"],
        stats["mongo_writes"], stats["rag_writes"], stats["lock_skips"], time.monotonic() - started, output,
    )


# Historical API retained for callers/tests.
def get_formula_of_populism(country: str | None = None) -> None:
    run_step6(country)


countries = ["fi", "sv", "pl", "pt", "de", "es", "hu", "hr", "fr", "bg"]


if __name__ == "__main__":
    if os.getenv("LACLAUGPT_INPUT_CSV"):
        run_step6(None)
    else:
        for country in countries:
            run_step6(country)
