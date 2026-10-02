#!/usr/bin/env python3
"""Step 7: Leifeld-compatible Discourse Network Analysis extraction for EP24.

Optional/experimental, but production-safe: the stage preserves the cumulative
dataframe, persists to country-scoped Mongo when enabled, uses researcher memory
and RAG only as normalization context, coordinates through optional Redis, and
exports a strict actor-concept-agreement-time event list for DNA/rDNA.

Methodology: Philip Leifeld (2017), "Discourse Network Analysis: Policy Debates
as Dynamic Networks", Oxford Handbook of Political Networks, chapter 25.
https://eprints.gla.ac.uk/121525/
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

from ep24_cli import configure_step_cli
from ep24_db import country_storage
from ep24_dna import (
    DNA_COLUMNS,
    PROMPT_VERSION,
    build_prompt_context,
    enrich_statement,
    statements_to_event_rows,
    utc_now,
    write_eventlist,
)
from ep24_entities import resolution_lookup
from ep24_memory import retrieve_researcher_memory
from ep24_models import ollama_model, ollama_model_source
from ep24_pipeline import load_cumulative_csv, write_cumulative_csv
from ep24_rag import retrieve_stage_rag, upsert_stage_rag
from ep24_redis import RedisCoordinator
from ep24_schema import stable_source_id
from roihu_storage import StorageConfig

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("step_7_roihu_discourse_network_analysis")

DEFAULT_MAX_CONTEXT_CHARS = 30000
DEFAULT_NUM_CTX = 16384
DEFAULT_NUM_PREDICT = 4096


def _models():
    """Import Pydantic models after CLI parsing."""
    from pydantic import BaseModel, Field

    class DNAStatement(BaseModel):
        actor_name: str = Field(description="Actor/person/organization explicitly making the claim")
        actor_type: str | None = None
        concept_label: str = Field(
            description="Analyzable claim, policy position, belief, justification, or narrative"
        )
        proposition: str
        stance: str = Field(description="support, oppose, neutral, mixed, or unknown")
        agreement: bool | None = Field(
            default=None,
            description="true only for explicit support; false only for explicit opposition",
        )
        evidence_quote: str
        evidence_source_fields: list[str] = Field(default_factory=list)
        confidence: float = Field(ge=0.0, le=1.0)

    class DNAResult(BaseModel):
        analysis_markdown: str
        statements: list[DNAStatement] = Field(default_factory=list)

    return DNAResult


SYSTEM = """You are coding statements for Discourse Network Analysis (DNA) following
Philip Leifeld's methodology (2017, Discourse Network Analysis: Policy Debates
as Dynamic Networks).

UNIT OF ANALYSIS
A DNA statement links:
1. one identifiable actor (person or organization);
2. one discourse concept: a claim, policy position, belief, justification,
   narrative, or other theoretically meaningful proposition;
3. an agreement qualifier: positive/support or negative/oppose when explicit;
4. a time/source context supplied by the pipeline.

EVIDENCE DISCIPLINE
- Extract only actor-concept claims supported by CURRENT SOURCE EVIDENCE.
- Do not infer a position merely because an actor/topic/entity co-occurs.
- Do not invent actors, concepts, quotations, or relationships.
- One output statement must correspond to one explicit actor-concept claim.
- Use agreement=true only for explicit support/affirmation.
- Use agreement=false only for explicit opposition/rejection.
- If support/opposition is not explicit enough, agreement=null and preserve the
  uncertainty in stance. Never force neutral/mixed/unknown into a binary value.
- evidence_quote must be a concise span/cue from CURRENT SOURCE EVIDENCE, not
  from memory, RAG, codebooks, or another document.
- evidence_source_fields must name the current-record fields supporting the
  statement when identifiable.

NORMALIZATION CONTEXT
Researcher memory, codebooks, entity normalization, Step 6 Laclau/Palonen
analysis, and retrieved RAG records may help normalize names and concepts or
interpret what to inspect. They are NOT evidence that the current actor made a
claim. Never copy a prior-record claim into the current record.

THEORETICAL BOUNDARIES
- Preserve the distinction between actor and concept.
- Concepts are not arbitrary named entities or generic topics.
- Agreement matters: two actors can mention the same concept with opposite
  positions and must remain distinguishable.
- Step 6 Laclau/Palonen analysis is upstream context; do not replace it and do
  not force every Laclau category into a DNA concept.
- Step 8 Social Network Analysis is a different layer. DNA congruence/conflict
  is not evidence of friendship, coordination, communication, affiliation,
  influence, or hidden social ties.
- Preserve ambiguity and source-grounded temporality.

OUTPUT
Return JSON exactly matching the supplied schema:
1. concise human-readable DNA interpretation;
2. structured statements suitable for projection into the Leifeld/rDNA
   organization-concept-agreement-time model.
"""


def source(lang: str) -> Path | None:
    explicit = os.getenv("LACLAUGPT_INPUT_CSV")
    if explicit:
        path = Path(explicit)
        return path if path.exists() else None
    for path in (Path(f"ep24_{lang}.csv"), Path(f"csv/tiktok_{lang}.csv")):
        if path.exists():
            return path
    return None


def _row_storage_id(row: pd.Series) -> str:
    return str(row.get("_storage_id", "") or "").strip() or stable_source_id(row)


def _retrieval_query(row: pd.Series) -> str:
    fields = (
        "entities",
        "themes",
        "ep24_entity_canonical_names",
        "ep24_theme_canonical_names",
        "asr_translated",
        "asr_transcript",
        "summary_analysis",
        "laclau_summary_md",
        "formula_of_populism_analysis",
    )
    return "\n".join(str(row.get(name, "") or "").strip() for name in fields if str(row.get(name, "") or "").strip())[:12000]


def _prompt_sha(context: str, model: str) -> str:
    return hashlib.sha256(
        f"{PROMPT_VERSION}\n{model}\n{SYSTEM}\n{context}".encode("utf-8")
    ).hexdigest()


def _persist_mongo(
    storage: Any,
    row: pd.Series,
    *,
    source_id: str,
    statements: list[dict[str, Any]],
    model: str,
    context_sha256: str,
) -> tuple[int, int]:
    """Patch cumulative row and store statement-level documents."""
    document = {str(k): v for k, v in row.to_dict().items() if str(k) != "_storage_id"}
    document["_storage_id"] = source_id
    document["step7_dna_provenance"] = {
        "pipeline_stage": "step_7_discourse_network_analysis",
        "method": "Leifeld_DNA",
        "reference": "Leifeld 2017, Oxford Handbook of Political Networks, chapter 25",
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "context_sha256": context_sha256,
        "network_layer": "discourse",
        "timestamp": utc_now(),
    }
    row_count = storage.patch_documents("dataframe", [document])

    statement_docs = []
    for statement in statements:
        doc = dict(statement)
        doc["_storage_id"] = statement["statement_id"]
        statement_docs.append(doc)
    statement_count = storage.upsert_documents("dna_statements", statement_docs)
    return row_count, statement_count


def _eventlist_output(input_path: Path, country: str) -> Path:
    explicit = os.getenv("LACLAUGPT_DNA_EVENTLIST_CSV")
    if explicit:
        return Path(explicit)
    return Path("exports") / "dna" / f"{input_path.stem}_{country}_dna_eventlist.csv"


def run_language(lang: str) -> None:
    import ollama

    DNAResult = _models()
    path = source(lang)
    if path is None:
        LOG.warning("input_missing language=%s", lang)
        return

    output = Path(os.getenv("LACLAUGPT_OUTPUT_CSV") or path)
    df = load_cumulative_csv(path, require_canonical=bool(os.getenv("LACLAUGPT_INPUT_CSV")))
    before = df.drop(columns=[c for c in DNA_COLUMNS if c in df.columns], errors="ignore").copy(deep=True)
    for column in DNA_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    config = StorageConfig.from_env()
    country = config.country or lang or "unknown"
    model = ollama_model()
    limit = int(os.getenv("LACLAUGPT_MAX_ROWS", "0") or 0)
    max_context = int(
        os.getenv("LACLAUGPT_STEP7_MAX_CONTEXT_CHARS", str(DEFAULT_MAX_CONTEXT_CHARS))
    )
    if limit > 0:
        work_indices = list(df.head(limit).index)
    else:
        work_indices = list(df.index)

    storage_cm = country_storage(country) if config.mongo_enabled else None
    storage = storage_cm.__enter__() if storage_cm is not None else None
    redis = RedisCoordinator(country, 7)
    all_statements: list[dict[str, Any]] = []
    stats = {
        "processed": 0,
        "failed": 0,
        "lock_skips": 0,
        "mongo_rows": 0,
        "mongo_statements": 0,
        "rag_writes": 0,
        "exportable": 0,
        "review": 0,
    }
    started = time.monotonic()

    LOG.info(
        "startup step=7 input=%s output=%s country=%s language=%s rows=%d columns=%d "
        "model=%s model_source=%s mongo_enabled=%s max_context_chars=%d",
        path,
        output,
        country,
        lang,
        len(work_indices),
        len(df.columns),
        model,
        ollama_model_source(),
        config.mongo_enabled,
        max_context,
    )
    LOG.debug("incoming_columns=%s", list(before.columns))

    try:
        for ordinal, index in enumerate(work_indices, start=1):
            row = df.loc[index]
            source_id = _row_storage_id(row)
            memory_items: list[dict[str, Any]] = []
            rag_items: list[dict[str, Any]] = []
            retrieval = _retrieval_query(row)

            if storage is not None:
                try:
                    memory_items = retrieve_researcher_memory(storage, retrieval, limit=12)
                    rag_items = retrieve_stage_rag(
                        storage,
                        retrieval,
                        exclude_source_record_id=source_id,
                        limit=6,
                    )
                except Exception:
                    LOG.exception("context_retrieval_failed source_id=%s", source_id)

            context, truncated = build_prompt_context(
                row,
                memory_items=memory_items,
                rag_items=rag_items,
                max_chars=max_context,
            )
            context_sha = _prompt_sha(context, model)
            actor_lookup = resolution_lookup(row.get("ep24_entity_resolution_json"))

            LOG.info(
                "row_start ordinal=%d total=%d index=%s source_id=%s memory_hits=%d "
                "rag_hits=%d entity_aliases=%d context_chars=%d truncated=%s",
                ordinal,
                len(work_indices),
                index,
                source_id,
                len(memory_items),
                len(rag_items),
                len(actor_lookup),
                len(context),
                truncated,
            )

            try:
                with redis.lock(source_id) as acquired:
                    if not acquired:
                        stats["lock_skips"] += 1
                        redis.mark(source_id, "skipped_locked")
                        LOG.info("redis_lock_skip source_id=%s", source_id)
                        continue

                    redis.mark(source_id, "running")
                    response = ollama.chat(
                        model=model,
                        messages=[
                            {"role": "system", "content": SYSTEM},
                            {"role": "user", "content": context},
                        ],
                        format=DNAResult.model_json_schema(),
                        options={
                            "temperature": 0.0,
                            "num_ctx": int(os.getenv("LACLAUGPT_STEP7_NUM_CTX", str(DEFAULT_NUM_CTX))),
                            "num_predict": int(
                                os.getenv("LACLAUGPT_STEP7_NUM_PREDICT", str(DEFAULT_NUM_PREDICT))
                            ),
                        },
                    )
                    raw = str(response["message"]["content"])
                    parsed = DNAResult.model_validate_json(raw)

                    statements: list[dict[str, Any]] = []
                    seen_ids: set[str] = set()
                    for statement in parsed.statements:
                        enriched = enrich_statement(
                            statement.model_dump(),
                            row=row,
                            source_record_id=source_id,
                            country=country,
                            actor_lookup=actor_lookup,
                            memory_items=memory_items,
                            rag_items=rag_items,
                            model=model,
                        )
                        statement_id = enriched["statement_id"]
                        if statement_id in seen_ids:
                            continue
                        seen_ids.add(statement_id)
                        statements.append(enriched)

                    exportable = sum(1 for item in statements if item["exportable_to_dna"])
                    review = len(statements) - exportable
                    all_statements.extend(statements)
                    stats["processed"] += 1
                    stats["exportable"] += exportable
                    stats["review"] += review

                    df.at[index, "dna_analysis_markdown"] = parsed.analysis_markdown
                    df.at[index, "dna_statements_json"] = json.dumps(
                        statements, ensure_ascii=False, sort_keys=True
                    )
                    df.at[index, "dna_exportable_count"] = exportable
                    df.at[index, "dna_review_count"] = review
                    df.at[index, "dna_prompt_version"] = PROMPT_VERSION
                    df.at[index, "dna_model_metadata_json"] = json.dumps(
                        {
                            "model": model,
                            "model_source": ollama_model_source(),
                            "network_layer": "discourse",
                            "methodology": "Leifeld 2017",
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    df.at[index, "dna_generated_at"] = utc_now()
                    df.at[index, "dna_context_sha256"] = context_sha
                    df.at[index, "dna_memory_context_json"] = json.dumps(
                        memory_items, ensure_ascii=False, default=str
                    )
                    df.at[index, "dna_rag_context_json"] = json.dumps(
                        rag_items, ensure_ascii=False, default=str
                    )
                    df.at[index, "dna_status"] = "completed"
                    df.at[index, "dna_error"] = ""
                    df.at[index, "dna_persistence_status"] = (
                        "pending_mongo" if storage is not None else "csv_only"
                    )

                    write_cumulative_csv(before, df, output)
                    LOG.info(
                        "local_checkpoint source_id=%s statements=%d exportable=%d review=%d output=%s",
                        source_id,
                        len(statements),
                        exportable,
                        review,
                        output,
                    )

                    if storage is not None:
                        row_count, statement_count = _persist_mongo(
                            storage,
                            df.loc[index],
                            source_id=source_id,
                            statements=statements,
                            model=model,
                            context_sha256=context_sha,
                        )
                        stats["mongo_rows"] += row_count
                        stats["mongo_statements"] += statement_count
                        rag_count = upsert_stage_rag(
                            storage,
                            df.loc[[index]].assign(_storage_id=source_id),
                            stage="discourse_network_analysis",
                        )
                        stats["rag_writes"] += rag_count
                        df.at[index, "dna_persistence_status"] = "mongo_and_csv"
                        write_cumulative_csv(before, df, output)
                        LOG.info(
                            "mongo_status source_id=%s row_writes=%d statement_writes=%d rag_writes=%d",
                            source_id,
                            row_count,
                            statement_count,
                            rag_count,
                        )

                    redis.mark(source_id, "completed")
            except Exception as exc:
                stats["failed"] += 1
                redis.mark(source_id, "failed")
                df.at[index, "dna_status"] = "failed"
                df.at[index, "dna_error"] = str(exc)
                write_cumulative_csv(before, df, output)
                LOG.exception("row_failed source_id=%s index=%s error=%s", source_id, index, exc)

        # Include valid statements from untouched rows too, making rerun/limited
        # event exports complete for the cumulative CSV.
        if len(work_indices) < len(df):
            for index in df.index:
                if index in work_indices:
                    continue
                try:
                    existing = json.loads(str(df.at[index, "dna_statements_json"] or "[]"))
                    if isinstance(existing, list):
                        all_statements.extend(item for item in existing if isinstance(item, dict))
                except (TypeError, ValueError, json.JSONDecodeError):
                    pass

        event_path = _eventlist_output(path, country)
        write_eventlist(all_statements, event_path)
        for index in work_indices:
            if str(df.at[index, "dna_status"]) == "completed":
                df.at[index, "dna_eventlist_path"] = str(event_path)
        write_cumulative_csv(before, df, output)

        LOG.info(
            "eventlist_written path=%s rows=%d",
            event_path,
            len(statements_to_event_rows(all_statements)),
        )
    finally:
        if storage_cm is not None:
            storage_cm.__exit__(None, None, None)
            LOG.debug("mongo_closed country=%s", country)

    LOG.info(
        "complete step=7 processed=%d failed=%d lock_skips=%d exportable=%d review=%d "
        "mongo_rows=%d mongo_statements=%d rag_writes=%d elapsed_seconds=%.3f",
        stats["processed"],
        stats["failed"],
        stats["lock_skips"],
        stats["exportable"],
        stats["review"],
        stats["mongo_rows"],
        stats["mongo_statements"],
        stats["rag_writes"],
        time.monotonic() - started,
    )


if __name__ == "__main__":
    selection = configure_step_cli(7, sys.argv[1:])
    if selection.remaining_argv:
        raise SystemExit(f"unrecognized arguments: {' '.join(selection.remaining_argv)}")
    if os.getenv("LACLAUGPT_INPUT_CSV"):
        run_language("")
    else:
        for lang in os.getenv(
            "LACLAUGPT_LANGUAGES", "fi,sv,pl,pt,de,es,hu,hr,fr,bg,en"
        ).split(","):
            run_language(lang.strip())
