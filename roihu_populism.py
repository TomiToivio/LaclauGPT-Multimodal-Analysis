from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from ep24_models import ollama_model, ollama_model_source
from ep24_pipeline import ensure_columns, load_cumulative_csv, write_cumulative_csv
from ep24_redis import RedisCoordinator
from ep24_schema import stable_source_id

LOG = logging.getLogger("ep24.step6")

PROMPT_VERSION = "ep24-laclau-palonen-v3-evidence-packet"
DEFAULT_MAX_CONTEXT_CHARS = 28000
DEFAULT_NUM_CTX = 16384
DEFAULT_NUM_PREDICT = 8192

STEP6_COLUMNS = (
    "formula_of_populism_analysis",
    "formula_of_populism_us",
    "formula_of_populism_frontier",
    "laclau_summary_md",
    "laclau_structured_json",
    "laclau_formula_conditions_met",
    "laclau_abstention_reason",
    "laclau_prompt_version",
    "laclau_model_metadata_json",
    "laclau_generated_at",
    "laclau_context_sha256",
    "laclau_context_truncated",
    "laclau_codebook_fingerprint",
    "laclau_codebook_context_json",
    "laclau_memory_context_json",
    "laclau_rag_context_json",
    "laclau_raw_response",
    "laclau_status",
    "laclau_error",
    "laclau_runtime_seconds",
    "laclau_persistence_status",
)

COUNTRY_ALIASES = {
    "fi": "finland", "finland": "finland",
    "pl": "poland", "poland": "poland",
    "pt": "portugal", "portugal": "portugal",
    "de": "germany", "germany": "germany",
    "es": "spain", "spain": "spain",
    "hu": "hungary", "hungary": "hungary",
    "hr": "croatia", "croatia": "croatia",
    "fr": "france", "france": "france",
    "bg": "bulgaria", "bulgaria": "bulgaria",
    "sv": "sweden", "se": "sweden", "sweden": "sweden",
}


class Step6ParseError(ValueError):
    def __init__(self, message: str, *, raw_response: str):
        super().__init__(message)
        self.raw_response = raw_response


class EvidenceCandidate(BaseModel):
    label: str
    text_span: str
    confidence: float = Field(ge=0.0, le=1.0)
    provenance: str = "current_source"
    uncertainty_notes: list[str] = Field(default_factory=list)
    counter_evidence: list[str] = Field(default_factory=list)


class UsConstruct(EvidenceCandidate):
    demands: list[str] = Field(default_factory=list)
    identities: list[str] = Field(default_factory=list)


class FrontierConstruct(EvidenceCandidate):
    us_side: str | None = None
    them_side: str
    relation: Literal[
        "opposition",
        "criticism",
        "blame",
        "threat_construction",
        "exclusion",
        "boundary_construction",
        "antagonistic_frontier",
    ]


class AffectObservation(EvidenceCandidate):
    affect: str
    target: str | None = None


class ChainRelation(BaseModel):
    relation_type: Literal["equivalence", "difference"]
    members: list[str] = Field(default_factory=list)
    text_span: str
    confidence: float = Field(ge=0.0, le=1.0)
    uncertainty_notes: list[str] = Field(default_factory=list)


class SignifierCandidate(EvidenceCandidate):
    candidate_type: Literal["nodal", "floating", "empty"]
    document_level_only: bool = True


class PalonenDynamicEvidence(BaseModel):
    dynamic: Literal["fringe", "mainstream", "competing"]
    description: str
    text_span: str
    confidence: float = Field(ge=0.0, le=1.0)
    uncertainty_notes: list[str] = Field(default_factory=list)


class RhetoricalPerformance(BaseModel):
    action: Literal[
        "connects_demands",
        "constructs_collective_subject",
        "redefines_frontier",
        "represents_wider_chain",
        "reframes_political_possibility",
        "other",
    ]
    description: str
    text_span: str
    confidence: float = Field(ge=0.0, le=1.0)


class EP24DiscourseAnalysis(BaseModel):
    analysis_md: str
    us_constructs: list[UsConstruct] = Field(default_factory=list)
    frontier_constructs: list[FrontierConstruct] = Field(default_factory=list)
    affects: list[AffectObservation] = Field(default_factory=list)
    chains: list[ChainRelation] = Field(default_factory=list)
    signifier_candidates: list[SignifierCandidate] = Field(default_factory=list)
    rhetorical_performances: list[RhetoricalPerformance] = Field(default_factory=list)
    palonen_dynamic_evidence: list[PalonenDynamicEvidence] = Field(default_factory=list)
    hegemonic_evidence_candidates: list[EvidenceCandidate] = Field(default_factory=list)
    formula_minimum_conditions_met: bool = False
    formula_abstention_reason: str | None = None
    counter_evidence: list[str] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)
    corpus_level_cautions: list[str] = Field(default_factory=list)


SYSTEM_PROMPT = """You are LaclauGPT, a social scientist at the University of Helsinki analysing
2024 European Parliament election TikTok/Instagram material using the generic
Laclau, Mouffe and Palonen framework.

Return JSON only and conform exactly to the supplied schema. Keep analysis_md under 200 words; avoid duplicating JSON fields in prose. Limit each evidence array to the strongest 2-3 source-grounded items, and use empty arrays when unsupported.

EVIDENCE DISCIPLINE
- Analyse the current document evidence first. Never force an Us, Frontier, affect,
  chain, nodal/floating/empty signifier or populist formula.
- Empty lists are valid and preferred when evidence is absent.
- Researcher/codebook memory, prior-stage model analysis and retrieved corpus
  context are context for normalization/comparison only. They are NOT evidence
  that a feature exists in the current document.
- Human entities/themes are authoritative canonical seeds and must not be silently
  renamed in your interpretation.
- Preserve uncertainty and counter-evidence.

THEORY
- Populism is a political logic, not a permanent party/actor label.
- A politically meaningful Us is a collective subject produced through articulation;
  a plural pronoun alone is insufficient.
- Criticism, disagreement, negative sentiment or opponent mention are not by
  themselves an antagonistic frontier. Use the relation labels precisely.
- Affect means affective investment/expression, not detachable sentiment. Never map
  Us mechanically to positive affect or Frontier to negative affect.
- Co-occurrence is not a chain of equivalence. Record difference where relevant.
- Frequency/prominence alone is not a nodal point.
- Polysemy alone is not floating signification.
- One heterogeneous use is not enough to establish an empty signifier.
- Hegemony and bipolar/hegemonic polarisation are corpus-level/dynamic claims and
  MUST NOT be inferred from a single video.
- Palonen fringe/mainstream/competing populist dynamics may be noted only as
  provisional evidence, never as permanent party labels.
- Rhetorical analysis should focus on what articulation performs: connecting demands,
  constructing a collective subject, redefining a frontier, making one signifier
  represent a wider chain, or reframing political possibility.

FORMULA / ABSTENTION
- Extract components independently.
- formula_minimum_conditions_met may be true only when BOTH a politically meaningful
  collective Us and an antagonistic_frontier are supported by current-source evidence.
- If minimum conditions are not met, set it false and explain briefly in
  formula_abstention_reason.
- Never output a populism score.
- Never infer a permanent populist identity for an actor or party.

EVIDENCE SPANS
- Every candidate must contain a concise text_span or source cue grounded in the
  CURRENT SOURCE EVIDENCE section. Do not cite memory/RAG/codebook text as the span.
"""


def _json_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.casefold() == "nan" else text


def _bounded(value: str, limit: int) -> tuple[str, bool]:
    if len(value) <= limit:
        return value, False
    return value[:limit], True


def _external_context(row: pd.Series, column: str, heading: str, role: str) -> str:
    value = _json_text(row.get(column, ""))
    return f"{heading}\nROLE: {role}\n{value or '[]'}"


def _render_fields(row: pd.Series, names: list[str]) -> str:
    lines = []
    for name in names:
        value = _json_text(row.get(name, ""))
        if value:
            lines.append(f"- {name}: {value}")
    return "\n".join(lines) if lines else "- <none>"


def build_prompt_context(row: pd.Series, *, max_chars: int | None = None) -> tuple[str, bool]:
    """Evidence-first, stage-specific input; preserves entire cumulative record.

    Bounded and provenance-labelled by the shared contract (``ep24_prompt_contract``),
    which is the single builder Steps 4-8 use (#241).
    """
    from ep24_prompt_contract import build_stage_packet

    packet = build_stage_packet(6, row, max_chars=max_chars)
    return packet.text, packet.any_truncation


def _prompt_hash(context: str) -> str:
    from ep24_prompt_contract import effective_prompt_fingerprint

    metadata = _model_metadata()
    return effective_prompt_fingerprint(
        SYSTEM_PROMPT, f"{PROMPT_VERSION}\n{context}", metadata["model"],
        {"temperature": metadata["temperature"], "num_ctx": metadata["num_ctx"],
         "num_predict": metadata["num_predict"]},
    )


def _model_metadata() -> dict[str, Any]:
    return {
        "provider": "ollama",
        "model": ollama_model(),
        "model_source": ollama_model_source(),
        "num_ctx": int(os.getenv("LACLAUGPT_STEP6_NUM_CTX", DEFAULT_NUM_CTX)),
        "num_predict": int(os.getenv("LACLAUGPT_STEP6_NUM_PREDICT", DEFAULT_NUM_PREDICT)),
        "temperature": 0.0,
    }


def analyze_context(context: str) -> tuple[str, EP24DiscourseAnalysis]:
    import ollama

    metadata = _model_metadata()
    parsed = None
    raw = ""
    max_attempts = max(1, min(5, int(os.getenv("LACLAUGPT_STEP6_MAX_ATTEMPTS", "3"))))
    for attempt in range(1, max_attempts + 1):
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": context}]
        if attempt > 1:
            messages.append({"role": "user", "content":
                "Retry: output ONE COMPLETE schema-compliant JSON object. "
                "Keep analysis_md below 100 words, use at most 2 items per array, "
                "and retain abstention and uncertainty. Do not invent evidence."})
        response = ollama.chat(
            model=metadata["model"],
            messages=messages,
            format=EP24DiscourseAnalysis.model_json_schema(),
            options={"temperature": 0.0, "num_ctx": metadata["num_ctx"],
                     "num_predict": metadata["num_predict"]},
        )
        raw = response["message"]["content"] or ""
        try:
            parsed = EP24DiscourseAnalysis.model_validate_json(raw)
            LOG.info("Step 6 JSON valid attempt=%s chars=%s done_reason=%s",
                     attempt, len(raw), getattr(response, "done_reason", None))
            break
        except Exception as exc:
            LOG.warning("Step 6 invalid JSON attempt=%s/%s chars=%s done_reason=%s: %s",
                        attempt, max_attempts, len(raw),
                        getattr(response, "done_reason", None), str(exc)[:240])
            if attempt == max_attempts:
                raise Step6ParseError(
                    f"Step 6 JSON invalid after {max_attempts} attempts: {exc}",
                    raw_response=raw) from exc
    assert parsed is not None

    # Mechanical theory guard: a fluent model cannot declare the Formula's
    # minimum conditions met unless both a collective Us and an explicitly
    # antagonistic frontier are present in its own structured evidence.
    minimum_supported = bool(parsed.us_constructs) and any(
        item.relation == "antagonistic_frontier"
        for item in parsed.frontier_constructs
    )
    if not minimum_supported:
        parsed.formula_minimum_conditions_met = False
        if not parsed.formula_abstention_reason:
            parsed.formula_abstention_reason = (
                "Minimum conditions not met: both a collective Us and an "
                "antagonistic frontier require current-source evidence."
            )
    return raw, parsed


def _affect_for(label: str, affects: list[AffectObservation]) -> str:
    target = label.casefold().strip()
    matches = [
        item for item in affects
        if item.target and item.target.casefold().strip() == target
    ]
    if not matches:
        return ""
    matches.sort(key=lambda item: item.confidence, reverse=True)
    return matches[0].affect.strip()


def compatibility_columns(result: EP24DiscourseAnalysis) -> tuple[str, str]:
    """Project the structured result onto the legacy ``element^affect`` surface.

    The legacy columns are a deliberate RDF-compatibility surface, so an element
    that is **evidenced but has no evidenced affect** must still reach it. Two
    wrong answers are available here and both have been in the tree:

    * ``f"{label}^{affect}"`` fabricates an emotion when there is none;
    * dropping the line loses the coding entirely, so abstention costs the
      analysis its finding (#180).

    A bare ``element`` line is neither: it carries the evidenced element and
    states no affect. Both downstream RDF consumers read it that way -- the CSV
    projection emits the coding with no affect triple, and the graph exporter's
    ``parse_populism_elements`` documents the same choice ("a line without the
    separator is kept as an element with no affect rather than dropped, because
    an unparsed researcher value is still a research value"). This stage must not
    import either of them, so the format is agreed by contract, not by import.

    An element with an evidenced affect keeps the historical ``element^affect``
    form unchanged, and an element with no label is skipped.
    """
    us_lines = []
    for item in result.us_constructs:
        label = item.label.strip()
        if not label:
            continue
        affect = _affect_for(item.label, result.affects)
        # An affect containing a newline or a second "^" would make the line
        # parse differently in the two RDF consumers, so it is omitted rather
        # than emitted ambiguously.
        if affect and "^" not in affect and "\n" not in affect:
            us_lines.append(f"{label}^{affect}")
        else:
            us_lines.append(label)

    frontier_lines = []
    for item in result.frontier_constructs:
        them_side = item.them_side.strip()
        if not them_side:
            continue
        affect = _affect_for(item.them_side, result.affects)
        if affect and "^" not in affect and "\n" not in affect:
            frontier_lines.append(f"{them_side}^{affect}")
        else:
            frontier_lines.append(them_side)

    return "\n".join(us_lines), "\n".join(frontier_lines)


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
    return COUNTRY_ALIASES.get(key, key or "unknown")


def _prepare_context(df: pd.DataFrame, country: str) -> tuple[pd.DataFrame, Any | None]:
    """Attach bounded Mongo-backed memory/RAG/codebook context when configured."""
    if os.getenv("LACLAUGPT_MONGO_ENABLED", "0").casefold() not in {"1", "true", "yes", "on"}:
        LOG.warning("MongoDB disabled; Step 6 will run without durable memory/RAG persistence")
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
        enriched = enrich_dataframe(storage, df)
        return enriched, (cm, storage, bootstrap)
    except Exception:
        cm.__exit__(*__import__("sys").exc_info())
        raise


def _persist_row(storage, row: pd.Series, *, country: str) -> int:
    doc = _clean_document(row)
    record_id = str(doc.get("_storage_id") or stable_source_id(row))
    doc["_storage_id"] = record_id
    doc["_pipeline_step_6"] = {
        "status": doc.get("laclau_status"),
        "prompt_version": PROMPT_VERSION,
        "generated_at": doc.get("laclau_generated_at"),
        "context_sha256": doc.get("laclau_context_sha256"),
        "country": country,
    }
    # Patch, never replace, so prior-stage fields cannot be erased.
    return storage.patch_documents("dataframe", [doc])


def _resume_from_mongo(storage, record_id: str, *, context_hash: str) -> dict[str, Any] | None:
    """Return Step 6 fields only when durable provenance exactly matches."""
    docs = storage.find("dataframe", {"_storage_id": record_id}, limit=1)
    if not docs:
        return None
    doc = docs[0]
    if str(doc.get("laclau_status", "")) != "ok":
        return None
    if str(doc.get("laclau_prompt_version", "")) != PROMPT_VERSION:
        return None
    if str(doc.get("laclau_context_sha256", "")) != context_hash:
        return None
    try:
        model_meta = json.loads(str(doc.get("laclau_model_metadata_json", "") or "{}"))
    except json.JSONDecodeError:
        return None
    if str(model_meta.get("model", "")) != ollama_model():
        return None
    return {column: doc.get(column, "") for column in STEP6_COLUMNS}


def _finalize_context_handle(handle: Any | None, df: pd.DataFrame) -> None:
    if not handle:
        return
    cm, storage, _ = handle
    try:
        from ep24_context import update_retrieval
        update_retrieval(storage, df, stage="step_06_discourse_analysis")
    finally:
        cm.__exit__(None, None, None)


def process_country(country: str | None = None) -> Path:
    normalized_country = _country(country)
    input_path = Path(os.getenv("LACLAUGPT_INPUT_CSV") or f"ep24_{country}.csv")
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
    ensure_columns(out, STEP6_COLUMNS)

    enriched, context_handle = _prepare_context(out, normalized_country)
    # Copy only context columns produced by the shared enrichment layer.
    for column in (
        "codebook_context_json",
        "memory_context_json",
        "rag_context_json",
        "entity_normalization_json",
        "theme_normalization_json",
        "context_evidence_role",
    ):
        if column in enriched.columns:
            out[column] = enriched[column]

    redis = RedisCoordinator(normalized_country, 6)
    checkpoint_every = max(1, int(os.getenv("LACLAUGPT_STEP6_CHECKPOINT_EVERY", "10") or 10))
    processed_since_checkpoint = 0

    try:
        storage = context_handle[1] if context_handle else None
        for index, row in out.iterrows():
            record_id = str(row.get("_storage_id") or stable_source_id(row))
            with redis.lock(record_id) as acquired:
                if not acquired:
                    LOG.info("record lock busy id=%s; skipped", record_id)
                    continue

                redis.mark(record_id, "processing")
                started = time.monotonic()
                context, truncated = build_prompt_context(row)
                context_hash = _prompt_hash(context)
                LOG.info(
                    "step6 country=%s row=%s id=%s model=%s prompt=%s context_sha256=%s truncated=%s",
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
                        out.at[index, "laclau_persistence_status"] = "mongo_resume"
                        redis.mark(record_id, "complete")
                        LOG.info(
                            "Step 6 Mongo resume hit country=%s row=%s id=%s context_sha256=%s",
                            normalized_country, index, record_id, context_hash,
                        )
                        continue

                    raw, result = analyze_context(context)
                    us_legacy, frontier_legacy = compatibility_columns(result)
                    generated_at = datetime.now(timezone.utc).isoformat()
                    model_metadata = _model_metadata()

                    out.at[index, "formula_of_populism_analysis"] = result.analysis_md
                    out.at[index, "formula_of_populism_us"] = us_legacy
                    out.at[index, "formula_of_populism_frontier"] = frontier_legacy
                    out.at[index, "laclau_summary_md"] = result.analysis_md
                    out.at[index, "laclau_structured_json"] = result.model_dump_json()
                    out.at[index, "laclau_formula_conditions_met"] = json.dumps(result.formula_minimum_conditions_met)
                    out.at[index, "laclau_abstention_reason"] = result.formula_abstention_reason or ""
                    out.at[index, "laclau_prompt_version"] = PROMPT_VERSION
                    out.at[index, "laclau_model_metadata_json"] = json.dumps(model_metadata, ensure_ascii=False, sort_keys=True)
                    out.at[index, "laclau_generated_at"] = generated_at
                    out.at[index, "laclau_context_sha256"] = context_hash
                    out.at[index, "laclau_context_truncated"] = json.dumps(truncated)
                    bootstrap_fingerprint = context_handle[2].get("codebook_fingerprint", "") if context_handle else ""
                    out.at[index, "laclau_codebook_fingerprint"] = bootstrap_fingerprint or _json_text(row.get("ep24_codebook_fingerprint", ""))
                    out.at[index, "laclau_codebook_context_json"] = _json_text(row.get("codebook_context_json", ""))
                    out.at[index, "laclau_memory_context_json"] = _json_text(row.get("memory_context_json", ""))
                    out.at[index, "laclau_rag_context_json"] = _json_text(row.get("rag_context_json", ""))
                    out.at[index, "laclau_raw_response"] = raw
                    out.at[index, "laclau_status"] = "ok"
                    out.at[index, "laclau_error"] = ""
                    out.at[index, "laclau_runtime_seconds"] = f"{time.monotonic() - started:.3f}"

                    if storage is not None:
                        out.at[index, "laclau_persistence_status"] = "mongo_patch:pending"
                        persisted = _persist_row(storage, out.loc[index], country=normalized_country)
                        out.at[index, "laclau_persistence_status"] = f"mongo_patch:{persisted}"
                    else:
                        out.at[index, "laclau_persistence_status"] = "mongo_disabled"
                    redis.mark(record_id, "complete")
                except Exception as exc:
                    out.at[index, "laclau_status"] = "error"
                    out.at[index, "laclau_error"] = f"{type(exc).__name__}: {exc}"
                    if isinstance(exc, Step6ParseError):
                        out.at[index, "laclau_raw_response"] = exc.raw_response
                    out.at[index, "laclau_runtime_seconds"] = f"{time.monotonic() - started:.3f}"
                    if storage is not None:
                        out.at[index, "laclau_persistence_status"] = "mongo_patch:error_record"
                        try:
                            _persist_row(storage, out.loc[index], country=normalized_country)
                        except Exception:
                            LOG.exception("Could not persist Step 6 error state id=%s", record_id)
                    redis.mark(record_id, "error")
                    LOG.exception("Step 6 failed country=%s row=%s id=%s", normalized_country, index, record_id)

                processed_since_checkpoint += 1
                if processed_since_checkpoint >= checkpoint_every:
                    write_cumulative_csv(original, out, output_path)
                    LOG.info("CSV checkpoint path=%s row=%s", output_path, index)
                    processed_since_checkpoint = 0

        write_cumulative_csv(original, out, output_path)
        LOG.info("Step 6 complete country=%s output=%s rows=%s", normalized_country, output_path, len(out))
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
    if os.getenv("LACLAUGPT_INPUT_CSV"):
        process_country(os.getenv("LACLAUGPT_COUNTRY"))
    else:
        for item in countries:
            process_country(item)