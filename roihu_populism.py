"""Step 6 — EP24 discourse analysis (Laclau / Mouffe / Palonen).

This is the theory-critical analytical stage of the EP24 pipeline. It replaces
the legacy ``formula_of_populism`` classifier, which presupposed a populist
narrative, mechanically mapped Us→positive / Frontier→negative affect and
equated the Frontier with a "disliked enemy".

Design (see ``docs/EP24_STEP6_POPULISM_AUDIT.md`` and ``THEORY.md``):

* evidence-first: the model may abstain; empty outputs are valid;
* candidate-only: one document never establishes hegemony or polarisation;
* affect is affective investment, not sentiment polarity;
* a Frontier is a constructed political boundary, not a disliked entity;
* the ``populist`` verdict is **computed deterministically** from the structured
  output (requires an evidenced Us and an evidenced antagonistic Frontier);
* the local SQLite database is a subordinate restart cache keyed by the stable
  record id plus a prompt/context hash — never an independent source of truth;
* MongoDB (via ``ep24_db``) is the durable cumulative persistence layer;
* memory/RAG/codebook context is explicitly *not* source evidence;
* Redis coordination is optional; the stage runs unchanged without it.

The four legacy Step 6 columns are retained and derived deterministically from
the structured output so downstream DNA/SNA/RDF stages keep working.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field

from ep24_models import ollama_model, ollama_model_source
from ep24_pipeline import (
    ensure_columns,
    load_cumulative_csv,
    metadata_context,
    write_cumulative_csv,
)
from ep24_redis import RedisCoordinator
from ep24_schema import stable_source_id
from ep24_schema import value as ep24_value
from roihu_storage import StorageConfig

logger = logging.getLogger(__name__)

PROMPT_VERSION = "ep24-step6-discourse-v1"
STEP = 6

# --- columns -----------------------------------------------------------------

# The first four are the declared Step 6 contract columns and are kept for
# downstream DNA/SNA/RDF compatibility. Everything after them is the richer
# evidence-linked structure added by this stage.
CONTRACT_COLUMNS = (
    "formula_of_populism_analysis",
    "formula_of_populism_us",
    "formula_of_populism_frontier",
    "laclau_summary_md",
)
STRUCTURED_COLUMNS = (
    "formula_of_populism_json",
    "formula_of_populism_us_json",
    "formula_of_populism_frontier_json",
    "formula_of_populism_affects_json",
    "formula_of_populism_chains_json",
    "formula_of_populism_signifiers_json",
    "formula_of_populism_populist_dynamic_json",
    "formula_of_populism_counter_evidence_json",
    "formula_of_populism_uncertainty_json",
    "formula_of_populism_populist",
    "formula_of_populism_codebook_context_json",
    "formula_of_populism_codebook_fingerprint",
)
PROVENANCE_COLUMNS = (
    "formula_of_populism_prompt_version",
    "formula_of_populism_context_sha256",
    "formula_of_populism_model",
    "formula_of_populism_generated_at",
    "formula_of_populism_status",
)
OUTPUT_COLUMNS = (*CONTRACT_COLUMNS, *STRUCTURED_COLUMNS, *PROVENANCE_COLUMNS)

DB_PATH = Path(os.getenv("LACLAUGPT_STEP6_SQLITE", "./database/formula_of_populism.db"))


def _db_path() -> Path:
    """Resolve the subordinate cache path at call time (not at import time)."""
    return Path(os.getenv("LACLAUGPT_STEP6_SQLITE", "./database/formula_of_populism.db"))

# Evidence field groups (the document under analysis).
TRANSCRIPT_FIELDS = (
    "asr_translated",
    "asr_transcript",
    "whisper_translated",
    "whisper_transcript",
    "whisperResult",
)
FRAME_FIELDS = ("frame_analysis_1", "frame_analysis_2", "frame_analysis_3")
OCR_FIELDS = ("ocr_1", "ocr_2")
VIDEO_FIELDS = ("vllm_video_analysis", "vllm_video_markdown_analysis", "vllm_structured_output")
# Derived prior-stage analysis (context, not new source evidence).
DERIVED_FIELDS = ("summary_analysis", "postprocess_summary_md", "summary_summary_md")
AUXILIARY_FIELDS = ("positive", "neutral", "negative", "ep24_sentiment_targets_json")
# Context-only material (normalization/retrieval, never source evidence).
CONTEXT_FIELDS = (
    "entities",
    "themes",
    "codebook_context_json",
    "memory_context_json",
    "rag_context_json",
    "entity_normalization_json",
    "theme_normalization_json",
)

# The only relation that counts as an antagonistic frontier for the computed
# ``populist`` verdict (THEORY.md §6.2/§6.4, invariant INV_POPULISM).
ANTAGONISTIC_RELATIONS = frozenset({"antagonistic_boundary"})

DEFAULT_CONTEXT_CHARS = 48000

countries = ["fi", "sv", "pl", "pt", "de", "es", "hu", "hr", "fr", "bg"]


# --- errors ------------------------------------------------------------------


class DiscourseParseError(ValueError):
    """Raw model output was not valid Step 6 JSON. Retains the raw response."""

    def __init__(self, message: str, *, raw_response: str, metadata: dict[str, Any] | None = None):
        super().__init__(message)
        self.raw_response = raw_response
        self.metadata = dict(metadata or {})


# --- schema ------------------------------------------------------------------


class EvidenceSpan(BaseModel):
    text_span: str
    modality: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class UsConstruct(BaseModel):
    label: str
    demands: list[str] = Field(default_factory=list)
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    uncertainty: str = ""


class FrontierConstruct(BaseModel):
    them_side: str
    us_side: str | None = None
    relation: Literal[
        "opposition",
        "exclusion",
        "antagonistic_boundary",
        "threat_construction",
        "blame",
        "boundary_construction",
    ] = "opposition"
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    uncertainty: str = ""


class AffectObservation(BaseModel):
    affect: str
    target: str | None = None
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    uncertainty: str = ""


class ChainCandidate(BaseModel):
    elements: list[str] = Field(default_factory=list)
    kind: Literal["equivalence", "difference"] = "equivalence"
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    uncertainty: str = ""


class SignifierCandidate(BaseModel):
    signifier: str
    role: Literal[
        "nodal_point_candidate",
        "floating_signifier_candidate",
        "empty_signifier_candidate",
    ] = "nodal_point_candidate"
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    uncertainty: str = ""


class PopulistDynamic(BaseModel):
    dynamic: Literal["fringe", "mainstream", "competing", "none"] = "none"
    evidence: list[EvidenceSpan] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    uncertainty: str = ""


class FormulaOfPopulism(BaseModel):
    """Evidence-linked, candidate-only Step 6 output."""

    populism_analysis: str = ""
    us_constructs: list[UsConstruct] = Field(default_factory=list)
    frontier_constructs: list[FrontierConstruct] = Field(default_factory=list)
    affects: list[AffectObservation] = Field(default_factory=list)
    chains: list[ChainCandidate] = Field(default_factory=list)
    signifier_candidates: list[SignifierCandidate] = Field(default_factory=list)
    populist_dynamic: PopulistDynamic = Field(default_factory=PopulistDynamic)
    counter_evidence: list[str] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)
    prompt_version: str = PROMPT_VERSION
    model_metadata: dict[str, Any] = Field(default_factory=dict)
    generated_at: str = ""


# --- prompt ------------------------------------------------------------------

SYSTEM_PROMPT = """You are LaclauGPT, a social scientist at the University of Helsinki \
performing cautious, evidence-first discourse analysis of 2024 European Parliament \
election campaign videos (TikTok / Instagram) in Finland, Sweden, Germany, France, \
Spain, Portugal, Croatia, Hungary, Bulgaria and Poland.

Return JSON only, conforming exactly to the supplied schema. Identify *candidate* \
relational structures. Never assert a final theoretical fact from a single document.

EVIDENCE AND ABSTENTION
- Every substantive coding must carry a short source span and a confidence.
- Empty lists are valid and preferable to forced coding. If the evidence does not
  support a construct, omit it.
- You are not required to find populism. There may be no Us, no Frontier, no affect,
  no chain. Report that honestly.

Us (UsConstruct)
- A politically meaningful collective subject constructed by articulation.
- A plural pronoun alone is not enough; there must be evidence of a shared subject.
- Never force an Us to exist.

Frontier (FrontierConstruct)
- A constitutive political boundary, opposition or exclusion, not merely a disliked
  entity. Classify the relation:
  opposition, exclusion, antagonistic_boundary, threat_construction, blame,
  boundary_construction.
- Ordinary disagreement or criticism is not an antagonistic boundary.
- Negative sentiment alone does not create a Frontier.
- The Frontier may be absent; the Us side may be absent if only an opponent is built.

Affect (AffectObservation)
- Affect is affective investment, not sentiment polarity.
- Do NOT map Us to positive affect or Frontier to negative affect.
- Anger may invest an Us; admiration may qualify an opponent; ambivalence may matter.
- Record the target when supported. If affect is not evidenced, leave it out.

Chains (ChainCandidate)
- Only where the discourse actually articulates a relation. Co-occurrence is not a
  chain of equivalence. Record difference where that is the meaningful relation.

Signifiers (SignifierCandidate)
- Nodal point, floating and empty signifier are candidates requiring evidence.
- Prominence or frequency alone is not a nodal point.
- Polysemy alone is not floating signification; one heterogeneous use is not an
  empty signifier. These are document-level candidates.

Hegemony, polarisation, dynamics
- Do NOT infer hegemony or polarisation from one document. Record at most evidence
  relevant to later corpus analysis.
- ``populist_dynamic`` (fringe / mainstream / competing / none) is a provisional
  relational heuristic, not a permanent label for an actor or party. Use "none" when
  the evidence does not support a dynamic.

Counter-evidence and uncertainty
- Record counter_evidence (material that argues against a construct) and
  uncertainty_notes. These are required when you make a substantive claim.

Do not output a populism score. Do not output a ``populist`` boolean; it is derived
deterministically downstream from your Us and Frontier constructs.

CONTEXT CLASSES
The user prompt separates four classes. Only the class labelled CURRENT SOURCE
EVIDENCE is evidence about this document. DERIVED PRIOR-STAGE ANALYSIS,
RESEARCHER/CODEBOOK MEMORY, and RETRIEVED CORPUS CONTEXT are aids for continuity and
normalization: never present them as evidence that something exists in this document.
"""


def get_step6_system_prompt() -> str:
    return SYSTEM_PROMPT


# --- context construction ----------------------------------------------------


def _first_nonempty(row: pd.Series, fields: tuple[str, ...]) -> str:
    for field in fields:
        text = str(row.get(field, "") or "").strip()
        if text:
            return text
    return ""


def _bounded(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit] + "\n[…truncated…]", True


def build_step6_context(
    row: pd.Series, *, max_chars: int | None = None
) -> tuple[str, str, dict[str, Any]]:
    """Build the deterministic Step 6 prompt context.

    Returns ``(context_text, sha256, info)`` where ``info`` records truncation and
    the evidence classes actually present (for reproducibility logging).
    """
    budget = int(max_chars or os.getenv("LACLAUGPT_STEP6_CONTEXT_CHARS", DEFAULT_CONTEXT_CHARS))
    sections: list[tuple[str, str]] = []

    metadata = metadata_context(
        row.drop(labels=[c for c in (*TRANSCRIPT_FIELDS, *FRAME_FIELDS, *OCR_FIELDS, *VIDEO_FIELDS) if c in row.index]),
        include_model_fields=False,
    )
    sections.append(("SOURCE METADATA AND RESEARCHER ANNOTATION", metadata))

    evidence: list[str] = []
    transcript = _first_nonempty(row, TRANSCRIPT_FIELDS)
    if transcript:
        evidence.append("### Speech / transcript\n" + transcript)
    frames = [str(row.get(f, "") or "").strip() for f in FRAME_FIELDS]
    frames = [f for f in frames if f]
    if frames:
        evidence.append("### Frame analysis\n" + "\n\n".join(frames))
    ocr = [str(row.get(f, "") or "").strip() for f in OCR_FIELDS]
    ocr = [o for o in ocr if o]
    if ocr:
        evidence.append("### On-screen / OCR text\n" + "\n\n".join(ocr))
    video = _first_nonempty(row, VIDEO_FIELDS)
    if video:
        evidence.append("### Full-video analysis\n" + video)
    if evidence:
        sections.append(("CURRENT SOURCE EVIDENCE (the document under analysis)", "\n\n".join(evidence)))

    derived: list[str] = []
    for field in DERIVED_FIELDS:
        text = str(row.get(field, "") or "").strip()
        if text:
            derived.append(f"### {field}\n{text}")
    for field in AUXILIARY_FIELDS:
        text = str(row.get(field, "") or "").strip()
        if text:
            derived.append(f"### {field} (auxiliary sentiment; NOT affective investment)\n{text}")
    if derived:
        sections.append(
            ("DERIVED PRIOR-STAGE ANALYSIS (context, not new source evidence)", "\n\n".join(derived))
        )

    context: list[str] = []
    for field in CONTEXT_FIELDS:
        text = str(row.get(field, "") or "").strip()
        if text:
            context.append(f"### {field}\n{text}")
    if context:
        sections.append(
            (
                "RESEARCHER/CODEBOOK MEMORY AND RETRIEVED CORPUS CONTEXT "
                "(normalization/retrieval context, NOT source evidence)",
                "\n\n".join(context),
            )
        )

    per_section = max(4000, budget // max(1, len(sections)))
    truncations: dict[str, bool] = {}
    rendered: list[str] = []
    for title, body in sections:
        body, truncated = _bounded(body, per_section)
        truncations[title] = truncated
        rendered.append(f"## {title}\n{body}")

    text = "\n\n".join(rendered)
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    info = {
        "context_chars": len(text),
        "context_sha256": sha,
        "sections": [title for title, _ in sections],
        "truncated": [title for title, flag in truncations.items() if flag],
        "budget_chars": budget,
    }
    return text, sha, info


def _prompt_sha256(system_prompt: str, user_prompt: str, model: str) -> str:
    return hashlib.sha256(f"{model}\n{system_prompt}\n{user_prompt}".encode()).hexdigest()


def get_step6_user_prompt(context_text: str) -> str:
    return (
        "### Step 6 — EP24 discourse analysis\n\n"
        "The material below is separated into four provenance classes. Only CURRENT "
        "SOURCE EVIDENCE is evidence about this document; the other classes must not be "
        "treated as evidence that something exists here.\n\n"
        f"{context_text}"
    )


# --- codebook context (opt-in, versioned) ------------------------------------

COUNTRY_CODEBOOK_CODES = {
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
}
_CODEBOOK_CACHE: dict[tuple[str, str, str], Any] = {}


def codebook_context_enabled() -> bool:
    return os.getenv("LACLAUGPT_ENRICHMENT_ENABLED", "0").casefold() in {"1", "true", "yes", "on"}


def add_codebook_context(country: str, source_text: str) -> tuple[str, str, str]:
    """Append the opt-in, versioned country-codebook background block.

    Returns ``(text, selected_entries_json, fingerprint)``. When the opt-in is off
    the text is returned unchanged with empty provenance. The selected entries and
    the codebook fingerprint are recorded so the prompt is reproducible (#170 §6).
    """
    if not codebook_context_enabled():
        return str(source_text), "", ""
    try:
        from roihu_codebooks import context_block, load_profile

        code, language = COUNTRY_CODEBOOK_CODES.get(country.casefold(), ("", ""))
        if not code:
            return str(source_text), "", ""
        private_root = os.getenv("LACLAUGPT_MULTIMODAL_PRIVATE_ROOT", ".")
        cache_key = (private_root, code, language)
        if cache_key not in _CODEBOOK_CACHE:
            _CODEBOOK_CACHE[cache_key] = load_profile(private_root, code, language=language)
        entries, profile = _CODEBOOK_CACHE[cache_key]
        block, selection = context_block(str(source_text), entries, country=code, language=language)
        fingerprint = str(profile.get("fingerprint", ""))
        selection_json = json.dumps(selection, ensure_ascii=False, sort_keys=True)
        if not block:
            return str(source_text), selection_json, fingerprint
        return f"{source_text}\n\n{block}", selection_json, fingerprint
    except Exception as exc:  # noqa: BLE001 - codebook context is optional
        logger.warning("codebook_context_unavailable country=%s error=%s", country, exc)
        return str(source_text), "", ""


# --- model call --------------------------------------------------------------


def _chat(*, model: str, system_prompt: str, user_prompt: str, options: dict, response_format: Any):
    """Single seam over the Ollama client so tests can stub the call."""
    import ollama

    return ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options=options,
        format=response_format,
    )


def call_model(system_prompt: str, user_prompt: str, *, model: str | None = None) -> tuple[str, FormulaOfPopulism]:
    """Call the model and parse the structured result. Raises DiscourseParseError."""
    selected_model = model or ollama_model(specific_env="LACLAUGPT_STEP6_MODEL")
    options = {
        "repeat_last_n": 64,
        "repeat_penalty": 1.1,
        "num_ctx": int(os.getenv("LACLAUGPT_STEP6_NUM_CTX", "32768")),
        "top_p": 0.9,
        "top_k": 40,
        "min_p": 0.0,
        "temperature": 0.0,
        "num_predict": int(os.getenv("LACLAUGPT_STEP6_NUM_PREDICT", "3072")),
    }
    logger.info(
        "model_call_start step=%d model=%s model_source=%s num_ctx=%s num_predict=%s",
        STEP,
        selected_model,
        ollama_model_source(specific_env="LACLAUGPT_STEP6_MODEL"),
        options["num_ctx"],
        options["num_predict"],
    )
    response = _chat(
        model=selected_model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        options=options,
        response_format=FormulaOfPopulism.model_json_schema(),
    )
    raw = str(response["message"]["content"])
    try:
        parsed = FormulaOfPopulism.model_validate_json(raw)
    except Exception as exc:  # noqa: BLE001 - re-raised with raw output retained
        raise DiscourseParseError(
            f"Step 6 JSON validation failed ({exc})",
            raw_response=raw,
            metadata={"model": selected_model, "prompt_version": PROMPT_VERSION},
        ) from exc
    parsed.model_metadata = {
        "provider": "ollama",
        "model": selected_model,
        "num_ctx": options["num_ctx"],
        "num_predict": options["num_predict"],
    }
    parsed.generated_at = datetime.now(timezone.utc).isoformat()
    parsed.prompt_version = PROMPT_VERSION
    logger.info("model_call_end step=%d model=%s response_chars=%d", STEP, selected_model, len(raw))
    return raw, parsed


# --- deterministic derivations ----------------------------------------------


def compute_populist(result: FormulaOfPopulism) -> bool:
    """``populist`` requires an evidenced Us AND an antagonistic Frontier.

    Deterministic and theory-bound (THEORY.md §6.4, invariant INV_POPULISM). It is
    never taken from the model, and it is never a score.
    """
    has_us = any(u.label.strip() for u in result.us_constructs)
    has_antagonistic = any(
        f.relation in ANTAGONISTIC_RELATIONS and f.them_side.strip() for f in result.frontier_constructs
    )
    return bool(has_us and has_antagonistic)


def legacy_pairs(labels: list[str], affects: list[AffectObservation]) -> str:
    """Deterministically serialize ``element^affect`` lines for the RDF contract.

    Only pairs with an *evidenced* affect are emitted, so no emotion is fabricated
    from polarity. Elements without a matching affect remain in the rich JSON but
    produce no legacy line. ``roihu_csv_rdf`` parses exactly this format.
    """
    lines: list[str] = []
    seen: set[str] = set()
    for label in labels:
        key = label.strip().casefold()
        if not key:
            continue
        for affect in affects:
            if (affect.target or "").strip().casefold() != key:
                continue
            emotion = affect.affect.strip()
            if not emotion:
                continue
            line = f"{label.strip()}^{emotion}"
            if line not in seen:
                seen.add(line)
                lines.append(line)
    return "".join(f"{line}\n" for line in lines)


def render_analysis_md(result: FormulaOfPopulism) -> str:
    """Deterministic markdown fallback when the model returns no prose analysis."""
    if result.populism_analysis.strip():
        return result.populism_analysis.strip()
    parts = ["## Step 6 discourse analysis (candidate structures)"]
    parts.append(f"- populist (deterministic): {str(compute_populist(result)).lower()}")
    parts.append(f"- Us candidates: {len(result.us_constructs)}")
    parts.append(f"- Frontier candidates: {len(result.frontier_constructs)}")
    parts.append(f"- Affect observations: {len(result.affects)}")
    if result.counter_evidence:
        parts.append("- Counter-evidence: " + "; ".join(result.counter_evidence))
    if result.uncertainty_notes:
        parts.append("- Uncertainty: " + "; ".join(result.uncertainty_notes))
    return "\n".join(parts)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _dump(models: list[BaseModel]) -> str:
    return _json([m.model_dump() for m in models])


# --- persistence helpers -----------------------------------------------------


@contextmanager
def _open_cache():
    """Subordinate SQLite restart cache (never an independent source of truth)."""
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS populism_cache (
            source_id TEXT NOT NULL,
            model TEXT NOT NULL,
            context_sha256 TEXT NOT NULL,
            payload TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (source_id, model, context_sha256)
        )
        """
    )
    conn.commit()
    try:
        yield conn
    finally:
        conn.close()


def _cache_lookup(conn: sqlite3.Connection, source_id: str, model: str, context_sha256: str) -> str | None:
    row = conn.execute(
        "SELECT payload FROM populism_cache WHERE source_id=? AND model=? AND context_sha256=?",
        (source_id, model, context_sha256),
    ).fetchone()
    return None if row is None else str(row[0])


def _cache_store(
    conn: sqlite3.Connection, *, source_id: str, model: str, context_sha256: str, payload: str
) -> None:
    conn.execute(
        """INSERT INTO populism_cache (source_id, model, context_sha256, payload, updated_at)
           VALUES (?, ?, ?, ?, ?) ON CONFLICT(source_id, model, context_sha256) DO UPDATE SET
             payload=excluded.payload, updated_at=excluded.updated_at""",
        (source_id, model, context_sha256, payload, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def _write_row(
    df: pd.DataFrame,
    index: Any,
    result: FormulaOfPopulism,
    *,
    status: str,
    context_sha256: str = "",
    codebook_context_json: str = "",
    codebook_fingerprint: str = "",
) -> None:
    us_labels = [u.label for u in result.us_constructs]
    frontier_labels = [f.them_side for f in result.frontier_constructs]
    analysis = render_analysis_md(result)
    meta = result.model_metadata
    values = {
        "formula_of_populism_analysis": analysis,
        "formula_of_populism_us": legacy_pairs(us_labels, result.affects),
        "formula_of_populism_frontier": legacy_pairs(frontier_labels, result.affects),
        "laclau_summary_md": analysis,
        "formula_of_populism_json": result.model_dump_json(),
        "formula_of_populism_us_json": _dump(result.us_constructs),
        "formula_of_populism_frontier_json": _dump(result.frontier_constructs),
        "formula_of_populism_affects_json": _dump(result.affects),
        "formula_of_populism_chains_json": _dump(result.chains),
        "formula_of_populism_signifiers_json": _dump(result.signifier_candidates),
        "formula_of_populism_populist_dynamic_json": result.populist_dynamic.model_dump_json(),
        "formula_of_populism_counter_evidence_json": _json(result.counter_evidence),
        "formula_of_populism_uncertainty_json": _json(result.uncertainty_notes),
        "formula_of_populism_populist": str(compute_populist(result)).lower(),
        "formula_of_populism_codebook_context_json": codebook_context_json,
        "formula_of_populism_codebook_fingerprint": codebook_fingerprint,
        "formula_of_populism_prompt_version": result.prompt_version,
        "formula_of_populism_context_sha256": context_sha256,
        "formula_of_populism_model": str(meta.get("model", "")),
        "formula_of_populism_generated_at": result.generated_at,
        "formula_of_populism_status": status,
    }
    for column, value in values.items():
        df.at[index, column] = value


def _mongo_patch(storage, row, *, source_id: str, model: str, context_sha256: str) -> str:
    document = {
        "_storage_id": source_id,
        "formula_of_populism_analysis": str(row.get("formula_of_populism_analysis", "")),
        "formula_of_populism_us": str(row.get("formula_of_populism_us", "")),
        "formula_of_populism_frontier": str(row.get("formula_of_populism_frontier", "")),
        "laclau_summary_md": str(row.get("laclau_summary_md", "")),
        "formula_of_populism_json": str(row.get("formula_of_populism_json", "")),
        "_provenance.step6_discourse": {
            "pipeline_stage": "step_6_discourse_analysis",
            "model": model,
            "context_sha256": context_sha256,
            "prompt_version": PROMPT_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }
    return f"mongo_ok:{storage.patch_documents('dataframe', [document])}"


def _configure_logging() -> None:
    if logging.getLogger().handlers:
        return
    Path("./logs").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        handlers=[logging.handlers.RotatingFileHandler("./logs/populism.log", encoding="utf-8", maxBytes=1000000, backupCount=5)],
        level=logging.DEBUG,
    )


# --- stage entry point -------------------------------------------------------


def analyze_discourse(country: str | None = None) -> None:
    """Run cumulative Step 6 without dropping any upstream data."""
    import logging.handlers  # noqa: F401 - ensure RotatingFileHandler is available

    _configure_logging()
    started = time.monotonic()
    filename = os.getenv("LACLAUGPT_INPUT_CSV") or f"ep24_{country}.csv"
    if not Path(filename).exists():
        logger.warning("input_missing path=%s country=%s", filename, country)
        return
    output = os.getenv("LACLAUGPT_OUTPUT_CSV") or filename
    model = ollama_model(specific_env="LACLAUGPT_STEP6_MODEL")
    max_rows = int(os.getenv("LACLAUGPT_MAX_ROWS", "0") or 0)

    df = load_cumulative_csv(filename, require_canonical=bool(os.getenv("LACLAUGPT_INPUT_CSV")))
    if max_rows > 0:
        logger.warning("demo_row_limit active=%d total=%d", max_rows, len(df))
        df = df.head(max_rows).copy()

    # The full incoming row is the contract; only stage-owned columns may change.
    before = df.drop(columns=[c for c in OUTPUT_COLUMNS if c in df.columns], errors="ignore").copy(deep=True)
    ensure_columns(df, OUTPUT_COLUMNS)

    from ep24_stage_contract import STAGE_CONTRACT

    expected_prior = [c for stage in STAGE_CONTRACT if stage.number < STEP for c in stage.appends]
    missing_prior = [c for c in expected_prior if c not in df.columns]

    config = StorageConfig.from_env()
    resolved_country = (country or config.country or "fi").casefold()
    system_prompt = get_step6_system_prompt()

    logger.info(
        "startup step=%d input=%s output=%s country=%s rows=%d columns=%d model=%s "
        "model_source=%s sqlite=%s mongo_enabled=%s max_rows=%d",
        STEP,
        filename,
        output,
        resolved_country,
        len(df),
        len(df.columns),
        model,
        ollama_model_source(specific_env="LACLAUGPT_STEP6_MODEL"),
        DB_PATH,
        config.mongo_enabled,
        max_rows,
    )
    if missing_prior:
        logger.warning("upstream_contract_missing=%s", missing_prior)

    stats = {"processed": 0, "failed": 0, "cache_hits": 0, "mongo_writes": 0, "skipped": 0}

    storage_cm = nullcontext(None)
    if config.mongo_enabled:
        from ep24_db import country_storage

        storage_cm = country_storage(resolved_country)
    coordinator = RedisCoordinator(resolved_country, STEP)

    with storage_cm as storage:
        if storage is not None:
            _bootstrap_context(storage, df, resolved_country)
            try:
                from ep24_context import enrich_dataframe

                df = enrich_dataframe(storage, df)
                ensure_columns(df, OUTPUT_COLUMNS)
            except Exception:
                logger.exception("context_enrich_failed country=%s", resolved_country)

        with _open_cache() as connection:
            total = len(df)
            for ordinal, (index, row) in enumerate(df.iterrows(), start=1):
                _process_row(
                    df,
                    index,
                    row,
                    ordinal=ordinal,
                    total=total,
                    system_prompt=system_prompt,
                    model=model,
                    output=output,
                    before=before,
                    connection=connection,
                    storage=storage,
                    coordinator=coordinator,
                    stats=stats,
                    country=resolved_country,
                )
            # Materialize stage columns even on an all-failure/empty run.
            write_cumulative_csv(before, df, output)

        if storage is not None:
            try:
                from ep24_context import update_retrieval

                update_retrieval(storage, df, stage=f"step_{STEP}_discourse_analysis")
            except Exception:
                logger.exception("rag_update_failed country=%s", resolved_country)

    logger.info(
        "complete step=%d processed=%d failed=%d cache_hits=%d mongo_writes=%d skipped=%d "
        "output=%s elapsed_seconds=%.3f",
        STEP,
        stats["processed"],
        stats["failed"],
        stats["cache_hits"],
        stats["mongo_writes"],
        stats["skipped"],
        output,
        time.monotonic() - started,
    )


def _bootstrap_context(storage, df: pd.DataFrame, country: str) -> None:
    try:
        from ep24_context import bootstrap_context

        private_root = Path(os.getenv("LACLAUGPT_MULTIMODAL_PRIVATE_ROOT", "."))
        info = bootstrap_context(storage, df, private_root=private_root, country=country)
        logger.info(
            "context_bootstrap country=%s codebooks=%s memory_seeds=%s fingerprint=%s",
            country,
            info.get("codebook_count"),
            info.get("memory_seed_count"),
            info.get("codebook_fingerprint"),
        )
    except Exception:
        logger.exception("context_bootstrap_failed country=%s", country)


def _process_row(
    df: pd.DataFrame,
    index: Any,
    row: pd.Series,
    *,
    ordinal: int,
    total: int,
    system_prompt: str,
    model: str,
    output: str,
    before: pd.DataFrame,
    connection: sqlite3.Connection,
    storage,
    coordinator: RedisCoordinator,
    stats: dict[str, int],
    country: str = "",
) -> None:
    source_id = stable_source_id(row)
    author = ep24_value(row, "author_username")
    video_id = ep24_value(row, "video_id")
    context_text, context_sha, info = build_step6_context(row)
    context_text, codebook_json, codebook_fingerprint = add_codebook_context(country, context_text)
    user_prompt = get_step6_user_prompt(context_text)
    context_sha256 = _prompt_sha256(system_prompt, user_prompt, model)
    logger.info(
        "row_start ordinal=%d total=%d index=%s source_id=%s author=%s video_id=%s "
        "context_chars=%d context_sha256=%s codebook=%s truncated=%s",
        ordinal,
        total,
        index,
        source_id,
        author,
        video_id,
        info["context_chars"],
        context_sha256,
        codebook_fingerprint,
        info["truncated"],
    )
    try:
        with coordinator.lock(source_id) as acquired:
            if not acquired:
                stats["skipped"] += 1
                logger.info("lock_busy source_id=%s", source_id)
                return
            _process_row_locked(
                df,
                index,
                row,
                source_id=source_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                context_sha256=context_sha256,
                model=model,
                output=output,
                before=before,
                connection=connection,
                storage=storage,
                coordinator=coordinator,
                stats=stats,
                codebook_context_json=codebook_json,
                codebook_fingerprint=codebook_fingerprint,
            )
    except DiscourseParseError as exc:
        stats["failed"] += 1
        logger.error(
            "row_parse_failed index=%s source_id=%s raw_chars=%d raw=%s",
            index,
            source_id,
            len(exc.raw_response),
            exc.raw_response[:2000],
        )
    except Exception as exc:  # noqa: BLE001 - one bad row must not kill the run
        stats["failed"] += 1
        logger.exception("row_failed index=%s source_id=%s video_id=%s error=%s", index, source_id, video_id, exc)


def _process_row_locked(
    df: pd.DataFrame,
    index: Any,
    row: pd.Series,
    *,
    source_id: str,
    system_prompt: str,
    user_prompt: str,
    context_sha256: str,
    model: str,
    output: str,
    before: pd.DataFrame,
    connection: sqlite3.Connection,
    storage,
    coordinator: RedisCoordinator,
    stats: dict[str, int],
    codebook_context_json: str = "",
    codebook_fingerprint: str = "",
) -> None:
    cached = _cache_lookup(connection, source_id, model, context_sha256)
    if cached is not None:
        result = FormulaOfPopulism.model_validate_json(cached)
        status = "cache"
        stats["cache_hits"] += 1
        logger.info("cache_hit source_id=%s", source_id)
    else:
        coordinator.mark(source_id, "running")
        _raw, result = call_model(system_prompt, user_prompt, model=model)
        _cache_store(
            connection,
            source_id=source_id,
            model=model,
            context_sha256=context_sha256,
            payload=result.model_dump_json(),
        )
        status = "ok"

    _write_row(
        df,
        index,
        result,
        status=status,
        context_sha256=context_sha256,
        codebook_context_json=codebook_context_json,
        codebook_fingerprint=codebook_fingerprint,
    )
    # Local cumulative CSV is the first durability boundary; never a head-only frame.
    write_cumulative_csv(before, df, output)
    stats["processed"] += 1
    logger.info(
        "local_checkpoint source_id=%s output=%s fields=%d",
        source_id,
        output,
        len(OUTPUT_COLUMNS),
    )

    if storage is not None:
        try:
            status_text = _mongo_patch(storage, df.loc[index], source_id=source_id, model=model, context_sha256=context_sha256)
            if status_text.startswith("mongo_ok:"):
                stats["mongo_writes"] += int(status_text.split(":", 1)[1])
            logger.info("mongo_status source_id=%s status=%s", source_id, status_text)
        except Exception:
            logger.exception("mongo_patch_failed source_id=%s local_checkpoint_is_safe=true", source_id)

    coordinator.mark(source_id, "done")


def main(argv: list[str] | None = None) -> int:
    from ep24_cli import configure_step_cli

    selection = configure_step_cli(STEP, argv)
    if selection.country:
        analyze_discourse(selection.country)
    elif os.getenv("LACLAUGPT_INPUT_CSV"):
        analyze_discourse(None)
    else:
        for code in countries:
            analyze_discourse(code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
