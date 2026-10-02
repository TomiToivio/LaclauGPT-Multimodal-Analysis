#!/usr/bin/env python3
"""Step 8: basic evidence-backed Social Network Analysis for EP24.

The model extracts explicit relations. Deterministic code then constructs stable
Node-Edge-Node tables, calculates a deliberately small set of transparent SNA
metrics, persists additive outputs, and writes a human-readable report with a
separate Castells-informed interpretation.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ep24_cli import configure_step_cli
from ep24_entities import fold_key, resolution_lookup
from ep24_models import ollama_model, ollama_model_source
from ep24_pipeline import (
    load_cumulative_csv,
    metadata_context,
    write_cumulative_csv,
)
from ep24_sna import (
    SNA_SCHEMA_VERSION,
    basic_metrics,
    graph_from_edges,
    graph_summary,
    render_markdown_report,
)

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("step_8_roihu_social_network_analysis")

PROMPT_VERSION = "ep24-sna-extraction/2.0"
CASTELLS_PROMPT_VERSION = "ep24-sna-castells/1.0"

EXTRACTION_SYSTEM = """You extract evidence-supported social/communication network relations
from an EP24 social-media analysis. Create an edge only when supplied evidence explicitly
supports a relation between two actors or organizations. Do not infer friendship,
ideology, coordination, influence, hidden ties, or causality. Distinguish mention,
reply, support, opposition, affiliation, co-appearance, quotation, and other explicit
relation types only where justified. The edge list is empirical coding: theory must not
change topology. Return a concise empirical Markdown summary and structured edges."""

CASTELLS_SYSTEM = """Interpret the supplied, already-constructed social network through a
limited Manuel Castells network-society / Communication Power lens. The graph topology is
fixed: NEVER add, remove, infer, or rename nodes or edges. Discuss communication networks,
flows, inclusion/exclusion, communication power and network-making power only to the
extent visible in the supplied graph. Use 'programmer' or 'switcher' only when the
observed relations provide specific evidence; otherwise explicitly say the role cannot be
established. Do not infer hidden coordination, motives, causality, strategic control, or
population-level influence from a small sample. Produce short Markdown prose labelled as
interpretation rather than empirical fact."""


def _models():
    """Import runtime-only dependencies after CLI parsing."""
    from pydantic import BaseModel, Field

    class SNAEdge(BaseModel):
        source_actor: str
        target_actor: str
        relation_type: str
        directed: bool = True
        evidence_quote: str
        confidence: float = Field(ge=0.0, le=1.0)

    class SNAResult(BaseModel):
        analysis_markdown: str
        edges: list[SNAEdge]

    class CastellsResult(BaseModel):
        interpretation_markdown: str

    return SNAResult, CastellsResult


def source(lang: str) -> Path | None:
    explicit = os.getenv("LACLAUGPT_INPUT_CSV")
    if explicit:
        path = Path(explicit)
        return path if path.exists() else None
    for path in (Path(f"ep24_{lang}.csv"), Path(f"csv/tiktok_{lang}.csv")):
        if path.exists():
            return path
    return None


def _output_path(input_path: Path) -> Path:
    return Path(os.getenv("LACLAUGPT_OUTPUT_CSV") or input_path)


def _enrich_edges(edges, row) -> list[dict]:
    lookup = resolution_lookup(row.get("ep24_entity_resolution_json"))
    enriched: list[dict] = []
    for edge in edges:
        item = edge.model_dump()
        item["network_layer"] = "social"
        source_hit = lookup.get(fold_key(edge.source_actor))
        target_hit = lookup.get(fold_key(edge.target_actor))
        if source_hit:
            item["source_actor_id"] = source_hit["entity_id"]
            item["source_actor_canonical_name"] = source_hit["canonical_name"]
        if target_hit:
            item["target_actor_id"] = target_hit["entity_id"]
            item["target_actor_canonical_name"] = target_hit["canonical_name"]
        enriched.append(item)
    return enriched


def _castells_interpretation(ollama, model: str, nodes, edges, metrics) -> str:
    if not edges:
        return (
            "The sample contains no evidence-supported social edges, so a Castellsian "
            "interpretation of network position or network-making power is not warranted."
        )
    _, CastellsResult = _models()
    summary = graph_summary(nodes, edges, metrics)
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": CASTELLS_SYSTEM},
            {"role": "user", "content": summary},
        ],
        format=CastellsResult.model_json_schema(),
        options={"temperature": 0.0, "num_ctx": 8192},
    )
    return CastellsResult.model_validate_json(
        response["message"]["content"]
    ).interpretation_markdown


def _write_graph_exports(output_path: Path, all_nodes: list[dict], all_edges: list[dict]) -> None:
    base = output_path.with_suffix("")
    nodes_path = Path(str(base) + ".sna_nodes.csv")
    edges_path = Path(str(base) + ".sna_edges.csv")
    nodes_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_nodes).to_csv(nodes_path, index=False)
    pd.DataFrame(all_edges).to_csv(edges_path, index=False)
    LOG.info("SNA graph exports nodes=%s edges=%s", nodes_path, edges_path)


def _persist_mongo(df: pd.DataFrame, all_nodes: list[dict], all_edges: list[dict], source_path: Path, model: str) -> None:
    from roihu_storage import MongoStorage, StorageConfig, dataframe_to_documents

    config = StorageConfig.from_env()
    if not config.mongo_enabled:
        return
    storage = MongoStorage(config)
    run_id = os.getenv("SLURM_JOB_ID", "local-step8")
    try:
        cumulative = dataframe_to_documents(
            df,
            config=config,
            stage="sna",
            run_id=run_id,
            source_hint=str(source_path),
            model=model,
        )
        storage.patch_documents("dataframe", cumulative)
        graph_docs = []
        for kind, rows in (("node", all_nodes), ("edge", all_edges)):
            for raw in rows:
                doc = dict(raw)
                graph_id = str(doc.get(f"{kind}_id") or "")
                if not graph_id:
                    continue
                doc["_storage_id"] = f"sna:{kind}:{graph_id}"
                doc["graph_record_type"] = kind
                doc["sna_schema_version"] = SNA_SCHEMA_VERSION
                graph_docs.append(doc)
        storage.upsert_documents("sna", graph_docs)
        LOG.info("Mongo SNA persistence cumulative=%s graph=%s", len(cumulative), len(graph_docs))
    finally:
        storage.close()


def run_language(lang: str) -> Path | None:
    import ollama

    SNAResult, _ = _models()
    input_path = source(lang)
    if input_path is None:
        LOG.warning("No CSV for %s", lang)
        return None
    output_path = _output_path(input_path)

    original = load_cumulative_csv(
        input_path, require_canonical=bool(os.getenv("LACLAUGPT_INPUT_CSV"))
    )
    limit = int(os.getenv("LACLAUGPT_MAX_ROWS", "100") or 100)
    if limit > 0:
        if output_path.resolve() == input_path.resolve() and limit < len(original):
            raise ValueError(
                "Refusing to truncate the input CSV: when LACLAUGPT_MAX_ROWS is set, "
                "LACLAUGPT_OUTPUT_CSV must be a different path."
            )
        working = original.head(limit).copy()
    else:
        working = original.copy()

    df = working.copy()
    output_columns = (
        "sna_analysis_markdown",
        "sna_castells_interpretation_markdown",
        "sna_nodes_json",
        "sna_edges_json",
        "sna_metrics_json",
        "sna_schema_version",
        "sna_prompt_version",
        "sna_castells_prompt_version",
        "sna_generated_at",
        "sna_status",
        "sna_error",
    )
    for column in output_columns:
        if column not in df.columns:
            df[column] = ""

    model = ollama_model()
    LOG.info("model=%s model_source=%s", model, ollama_model_source())
    all_nodes: list[dict] = []
    all_edges: list[dict] = []

    for index, row in df.iterrows():
        evidence = metadata_context(row) + "\n\nANALYTICAL EVIDENCE:\n" + "\n\n".join(
            str(row.get(key, ""))
            for key in (
                "summary_analysis",
                "formula_of_populism_analysis",
                "dna_analysis_markdown",
                "dna_statements_json",
                "entities",
                "themes",
            )
            if str(row.get(key, "")).strip()
        )
        if not evidence.strip():
            df.at[index, "sna_status"] = "no_evidence"
            continue

        try:
            response = ollama.chat(
                model=model,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM},
                    {"role": "user", "content": evidence},
                ],
                format=SNAResult.model_json_schema(),
                options={"temperature": 0.0, "num_ctx": 8192},
            )
            extracted = SNAResult.model_validate_json(response["message"]["content"])
            raw_edges = _enrich_edges(extracted.edges, row)
            nodes, edges = graph_from_edges(raw_edges, row)
            metrics = basic_metrics(nodes, edges)
            castells = _castells_interpretation(ollama, model, nodes, edges, metrics)
            report = render_markdown_report(
                row,
                nodes,
                edges,
                metrics,
                extracted.analysis_markdown,
                castells,
            )

            df.at[index, "sna_analysis_markdown"] = report
            df.at[index, "sna_castells_interpretation_markdown"] = castells
            df.at[index, "sna_nodes_json"] = json.dumps(nodes, ensure_ascii=False, sort_keys=True)
            df.at[index, "sna_edges_json"] = json.dumps(edges, ensure_ascii=False, sort_keys=True)
            df.at[index, "sna_metrics_json"] = json.dumps(metrics, ensure_ascii=False, sort_keys=True)
            df.at[index, "sna_schema_version"] = SNA_SCHEMA_VERSION
            df.at[index, "sna_prompt_version"] = PROMPT_VERSION
            df.at[index, "sna_castells_prompt_version"] = CASTELLS_PROMPT_VERSION
            df.at[index, "sna_generated_at"] = datetime.now(timezone.utc).isoformat()
            df.at[index, "sna_status"] = "ok"
            df.at[index, "sna_error"] = ""
            all_nodes.extend(nodes)
            all_edges.extend(edges)
        except Exception as exc:
            df.at[index, "sna_status"] = "error"
            df.at[index, "sna_error"] = f"{type(exc).__name__}: {exc}"
            LOG.exception("SNA failed row=%s file=%s", index, input_path)

    write_cumulative_csv(working, df, output_path)
    _write_graph_exports(output_path, all_nodes, all_edges)
    try:
        _persist_mongo(df, all_nodes, all_edges, output_path, model)
    except Exception:
        LOG.exception("Mongo SNA persistence failed; CSV graph artifacts remain authoritative compatibility outputs")
    LOG.info("Step 8 complete input=%s output=%s rows=%s nodes=%s edges=%s",
             input_path, output_path, len(df), len(all_nodes), len(all_edges))
    return output_path


if __name__ == "__main__":
    selection = configure_step_cli(8, sys.argv[1:])
    if selection.remaining_argv:
        raise SystemExit(f"unrecognized arguments: {' '.join(selection.remaining_argv)}")
    if os.getenv("LACLAUGPT_INPUT_CSV"):
        run_language("")
    else:
        for language in os.getenv(
            "LACLAUGPT_LANGUAGES", "fi,sv,pl,pt,de,es,hu,hr,fr,bg,en"
        ).split(","):
            run_language(language.strip())
