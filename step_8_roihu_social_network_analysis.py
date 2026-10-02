#!/usr/bin/env python3
"""Step 8: basic Social Network Analysis for EP24, with Castells interpretation.

Two layers, deliberately separated:

1. **Relation extraction** (pre-existing, preserved). An LLM proposes
   evidence-supported ties between actors; they land in ``sna_edges_json``. Per
   AGENTS.md this human-authored implementation is kept as-is.
2. **Deterministic graph construction** (added for issue #194). Node/edge tables,
   basic metrics and a human-readable Markdown report are derived from the
   accumulated record plus the layer-1 edges. No theory is involved here.

The Castells / *Communication Power* reading is applied only to the *computed*
summary, is labelled as interpretation, and cannot add topology.

Scope is intentionally modest per the issue: **Node – Edge – Node + basic metrics
+ Castells-informed human interpretation**. Advanced SNA is a later issue.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ep24_pipeline import load_cumulative_csv, metadata_context, write_cumulative_csv
import roihu_sna as SNA

try:  # optional: the deterministic layer must work with no LLM dependency
    import ollama
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover - exercised only in a minimal environment
    ollama = None

    class BaseModel:  # type: ignore[no-redef]
        pass

    def Field(**kwargs):  # type: ignore[no-redef]
        return None

logging.basicConfig(level=os.getenv("LACLAUGPT_LOGLEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("step_8_roihu_social_network_analysis")

STAGE = "sna"
DEFAULT_LANGUAGES = "fi,sv,pl,pt,de,es,hu,hr,fr,bg,en"


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


SYSTEM = """You are extracting evidence-supported social/communication network relations
from an EP24 social-media analysis, following the Phase 2 LaclauGPT SNA layer.
Create an edge only when the supplied evidence explicitly supports a relation
between two actors or organizations. Do not infer friendship, ideology,
coordination, influence, or hidden ties. Distinguish mention, reply, support,
opposition, affiliation, co-appearance, quotation, and other explicit relation
types where justified. Return a short Markdown explanation and structured edges."""


# --------------------------------------------------------------------------- #
# Layer 1: relation extraction (preserved)
# --------------------------------------------------------------------------- #

def source(lang: str) -> Path | None:
    explicit = os.getenv("LACLAUGPT_INPUT_CSV")
    if explicit:
        path = Path(explicit)
        return path if path.exists() else None
    for candidate in (Path(f"ep24_{lang}.csv"), Path(f"csv/tiktok_{lang}.csv")):
        if candidate.exists():
            return candidate
    return None


def extract_edges(frame: pd.DataFrame, *, model: str, limit: int) -> int:
    """Run the preserved LLM edge extraction. Returns the number of rows updated."""
    if ollama is None:
        LOG.warning("ollama is unavailable; skipping LLM edge extraction")
        return 0
    updated = 0
    for index, row in frame.head(limit).iterrows():
        evidence = metadata_context(row) + "\n\nANALYTICAL EVIDENCE:\n" + "\n\n".join(
            str(row.get(key, "")) for key in
            ("summary_analysis", "formula_of_populism_analysis", "dna_analysis_markdown",
             "dna_statements_json", "entities", "themes")
            if str(row.get(key, "")).strip()
        )
        if not evidence.strip():
            continue
        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": evidence}],
                format=SNAResult.model_json_schema(),
                options={"temperature": 0.0, "num_ctx": 8192},
            )
            out = SNAResult.model_validate_json(response["message"]["content"])
            frame.at[index, "sna_analysis_markdown"] = out.analysis_markdown
            frame.at[index, "sna_edges_json"] = json.dumps(
                [edge.model_dump() for edge in out.edges], ensure_ascii=False
            )
            updated += 1
        except Exception:
            LOG.exception("SNA extraction failed row=%s", index)
    return updated


# --------------------------------------------------------------------------- #
# Layer 2: deterministic graph + report (added)
# --------------------------------------------------------------------------- #

def build_and_write(
    frame: pd.DataFrame,
    *,
    language: str,
    country: str,
    output_dir: Path,
    interpret: bool = True,
    model: str = "",
) -> dict:
    """Build the graph, write the tables/report, and append the per-row fields."""
    rows = frame.to_dict(orient="records")
    network = SNA.build_network(rows, country=country, language=language)
    summary = SNA.graph_summary(network)

    interpretation = ""
    if interpret:
        interpretation = SNA.castells_interpretation(
            summary, client=_interpretation_client(model), model=model
        )

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "country": country,
        "language": language,
        "rows": len(frame),
        "model": model or "none",
    }
    paths = SNA.write_outputs(
        network, language=language, country=country, output_dir=output_dir,
        run_metadata=metadata, interpretation=interpretation,
    )
    SNA.append_to_dataframe(frame, network)
    LOG.info(
        "[%s] %s: %s node(s), %s edge(s), density=%s, components=%s",
        STAGE, language or country, summary["nodes"], summary["edges"],
        summary["density"], summary["components"],
    )
    for name, path in paths.items():
        LOG.debug("[%s] wrote %s -> %s", STAGE, name, path)
    return {"summary": summary, "paths": paths}


def _interpretation_client(model: str):
    """Return a callable for the Castells layer, or None when no model is usable.

    Returning ``None`` makes ``roihu_sna`` use its deterministic structural
    reading, so the report always has an interpretation section and the stage never
    depends on an LLM being reachable.
    """
    if ollama is None or not model or os.getenv("LACLAUGPT_SNA_INTERPRET", "1") == "0":
        return None

    def client(prompt: str, system: str) -> str:
        response = ollama.chat(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": prompt}],
            options={"temperature": 0.0, "num_ctx": 8192},
        )
        return response["message"]["content"]

    return client


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def run_language(
    lang: str,
    *,
    input_path: str = "",
    output_dir: str = "",
    limit: int | None = None,
    country: str = "",
    extract: bool = True,
    interpret: bool = True,
) -> dict:
    path = Path(input_path) if input_path else source(lang)
    if path is None or not path.exists():
        LOG.warning("No CSV for %s", lang or "(explicit)")
        return {"language": lang, "status": "no_input"}

    frame = load_cumulative_csv(path)
    for column in ("sna_analysis_markdown", "sna_edges_json"):
        if column not in frame.columns:
            frame[column] = ""

    row_limit = limit if limit is not None else int(os.getenv("LACLAUGPT_MAX_ROWS", "100") or 100)
    model = os.getenv("LACLAUGPT_MULTIMODAL_MODEL", "gemma4:12b")

    if extract:
        updated = extract_edges(frame, model=model, limit=row_limit)
        LOG.info("[%s] %s: LLM edge extraction updated %s row(s)", STAGE, lang, updated)

    resolved_country = country or str(
        frame["country"].iloc[0] if "country" in frame.columns and len(frame) else ""
    ).strip()
    out_dir = Path(output_dir) if output_dir else SNA.SNA_DIR
    result = build_and_write(
        frame, language=lang, country=resolved_country,
        output_dir=out_dir, interpret=interpret, model=model,
    )

    # Additive write: fails loudly if any incoming column was dropped or mutated.
    write_cumulative_csv(frame, frame, path)
    result.update({"language": lang, "status": "ok", "source": str(path)})
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="", help="explicit input CSV (overrides discovery)")
    parser.add_argument("--output-dir", default="", help="directory for node/edge tables and report")
    parser.add_argument("--language", default="", help="single language to process")
    parser.add_argument("--country", default="", help="country label recorded in the graph")
    parser.add_argument("--limit", type=int, default=None, help="max rows for LLM extraction")
    parser.add_argument("--no-extract", action="store_true", help="skip LLM edge extraction")
    parser.add_argument("--no-interpret", action="store_true", help="skip the Castells LLM call")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    languages = [args.language] if args.language else (
        [""] if args.input else
        [item.strip() for item in os.getenv("LACLAUGPT_LANGUAGES", DEFAULT_LANGUAGES).split(",")
         if item.strip()]
    )
    failures = 0
    for language in languages:
        try:
            result = run_language(
                language,
                input_path=args.input,
                output_dir=args.output_dir,
                limit=args.limit,
                country=args.country,
                extract=not args.no_extract,
                interpret=not args.no_interpret,
            )
            if result.get("status") == "ok":
                summary = result["summary"]
                print(
                    f"[{STAGE}] {language or '(explicit)'}: {summary['nodes']} node(s), "
                    f"{summary['edges']} edge(s), density={summary['density']}"
                )
        except Exception:
            LOG.exception("SNA failed for language=%s", language)
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
