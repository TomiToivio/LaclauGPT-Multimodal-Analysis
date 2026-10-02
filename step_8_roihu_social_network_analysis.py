#!/usr/bin/env python3
"""Step 8: structured Social Network Analysis relation extraction for EP24.

Standalone Roihu adapter inspired by the Phase 2 sna package. It extracts only
evidence-supported social/communication ties; aggregate graph metrics can be
computed later from the emitted edge JSON without changing the CSV contract.
"""
from __future__ import annotations
import json, logging, os
from pathlib import Path
import pandas as pd
from ep24_models import ollama_model, ollama_model_source
from ep24_pipeline import load_cumulative_csv, metadata_context
from ep24_entities import fold_key, resolution_lookup
from ep24_cli import configure_step_cli
import sys

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
LOG=logging.getLogger("step_8_roihu_social_network_analysis")

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

    return SNAResult

SYSTEM="""You are extracting evidence-supported social/communication network relations
from an EP24 social-media analysis, following the Phase 2 LaclauGPT SNA layer.
Create an edge only when the supplied evidence explicitly supports a relation
between two actors or organizations. Do not infer friendship, ideology,
coordination, influence, or hidden ties. Distinguish mention, reply, support,
opposition, affiliation, co-appearance, quotation, and other explicit relation
types where justified. Return a short Markdown explanation and structured edges."""

def source(lang):
    explicit = os.getenv("LACLAUGPT_INPUT_CSV")
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    for p in (Path(f"ep24_{lang}.csv"), Path(f"csv/tiktok_{lang}.csv")):
        if p.exists():
            return p
    return None

def run_language(lang):
    import ollama
    SNAResult = _models()
    p=source(lang)
    if p is None:
        LOG.warning("No CSV for %s",lang); return
    df=load_cumulative_csv(p)
    for col in ("sna_analysis_markdown","sna_edges_json"):
        if col not in df.columns: df[col]=""
    limit=int(os.getenv("LACLAUGPT_MAX_ROWS","100") or 100)
    model=ollama_model()
    LOG.info('model=%s model_source=%s', model, ollama_model_source())
    for i,row in df.head(limit).iterrows():
        evidence = metadata_context(row) + "\n\nANALYTICAL EVIDENCE:\n" + "\n\n".join(
            str(row.get(k, "")) for k in
            ("summary_analysis", "formula_of_populism_analysis", "dna_analysis_markdown",
             "dna_statements_json", "entities", "themes")
            if str(row.get(k, "")).strip()
        )
        if not evidence.strip(): continue
        try:
            r=ollama.chat(model=model,messages=[{"role":"system","content":SYSTEM},
              {"role":"user","content":evidence}],format=SNAResult.model_json_schema(),
              options={"temperature":0.0,"num_ctx":8192})
            out=SNAResult.model_validate_json(r["message"]["content"])
            df.at[i,"sna_analysis_markdown"]=out.analysis_markdown
            lookup=resolution_lookup(row.get("ep24_entity_resolution_json"))
            edges=[]
            for edge in out.edges:
                item=edge.model_dump()
                item["network_layer"]="social"
                source_hit=lookup.get(fold_key(edge.source_actor))
                target=lookup.get(fold_key(edge.target_actor))
                if source_hit:
                    item["source_actor_id"]=source_hit["entity_id"]
                    item["source_actor_canonical_name"]=source_hit["canonical_name"]
                if target:
                    item["target_actor_id"]=target["entity_id"]
                    item["target_actor_canonical_name"]=target["canonical_name"]
                edges.append(item)
            df.at[i,"sna_edges_json"]=json.dumps(edges,ensure_ascii=False)
        except Exception:
            LOG.exception("SNA failed row=%s file=%s",i,p)
    df.to_csv(p,index=False)

if __name__=="__main__":
    selection = configure_step_cli(8, sys.argv[1:])
    if selection.remaining_argv:
        raise SystemExit(f"unrecognized arguments: {' '.join(selection.remaining_argv)}")
    if os.getenv("LACLAUGPT_INPUT_CSV"):
        run_language("")
    else:
        for lang in os.getenv("LACLAUGPT_LANGUAGES","fi,sv,pl,pt,de,es,hu,hr,fr,bg,en").split(","):
            run_language(lang.strip())
