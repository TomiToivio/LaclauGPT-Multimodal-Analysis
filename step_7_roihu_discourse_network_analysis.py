#!/usr/bin/env python3
"""Step 7: structured Discourse Network Analysis extraction for EP24.

Standalone Roihu script adapted from the Phase 2 discourse_network concepts in
TomiToivio/LaclauGPT-Data-Analysis. It keeps the CSV contract human-readable
while adding typed JSON statements for later graph construction.
"""
from __future__ import annotations
import json, logging, os
from pathlib import Path
import pandas as pd
from ep24_pipeline import load_cumulative_csv, metadata_context
from ep24_entities import fold_key, resolution_lookup
from ep24_cli import configure_step_cli
import sys

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
LOG=logging.getLogger("step_7_roihu_discourse_network_analysis")

def _dna_models():
    """Build the DNA structured-output models on demand.

    pydantic is imported here rather than at module scope so that ``--help`` and
    argument validation work in a minimal environment without ollama/pydantic
    (issue #152). The Ollama client is deferred the same way in ``run_language``.
    """
    from pydantic import BaseModel, Field

    class DNAStatement(BaseModel):
        actor_name: str
        concept_label: str
        proposition: str
        stance: str = Field(description="support, oppose, neutral, mixed, or unknown")
        agreement: bool | None = None
        evidence_quote: str
        confidence: float = Field(ge=0.0, le=1.0)

    class DNAResult(BaseModel):
        analysis_markdown: str
        statements: list[DNAStatement]

    return DNAResult

SYSTEM="""You are extracting evidence-linked Discourse Network Analysis (DNA) statements
from an EP24 social-media analysis. Follow the Phase 2 LaclauGPT DNA logic:
actor + concept/proposition + stance/agreement + evidence + uncertainty.
Do not invent actors, concepts, or quotations. One statement must correspond to
one explicit actor-concept claim. Use agreement=true only for explicit support,
false only for explicit opposition, otherwise null. Preserve ambiguity.
Return a short human-readable Markdown analysis plus structured statements."""

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
    p=source(lang)
    if p is None:
        LOG.warning("No CSV for %s",lang); return
    df=load_cumulative_csv(p)
    for col in ("dna_analysis_markdown","dna_statements_json"):
        if col not in df.columns: df[col]=""
    limit=int(os.getenv("LACLAUGPT_MAX_ROWS","100") or 100)
    model=os.getenv("LACLAUGPT_MULTIMODAL_MODEL","gemma4:12b")
    import ollama
    DNAResult=_dna_models()
    for i,row in df.head(limit).iterrows():
        evidence = metadata_context(row) + "\n\nANALYTICAL EVIDENCE:\n" + "\n\n".join(
            str(row.get(k, "")) for k in
            ("summary_analysis", "formula_of_populism_analysis", "entities", "themes", "topics")
            if str(row.get(k, "")).strip()
        )
        if not evidence.strip(): continue
        try:
            r=ollama.chat(model=model,messages=[{"role":"system","content":SYSTEM},
              {"role":"user","content":evidence}],format=DNAResult.model_json_schema(),
              options={"temperature":0.0,"num_ctx":8192})
            out=DNAResult.model_validate_json(r["message"]["content"])
            df.at[i,"dna_analysis_markdown"]=out.analysis_markdown
            lookup=resolution_lookup(row.get("ep24_entity_resolution_json"))
            statements=[]
            for statement in out.statements:
                item=statement.model_dump()
                linked=lookup.get(fold_key(statement.actor_name))
                if linked:
                    item["actor_id"]=linked["entity_id"]
                    item["actor_canonical_name"]=linked["canonical_name"]
                statements.append(item)
            df.at[i,"dna_statements_json"]=json.dumps(statements,ensure_ascii=False)
        except Exception:
            LOG.exception("DNA failed row=%s file=%s",i,p)
    df.to_csv(p,index=False)

if __name__=="__main__":
    selection = configure_step_cli(7, sys.argv[1:])
    if selection.remaining_argv:
        raise SystemExit(f"unrecognized arguments: {' '.join(selection.remaining_argv)}")
    if os.getenv("LACLAUGPT_INPUT_CSV"):
        run_language("")
    else:
        for lang in os.getenv("LACLAUGPT_LANGUAGES","fi,sv,pl,pt,de,es,hu,hr,fr,bg,en").split(","):
            run_language(lang.strip())
