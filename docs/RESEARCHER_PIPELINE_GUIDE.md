# Researcher's map of the EP24 analysis pipeline

> First, documentation-only increment for [issue #411](https://github.com/TomiToivio/LaclauGPT-Multimodal-Analysis/issues/411). Based on the `main` repository on 2026-11-16. This guide describes the existing pipeline; it does **not** approve methodological changes or attest that all stages have passed a full Roihu run.

## Start here: the route from source to claim

A source video and researcher feed metadata enter Step 1. Every step **adds columns to the same cumulative record**, preserving previous columns and their meanings. A researcher tracing an interpretation should start with the source ID and media key, inspect the evidence columns, check the prompt and model outputs for the relevant stage, then check any deterministic transformations and export. An empty or failed stage is not negative evidence about political content.

| Stage | Researcher's question | Input / evidence | Output to inspect | Implementation to read |
| --- | --- | --- | --- | --- |
| 1 Preprocess | What does the source video actually contain? | Allas video + source row | ASR transcript/translation, one keyframe, OCR, backend provenance, status | `step_1_roihu_preprocess.py` → `roihu_preprocess.py`; `asr_backend.py`, `ocr_backend.py` |
| 2 Frame | What can be observed in the selected still? | Step 1 keyframe + cumulative context | `frame_analysis_1`, frame model and QA fields | `step_2_roihu_frame.py` → `roihu_frame.py` |
| 3 Video | What happens over time? | Trimmed video + record context | `vllm_video_analysis`, Markdown, structured JSON, scroll/quality status, prompt/model provenance | `step_3_roihu_video.py` → `experiments/vllm_video_test.py`; `ep24_video.py` |
| 4 Summary | How do modalities combine without losing evidence? | Frame, video, ASR, OCR, source context | `summary_analysis`, `summary_summary_md`, quality fields | `step_4_roihu_summary.py` → `roihu_summary.py` |
| 5 Postprocess | Which structured mentions, themes and targets were extracted? | Cumulative analysis | `postprocess_entities`, `postprocess_themes`, additional postprocess fields | `step_5_roihu_postprocess.py` → `roihu_postprocess.py` |
| 6 Discourse | What Laclau/Palonen interpretation is warranted? | Source evidence and accumulated analysis; bounded prompt | `laclau_*`, discourse outputs and status | `step_6_roihu_discourse_analysis.py` → `roihu_populism.py` |
| 7 DNA | Who explicitly supports/opposes which concept? | Current-source claims + bounded context | `dna_*` statements, stance, provenance and human-readable output | `step_7_roihu_discourse_network_analysis.py`, `ep24_dna.py` |
| 8 SNA | Which evidenced actor relations exist, and what does their topology show? | Source evidence + upstream coded statements | `sna_nodes_json`, `sna_edges_json`, metrics, separate Castells interpretation, sibling CSV graph tables | `step_8_roihu_social_network_analysis.py`, `ep24_sna.py`, `ep24_graph_aggregate.py` |
| 9 RDF | How can the results be queried as a graph? | Cumulative row, existing SNA graph and provenance | `rdf_*` metadata + RDF files | `step_9_roihu_rdf.py` → `roihu_rdf.py`; `roihu_csv_rdf.py` |

The table gives **examples**, not exhaustive column inventories. The machine-readable contracts are `ep24_stage_contract.py` and `config/ep24_pipeline_columns.json`; consult both when changing fields. See [pipeline contract](EP24_PIPELINE_CONTRACT.md) and [numbered pipeline](ROIHU_NUMBERED_PIPELINE.md).

## Where things live

- **Directly executable stages:** `step_1_roihu_*.py` through `step_9_roihu_*.py`. Keep these commands and matching `scripts/roihu/step_N_*.sbatch` interfaces intact.
- **Compatibility implementations:** `roihu_preprocess.py`, `roihu_frame.py`, `roihu_summary.py`, `roihu_postprocess.py`, `roihu_populism.py`. The numbered wrappers are intentional, **not** proof of removable duplication. Historical reproducibility is preserved independently on the immutable `legacy` branch.
- **Cumulative CSV and identity:** `ep24_pipeline.py`, `ep24_schema.py`, `ep24_input_schema.py`, `ep24_stage_contract.py`.
- **Operation, resumability, state and storage:** `ep24_cli.py`, `ep24_stage_orchestrator.py`, `ep24_stage_state.py`, `ep24_settings.py`, `roihu_storage.py`, `ep24_db.py`, `ep24_redis.py`, `ep24_allas.py`; scripts under `scripts/ep24/` and `scripts/roihu/`.
- **Model/prompt decisions:** `ep24_models.py`, `ep24_prompt_contract.py`, stage-local prompt constants and runtime callers, `roihu_codebooks.py` and `roihu_codebook_sources.py`. Do not publish private codebooks or researcher notes.
- **Graph outputs:** `ep24_dna.py`, `ep24_sna.py`, `ep24_sna_aggregate.py`, `ep24_graph_aggregate.py`, `roihu_rdf.py`, `roihu_csv_rdf.py`.
- **Validation:** `tests/`, CI in `.github/workflows/`, [coding style](CODING_STYLE.md), [Roihu runbook](ROIHU_RUNBOOK.md), and [prompt evidence runbook](EP24_PROMPT_EVIDENCE_RUNBOOK.md).

## What is interpreted and what is calculated?

**Source evidence** is text, image, video and metadata traceable to the current record. **LLM-derived claims** include narrative descriptions, discourse interpretations, DNA statement candidates and extracted SNA relations; their correctness must be evaluated against source evidence. **Researcher context** (including codebooks, memory and RAG) can guide interpretation or normalization but does **not** independently prove that a source actor made a claim or has a tie. **Deterministic transformations** include schema checks, stable identity, graph metrics after extraction, and RDF projection; these still need input and code verification.

An illustration, **not an observed EP24 result**:

```text
video_id: synthetic-example-001
asr_transcript: "We support the proposal."
dna statement: {actor: "speaker", concept: "proposal", agreement: "support"}
sna edge: none unless an explicit relation is evidenced
```

A supporting quote for a DNA statement does not automatically establish a social edge. Model outputs should be treated as hypotheses to audit against evidence, not transparent windows into model reasoning.

## Inspect one record without changing a run

1. Identify the original `video_id`, `allas_filename` and, if present, `_storage_id`. Preserve them as strings. Inspect `preprocess_status`, OCR/ASR provenance and media QA before interpreting text.
2. Compare frame and video outputs with source material. Initial feed scroll is a known visual artifact; review `SCROLL`, `SCROLL_SECONDS` and `needs_resplit` where applicable. ASR is intentionally full-video in the current Step 1 contract.
3. Read `summary_analysis` and the Step 6 interpretation with the original transcript/translation visible. A summary is model-generated context, not a substitute for an observed quotation.
4. For each Step 7 statement, verify actor, exact proposition, positive/negative stance, evidence and source/time identity. For each Step 8 edge, verify the relation quote; separately inspect calculated metrics and Castells interpretation.
5. Check model name, prompt version/hash (when recorded), parse/validation status, missing values, retry/fallback evidence and stage logs. Keep raw model output when authorized and available. Do **not** infer a model's hidden reasoning.
6. Confirm Step 9 RDF reuses Step 8 identities and is consistent with the cumulative CSV; inspect sibling `.sna_nodes.csv` and `.sna_edges.csv` exports.

For isolated comparisons, use the [prompt evidence runbook](EP24_PROMPT_EVIDENCE_RUNBOOK.md): fixed source IDs, unchanged source material, independent review and no bulk force reset.

## Prompt inventory: entry points, not an unsupported completeness claim

The active system prompt can be **inline in a stage implementation** or assembled with shared bounded evidence packets. Begin with `SYSTEM_PROMPT` and analogous prompt constants/functions in `roihu_frame.py`, `experiments/vllm_video_test.py`, `roihu_summary.py`, `roihu_postprocess.py`, `roihu_populism.py`, `step_7_roihu_discourse_network_analysis.py` and `step_8_roihu_social_network_analysis.py`. Step 1 is primarily backend inference and OCR, while Step 9 is deterministic RDF export. Trace actual call sites, not just constant names: the inventory is **provisional** until each model call is reviewed.

`ep24_prompt_contract.py` already centralizes evidence-versus-context labelling and packet budgets for Steps 4–8; [prompt contract consolidation](EP24_PROMPT_CONTRACT_CONSOLIDATION.md) documents prior overlapping implementations that have **already** been merged. Do not reintroduce old modules as part of a new refactor. Preserve the existing stage budgets (4: 20,000; 6: 28,000; 7: 20,000; 8: 12,000 model-visible characters). Prompt readability edits require **schema and same-input output comparisons** before adoption; codebook/theory changes belong in separate proposals.

## Glossary

- **Cumulative dataframe:** one record with original fields plus fields appended by every completed stage.
- **Provenance:** the trace from source identity, media, model, prompt and processing status to a derived output.
- **ASR / OCR / VLM:** automatic speech recognition / optical character recognition / vision-language model.
- **DA / DNA / SNA:** discourse analysis (Step 6) / discourse network analysis of actor-concept statements (Step 7) / social network analysis of evidenced relations (Step 8).
- **Codebook:** researcher-defined concepts/normalization rules; context is not automatically source evidence.
- **RDF:** graph serialization of already-derived records, not a fresh LLM classification.
- **Checkpoint / resume:** persistent progress and recoverable cumulative output; not permission to silently overwrite original inputs.

## Issue #411 audit: what remains to be proved

The initial tree review confirms nine numbered entry points, compatibility modules, extensive `ep24_*` helpers, numerous Slurm wrappers, and a substantial existing test suite. This is a **structural review**, not a function-by-function duplicate or dead-code analysis.

| Candidate for further review | Why inspect it | Required evidence before cleanup |
| --- | --- | --- |
| `ep24_sna.py`, `ep24_sna_aggregate.py`, `ep24_graph_aggregate.py` | Overlapping-sounding graph responsibilities | Compare imports and callers, semantic purpose, outputs and graph tests. Do not merge merely because names overlap. |
| `ep24_cleaner.py`, `roihu_clean.py` | Multiple cleaning generations | Check dataset-specific meaning, CLI workflows, provenance and regression fixtures. |
| `ep24_mongo_export.py`, `ep24_mongo_dashboard_export.py`, `scripts/ep24/export_mongo_results.py` | Multiple export surfaces | Inspect output consumers and private dashboard contracts before sharing helpers. |
| `roihu_memory.py`, `ep24_memory.py`, `roihu_codebooks.py` | Similar-sounding persistence/config concepts | Distinguish memory store from codebook semantics, private boundaries and runtime responsibilities. |
| `scripts/roihu/run_step_*.sh`, `submit_step_*.sh`, `setup_step_*.sh` | Many near-identical short wrappers | Check dispatch and production script invocation; keep backward-compatible filenames. |
| Stage-local prompts versus `ep24_prompt_contract.py` | Potentially duplicated assembly or undocumented versioning | Trace each real model call; snapshot prompts, schemas, hashes, and compare sample outputs. |

### Prioritized, bounded follow-on PRs

1. **Stage audit:** enumerate exact call graphs and prompt/model call sites; create a verified inventory of each public interface, input/output column, side effect and failure state. Record existing test coverage and which branches cannot be exercised without Roihu.
2. **Low-risk readability:** add focused docstrings and descriptive local names to one stage at a time; preserve literal prompts, defaults, exception types, output fields and ordering. Include before/after tests and review.
3. **Shared infrastructure:** consolidate only proven equivalent code paths, one pair per PR, while retaining compatibility adapters. Validate CLI/resume, exports and storage.
4. **Prompt catalogue:** record exact prompt origin, codebook dependency, schema, model, version/hash and uncertainty handling per live call. Reword only after a traceable regression benchmark and methodological approval.
5. **End-to-end acceptance:** run fixed synthetic fixtures in CI, then identical approved source IDs on Roihu with manual evaluation of DA/DNA/SNA claims. Document differences, including LLM variation.

### Verification status of this increment

This document changes no Python, pipeline prompts, data contracts, CLI commands or runtime configuration. No behavioral tests or Roihu run were executed by the GitHub connector in this documentation-only increment. The issue is **not complete** until stage-by-stage code changes and validation are separately merged.
