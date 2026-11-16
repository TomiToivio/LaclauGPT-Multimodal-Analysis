# Repository map

Where everything lives and why. Written for a researcher who needs to follow one record from source
material to an analytical claim, and for a reviewer who needs to know which module is authoritative
before changing it.

The map is checked by `tests/test_repository_map_is_current.py`, which fails if a top-level module exists
that this file does not name — so it cannot quietly rot.

## 1 What the pipeline is

A nine-stage, cumulative EP24 pipeline. Each stage reads the previous stage's dataframe, appends its own
columns, and persists; a record accumulates evidence and interpretation as it goes down the steps, and
every stage can be re-run for one country without touching the others.

Stage numbering, entry points, appended columns and the per-stage notes are declared in
`ep24_stage_contract.py` — that module is the source of truth. This table is generated from it:

| step | stage | entry point | Slurm wrapper | appended columns |
| ---: | --- | --- | --- | ---: |
| 1 | `preprocess` | `step_1_roihu_preprocess.py` | `scripts/roihu/step_1_roihu_preprocess.sbatch` | 16 |
| 2 | `frame` | `step_2_roihu_frame.py` | `scripts/roihu/step_2_roihu_frame.sbatch` | 9 |
| 3 | `video` | `step_3_roihu_video.py` | `scripts/roihu/step_3_roihu_video.sbatch` | 54 |
| 4 | `summary` | `step_4_roihu_summary.py` | `scripts/roihu/step_4_roihu_summary.sbatch` | 7 |
| 5 | `postprocess` | `step_5_roihu_postprocess.py` | `scripts/roihu/step_5_roihu_postprocess.sbatch` | 26 |
| 6 | `discourse_analysis` | `step_6_roihu_discourse_analysis.py` | `scripts/roihu/step_6_roihu_discourse_analysis.sbatch` | 21 |
| 7 | `discourse_network_analysis` | `step_7_roihu_discourse_network_analysis.py` | `scripts/roihu/step_7_roihu_discourse_network_analysis.sbatch` | 20 |
| 8 | `social_network_analysis` | `step_8_roihu_social_network_analysis.py` | `scripts/roihu/step_8_roihu_social_network_analysis.sbatch` | 2 |
| 9 | `rdf` | `step_9_roihu_rdf.py` | `scripts/roihu/step_9_roihu_rdf.sbatch` | 0 |

Two conventions matter when reading the code:

* **`step_N_roihu_*.py` is the executable stage; `roihu_<stage>.py` is its implementation.** The step file
  parses arguments, opens the log, and calls into the implementation module. When you want to know what a
  stage *does*, read `roihu_<stage>.py`; when you want to know how it is *invoked*, read the `step_` file.
* **A stage appends columns, it does not rewrite them.** The `appends` tuple in the contract is the
  authoritative list, and `tests/test_ep24_stage_contract.py` guards it. Human-coded legacy columns are
  never overwritten by a model-derived stage: model output lands in its own columns.

## 2 Which module is which

### Orchestration, state and contracts

| module | role |
| --- | --- |
| `ep24_stage_contract.py` | Declares stages, entry points, appended columns, per-stage notes. Source of truth for the table above. |
| `ep24_stage_orchestrator.py` | Runs one stage over only the records that are eligible in MongoDB; the batch driver. |
| `ep24_stage_state.py` | Durable per-record stage eligibility, claims and completion. Resume depends on this. |
| `ep24_scope.py` | Which countries/stages the active numbered pipeline covers. |
| `ep24_cli.py` | The shared command-line contract every `step_N_*` follows. |
| `ep24_eligibility.py` | Non-blocking stage eligibility rules. |

### Stage implementations

| module | stage | notes |
| --- | --- | --- |
| `roihu_preprocess.py` | 1 | Media staging, one keyframe at t=1.0s, OCR, backend-neutral ASR. |
| `roihu_frame.py` | 2 | Cumulative single-keyframe multimodal analysis. |
| `roihu_summary.py` | 4 | Whole-record summary; prompt built from accumulated context, cached per (prompt, model). |
| `roihu_postprocess.py` | 5 | Structured entity/theme/sentiment extraction, kept separate from immutable human columns. |
| `roihu_populism.py` | 6 | Laclau/Mouffe/Palonen discourse analysis: evidence-first, abstention-capable. |
| `roihu_enrich.py` | — | Additive codebook/memory enrichment between steps 5 and 6. |
| `ep24_dna.py` | 7 | Leifeld DNA/rDNA coding and compatibility layer. |
| `ep24_sna.py`, `ep24_sna_aggregate.py` | 8 | Deterministic social-network construction, persistence and cross-video aggregation. |
| `roihu_rdf.py`, `roihu_csv_rdf.py` | 9 | Deterministic RDF/knowledge-graph projection of the finished dataframe. |
| `ep24_video.py` | 3 | Shared skip/trim rules for the video stage (a library, not an entry point). |
| `ep24_models.py` | — | Per-stage model resolution: which model a stage asks for, and why. |
| `asr_backend.py`, `ocr_backend.py` | 1 | Backend-neutral ASR and OCR adapters. |

### Prompting, codebooks and context

| module | role |
| --- | --- |
| `ep24_prompt_contract.py` | Shared prompt/context contract for steps 4–8: what the model may see, in what order, with which provenance. |
| `roihu_codebooks.py` | Loads and retrieves the private bilingual EP24 codebooks. |
| `roihu_codebook_sources.py` | Loads the private `codebook_sources` replacement workbooks into registry entries. |
| `roihu_identity.py` | Conservative codebook identity resolution and human seed recycling. |
| `ep24_context.py` | Mongo-backed codebook, memory, RAG and normalization context for a stage run. |
| `ep24_memory.py`, `roihu_memory.py` | Research memory: retrieval, seeding, persistence. |
| `ep24_rag.py` | RAG persistence and retrieval helpers. |
| `ep24_result_reporting.py` | The human-readable per-stage report a researcher reads after a run. |

### Storage, export and audit

| module | role |
| --- | --- |
| `roihu_storage.py` | External MongoDB storage plus dataframe/CSV interoperability. |
| `ep24_db.py` | Thin EP24-facing helpers over `roihu_storage`. |
| `ep24_pipeline.py` | The shared dataframe contract for the active pipeline. |
| `ep24_input_schema.py` | Inspects the real incoming schema before preprocessing. |
| `ep24_mongo_export.py` | Read-only export of finished results from shared MongoDB. |
| `ep24_mongo_dashboard_export.py` | Read-only export shaped for the dashboards. |
| `ep24_graph_aggregate.py` | Read-only aggregation of graph layers across videos and countries. |
| `ep24_coverage_audit.py`, `scripts/ep24/audit_*` | Coverage, contract, diff and bundle audits. |
| `laclaugpt_quality.py` | The deterministic OK/REPROCESS/DELETE quality gate shared by steps 2–5. |

### Settings, logging, diagnostics

| module | role |
| --- | --- |
| `ep24_settings.py` | Loads private runtime settings without logging secrets. |
| `ep24_logging.py` | One log directory and one filename convention for every step (the one the steps use). |
| `ep24_logs.py` | Older implementation of the same convention, still used by the orchestrator. See `AUDIT_411.md` — it is a consolidation candidate, not a second convention. |
| `ep24_backups.py` | Atomic CSV/SQLite/manifest checkpoints for cumulative frames. |
| `ep24_redis.py` | Optional Redis coordination; MongoDB remains the durable source of truth. |
| `ep24_result_reporting.py` | Human-readable, private per-stage reporting. |

### Historical implementations kept for compatibility

`puhti_*.py` and the historical `roihu_*.py` compatibility modules exist so that published results remain
reproducible and comparable. `AGENTS.md` is explicit: the immutable historical record of the original
human-coded EP24 / CSC Puhti pipeline lives on the **`legacy` branch** and must not be modernised; on
`main`, `puhti_*.py` are reference implementations whose correspondence to the active `roihu_*.py` files is
pinned by `tests/test_legacy_safety.py`.

Two rules follow, and they are the ones a refactor most often breaks:

1. Do not delete or "clean up" a `puhti_*` module. If it stops being needed, that is a decision, recorded.
2. Do not make a `roihu_*` module import a `puhti_*` module, or the other way round: the pairing is
   deliberate duplication whose value is that the two can be compared.

### Scripts, tests, examples, experiments

| path | role |
| --- | --- |
| `scripts/ep24/` | Task-shaped tools: audits, coverage checks, hygiene scans, bootstrap. Each has a one-line purpose in its module docstring and is listed in `docs/` where it is used. |
| `scripts/roihu/` | Slurm wrappers per stage. |
| `scripts/asr_bench/` | ASR benchmark harness. |
| `tests/` | pytest suite; `tests/test_ep24_stage_contract.py`, `tests/test_legacy_safety.py` and `tests/test_repository_map_is_current.py` guard the structure itself. |
| `examples/` | Runnable demos on synthetic fixtures (for instance the step-7 DNA export demo). |
| `experiments/` | Not part of the pipeline. The video stage wraps `experiments/vllm_video_test.py`; treat the rest as scratch code. |
| `docs/` | Contracts, audits and policies. `docs/EP24_CANONICAL_SCHEMA.md` is the record schema; `docs/CODING_STYLE.md` is the house style. |

## 3 How to run, resume and inspect

```bash
# one stage for one country (the entry point is always the step_ file)
python step_4_roihu_summary.py --country FI

# one stage through the orchestrator, which only touches eligible records
python ep24_stage_orchestrator.py --step 4 --country FI

# on CSC Roihu, the same thing through the stage's Slurm wrapper
sbatch scripts/roihu/step_4_roihu_summary.sbatch
```

* **Resume.** `ep24_stage_state.py` records stage eligibility and completion per record. Re-running a
  stage skips completed records; a failed record stays failed until its eligibility is re-evaluated, so a
  re-run is safe and does not silently re-do good work.
* **Inspect one record.** The dataframe columns are the trail: `metadata` (prompt identity and context
  hashes), `<stage>_prompt_sha256`, `<stage>_summary_md`, `<stage>_structured_json`, and
  `processing_status`/`processing_status_reason`. The same values are persisted in MongoDB, which is the
  durable source of truth.
* **Debug a stage.** Set `EP24_LOG_LEVEL=DEBUG`; logs land where `ep24_logging` says they do, one file per
  (step, country). `ep24_result_reporting.py` produces the human-readable per-stage report.
* **Test.** `python -m pytest -q` from the repository root. The suite needs no network and no CSC
  credentials; anything that would reach a cluster is faked in the test.

## 4 Where the analytical decisions live

| decision | where it is written down |
| --- | --- |
| What each stage operationalises | the `notes=` field of each stage in `ep24_stage_contract.py`, plus the module docstring of its implementation |
| Prompt content and ordering | `ep24_prompt_contract.py` and the stage's prompt builder (`build_postprocess_prompt`, `_prompt_*` helpers) |
| Codebook definitions and versioning | `roihu_codebooks.py` plus the private workbooks; `docs/EP24_CODEBOOK_*` |
| What counts as evidence | step 6's evidence-first rules in `roihu_populism.py`; `docs/EP24_ISSUE306_PRIVACY_GATE.md` for what may leave |
| What is a quality failure | `laclaugpt_quality.py` |
| What is a known defect | `docs/EP24_DEFECT_REGISTER.md` |

## 5 Glossary

* **EP24** — the electoral-politics media corpus and its pipeline; nine cumulative stages.
* **Roihu** — CSC's current supercomputer; the active execution environment. **Puhti** is the previous
  one, used by the historical implementation on the `legacy` branch.
* **Cumulative dataframe** — the per-country table that each stage reads and appends to; the record's
  working state.
* **Stage contract** — the declaration of what each stage appends, and what it must not overwrite.
* **Human-coded columns** — values produced by researchers, never overwritten by model output.
* **DNA / rDNA** — Leifeld's discourse network analysis and its relational form; step 7.
* **SNA** — social network analysis over actor-to-actor ties; step 8.
* **Allas** — CSC object storage for media; staged before step 1 runs.
* **Eligibility** — whether a record may run a stage yet; stored in `ep24_stage_state`.

The grouped tables above explain the modules a reader is most likely to need. This index is the complete
list, so that nothing in the repository is unnamed — `tests/test_repository_map_is_current.py` fails if a
top-level module is missing from this file.

| module | purpose |
| --- | --- |
| `ep24_allas.py` | Stage EP24 media from CSC Allas before Step 1 runs. |
| `ep24_bootstrap.py` | EP24 country bootstrap for MongoDB-first restartable processing. |
| `ep24_cleaner.py` | Auditable EP24 legacy-data cleaner (issue #35). |
| `ep24_entities.py` | Multilingual entity normalization and resolution (issue #142). |
| `ep24_schema.py` | Researcher-feed metadata helpers. |
| `ep24_status_identity.py` | Identity arithmetic for the status tool, kept out of a script's import side effects (#379). |
| `roihu_clean.py` | Non-destructive cleaning of legacy per-country dataframes (issue #35). |
| `step_0_roihu_import.py` | Step 0: seed the Mongo dataframe from private canonical country CSVs. |
| `puhti_frame.py` | Historical step 2 implementation (CSC Puhti). Reference only — paired with `roihu_frame.py` by `tests/test_legacy_safety.py`. |
| `puhti_preprocess.py` | Historical step 1 implementation (CSC Puhti). Reference only. |
| `puhti_summary.py` | Historical step 4 implementation (CSC Puhti). Reference only. |
| `puhti_postprocess.py` | Historical step 5 implementation (CSC Puhti). Reference only. |
| `puhti_populism.py` | Historical step 6 implementation (CSC Puhti). Reference only. |
