# EP24 Step 5 Roihu execution readiness

Companion to Step 4 pilot guide `docs/EP24_STEP4_ROIHU_EXECUTION.md`.
Step 5 extracts structured entities, themes, sentiment targets and video
quality dispositions from completed Step 4 multimodal summaries.
It is **not** the Laclau discourse-analysis stage.

## Confirm readiness before allocating GPU

- Update the Roihu public checkout on the login node; record `git rev-parse HEAD`.
  Ensure the nested-JSON SQLite checkpoint fix in PR #267 is present
  in the checkout, since Step 5 uses the same `ep24_backups.write_checkpoint`.
- Use the same private `.env`, `LACLAUGPT_DATASET`, Mongo database, and
  per-country collection namespace used for Steps 3 and 4. Never log secrets.
- Confirm the selected country's MongoDB documents have
  `_pipeline.step_04.status=complete` and nonempty `summary_analysis`.
  Check `summary_quality_status` and any failed Step 4 rows before proceeding.
  The orchestrator's eligible query checks upstream completion but does **not**
  by itself enforce nonempty `summary_analysis`: quality review remains required.
- Confirm the correct ARM64 PyTorch environment and Ollama model are installed.
  `scripts/roihu/step_5_roihu_postprocess.sbatch` loads PyTorch, starts Ollama,
  runs preflight 5 and dispatches `step_5_roihu_postprocess.py` through the
  Mongo-backed stage orchestrator.
- Inspect `LACLAUGPT_STEP5_MAX_PROMPT_CHARS` (default 12000),
  `LACLAUGPT_POSTPROCESS_NUM_CTX` (32768), and
  `LACLAUGPT_POSTPROCESS_NUM_PREDICT` (2048). Do not include cumulative
  memory/RAG or earlier generated prompts as evidence by accident.

## Pilot: ten eligible Finland records

On a Roihu login node, once Step 4's pilot has completed and been reviewed:

```bash
cd /scratch/project_2009497/LaclauGPT-Multimodal-Analysis
git pull --ff-only
export LACLAUGPT_COUNTRY=finland
export LACLAUGPT_MAX_ROWS=10
export LACLAUGPT_STEP5_MAX_PROMPT_CHARS=12000
bash scripts/roihu/run_step.sh 5 --country finland --limit 10
```

This is a **real** job submission, not a dry run. It selects the next ten
eligible Mongo claims, **not necessarily the exact Step 4 pilot IDs**.
Never add `--force` to select sample rows. To compare exact paired IDs,
record the claimed IDs and compare only common source identities; an isolated
ID-filtered sample is needed for the fixed FI/PL/PT benchmark.

Review job stdout/stderr at
`/scratch/project_2009497/logs/ep24_s5_<jobid>.out` and `.err`;
private runtime details at `logs/postprocess.log`, plus the MongoDB
`_pipeline.step_05` statuses and the cumulative checkpoint.
Check per-record failure flags rather than treating subprocess exit 0 as proof
that every extraction passed. Preserve the original annotations and upstream
columns.

## Manual quality gate

For each sampled source video and Step 4 summary:
1. Compare extracted entity and theme arrays to directly supported
   original-language speech, OCR, source summary, and researcher seeds.
2. Check sentiment polarity **and its target**: no invented targets or
   inferred affiliation from source metadata alone.
3. Ensure missing evidence produces empty lists, not fabricated entities.
4. Check canonical labels, multilingual cases, duplicate merging, contradictory
   evidence, and valid JSON/schema-constrained fields.
5. Compare the `OK/REPROCESS/DELETE` disposition to upstream quality signals.
   A status of DELETE is a review disposition, not permission to delete
   source media or rows without a separate approved operation.
6. Record model identity/options, run SHA, source IDs, errors, and quality
   decisions; compare with historical outputs on matched source IDs where present.

Do not submit a full-country run until this sample is reviewed. If there
are no eligible rows, inspect `step_04.status`, `step_05.status` and
country/dataset namespace. Completed source rows are not reprocessed by
default; do not mass-reset them just to create a pilot.

## Scope and completion

Code for bounded Step 5 extraction is already present, and issue #237 was
closed. This document does not reopen the issue or claim that the Roihu pilot
has happened. Cluster execution and the manual quality gate are outstanding.
