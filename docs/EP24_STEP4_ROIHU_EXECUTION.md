# EP24 Step 4: Roihu execution checklist (#236)

Step 4 is the **bounded multimodal social-semiotic summary**, not Laclau discourse analysis.
The existing implementation and prompt-budget tests landed in #244 and #262. This
checklist is for the **first runtime quality gate**, not a mass rerun.

## Before allocating a GPU

1. On Roihu, update the public checkout and check the branch/commit:
   ```bash
   cd /scratch/project_2009497/LaclauGPT-Multimodal-Analysis
   git pull --ff-only
   git rev-parse --short HEAD
   ```
2. Confirm the SQLite nested-value checkpoint repair (#267) is merged into
   the checked-out commit before launching Stage 4. Stage 2/3 had successful
   MongoDB commits but their SQLite exports failed on dictionary cells.
   Step 4 shares the same `ep24_backups.write_checkpoint` path.
3. Inspect the private environment **without printing secrets**:
   `/scratch/project_2009497/LaclauGPT-Private/analysis/ep24_reprocess/.env`.
   Step 4 needs the same dataset/database/collection namespace as Step 3,
   a reachable MongoDB, the Step 4 Roihu Python environment, and the selected
   local Ollama model. Do not paste the Mongo URI or credentials into issues.
4. Check Step 3 **MongoDB** upstream completion (not just its possibly missing
   SQLite checkpoint). The orchestrator permits Step 4 only for documents with
   `_pipeline.step_03.status=complete`. Check summary input completeness,
   especially `vllm_video_analysis` and previous ASR/frame/OCR evidence.
   Any upstream failures must be reviewed separately. No global status reset.

## First bounded run: Finland, 10 rows

From the repository root on a **Roihu login node**:

```bash
export LACLAUGPT_COUNTRY=finland
export LACLAUGPT_MAX_ROWS=10
export LACLAUGPT_SUMMARY_INCLUDE_RETRIEVAL=0

# This submits a real job, *not* a dry run. Stage 4's Mongo claims
# select up to ten eligible records. Never add --force for a first run.
bash scripts/roihu/run_step.sh 4 --country finland --limit 10
```

The submitter loads the private .env and passes `--country`/`--limit` to
`ep24_stage_orchestrator.py` via `step_4_roihu_summary.sbatch`. Verify the
effective namespace and batch count in the Slurm output before extending work.
Output: `/scratch/project_2009497/logs/ep24_s4_<jobid>.out` and `.err`,
plus the private cumulative Step 4 outputs and MongoDB records. Inspect the
`context_budget` log lines and the per-record success/quality fields.

**Important sampling caveat:** `--limit 10` takes the next 10 **eligible Mongo
claims**, not necessarily the same deterministic IDs selected by the offline
`step4_evidence_coverage.py` worklist. Record the *actual* claimed
`_storage_id`/`video_id` from the batch and compare old/new outputs for
those exact IDs; do not pretend that unrelated first-10 samples are paired.
For the runbook's cross-country 30-item deterministic benchmark, use an
isolated sample with explicit ID filtering, not `--force` on production Mongo.

## Quality gate

For each analyzed video, inspect the original-language ASR, translated ASR,
video narrative, first-frame analysis, OCR and generated Step 4 output.
Record source ID, model and options, packet size/truncation, and scores for:
faithful temporal account; supported entities/quotations; multilingual
accuracy; contradictory evidence kept visible; no invented visuals/speech;
Halliday/SFL and Kress/van Leeuwen distinctions; no premature Laclau discourse
classification; quality disposition (OK/REPROCESS/DELETE).

If an upstream Step 3 record is marked complete but its analysis is unusable,
review/requeue that record specifically before Step 4. Never reset all rows.
Review the batch side by side with historical output; an old output comparison
is impossible when a matching baseline does not exist, so mark it as such.
**Do not launch full-country runs until this manual review passes.** Post
the sample IDs, aggregate review observations and go/no-go to #236 without
exposing private research content.

For a read-only coverage worklist, using an **existing locally available CSV**
(no Mongo writes):

```bash
python3 scripts/ep24/step4_evidence_coverage.py /path/to/existing-step4-input.csv --per-country 10
```

The CSV argument must be a real input file, not an invented export path.
If Step 4 reports no eligible records, verify `step_03.status` counts,
country namespace and already-completed Stage 4 statuses. Do not use
`--force` to get around eligibility.

## Outcome

A bounded 10-row *pilot is prepared*, but not run by these instructions.
Passing unit tests or generating the worklist does not satisfy #236's manual
Roihu comparison. The issue remains open until evidence fidelity has been
reviewed with the real corpus.
