# EP24 Step 6 Roihu execution: Laclau/Palonen pilot (#239)

Step 6 invokes `step_6_roihu_discourse_analysis.py` -> `roihu_populism.py`
through `ep24_stage_orchestrator.py`. It applies the **locked**
Laclau/Mouffe/Palonen research methodology and Formula of Populism.
Do not redesign the theory, status schema or thresholds for this pilot.

## Prerequisites before a GPU allocation

1. On the Roihu login node, update the public checkout, record
   `git rev-parse HEAD`, and verify the nested-object SQLite checkpoint
   repair (#267) is present. This stage shares `ep24_backups.write_checkpoint`
   with Steps 2–5.
2. Check that the chosen country and private `.env` point to the **same**
   dataset, Mongo database and collection namespace used by Steps 0–5.
   Do not echo credentials or private research evidence into GitHub.
3. Verify `_pipeline.step_05.status=complete` for the pilot source records.
   The generic orchestrator checks upstream completion only, not whether
   structured `postprocess_entities` / `postprocess_themes` or
   `summary_analysis` are semantically correct. Review their quality first.
4. Confirm the PyTorch ARM64 environment, Ollama installation and configured
   `LACLAUGPT_MULTIMODAL_MODEL`. The existing Step 6 sbatch script loads
   PyTorch, starts Ollama, runs preflight 6, and requests a GH200 GPU.
5. Confirm budget/model options: `LACLAUGPT_STEP6_MAX_CONTEXT_CHARS`
   (default 28000), `LACLAUGPT_STEP6_NUM_CTX` (default 16384),
   `LACLAUGPT_STEP6_NUM_PREDICT` (default 4096), and
   `LACLAUGPT_STEP6_CHECKPOINT_EVERY` (default 10).
   The source-first packet excludes unrelated cumulative content;
   inspect the logged prompt version, context hash and truncation.

## First pilot: at most 10 Finland records

Run on the Roihu **login node**, only after Step 5's pilot is reviewed:

```bash
cd /scratch/project_2009497/LaclauGPT-Multimodal-Analysis
git pull --ff-only
export LACLAUGPT_COUNTRY=finland
export LACLAUGPT_MAX_ROWS=10
export LACLAUGPT_STEP6_MAX_CONTEXT_CHARS=28000
export LACLAUGPT_STEP6_NUM_CTX=16384
export LACLAUGPT_STEP6_NUM_PREDICT=4096
bash scripts/roihu/run_step.sh 6 --country finland --limit 10
```

This submits a **real** SLURM job via
`scripts/roihu/step_6_roihu_discourse_analysis.sbatch`, not a dry run.
Do not use `--force`. `--limit 10` selects the next ten eligible Mongo
claims (sorted by `_storage_id`), **not** automatically the same ten source
IDs as the Step 4 or Step 5 pilot. Log the actual IDs and do matched-record
comparisons. For fixed benchmark IDs, use an isolated ID-scoped copy.

Logs are `/scratch/project_2009497/logs/ep24_s6_<jobid>.out` and
`.err`; inspect the private Step 6 logs, MongoDB stage status, complete
structured outputs, row-level errors and CSV/SQLite manifest checksums.
`stage_subprocess_end returncode=0` alone does not prove every Laclau
analysis meets the schema or quality gate.

## Theoretical and evidence review

For every pilot record, compare its **original speech, translation, OCR,
video narrative, Step 4 fusion, Step 5 extracted annotations** and Step 6
analysis with source provenance retained. Score:

- Source quotations and original-language signifiers are grounded in the
  current video; retrieved codebook and prior analyses are not source proof.
- A political collective **Us** is not inferred from a pronoun alone.
- **Antagonistic frontier** is distinguished from criticism, blame, opposition
  or negative sentiment.
- The **Formula of Populism** is true only with independently evidenced
  collective Us **and** antagonistic frontier. Otherwise, explicit abstention
  is valid and expected.
- Affects are not mechanically mapped onto positive/negative sentiment;
  chains require articulation rather than mere co-occurrence.
- Nodal/floating/empty signifier and Palonen claims express appropriate
  uncertainty; corpus-level hegemony is not asserted for one source.
- Output respects `EP24DiscourseAnalysis` schema, candidate provenance,
  confidence bounds, empty-list semantics, error/status fields and
  source-level counter-evidence.
- The Step 6 output leaves previous researcher annotations, all preceding
  columns and the existing locked methodology intact.

Record the exact source IDs, model/options, prompt version/context hash,
missing or truncated evidence, source-grounding errors, and comparison to
previous output **for matching IDs**. Review Finnish first, then extend the
same gate to Poland and Portugal (ten each). Stop on substantive quality
failures; inspect records individually instead of mass resetting statuses.
`DELETE`/reprocess flags do not authorize deletion of original source media.

## Readiness versus acceptance

The Step 6 model-evidence implementation was previously delivered under
#239 (notably #257/#260). This document prepares execution only: no Roihu
job is submitted and no manual corpus validation is claimed. Keep #239
open until the source-grounded runtime comparison and CI acceptance pass.
