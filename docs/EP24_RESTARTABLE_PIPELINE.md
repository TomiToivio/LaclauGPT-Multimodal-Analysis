# Restartable EP24 Roihu pipeline

Issue #64 adds a MongoDB-first orchestration layer around the existing numbered EP24 scripts. The analysis scripts remain readable and independently runnable. The orchestration layer decides which records are eligible, writes small durable batches, and preserves the cumulative dataframe contract.

## Bootstrap before Step 1

Run from the public repository checkout:

```bash
python scripts/ep24/bootstrap_ep24_mongodb.py
```

Bootstrap discovers `LaclauGPT-Private/analysis/ep24_reprocess/data/to_reprocess/ep24_*.csv` and processes countries in this order:

1. Finland
2. Poland
3. Portugal
4. all remaining countries alphabetically

Before Step 1, the private country CSVs are already canonicalized by the issue #21 migration. Human annotations arrive only in `entities` and `themes`; the four legacy annotation columns have been removed after verification.

Bootstrap validates that canonical schema, creates stable `_storage_id` values, imports the cumulative records to MongoDB, loads available bilingual private codebooks into MongoDB, seeds normalization memory, and writes Step 0 CSV + SQLite backups with checksums.

## Storage roles

MongoDB is the durable source of truth. Country collections use the existing `laclaugpt_ep2024_reprocess_<country>_<purpose>` convention.

Redis is optional transient coordination/cache. It mirrors per-record status and locks but is never the only durable state.

CSV and SQLite checkpoints are written under:

```
$LACLAUGPT_EP24_OUTPUT_ROOT/<country>/
```

after every successful batch.

## Active pipeline scope (issue #380)

Steps 1–9 are active by default. Step 8 performs descriptive Social Network Analysis only (no Castells or Luhmann interpretation), and Step 9 deterministically exports RDF on CPU. Historical code and prior stored results remain intact; enabling these stages does not authorize automatic production-job submission. Use `bash scripts/roihu/run_country_pipeline.sh COUNTRY --from-step 8 --to-step 9` after verifying prior stage completion, if only the previously paused steps need running. Methodological reference: Borgatti, Everett and Johnson, *Analyzing Social Networks*, 2nd ed. The privately provided ebook is not included in the repository.

## Partial-pipeline semantics

A Step N record is eligible only when Step N-1 is complete. Each stage claims a small batch atomically in MongoDB, runs the unchanged numbered analysis script against a temporary cumulative CSV, persists the output, marks those rows complete, updates RAG, and writes a cumulative checkpoint.

This means Step 2 can process completed Step 1 rows while Step 1 still has pending rows.

Default batch size is 25 rows:

```bash
export LACLAUGPT_STAGE_BATCH_SIZE=25
```

The orchestrator stops at a soft deadline of 126000 seconds (35 hours) by default, before the Roihu `gpumedium` 36-hour limit:

```bash
export LACLAUGPT_STAGE_SOFT_DEADLINE_SECONDS=126000
```

Stale Mongo claims are recovered as retryable work on the next run.

## One-command Roihu operation

The launcher reads the private `.env`, exports the Mongo/Redis/runtime settings, and submits the matching sbatch job:

```bash
bash scripts/roihu/run_step_1.sh
bash scripts/roihu/run_step_2.sh
bash scripts/roihu/run_step_3.sh
...
bash scripts/roihu/run_step_9.sh
```

Useful selectors pass through to the stage orchestrator:

```bash
bash scripts/roihu/run_step_1.sh --country finland
bash scripts/roihu/run_step_2.sh --country finland --limit 100
bash scripts/roihu/run_step_4.sh --retry-errors
bash scripts/roihu/run_step_6.sh --country poland --force
```

Direct execution of the analysis files remains available for manual tuning, for example:

```bash
python3 step_4_roihu_summary.py
```

## Progress

```bash
python scripts/ep24/ep24_status.py
python scripts/ep24/ep24_status.py --country finland
```

### Identity in the progress view (#379)

A total is not a count of records: duplicate rows inflate it, and a row without `_storage_id` can neither
be resumed nor exported. The progress view therefore prints the dataset and collection it actually read,
then the total, the distinct id count, duplicates and rows missing an id:

```
finland: dataset=ep24_monday500_20261031 collection=laclaugpt_ep24_monday500_20261031_finland_dataframe
finland: total=500 unique=500 duplicates=0 missing_id=0
```

That is reporting. The **gate** before a resubmission is
[`scripts/ep24/preflight_pilot.py`](../scripts/ep24/preflight_pilot.py), which asserts the pilot's identity,
dataset isolation and stale claims and exits non-zero — one gate, one home. Do not add a second pass/fail
check to the status command: two gates drift apart, and the one an operator did not run is the one that was
right.

## Codebook, Memory and RAG context

Before each batch, the orchestrator appends bounded derived-context columns:

- `codebook_context_json`
- `memory_context_json`
- `rag_context_json`
- `entity_normalization_json`
- `theme_normalization_json`
- `context_evidence_role`

These are explicitly normalization/retrieval context, not source evidence. The current row is excluded from its own RAG retrieval. Successful stage output is added back to Mongo RAG for subsequent batches.

## Allas staging

Step 1 stages media before invoking the legacy preprocess. If `allas_filename` is an HTTP(S) URL it is downloaded directly. For CSC object keys, define a private command template in the private `.env`:

```bash
LACLAUGPT_ALLAS_FETCH_COMMAND='YOUR_PRIVATE_FETCH_COMMAND {object} {destination}'
```

The public repository deliberately does not embed credentials or a project-specific Allas command.

## RDF

Step 9 consumes the cumulative orchestrated input. RDF identity prefers `_storage_id` and canonical `video_id`, so graph identity remains joinable to MongoDB. The historical `graph.nt` output is retained and the same deterministic graph is also emitted as `graph.ttl`.

## Recovery

If a job dies:
- completed batches stay complete;
- the current claimed batch becomes stale and is recovered to `retry`;
- rerunning the same step skips completed rows unless `--force` is used;
- cumulative CSV/SQLite checkpoints remain available independently of MongoDB.

The machine-readable boundary contract is `config/ep24_pipeline_columns.json`.

### Resuming a sample pilot, without redoing the full corpus (#379)

A resumed chain inherits whatever dataset its environment resolves to, and `roihu_job_common.sh`
defaults `LACLAUGPT_DATASET` to `ep2024_reprocess` when the env file omits it. A pilot resume therefore
needs two guards, both now enforced:

```bash
# 1. Verify identity, isolation and claim state. Read-only; exits non-zero on any HARD STOP condition.
python scripts/ep24/preflight_pilot.py --countries finland poland portugal \
    --dataset ep24_monday500_20261031 --expect-unique 500

# 2. Refuse to submit unless every Slurm job will inherit the pilot dataset.
bash scripts/roihu/run_country_pipeline.sh portugal --from-step 1 --to-step 7 --limit 0 \
    --require-dataset ep24_monday500_20261031
```

The preflight checks what `ep24_status.py` cannot: **distinct `_storage_id` records** (500 rows over 480
identities is a duplicate-identity resume), dataset isolation (the full-corpus namespace is refused by
name), and stale claims per step, which explain why records are ineligible for up to six hours after a
killed job. The submission guard resolves the dataset exactly as the job will — from
`LACLAUGPT_EP24_ENV_FILE` — prints it, writes it into the run manifest, and refuses on mismatch; put
`LACLAUGPT_DATASET` in the env *file*, not only in your shell, because `sbatch` copies the job's
environment from submission time.

