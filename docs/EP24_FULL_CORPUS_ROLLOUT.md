# EP24 full-corpus rollout (#373)

**Status:** Prepared for operator review. Nothing in this change submits a real job by default. The FI/PL/PT Monday500 sample must be validated and explicitly approved before processing the rest.

## 1. Verify the actual sample and MongoDB namespace

The canonical full inputs are private CSVs in `LaclauGPT-Private/analysis/ep24_reprocess/data/to_reprocess/`. The new controller uses the public `ep24_bootstrap.prepare_dataframe` canonical schema and the active `roihu_storage.StorageConfig` dataset/country prefix. A separate Monday500 collection is **not** silently assumed equivalent to the active production namespace.

Before proceeding, inspect live stage coverage (including error/partial data and duplicate source video IDs), accepted prompts, models and codebooks, and verify that the selected `LACLAUGPT_DATASET` / `LACLAUGPT_MONGO_DATABASE` addresses the actual accepted sample records. If the 500 sample is in another namespace, **migrate/reconcile it through a separately reviewed procedure** before running `--execute`; do not start a fresh full run in an empty namespace.

Examples, from the Roihu checkout with the private environment configured:

```bash
export LACLAUGPT_MONGO_ENABLED=1
export LACLAUGPT_DATASET=ep2024_reprocess
python scripts/ep24/ep24_status.py --country finland
bash scripts/ep24/run_remaining_fi_pl_pt.sh --dry-run
bash scripts/ep24/run_remaining_countries.sh --dry-run
```

Read-only plans show total input rows, unique media, exact duplicate/invalid inputs, number matched by **(video_id, allas_filename)**, missing rows, existing statuses for steps 1–7 and the resolved Mongo collection. They refuse ambiguous identifiers, missing existing media identities, or an existing Mongo sample not represented in the canonical full input. This is stricter than trusting a positional 500-row offset.

**Before approval:** review the matched sample rows and individual stage validity, not only `_pipeline.step_07.status=complete`. In particular, confirm no degenerate video outputs, no duplicate video IDs counted as independent videos, no inconsistent prompt/model/codebook versions, and no source/provenance drift. The planning command audits stage status, **not** semantic validity. Obtain a documented sample sign-off in a private, nonempty approval file.

## 2. Launch remaining analysis only after sign-off

```bash
# Write the signed/approved sample decision outside the public repository.
bash scripts/ep24/run_remaining_fi_pl_pt.sh --execute --approval-file /private/ep24_sample_approved.txt
bash scripts/ep24/run_remaining_countries.sh --execute --approval-file /private/ep24_sample_approved.txt
```

Each launcher first performs the same reconciliation, then uses **insert-only** Mongo updates (`$setOnInsert`) for missing canonical input rows. Existing sample documents, stage fields and run provenance are left untouched. The country runner `scripts/roihu/run_country_pipeline.sh` is reused for steps **1–7 only**, `--limit 0`, without `--force` or automatic `--retry-errors`. Already complete stage outputs are not reprocessed; stages with missing predecessor outputs wait for upstream stages. Existing errored rows require an explicit selective retry, after review. For the other countries an existing nonempty collection blocks the new-country write instead of destructively resetting it.

Do not use `scripts/roihu/run_pipeline.sh --full` for this rollout: its general legacy compatibility controller includes steps 8–9. Normal rollout excludes those paused steps. Step chain logs/Slurm IDs are emitted by the existing country runner. Slurm `afterany` dependencies are now accepted so incomplete stages can leave data for subsequent stages to inspect; **this does not mean all previous stages succeeded**.

If a submission fails partway through, inspect the partial Slurm manifest, current jobs and Mongo stage state before rerunning. Existing jobs are not automatically cancelled, nor is duplicate concurrent submission prevented by the launcher. Do not submit two live chains for the same country at once.

## 3. Final full-dataframe CSV and SQLite exports

This is distinct from the lean dashboard exporter in `scripts/ep24/export_mongo_results.py`, which moves long text into separate bundles. The final researcher export keeps the **entire cumulative dataframe, all text, stage statuses, provenance and source fields**.

```bash
# Interim run: explicitly labeled as incomplete.
python scripts/ep24/export_full_corpus.py --output /private/ep24_exports/interim_001 --allow-incomplete

# Final run: fails if any country has an uncompleted Step 7 status.
python scripts/ep24/export_full_corpus.py --output /private/ep24_exports/final_001
```

Exports include `<country>_full_steps_1_7.csv`, `all_countries_full_steps_1_7.csv`, SQLite companions, SHA-256 manifests and `full_export_manifest.json`. They are UTF-8, deterministic by `_storage_id`, preserve all columns, JSON-encode nested cells and verify CSV read-back IDs, row/column counts and serialized provenance. No original private input is overwritten. CSV source-universe completeness is not inferred from Mongo counts alone: reconcile `input_rows`, `unique_media`, `invalid_input_rows` and expected dataset coverage in the plan against final export manifest before declaring final delivery. The exporter fails on missing IDs or duplicate `_storage_id`, and final mode refuses partial Step 7 coverage.

## 4. Independent off-node backup and restore verification

`ep24_backups.py` already creates atomic cumulative CSV, SQLite and JSON/checksum checkpoints. It now uses JSON serialization for nested cells in **both** CSV and SQLite and creates files with mode 0600.

Set up a restricted private CSC Allas/rclone remote and verify the remote credentials and encryption/security policy independently. **Do not** place credentials or raw data in GitHub, public CI artifacts, shared dashboards or job logs.

```bash
bash scripts/ep24/backup_and_verify.sh \
  /scratch/project_2009497/LaclauGPT-Private/analysis/ep24_reprocess/outputs \
  allas-private:ep24-backups
bash scripts/ep24/backup_and_verify.sh \
  /private/ep24_exports/final_001 \
  allas-private:ep24-final-exports
```

The script creates a distinct, timestamped version, calculates SHA-256 for each source file, copies it off-node, **downloads it into a new temporary directory** and checks every file against the manifest before reporting success. Keep the remote version name, timestamp, log and restoration proof in a private operator record. Retention, access control, encryption at rest, TLS and key rotation are infrastructure policies that must be verified on CSC, not assumed from the repository.

Periodic stage checkpoints are generated during the existing orchestration; schedule independent off-node copying during long runs. A snapshot on the same scratch filesystem in a second format is *not* disaster recovery. Restore test: download version to a clean private directory, run its SHA checks, load the CSV and SQLite, compare row IDs and provenance/stage samples with the Mongo collection, and record result.

## Remaining live acceptance checks

- [ ] Documented researcher sign-off for each FI/PL/PT 500-sample stage coverage/validity and source/model/codebook fingerprint.
- [ ] Actual source inputs and active sample collection namespaces reconciled, with no unexplained orphan/duplicate records.
- [ ] Dry-run and unit tests executed in an appropriate Python environment.
- [ ] One small controlled Roihu/Slurm continuation run successfully resumes missing stages without rerunning completed outputs.
- [ ] Fresh-country test preserves source schema and safely resumes after forced interruption.
- [ ] Read/write privilege, MongoDB durability and independently stored backup validated on CSC.
- [ ] Successful off-node clean restore with checksum and representative stage-field comparison.
- [ ] Full input-vs-export row/identity reconciliation and final combined CSV validation.

Do **not close issue #373** merely because the scripts exist: real data availability, successful trial processing, off-node backups and sign-off must still be proven.
