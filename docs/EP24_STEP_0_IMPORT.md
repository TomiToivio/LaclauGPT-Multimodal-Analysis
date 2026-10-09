# EP24 Step 0: private CSV → MongoDB

Step 1 in Mongo-backed mode **cannot process records until its dataframe
collection has been seeded**. Step 0 imports already prepared, human-annotated
country CSVs from the private companion checkout. No source data or credentials
are committed to the public repository.

## Source of truth

`LaclauGPT-Private/analysis/ep24_reprocess/data/to_reprocess/ep24_<country>.csv`

Verified country CSVs include Finland, Poland, Portugal and seven other
countries. This is NOT the raw 20,842-row export or researcher-review worklist:
it is the canonical Step 1 input split.

Private settings live at
`LaclauGPT-Private/analysis/ep24_reprocess/.env`:

- `LACLAUGPT_MONGO_ENABLED=1`
- `LACLAUGPT_MONGO_URI` with the authentication source supplied privately
- `LACLAUGPT_MONGO_DATABASE` (e.g. `spectacleScraper`)
- `LACLAUGPT_DATASET=ep2024_reprocess`

## Commands

On a Roihu login or CPU node, with both repositories checked out:

```bash
cd /scratch/project_2009497/LaclauGPT-Multimodal-Analysis
git pull origin main
git -C /scratch/project_2009497/LaclauGPT-Private lfs pull

bash scripts/roihu/setup_step_0_import.sh
bash scripts/roihu/run_step_0_import.sh --country finland --limit 10 --dry-run
bash scripts/roihu/run_step_0_import.sh --country finland --limit 10
bash scripts/roihu/run_step_0_import.sh --country poland --limit 10
bash scripts/roihu/run_step_0_import.sh --country portugal --limit 10
# Full import of every country:
bash scripts/roihu/run_step_0_import.sh
```

Step 0 uses the same per-country Mongo collection name as the existing
orchestrator, `laclaugpt_ep2024_reprocess_<country>_dataframe`.
It preserves the established `_storage_id` formula from `ep24_bootstrap`
and the original CSV columns, including researcher entities/themes.
It inserts only new documents, never resets `_pipeline.step_01` etc.,
so running it repeatedly will not erase completed analyses.

Use `--input-root` for a nonstandard private checkout, `--batch-size` for
Mongo write batch tuning, and `--dry-run` to validate without contacting
MongoDB. Step 0 prints count/progress and writes per-run logs into
`<private-root>/logs/step0/`. No model or GPU is needed.

Once Step 0 is complete, run Step 1 as usual. Step 0 is not an analysis
stage and does not consume GPUs.
