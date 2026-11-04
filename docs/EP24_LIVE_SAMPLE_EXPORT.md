# Friday EP24 FI/PL/PT live CSV export (issue #373)

**Presentation: Friday 6 November 2026.** This is a **read-only interim export**,
not permission to launch full-corpus GPU jobs or a declaration that all 500
videos/country have completed Steps 1–7.

On CSC Roihu, from the current public repository checkout with the private
environment configured:

~~~bash
python3 scripts/ep24/export_live_sample.py \
  --dataset YOUR_ACTUAL_SAMPLE_DATASET \
  --output-root /scratch/project_2009497/LaclauGPT-Private/analysis/ep24_reprocess/outputs/presentation_snapshots
~~~

**Check the MongoDB collection names before invoking.** Set --dataset to the
exact LACLAUGPT_DATASET used by the running 500-video sample. A Monday500
run-specific namespace may differ from ep2024_reprocess. The script deliberately
does not guess or fall back to the wrong dataset. Collection names are derived
using the project's existing roihu_storage.StorageConfig.

Output is a uniquely named private live_TIMESTAMP_random directory containing:
- finland_live_steps_1_7.csv, poland_live_steps_1_7.csv, portugal_live_steps_1_7.csv
- fi_pl_pt_live_combined.csv
- CSV companion SQLite checkpoints and SHA-256 manifests
- live_manifest.json with per-country stage status coverage, row counts, hashes

All persisted rows are included, regardless of Step 7 status. Every accumulated
input/output field and nested provenance is preserved; nested values are
JSON-encoded in the CSV. Step statuses are additionally provided as friendly
step_01_status through step_07_status columns. Missing stages are "pending".
The script fails rather than silently reporting zero rows for a missing collection
or treating an invalid ID as a valid one. Never assume exactly 500 observed records.

For a presentation subset you can inspect only valid Step 7 rows in pandas:

~~~python
import pandas as pd
df = pd.read_csv("fi_pl_pt_live_combined.csv", dtype=str, keep_default_na=False)
ready = df[df["step_07_status"] == "complete"]
print(df.groupby("country")["step_07_status"].value_counts())
~~~

**Important:** Mongo writes may be happening while each cursor is read, so
different documents and countries are not guaranteed to represent the exact same
instant. Manifest coverage reflects the observed read. Report counts and snapshot
time in the presentation and do not imply unfinished work is complete.

Exports contain private research information. Keep files under private access-
controlled CSC storage, never commit them or publish them as GitHub artifacts.
The export does not authenticate Allas; off-node backup and restore verification
remain separately required in issue #373. Full continuation requires explicit
validated-sample approval, via the existing rollout scripts.

## Validation checklist

1. Verify exact sample namespace and each per-country Mongo collection before export.
2. Inspect manifest country counts, stage failures and pending/claimed rows.
3. Open the combined CSV and confirm its rows equal summed country counts;
   confirm each per-country CSV and manifest hash.
4. If rerunning while jobs continue, compare two timestamped snapshots; do
   not overwrite either. Do not release researcher data publicly.
