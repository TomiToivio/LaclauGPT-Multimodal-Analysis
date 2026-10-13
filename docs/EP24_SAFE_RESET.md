# Rebuild EP24 processing state from canonical source CSVs

Run **only after stopping all EP24 SLURM jobs**. This tool is an archive-and-restart operation, not a database drop. The public source repo and private canonical inputs remain unchanged.

From the public checkout on `roihu-gpu.csc.fi`, first preview:

```bash
python scripts/roihu/reset_ep24_and_restart.py --country finland
python scripts/roihu/reset_ep24_and_restart.py --country poland
# or: python scripts/roihu/reset_ep24_and_restart.py --all-countries
```

After verifying the namespace, source files and collections printed by the dry-run:

```bash
python scripts/roihu/reset_ep24_and_restart.py --country finland --country poland \
  --execute --confirm-dataset ep2024_reprocess
```

The script archives each selected country's Mongo collections under `*_reset_archive_<UTC timestamp>`, except `*_memory` and `*_entities`, which are deliberately **preserved**. It moves existing country output directories and recognized local SQLite restart caches to `<private-root>/reset-archives/<timestamp>/`; it then invokes `step_0_roihu_import.py` and submits a fresh Step 1 SLURM job per country. The manifest at that archive path records each completed operation.

For a reimport without submitting Step 1, append `--no-submit`. Never run two resets concurrently or while any EP24 processing job is active. The script does not attempt automatic rollback if the process fails halfway: inspect its manifest and archived Mongo names before retrying. In particular, it does **not** remove researcher-seeded memory, original Allas media, original Git LFS inputs, or private codebooks. It also does not reset Redis keys or every unknown SQLite database. Those should be handled by separate audited procedures if needed.

To restore an archived Mongo collection, first back up or remove any replacement live collection, then rename the archived collection to the original name using your approved Mongo administration procedure. For country outputs and SQLite files, use the manifest's `from` and `to` paths.
