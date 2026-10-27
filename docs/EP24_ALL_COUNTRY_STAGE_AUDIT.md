# EP24 all-country read-only stage inventory (#306)

Run on an **authorized private** host with a read-only Mongo account. This is
distinct from result-quality evaluation and does not replace the FI/PL/PT
exporter or visualization dashboard.

```sh
# Supply LACLAUGPT_MONGO_URI and LACLAUGPT_MONGO_DATABASE privately.
python scripts/ep24/audit_country_stages.py --output "$PRIVATE_AUDIT_OUTPUT"
python -m unittest tests.test_ep24_country_stage_audit
```

The script discovers live `laclaugpt_ep2024_reprocess_<country>_dataframe`
collections case-insensitively; reset/archived variants are excluded. Countries
are discovered from Mongo, not guessed from input CSVs or country-audit docs.

The JSON/CSV output distinguishes stages 0–9 and missing stage fields, counts
all `complete`, `claimed`, `error`, `retry`, `skipped`, and other
literal states without converting missingness to a substantive negative. Step 0
only measures imported records and media identity. The chain metric requires
all nine status flags to be complete in the same record.

**Important limitations:** Status flags are not proof of substantive model
results, stage-run coherence, or completed downstream artifacts. A Mongo
collection may report estimated documents while a cursor reads zero; such
anomalies are surfaced as `metadata_nonzero_but_cursor_empty`, never as
verified empty data. The tool does not expose record-level contents, and
does not use write queries, but its manifests still belong in private storage
until aggregate disclosure review. Never commit real exports, credentials or
internal endpoints to public GitHub. FI/PL/PT source-run, manual codebook
review, actual payload validation, reruns and dashboard deployment remain
separate acceptance gates.

## Dependency diagnostics

For each stage N, `eligible_by_previous_stage` counts rows with a completed
immediate previous stage (or a recognized identity for stage 1).
`complete_with_upstream_gap` flags a completed stage with an immediate
upstream non-complete status, while
`upstream_blocked_or_identity_missing` flags pending/missing downstream
work without the prerequisite. These are **diagnostic flags**, not mutually
exclusive pipeline states; do not sum them with `complete/error/missing`.

`identity_valid` permits a Mongo storage key; the separate
`source_media_identity_present` / `source_media_identity_missing`
metrics require actual `video_id` or `allas_filename`. This prevents a
storage-generated identifier from being mistaken for resolvable source media.
Run IDs are inventoried from stage markers as well as row-level provenance;
this does not establish same-run consistency or verify the result payload.
