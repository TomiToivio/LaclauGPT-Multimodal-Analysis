# EP24 MongoDB export (issue #306, section 1)

`ep24_mongo_dashboard_export.py` turns the reanalysis results held in shared MongoDB into the artifact set the
researcher dashboard already reads. It is the missing link between the pipeline and
`LaclauGPT-Data-Visualization`'s exporter contract: the visualization repository exports from **CSV
frames**, and this produces those frames from the database.

## What one record is

**One row per video.** Identity is the country-scoped media key (`allas_filename`, falling back to
`_storage_id`), plus `video_id`, recording, segment and `source_type`. Several reanalysis runs of one
video therefore collapse to the row carrying the most stage output, and the collapse is counted as
`duplicates` in the manifest. This is the `record_unit: video` rule of the #195 contract, so both
exporters agree on denominators.

A document without a `video_id` **and** a media key is quarantined and counted as `invalid_ids`. No
identifier is fabricated, and no row is dropped silently.

## Usage

```bash
export LACLAUGPT_MONGO_URI=...            # or LACLAUGPT_MONGODB_URI (private runtime config)
export LACLAUGPT_MONGO_DATABASE=...

python ep24_mongo_dashboard_export.py --countries fi pl pt \
    --output /private/LaclauGPT-Private/analysis/ep24/mongo-export
```

Options:

| flag | effect |
| --- | --- |
| `--countries fi pl pt` | which countries to export (default: all three) |
| `--output DIR` | destination; real data belongs in private storage, never in this repository |
| `--limit N` | read at most N documents per collection, for a fast development subset |
| `--batch-size N` | cursor batch size (default 500) |
| `--database NAME` | override the database from the environment |
| `--query JSON` | an explicit Mongo filter, for example a single run id |

Collections are addressed as `laclaugpt_ep2024_reprocess_<country>_<purpose>` through
`roihu_storage.collection_prefix`, not by a hard-coded string, so a dataset rename fails loudly
instead of exporting nothing. Purposes read: `dataframe`, `dna_statements`, `sna`, `entities`,
`themes`.

## Read-only guarantees

Every read is `find`/`aggregate`. There is no write path in the module and a test asserts it: the
fake collection raises if `insert_one` is ever called. The URI is never written to the manifest —
only the **name of the environment variable** that supplied it, because the manifest is copied into
issues and notes.

## Artifacts and the stable export contract (#330, #331, #332)

This section is the contract a consumer may build on. It describes what `ep24_mongo_export.py` writes
through `scripts/ep24/export_mongo_results.py`; `ep24_mongo_dashboard_export.py` is a separate
producer with its own manifest (see *Compatibility*).

```
<country>.csv  combined.csv      one row per record at record_unit; the lean index
<country>_text.jsonl.gz          cells longer than text_bundle.threshold, keyed by record_id
children/<country>_<role>.csv    normalized one-to-many rows: entities, themes, dna_statements, sna
MANIFEST.json  DATA_DICTIONARY.md
```

### The lean index is lossy on purpose, and the manifest says how

Any cell longer than `text_bundle.threshold` (default **1,000** characters) is removed from the row
CSV and written to `<country>_text.jsonl.gz` under the row's own `record_id`. The bundle is JSON per
line: `{"record_id": ..., "<field>": "<original cell>", ...}`.

```json
"text_bundle": {"suffix": "_text.jsonl.gz", "key": "record_id", "threshold": 1000},
"heavy_fields": {"finland": {"dna_statements_json": 517, "asr_transcript": 2125}}
```

`heavy_fields[country][field]` is the number of rows whose cell moved to the bundle. Read it before
concluding that a stage produced nothing: on a real FI/PL/PT export **every** record with DNA
statements had its `dna_statements_json` elided (mean 4,176 characters), so the index CSV alone cannot
show them. Cells are elided, never truncated or replaced by a sentinel, so the bundle is the only
source of truth for their content.

### Child tables carry an explicit parent key

| column | meaning |
|---|---|
| `parent_record_id` | **the join key.** The parent's `record_id` at this export's `record_unit` (`video:allas_filename` for the default unit), or `""` when no parent survives in this export |
| `source_record_id` | the parent's `_storage_id` as recorded by the producing step — ingest-level, kept for traceability |
| `_storage_id` | the **child's own** id (for statements, the statement id) — *not* a parent key |
| `statement_id`, `evidence_quote`, `actor_name_raw`, `concept_label_raw`, `agreement`, `stance`, `confidence`, `provenance`, `extraction_prompt_version`, `context_sha256`, `uncertainty_notes` | the evidence a reader must show next to the row; absent values are empty strings, never invented |

Children are produced by Mongo collections keyed at pipeline-run level, so a child table can hold runs
that the export's unit-of-analysis collapse dropped. Those rows are **kept** (`parent_record_id: ""`)
and counted, not silently deleted:

```json
"children": {"finland": {"dna_statements": 1583}},
"children_unit": "video",
"children_detail": {"finland": {"dna_statements": {"records": 1583, "matched": 1490, "orphan": 93}}}
```

`children.<country>.<role>` is the row count in the file (backward compatible). `children_detail`
separates `records` read from `matched` (parent present at `children_unit`) and `orphan`. On the real
export the manifest's 1,583 statements reconcile to the **1,490** the per-record `dna_statement_count`
counters report only after the 93 orphans are excluded — reconcile against `matched`, never against the
file row count.

A consumer that joins children must therefore: read `children_detail` for the denominator, join on
`parent_record_id`, drop `""`, and deduplicate on `statement_id` within a parent.

### Coverage, and the invariant that is checked

```json
"record_counts": {"exported_total": 0, "finland": {"read": 0, "exported": 0,
                                                   "duplicates": 0, "invalid_ids": 0,
                                                   "distinct_media": 0}},
"stage_coverage": {"finland": {"6_discourse": {"records": 0, "available": true,
                                               "failed_or_other": 0, "pct_of_exported": 0.0}}},
"inputs": {"finland": {"profile": "mongo", "stage_unavailable": ["9_rdf"]}}
```

Step 7 coverage counts rows carrying `dna_statements_json`, `dna_analysis_markdown` **or a positive
`dna_statement_count`**. The counter is short, so it survives both moves and coverage stays reproducible
from the delivered CSV; a `0` counter is an answer, not output.

`contract_warnings` is written by `coverage_reconciliation()` and lists every place where coverage and
the child tables disagree — a step claiming records with no attached child row (`--children` omitted,
or a broken identity), a step marked `available: false` while children prove output, or children
attached while coverage claims nothing. An empty list is the healthy answer, and a test asserts it for
a synthetic export. Disagreements are **recorded, not resolved**: the exporter does not rewrite one
number to match the other.

## Missingness is not a negative finding

A step whose defining columns are absent from **every** source document is reported
`available: false` with a reason and listed in `inputs[country].stage_unavailable`. A step that ran
and produced nothing is `records: 0` with `available: true`. Rendering both as `0 %` would turn an
absent stage into a null finding, which is the failure mode issue #195 already fixed once.

## Access gaps

Running this requires the shared MongoDB URI. Where it is not available, the module is testable
without it (synthetic documents, no network) and the export is reported as not produced — never
simulated.

## Compatibility

The existing `ep24_mongo_export.py` API and `scripts/ep24/export_mongo_results.py` CLI remain unchanged. Use `ep24_mongo_dashboard_export.py` explicitly for the dashboard artifact contract described here. Their identity rules and manifests differ; do not combine their record counts or substitute one export directory for the other without validating the consuming application.

Subset exports (`--query` or `--limit`) deliberately omit child tables and mark them unavailable because child-to-parent run alignment is unverified. Cursor failures after emitting records abort rather than replaying documents.
