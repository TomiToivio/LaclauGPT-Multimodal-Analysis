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

## Historical normalized export contract

The v1 normalized exporter attached parent IDs opportunistically, retained orphan
children, and exposed `children_detail` and `contract_warnings`. Those helper APIs
remain available for existing consumers. The v2 CLI contract below supersedes that
layout: DNA exports contain validated children only, with exclusions in
`child_join_qa`, while registry and graph collections retain their own grain.
Both row markers (`heavy_fields_in_text_bundle` and `_text_bundle_fields`) remain
available. The offline audit supports both generations of exported artifacts.

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


## Normalized Mongo CLI contract v2 (#330–#333)

The separate `scripts/ep24/export_mongo_results.py` CLI writes
`ep24-mongo-export-2`. The usage sections above describe the dashboard exporter;
this section describes the normalized CLI actually used by the reported export.
Do not mix the two layouts.

Run with `--country finland poland portugal --output <new-private-directory>`.
The destination must be empty so a missing collection cannot leave a stale child
file from an older export. Default record unit is `video`. The CLI selects the
most complete available row, **not necessarily the latest complete run**. Record
and verify run provenance independently before calling this a latest-run audit.

| Artifact | Contract |
| --- | --- |
| `<country>.csv`, `combined.csv` | Canonical parent rows; join key `(country, record_id)` |
| `<country>_text.jsonl.gz` | JSON lines keyed by `record_id` within that country's file |
| `children/<country>_dna_statements.csv` | Selected-ingest DNA statements; join `(parent_country, parent_record_id)` to parent key |
| `children/<country>_{entities,themes,sna}.csv` | Original source collection records; **not** guaranteed video children |
| `MANIFEST.json` | Counts, hashes, coverage, status matrix, provenance, elision and join QA |

Country names are `finland`, `poland`, `portugal`. Record IDs retain the historical
`video:` / `ingest:` prefix and must always be scoped by country. Equal IDs in two
countries are not a license to join across countries. A video child points to the
selected parent ingest; children from discarded ingests are excluded and counted.
Do not union them into a newer run.

DNA identity fields are `statement_id`, `source_record_id` (source ingest),
`parent_record_id`, `parent_country`; statement identity is unique within the
country/parent. Original `_storage_id` is the child storage identifier, **not**
the parent. Its Mongo meaning is preserved: changing it to the parent ID would
make default upserts overwrite multiple statements from one video. No database
migration is needed or performed.

Preserved analytical/provenance columns include `actor_name_raw`,
`concept_label_raw`, `agreement`, `stance`, `evidence_quote`, `confidence`,
`provenance` and prompt/model version fields when carried by the source. Missing
values stay missing; CSV null is an empty cell, JSON containers are encoded JSON,
and an empty child file represents zero rows. Missing IDs are never generated.
Missing evidence/provenance is counted in QA but not silently repaired or discarded.
Ambiguous parents, missing IDs, foreign-country statements and duplicate statement
identities are excluded with separate counters; all occurrences of a duplicate
identity are excluded rather than choosing an arbitrary research result.

Strings longer than `--text-threshold` (default 1000 characters) move to the
country text bundle. Identity fields never move. `_text_bundle_fields` is a JSON
array on each affected CSV row identifying the elided cells; restore exactly those
fields from the bundle before interpreting empty cells. Manifest `heavy_fields`
counts elisions by country and field; `text_bundle` records threshold and keys.
No sentinel text is inserted into analytical JSON columns.

`child_schema_version` is `ep24-dna-children-2`. `children_unit` is per role:
DNA uses the export record unit; other roles use `source_collection`, since entity
registries, codebooks and graph records have different identities.
`children.<country>.<role>` counts actual emitted rows. `child_join_qa` records
source counts and exclusions. All emitted child files are hashed in `outputs`.

When the DNA collection is requested, Step 7 `stage_coverage.records` counts
**distinct parents with exported DNA children**, and `statement_rows` equals
`children.<country>.dna_statements`. A missing collection has `available: false`;
an existing empty collection has `available: true`. `source_output_records`
retains pre-split row evidence coverage; `parent_count_mismatches` exposes disagreement
with source counters. An omitted DNA role retains inline coverage; it is not a
statement-table audit. `--limit` limits parents only: children are read fully so
the subset is not randomly truncated.

`stage_states` adds a per-country Steps 0–9 matrix with explicit denominators,
counts and percentages. It distinguishes known success with output/empty,
not-run, partial, retry, skipped, failed, unavailable and unknown statuses.
Output without a completion status is `output_status_unknown`, not success.
These are source-field observations: RDF paths do not prove files exist, absence
of fields does not prove not-run, and legacy `failed_or_other` is not a failure count.

Consumer coordination: [Visualization PR #222](https://github.com/TomiToivio/LaclauGPT-Data-Visualization/pull/222)
implements the earlier reader. Existing `source_record_id` remains intact;
new readers should use the explicit compound parent key and check manifest QA.
The synthetic CLI test in `tests/test_ep24_export_contract_330_333.py` is an
executable three-country consumer fixture. No restricted records are included.

