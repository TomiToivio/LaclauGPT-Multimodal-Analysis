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

## Artifacts (the #195 contract)

```
fi.csv  pl.csv  pt.csv  combined.csv        one row per video (the index)
fi_text.jsonl.gz  pl_text.jsonl.gz  ...     heavy cells, keyed by record_id
children/<country>_<role>.csv               normalized one-to-many tables
MANIFEST.json  DATA_DICTIONARY.md
```

### The index is lossy by design, and says so

Any cell longer than **1,000 characters** is removed from the index CSV and written to
`<country>_text.jsonl.gz` under the row's `record_id`. Without this the Finland frame alone is ~900 MB of
transcript JSON. Two consequences a consumer must know:

1. **The canonical cell is replaced by a sentinel**, `<<elided: N chars in <country>_text.jsonl.gz>>`,
   not deleted. An empty cell means the stage produced nothing; a sentinel means "the value exists,
   fetch the bundle". Conflating them is how a dashboard reports "no DNA evidence" for a country with
   1,583 statements.
2. **`MANIFEST.json.heavy_fields` echoes the split per country and field**, so elision is data:
   `{"finland": {"dna_statements_json": 517}}`.

### Children: unit, key and QA

`children/<country>_<role>.csv` is written from the pipeline's own collections, which are keyed at
**ingest** level even when the parent export is `record_unit: video`. The manifest states this
explicitly as `children_unit: "ingest"`; comparing `children` counts against `record_counts` without
accounting for the unit is measuring two populations.

| Field | Meaning |
| --- | --- |
| `statement_id` (or `node`/`entity` id) | identity **within** the child table |
| `source_record_id` | **the parent key** — matches the parent row's `_storage_id` |
| `_storage_id` | the child's own storage id; **do not join on this** |

`MANIFEST.json.children_qa` reports, per country and role: `rows`, `unique_statement_ids`,
`duplicate_statement_ids`, `parent_field`, `parents_resolved`, `orphans`, `missing_parent_field` and
`join_complete`. Orphans are counted, never given invented parents.

### Coverage source

Each `stage_coverage` entry carries `source`: `rows` when the step's output is a row cell, `children`
when it lives in a child table (DNA, SNA), `rows+children` when both, `none` when neither. A step whose
child table has rows is **not** `available: false` merely because the index cell is empty — that
contradiction between `stage_coverage` and `children` in the same manifest is #332.

`MANIFEST.json` carries the fields the dashboard's `coverage_rows()` reads:

```json
{
  "record_unit": "video",
  "records": 0, "distinct_media": 0,
  "record_counts": {"exported_total": 0, "fi": {"read": 0, "exported": 0,
                                                "duplicates": 0, "invalid_ids": 0,
                                                "distinct_media": 0}},
  "children_unit": "ingest",
  "heavy_fields": {"fi": {"dna_statements_json": 517}},
  "children_qa": {"fi:dna_statements": {"rows": 1583, "orphans": 93, "join_complete": false}},
  "stage_coverage": {"fi": {"6_discourse": {"records": 0, "available": true,
                                            "failed_or_other": 0, "pct_of_exported": 0.0,
                                            "source": "rows"}}},
  "inputs": {"fi": {"country": "finland", "profile": "mongo",
                     "source": "laclaugpt_ep2024_reprocess_finland_dataframe",
                     "stage_unavailable": ["9_rdf"]}}
}
```

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
