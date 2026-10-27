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
fi.csv  pl.csv  pt.csv  combined.csv        one row per video
records.jsonl                               the same rows, JSON per line
dna_statements.csv  sna_nodes.csv  entities.csv  themes.csv   normalized children
MANIFEST.json  DATA_DICTIONARY.md
```

`MANIFEST.json` carries the fields the dashboard's `coverage_rows()` reads:

```json
{
  "record_unit": "video",
  "records": 0, "distinct_media": 0,
  "record_counts": {"exported_total": 0, "fi": {"read": 0, "exported": 0,
                                                "duplicates": 0, "invalid_ids": 0,
                                                "distinct_media": 0}},
  "stage_coverage": {"fi": {"6_discourse": {"records": 0, "available": true,
                                            "failed_or_other": 0, "pct_of_exported": 0.0}}},
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
