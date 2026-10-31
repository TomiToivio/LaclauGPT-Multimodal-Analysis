# EP24 child-table export contract (issues #330, #331, #332)

This is the producer-side contract for the normalized one-to-many tables written next to the
video-level frame. It is what a consumer may rely on.

## Units

| Artifact | Unit | Identity |
|---|---|---|
| `<country>.csv`, `combined.csv` | one row per exported **video** record | `record_id` = `video:<media key>` |
| `children/<country>_<role>.csv` | one row per **ingest-level statement/entity/edge** | `_storage_id` is the child's own id |
| `<country>_text.jsonl.gz` | one line per video with heavy fields | `record_id` |

The child collections are written by the pipeline at pipeline-run (ingest) level, so they may
contain statements from ingest runs that lost the video-unit collapse. They are **not** dropped:
each row is joined where possible and marked where not.

## Child row keys

| Column | Meaning |
|---|---|
| `_storage_id` | the child document's own id (for DNA: the statement id) — **never the parent** |
| `source_record_id` | the ingest id of the parent record the pipeline attributed it to |
| `parent_record_id` | **the join key**: the exported video's `record_id`, or `""` when unjoined |
| `parent_join` | `source_record_id` when resolved, `unjoined` otherwise |

Join: `child.source_record_id` → parent `_storage_id`/`record_id`/`video_id`/`allas_filename`,
resolved by the exporter. A consumer must use `parent_record_id`; joining child `_storage_id` to a
parent key returns nothing by construction.

## `MANIFEST.json`

```jsonc
"children": { "finland": { "dna_statements": 1583 } },   // raw child rows written
"children_contract": {
  "unit": "video",
  "join": "source_record_id -> parent_record_id",
  "source_unit": "ingest",
  "cross_country_duplicate_statement_ids": 0,
  "countries": {
    "finland": {
      "dna_statements": {
        "rows": 1583, "joined_rows": 1490, "unjoined_rows": 93,
        "duplicate_statement_ids": 0,
        "parent_key_field": "parent_record_id", "join_field": "source_record_id"
      }
    }
  }
},
"heavy_field_splits": {
  "finland": { "threshold": 1000, "records_with_heavy_fields": 517,
               "fields": { "dna_statements_json": 517, "asr_transcript": 490 } }
}
```

- `children` counts are **child rows**, not videos. `children_contract.<country>.<role>.joined_rows`
  is the number reachable from an exported video.
- `heavy_field_splits` explains why an index CSV cell is empty: the value moved to the text bundle
  keyed by `record_id`.
- Coverage (`stage_coverage`) is row-level: for `7_dna` it reads `dna_statements_json` **or**
  `dna_statement_count`, so a child-table export reports coverage instead of "not available".

## Consumer rules

1. Read evidence from `children/`, join on `parent_record_id`.
2. Report `unjoined_rows` and `duplicate_statement_ids`; never silently drop them.
3. Do not infer coverage from a blanked index cell; use `stage_coverage`/`dna_statement_count`.
4. For heavy text, read `<country>_text.jsonl.gz` by `record_id`.

## QA

`python scripts/ep24/audit_export_contract.py <export-dir>` re-derives the child counts, join
status, duplicates, coverage agreement and heavy-split bookkeeping from the delivered files and
exits non-zero on any mismatch. It needs no database and no research data.
