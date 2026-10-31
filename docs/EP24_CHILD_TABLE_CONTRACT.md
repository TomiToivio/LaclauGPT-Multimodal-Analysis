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
"children_unit": { "dna_statements": "video" },
"child_parent_join_key": { "dna_statements": ["parent_country", "parent_record_id"] },
"children_detail": {                                     // re-derived by the auditor
  "finland": { "dna_statements": { "records": 1583, "matched": 1490, "orphan": 93 } }
},
"heavy_fields": { "finland": { "dna_statements_json": 517, "asr_transcript": 490 } },
"text_bundle": { "suffix": "_text.jsonl.gz", "key": "record_id", "threshold": 1000 }
```

- `children` counts are **child rows**, not videos. `children_detail.<country>.<role>.matched`
  is the number reachable from an exported video; `orphan` is what could not be attached and is kept
  rather than dropped. (Pre-#352 exports wrote the same numbers as
  `children_contract.countries.<country>.<role>.{rows,joined_rows,unjoined_rows}`; the auditor reads
  either spelling and compares like with like.)
- `heavy_fields` explains why an index CSV cell is empty: the value moved to the text bundle keyed by
  `record_id`, and the count says how many rows were affected.
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
