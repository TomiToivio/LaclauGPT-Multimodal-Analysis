# EP24 FI/PL/PT export audit (#333)

Deliverable 1 of #333: a reproducible, privacy-safe coverage and QA report for the latest authorized
export, separating **confirmed missing output** from **stages that are legitimately empty or not
carried**, with the root cause or the unresolved blocker named for each anomaly.

Reproduce with:

```bash
python scripts/ep24/audit_export.py --export /private/ep24/mongo_export_210 --json report.json
```

The tool is offline and stdlib-only: no MongoDB, no network, no Roihu. It prints counts and column
names, never cell contents, so its output can be pasted into a public issue. Exit status is 1 when an
ERROR-level anomaly exists.

## What was audited

`ep24-mongo-export-1`, `record_unit: video`, generated `2026-10-27T10:39:35+00:00`.

| Artifact | Bytes | sha256 vs manifest |
|---|---|---|
| `combined.csv` | 32,880,763 | match |
| `finland.csv` / `poland.csv` / `portugal.csv` | 13.1 MB / 13.3 MB / 6.4 MB | match |
| `finland_text.jsonl.gz` / `poland_text.jsonl.gz` / `portugal_text.jsonl.gz` | 134 MB / 101 MB / 249 MB | match |

All seven artifacts agree with the manifest's hashes, every bundle entry is keyed by a `record_id`
that exists in the index (`ids_not_in_index = 0` in all three), and there is exactly one row per video
(2,125 FI + 1,781 PL + 960 PT = 4,866). The artifact set is internally consistent; the defects below
are about what it can *prove*, not about corruption.

## Coverage matrix (manifest vs what the delivered CSV can reproduce)

`csv` is recomputed from `combined.csv` using the producer's own `COVERAGE_STEPS` definitions. A gap
means the stage's output exists only in the text bundle.

| country | step | manifest | csv | available |
|---|---|---|---|---|
| finland | 0_ingest | 2125 | 2125 | yes |
| finland | 1_preprocess | 2125 | 2125 | yes |
| finland | 2_frame | 2125 | 2125 | yes |
| finland | **3_video** | 1275 | **0** | yes |
| finland | 4_summary | 1047 | 1047 | yes |
| finland | 5_postprocess | 1034 | 1034 | yes |
| finland | **6_discourse** | 671 | **80** | yes |
| finland | 7_dna | 575 | 575 | yes |
| finland | **8_sna** | 615 | **261** | yes |
| finland | 9_rdf | 0 | 0 | **no** |
| poland | 0_ingest | 1781 | 1781 | yes |
| poland | 1_preprocess | 1781 | 1781 | yes |
| poland | 2_frame | 1781 | 1781 | yes |
| poland | **3_video** | 1781 | **0** | yes |
| poland | 4_summary | 1099 | 1099 | yes |
| poland | 5_postprocess | 1763 | 1763 | yes |
| poland | **6_discourse** | 10 | **1** | yes |
| poland | 7_dna | 0 | 0 | **no** |
| poland | 8_sna | 0 | 0 | **no** |
| poland | 9_rdf | 0 | 0 | **no** |
| portugal | 0_ingest | 960 | 960 | yes |
| portugal | 1_preprocess | 960 | 960 | yes |
| portugal | 2_frame | 960 | 960 | yes |
| portugal | **3_video** | 960 | **0** | yes |
| portugal | 4_summary | 600 | 600 | yes |
| portugal | 5_postprocess | 960 | 960 | yes |
| portugal | **6_discourse** | 436 | **84** | yes |
| portugal | 7_dna | 16 | 16 | yes |
| portugal | 8_sna | 0 | 0 | **no** |
| portugal | 9_rdf | 0 | 0 | **no** |

Child tables as delivered (this export predates the #330 contract, so it has no `children_detail` and
no `parent_record_id`):

| country | entities | themes | dna_statements | sna |
|---|---|---|---|---|
| finland | 1,302 | 1,460 | 1,583 | 3,359 |
| poland | 1,268 | 1,440 | **0** | **0** |
| portugal | 1,564 | 1,507 | 4 | **0** |

## Defect register

### D1 — The index CSV cannot reproduce steps 3, 6 and 8 (2 ERROR, 3 gap)

`3_video` claims 1,275 / 1,781 / 960 records and **zero** of them are visible in `combined.csv` — an
ERROR, because the whole stage is invisible to a consumer that reads the index. `6_discourse` shows 80 of
671 (12 %) for Finland, 84 of 436 (19 %) for Portugal and 1 of 10 for Poland; `8_sna` shows 261 of 615
(42 %) for Finland. Those are gaps in the matrix rather than errors: some rows are reproducible, so the
step is not invisible, only under-counted by whatever the split moved.

The missing values are not lost — they are in `<country>_text.jsonl.gz` — but **nothing in this manifest
says so**, so a consumer reading only the index concludes the stage produced less than it did.

Root cause: `split_heavy_fields()` moves every cell over 1,000 characters out of the CSV and, in this
exporter version, records no trace. Fixed at three levels since: #335 writes a per-row
`heavy_fields_in_text_bundle` column, the contract PR adds `heavy_fields` counts and `text_bundle` to the
manifest (#331.3, #330, #332), and the consumer-side reader is Data-Visualization #224. **This export
predates all three**, so D1 stays true for it — a re-export retires it, not a documentation change.

### D2 — Poland has no DNA and no SNA output, and that is a real gap (ERROR for analysis, not for the exporter)

Poland: 1,781 videos, 0 DNA statements, 0 SNA rows, 10 discourse records (0.6 %). Portugal: 960 videos,
4 DNA statements, 0 SNA, 436 discourse (45 %). Finland: 575/2,125 DNA, 615 SNA, 671 discourse (32 %).

These are **not** exporter defects — the manifest marks the stages `available: false` where the source
cannot express them, and the child collections are absent in Mongo, which is a pipeline outcome. What
they are is the top of the #333 agenda: Step 6 for Poland and Steps 7–8 for Poland/Portugal did not
produce usable output for an entire country.

Blocker for the fix: reaching this conclusion *and acting on it* needs the authorized run, the pipeline
commit and the step logs on Roihu plus the shared MongoDB. Neither is reachable from this session;
nothing here is inferred from row counts alone.

### D3 — Structured cells stored as Python repr, not JSON (WARN, 6,832 cells)

`ep24_human_seed_context` (2,814), `step4_summary_provenance` (2,726), `_pipeline_step_6` (676) and
`_pipeline_step_7` (616) contain `{'key': 'value'}`-style text that `json.loads` rejects. A consumer
using a JSON parser fails on these cells; a consumer using `ast.literal_eval` trusts content that was
never validated as JSON.

Root cause is upstream of the exporter: those strings arrive from the producing steps already in repr
form, and `_csv_safe()` only serializes real dict/list objects. Fix belongs in the step that writes
them, with a round-trip test; it is tracked here rather than patched blindly, because rewriting stored
provenance text is a data change, not a display change.

### D4 — Child tables predate the parent key (WARN, informational)

7,704 FI / 2,708 PL / 3,075 PT child rows carry no `parent_record_id`: the export was produced before
the #330 contract existed. A consumer must fall back to `source_record_id` → parent `_storage_id`, and
cannot distinguish an orphan from a matched row. The known consequence is the 1,583-vs-1,490 statement
gap (93 ingest-level orphans) reported in #331. A re-export with the new exporter resolves this and
records `children_detail` so the gap is visible instead of inferred.

### D5 — Nothing else disagrees

Hashes match, no duplicate `record_id`, no cross-country contamination, no bundle key without an index
row, no manifest claim that is silently impossible. The artifact set is honest about what it contains;
it just does not *describe* two of its own transformations (D1, D4).

## What remains blocked, and why

| #333 item | State |
|---|---|
| Coverage matrix, denominators, anomaly inventory | **done** (this document) |
| Confirm/repair producer bugs in the exporter | exporter fixes are in the contract PR; D3 needs the producing steps |
| Rerun affected steps on authorized infrastructure | **blocked**: needs Roihu + shared MongoDB credentials and a run window |
| Codebook/prompt improvement (FI/PL/PT) | **blocked**: needs the authorized corpus and researcher review before/after comparison |
| Sample review of Steps 1–9 outputs | **blocked**: needs the authorized records; nothing was sampled or quoted here |
| Updated exports / dashboard coordination | consumer side tracked in Data-Visualization #220/#224 |

No live rerun is claimed. Everything above was measured on the delivered artifact set.
