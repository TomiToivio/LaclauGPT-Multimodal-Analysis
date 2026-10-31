# EP24 record-level QA, and two corrections to the stage audit (#333)

Companion to `docs/EP24_STAGE_AUDIT_333.md` (coverage from `stage_coverage()`) and
`docs/EP24_EXPORT_QA_330_333.md` (export-to-export diff). This document adds what only the delivered
*records* can say — statuses per stage, output-versus-status agreement, language, freshness, evidence —
and corrects two readings that the index CSV alone produces.

Reproduce:

```bash
python scripts/ep24/audit_export.py --export /private/ep24/mongo_export_210 --records --json report.json
```

Offline, stdlib only, counts and column names only — nothing here quotes a record.

## Correction 1 — Step 3 video is **not** unavailable in this export; it was relocated

The stage audit reads `3 Video analysis | 0 | 0.0% | available: no` for all three countries, because no
delivered **CSV** column carries `vllm_video_analysis`. The manifest and the bundles disagree with that
reading:

| Evidence | FI | PL | PT |
|---|---|---|---|
| `MANIFEST.json` → `stage_coverage.<country>.3_video.records` | 1,275 | 1,781 | 960 |
| `...3_video.available` | true | true | true |
| `vllm_video_analysis` present in `<country>_text.jsonl.gz` | yes | yes | yes |
| `vllm_video_analysis` non-empty in `combined.csv` | 0 rows | 0 rows | 0 rows |

The cell is over the 1,000-character text threshold, so `split_heavy_fields()` moved it into the bundle.
"Unavailable" is therefore the wrong verdict for Step 3 — the correct one is "output exists, index cannot
show it", which is D1 in `docs/EP24_STAGE_AUDIT_333.md`'s own terms. The audit tool now decides this from
the bundle's field names when `heavy_fields` is absent (this export predates both).

This matters beyond bookkeeping: #342's baseline table inherits the "unavailable" reading and would plan a
full Step 3 re-run on the strength of it. Step 3 output exists for 4,016 of 4,866 records; what is unknown
is not whether it ran but whether the bundle's content is good.

## Correction 2 — SNA "57 % without evidence" was a node/edge artifact

`children/finland_sna.csv` mixes two row types: **1,915 nodes** (no actor pair, no quote — correctly) and
**1,444 edges**. Measuring quote coverage over all 3,359 rows gives 1,915 "missing", which reads like an
evidence failure. Measured on edges only:

| | FI edges |
|---|---|
| rows | 1,444 |
| with `source_actor` + `target_actor` + `relation_type` | **1,444 (100 %)** |
| with `evidence_quote` | **1,444 (100 %)** |

So Finland's SNA edges are fully evidenced. The audit separates the two row types now; any future metric
that averages nodes into edge completeness will reproduce the false alarm.

## Record-level QA

Status vocabulary per stage, from the delivered rows (values as recorded; `(empty)` means no status was
written):

| Country | 3_video | 4_summary | 6_discourse | 7_dna | 8_sna |
|---|---|---|---|---|---|
| finland | ok 1,275 / empty 850 | ok 710 / reprocess 330 / empty 1,078 / delete 7 | ok 671 / error 31 / empty 1,423 | ok 575 / error 76 / empty 1,474 | ok 615 / error 1 / empty 1,509 |
| poland | ok **1,781** | ok 960 / reprocess 131 / empty 682 / delete 8 | ok **10** / empty 1,771 | empty **1,781** | empty **1,781** |
| portugal | ok 960 | ok 447 / reprocess 137 / empty 360 / delete 16 | ok 436 / empty 524 | ok 16 / error 9 / empty 935 | empty **960** |

* **Status and output are different facts.** FI `3_video` is `ok` on 1,275 rows and shows 0 output in the
  CSV; PL `3_video` is `ok` on **all 1,781**. Both are true at once. Any coverage report that reads a
  status as "produced something" will be wrong; so will any that reads a blank CSV cell as "produced
  nothing". This is the pair of errors #332 named.
* **Language tags are clean.** `asr_language` is `fi` for all 2,125 FI rows, `pl` for all 1,781 PL rows and
  `pt` for all 960 PT rows — no cross-country contamination in the delivered corpus.
* **DNA quotes.** 465 of 1,583 FI statements carry `evidence_quote`; Portugal's 4 of 4 do. The remaining
  1,118 FI statements were generated 2026-10-13T20:51 – 2026-10-14T03:41, and Step 7 has refused to
  persist an ungrounded statement since commit `f2ec1e0` (#289, **2026-10-14**). Verified in the bundle as
  well as the child table: the quote is absent from the pipeline document, so this is pre-guard output, not
  an exporter loss — the action is a Step 6–8 re-run for Finland, not a patch.
* **Freshness.** FI Steps 6–8 end 2026-10-14 / 03:41 / 03:56 while `preprocess_completed_at` reaches
  2026-10-23; PL discourse ends 2026-10-16 against a 2026-10-20 ingestion; Portugal is aligned
  (2026-10-25). The delivered analysis describes an **older slice** than the corpus it ships with. Stage
  coverage cannot express that, which is why the audit reports it separately as
  `downstream-older-than-upstream`.
* **Ingest denominators.** 1,038 of 4,866 rows (21 %) are `reprocess` (835) or `delete` (203) at Step 0 —
  FI 518+26, PL 217+22, PT 184+71 — and they sit inside every downstream denominator. Whether the
  researcher-facing denominator should exclude them is a methods decision; the audit reports the counts and
  changes nothing.

## Fixed here

`write_cumulative_csv` now serializes container cells with `json_safe_cell()` instead of letting
`DataFrame.to_csv` write a Python `repr`. Measured on this export: **6,832** cells across
`ep24_human_seed_context` (2,814), `step4_summary_provenance` (2,726), `_pipeline_step_6` (676) and
`_pipeline_step_7` (616) look structured and do not parse as JSON, so every consumer that calls
`json.loads` on them fails. Round-trip test in `tests/test_issue333_csv_json_cells.py`.

## Blocked

The re-runs the findings call for (FI Steps 6–8 after the grounding guard; PL Steps 6–8; PT Steps 7–8)
need the authorized Roihu run and the shared MongoDB. #342 carries the planner that selects only the
records and stages that need it; nothing here claims a live repair.
