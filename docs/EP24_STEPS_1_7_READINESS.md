# EP24 Steps 1–7 readiness — FI/PL/PT

**Date:** 2026-10-31 · **Scope decision:** #346 (Steps 8–9 paused, Step 7 terminal) · **Baseline:** the
pinned audit export (`mongo_export_210`, `ep24-mongo-export-1`, `record_unit: video`, generated
`2026-10-27T10:39:35Z`)

This is a readiness report for the **active** pipeline, not a research report. Every number comes from the
delivered export; **no live Roihu or MongoDB run was available to this session**, so nothing below claims a
live repair. Where a number is missing because the export cannot express it, the row says so.

## Scope: what a normal run submits

```bash
bash scripts/roihu/run_country_pipeline.sh finland          # steps 1-7, the default
bash scripts/roihu/run_country_pipeline.sh poland --limit 50 # bounded smoke test
bash scripts/roihu/run_country_pipeline.sh portugal --to-step 9 --include-paused  # explicit opt-in
```

* Default `TO_STEP=7`. Steps 8–9 need `--include-paused` **and** an explicit `--to-step`; without it the
  launcher exits 2 and prints why.
* `LACLAUGPT_PAUSED_STAGES=8,9` is exported for the run and written beside the manifest
  (`<run>.scope`) and into `MANIFEST.json` as `paused_stages`. Coverage numbers are **not** rewritten:
  a paused stage stays visible as paused, and historical exports keep their historical values.
* Step 8/9 code, tests, outputs and provenance are untouched. Reactivation is `LACLAUGPT_PAUSED_STAGES=""`.

## Steps 1–7 by country (measured on the pinned export)

| Country | Records | 1 Preprocess | 2 Frame | 3 Video | 4 Summary | 5 Postprocess | 6 Discourse | 7 DNA |
|---|---|---|---|---|---|---|---|---|
| Finland | 2,125 | 2,125 output / 19 failed-or-other | 2,125 / 0 | 1,275 / 850 empty | 1,047 (49 %) / 337 | 1,034 (49 %) | 671 (32 %) / 31 error | 575 (27 %) / 76 error |
| Poland | 1,781 | 1,781 / **639** | 1,781 / 16 | **1,781** / 0 | 1,099 (62 %) / 139 | 1,763 (99 %) | **10 (0.6 %)** | **0** |
| Portugal | 960 | 960 / **300** | 960 / 66 | **960** / 0 | 600 (63 %) / 153 | 960 (100 %) | 436 (45 %) | 16 (1.7 %) / 9 error |

Three readings of that table matter more than the numbers:

1. **Step 3 output exists even where the CSV shows nothing.** `stage_coverage` reports 1,275 / 1,781 / 960
   records with video output and `vllm_video_analysis` is present in all three text bundles. The CSV shows
   0 because every such cell is over the 1,000-character split threshold. "Video unavailable" — as the
   first stage audit read it — is wrong, and a Step 3 rerun planned from the CSV alone would reprocess
   **4,016 records that already have output**.
2. **Status is not output.** Poland's Step 3 is `ok` on all 1,781 rows; Finland's Step 1 is `ok`/`cached` on
   every row while 19 records are failed-or-other. Both facts are true at once.
3. **Steps 6–7 are the bottleneck, and one of them is stale rather than broken** (below).

## Verified fixes landed for this scope

| Fix | Evidence | Reference |
|---|---|---|
| Export states its own joins, elisions and disagreements (`parent_record_id`, `heavy_fields`, `text_bundle`, `contract_warnings`) | exporter tests; contract documented | #330/#331/#332, PR #338 |
| Coverage counts a positive `dna_statement_count`, so a blanked cell is no longer "DNA unavailable" | same PR + #335 | #332 |
| Stage CSVs write JSON, not Python `repr` (6,832 cells on the pinned export) | round-trip test | #333, PR #345 |
| Retry planner never schedules verified output, and no longer schedules elided output as "missing" | `unverifiable` bucket, tests | #342, PR #343 + this change |
| Non-blocking predecessor gate: a stage may accept usable upstream **output** when its status is not `complete` | `predecessor_clause(allow_partial=True)`, existing gate unchanged by default | #342, this PR |
| Steps 8–9 paused by default, visibly, without rewriting coverage | launcher guard + `paused_stages` in the manifest | #346, this PR |
| Consumer side: DNA view reads the text bundle (1,490 FI statements from 517 records), service deployed and health-checked | live service verification | Visualization #215/#224 |

## Outstanding blockers (per stage)

**Step 1 — PL 639 / PT 300 failed-or-other while output coverage is 100 %.** The two facts disagree, which
means the status field and the output fields are describing different things (partial ASR/OCR, cached
runs, or a retried record). Needs live status reasons from the collection; the export carries no reason
column for step 1. *Blocker: no Mongo access.*

**Step 2 — PL 16 / PT 66 failed-or-other.** Same class as Step 1; frame output exists, status disagrees.

**Step 3 — output present, quality unverified.** The bundles carry `vllm_video_analysis`; what is unknown is
whether it is grounded and whether any record's status `ok` hides an empty analysis. A sample review needs
the real records. *Blocker: authorized data + a reviewer.*

**Step 4 — PT 600/960 (63 %) and FI 1,047 (49 %).** PT's gap is the anomaly #346 asks about: summary
coverage is *lower* than PT's postprocess (100 %) even though Step 5 consumes Step 4 output. Either Step 4
was skipped for 360 records or its status vocabulary is misleading. *Blocker: step log.*

**Step 5 — FI 1,034 (49 %) vs PL 99 % / PT 100 %.** With Step 4 at 49 % for FI, the step-5 ceiling is
inherited rather than a Step 5 defect. Fix Step 4 first, then re-measure.

**Step 6 — the critical bottleneck.** FI 671 (32 %), PT 436 (45 %), PL **10** records. Every Polish
discourse failure is upstream of DNA, which is why Poland has 0 statements. The delivered discourse output
is also **older than the corpus**: FI ends 2026-10-14 against preprocess 2026-10-23, PL 2026-10-16 against
2026-10-20.

**Step 7 — stale, not broken.** 1,118 of 1,583 FI DNA statements carry no `evidence_quote` or
`evidence_fields`. They were generated 2026-10-13/14, and Step 7 has refused to persist an ungrounded
statement since commit `f2ec1e0` (#289, 2026-10-14). Verified in the bundle as well as the child table, so
this is pre-guard output: **re-run, do not patch**. Portugal's 4 statements are all quoted.

## What Monday needs, in order

1. **Access**: a Roihu login plus the shared-MongoDB URI. Every blocker above needs it; none is a code
   question any more.
2. **Step 6 first** (it gates 7, and gates Poland entirely), then Step 7, then Step 4 → 5 for FI/PT.
3. **Step 3 sample review** on real records before any Step 3 rerun: the output exists, and rerunning
   blind is exactly what #342 exists to prevent.
4. Run the planner before any of it:

```bash
python scripts/ep24/plan_ep24_retry.py --export /private/ep24/<export> --out retry_plan.json
# steps 8-9 are excluded by default; counts marked 'unverifiable' are NOT work
```

A plan that schedules 4,016 Step 3 records is reading the CSV as if the bundles did not exist — check the
`unverifiable` column before submitting anything.
