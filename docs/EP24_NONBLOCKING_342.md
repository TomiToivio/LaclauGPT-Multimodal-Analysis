# Non-blocking stages and selective reprocessing (#342)

The pipeline gated every stage on its predecessor's `status == "complete"`. That is stricter than the
research needs and it is why one weak stage stalled everything behind it: a record with a usable
transcript but no video never reached discourse, DNA, SNA or RDF.

## Four separate facts, not one

`ep24_eligibility.py` keeps apart what the old two-state model collapsed:

| Fact | Values | Why it is separate |
| --- | --- | --- |
| Execution status | `not_run`, `running`, `succeeded`, `partial`, `failed`, `skipped` | what the stage did |
| Usable output | yes / no | what a downstream consumer can actually read |
| Diagnostic | one line | why, without rewriting history |
| Retry eligibility | yes / no | whether to pick it up again |

The rule that matters most: **a record with usable output is never retried**, even when an old failure
flag or a blank index cell says otherwise. A stale `error` does not make work retryable if a valid
result exists; a blank CSV cell does not mean nothing was produced (#331/#332).

## Inputs are data, not status words

A stage's input requirement is expressed as alternate routes over fields. Any route satisfied is enough:

| Stage | Routes (any one suffices) |
| --- | --- |
| 3 video | `frame_file` + `frame_timestamp_seconds` |
| 4 summary | transcript, or OCR, or frame analysis |
| 6 discourse | postprocess themes/entities, summary, or transcript |
| 7 DNA | discourse JSON, discourse markdown, themes, or transcript |
| 8 SNA | DNA statements, or postprocess entities |

This is what lets a text-only record reach discourse without a video, and lets DNA run on *partial*
discourse. When no route is satisfied the record is **skipped** — recorded, not retried.

## Opting in

```bash
export LACLAUGPT_EP24_NONBLOCKING=1     # default off: existing runs keep their exact behaviour
```

With it set, `eligible_query()` prefilters on consumed data instead of a predecessor's `complete`
status. The prefilter is deliberately wider than the row-level decision, because a Mongo query cannot
express "any of four routes" without risking a false negative.

## Selective reprocessing

```bash
# what would run? (dry run is the default)
python scripts/roihu/reprocess_selected.py \
    --export /private/ep24/mongo_export_210 --steps 3,6,7

# record the decision for the audit trail
python scripts/roihu/reprocess_selected.py \
    --export /private/ep24/mongo_export_210 --steps 3 --apply --manifest /private/ep24/rerun.json
```

Historical preview on the 2026-10-28 snapshot (`fi 2125 / pl 1781 / pt 960`),
**invalid as a retry worklist** because this version ignored heavy-field elision:

```
selected 11814 record-stage pairs, skipped 2784 (3:4866, 6:3217, 7:3731)
  failed_without_output 4047
  not_run              7767
```

The selection is deterministic (sorted by country, record, stage), so two runs produce the same
manifest and a diff between them is meaningful.

### What it will never do

- select a record whose stage output exists;
- treat a blank index cell as proof of absence (#331);
- rerun a stage whose status says success but produced nothing, unless `--include-empty` is given;
- submit anything: **live submission is not implemented**; `--execute` and `--commands` explicitly fail closed, rather than silently succeeding or printing placeholder `sbatch` paths. `--apply` only writes a local manifest.

### Before any live run

The script reads an **authorized export**, not live Mongo. Its manifest is **not** a validated live retry plan: the exporter can elide data into sidecar bundles, and both preview commands now use the country-specific elision detection from `plan_ep24_retry.py`.
The Roihu manifest records these pairs as `unverifiable_output` and does not select them, even with
`--records` or `--include-empty`. This conservative exclusion does not prove the output is valid. Do not pass this preview to Slurm as a worklist without live output verification. A real run must additionally pin: source
collection identity, repo SHA, run/checkpoint ids, model and prompt versions, and the destination.
Those are job-level facts, so they belong in the manifest the operator attaches to the Slurm job, not in
a CSV-derived selection.

## What remains unproven without Roihu access

- Step 3 results are present in FI/PL/PT text bundles; the index CSV alone cannot establish which individual results are absent. Live output checks and quality review are required before any retry.
- Steps 6–7 producer diagnoses, current live coverage and selective execution (Steps 8–9 are intentionally paused by #346). Historical Poland discourse metrics are superseded by the refreshed pinned-export readiness report; consult `docs/EP24_STEPS_1_7_READINESS.md`.
- Any before/after quality claim for codebook or prompt changes (#333 deliverable 4).

## Evidence-preserving preview correction (#333 / #342)

The preview now recognizes the canonical `vllm_video_structured_json` output and
`vllm_video_status`, retaining the older video field names for compatibility. A failed stage with
output keeps its failure status and diagnostic, remains available to downstream consumers, and is
not selected for retry. Manifest entries carry execution status, output availability and retry
eligibility separately. A success label without an artifact does not make downstream output available.

Synthetic regressions cover failure-with-output, success-without-output, canonical video fields,
country-isolated elision, and explicitly named records whose outputs were relocated. These are
preview fixes only: `--records` filters the export manifest; it is not a record selector in the
Slurm stage runners. No live submission backend, live audit, or blinded quality comparison is claimed.
