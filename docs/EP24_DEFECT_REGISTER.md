# EP24 defect register — 2026-10-31 (#333 deliverable 2)

Prioritised defects found by auditing the delivered artifact sets, with the evidence that establishes
each one and the component responsible. Produced offline: no MongoDB, no Roihu, no network.

Reproduce:

```bash
python scripts/ep24/audit_export.py <export-dir>
python scripts/ep24/audit_export_diff.py <older-export> <newer-export>
```

Both print counts and column names only — never cell contents — so their output is safe to paste in a
public issue.

## Provenance first: there is only one pipeline result set here

Four export directories exist locally. They are **two packages of one snapshot**, not four runs:

| Export | MANIFEST `generated_at` | `combined.csv` md5 |
| --- | --- | --- |
| `mongo_export` | 2026-10-27T07:54:15Z | `317bdbf3…` |
| `mongo_export_306` | 2026-10-27T07:56:02Z | `317bdbf3…` |
| `mongo_export_20261028` | 2026-10-28T13:59:39Z | `44fc9f93…` |
| `mongo_export_210` | 2026-10-28T14:09:48Z | `44fc9f93…` |

Diffing `mongo_export` → `mongo_export_210` across all three countries and eight measurable stages:
**0 records gained, 0 lost, 0 changed** (`content_changed = 0` for every stage; per-record SHA-256 of
each stage's values, so this is not a presence-only comparison).

**Consequence:** "the latest authorized results" for this corpus are the 2026-10-27/28 snapshot. Anything
reported as an improvement after that date — in an issue, a dashboard or a paper draft — is a
re-packaging, not new computation. Establishing a genuinely newer run needs Roihu access and CSC
credentials, which this session does not have.

## Defects, in priority order

### D1 — Step 3 produces no output for any country (confirmed missing output)

`vllm_structured_status` is `off` on 1,275 Finnish rows and empty on the remaining 850; the output
columns are non-empty on **0** rows in every export. Columns are present, so this is not a mapping bug in
the exporter: the stage genuinely did not produce.

**Owner:** Step 3 (`experiments/vllm_video_test.py`, GPU/vLLM environment).
**Needs:** a run with the vLLM environment available; cannot be repaired from artifacts.

### D2 — Coverage and index disagree for Steps 6 and 7 (fixed in the exporter, present in the data)

517 Finnish rows carry `dna_statement_count > 0`, 671 carry `laclau_status = ok`, yet the index shows
80 rows with discourse output and 0 with DNA JSON. Both are consequences of the >1,000-character split
plus, historically, coverage measured on a field the split had emptied.

**Owner:** exporter (fixed — see #331/#332). **Residual:** exports produced before the fix still carry
the silent hole; a re-export is required for the manifest and index to agree.

### D3 — Poland's Step 6 produced 1 record out of 1,781 (confirmed near-total failure)

`laclau_status` non-empty on one row; `children/poland_dna_statements.csv` absent; no SNA collection.
The absence of DNA/SNA for Poland is recorded by the manifest as "absent", which is honest — but
Step 6 running on a corpus and yielding one row is a failure mode, not an empty corpus.

**Owner:** Step 6 orchestration for Poland. **Needs:** run logs for the Polish Step 6 batch to separate
input-mapping failure from model refusal; not answerable from artifacts.

### D4 — Step 9 (RDF) is unmeasurable from the export (design gap)

No `rdf_*` columns exist in the index; RDF is file-based. A consumer following the CSV cannot see whether
Step 9 ran, and the audit must report "columns absent" rather than a percentage.

**Owner:** exporter contract. **Cheap fix:** emit a per-record `rdf_exported` flag or a manifest
`stage_coverage["9_rdf"]` sourced from the RDF output directory listing.

### D5 — Portugal's Step 4 produced nothing (0 of 960)

`summary_quality_status` is non-empty on 0 rows, with 153 rows in a non-success state. Unlike Finland
(17) and Poland (17), Portugal has no summary output at all.

**Owner:** Step 4. **Needs:** logs; a model-side refusal and a missing-input condition look identical in
the artifact.

## What is *not* a defect

- **Polish DNA statements: 0.** The collection does not exist for Poland. That is a finding, and the
  manifest says `absent`; it is not a silent zero.
- **`processing_status` values `REPROCESS` (518 FI) and `DELETE` (26 FI).** These are queue states, not
  failures. Counting them as errors would inflate the apparent failure rate; the audit reports them as
  "status not OK" and names the vocabulary.
- **Step 2 frame analysis at 100 %** with `frame_analysis_status` non-success on 16 Polish rows: the
  outputs exist, so the stage did run.

## Deliverables status against #333

| Deliverable | State |
| --- | --- |
| 1. Coverage and QA report | **done** — `docs/EP24_EXPORT_AUDIT.md`, `audit_export.py`, and this register |
| 2. Prioritised defect register | **this document** |
| 3. Tested repairs + targeted reruns | repairs: exporter fixed (#331/#332/#338). **Reruns: blocked** — no Roihu/CSC access this session |
| 4. Evaluated codebook/prompt improvements | **blocked** — requires model reruns on a fixed sample to compare before/after; proposing edits without that evaluation would be prompt churn |
| 5. Final verification note | this section |

## What would unblock the rest

1. CSC credentials and a Roihu allocation → rerun Steps 3, 4 (PT) and 6 (PL) on affected records only,
   preserving provenance, then re-export and re-diff.
2. A fixed, blinded sample of ~50 records per country and language → evaluate codebook/prompt edits with
   quote fidelity and attribution checks rather than output volume.
