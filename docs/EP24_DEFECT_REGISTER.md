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

### D6 — Step 7 DNA: the child table and the index describe different populations

Finland has 1,583 DNA statement rows in `children/finland_dna_statements.csv` against 1,490 in-frame
parents: the child table is at ingest grain, the index at video grain. Poland has no DNA collection at
all (the manifest records `absent`, which is honest) and Portugal has 4 statements for 960 records.

**Verified from artifacts:** the grain mismatch is real and the exporter now states both numbers
(`children` vs `children_detail`, `child_parent_join_key`). Step 7's own behaviour — why 27.1 % of Finnish
records have statements rather than 100 % — is **not** answerable from the export: a record whose
discourse step failed and a record whose DNA step was skipped look the same in the index, and the
per-record status fields do not distinguish them.
**Owner:** Step 7 gating on partial discourse. **Blocker:** Roihu run logs for the Finnish Step 7 batch.

### D7 — Step 8 SNA: absent for Poland and Portugal, thin for Finland, and the export cannot say why

`sna_nodes_json` / `sna_edges_json` are non-empty on 261 of 2,125 Finnish records (12.3 %), 0 of 1,781
Polish and 0 of 960 Portuguese. The Polish and Portuguese SNA collections do not exist (manifest:
`absent`); Finland's 12.3 % is consistent with SNA running only where DNA statements exist (1,583 → 261),
which is a dependency, not a defect — but the export cannot prove that ordering because Step 8 has no
status field.

**Owner:** Step 8 dependency selection. **Blocker:** live collection listing on Roihu to separate "not
scheduled" from "scheduled and produced nothing".
**Cheap fix (done in the planner):** Step 8 is paused by default (#346) so a retry plan does not queue
work whose output the export cannot verify.

### D8 — Step 3 is invisible to the export *and* paused, which is why the planner cannot triage it

Step 3 (video) has columns but no output anywhere, and is paused by default. Two consequences worth
stating plainly: the audit must report "columns present, output absent" rather than 0 %, and a dry-run
plan will not schedule Step 3 until the pause is lifted deliberately. Neither is a repair; both are
honesty about what the artifacts can support.

## Steps 3 and 6–9: cause or blocker, one line each (#342 acceptance)

The acceptance line asks for a *verified cause or an explicit unresolved live-access blocker* per step.
This is that table; the detail is in the defects above.

| Step | State | Verified from | Cause or blocker | Next action |
| --- | --- | --- | --- | --- |
| 3 Video | columns present, zero output in every country | export (D1, D8) | producer never wrote; vLLM/GPU environment not available this session | run Step 3 on Roihu with the vLLM env, then re-export |
| 6 Discourse | FI 3.8 %, PL 0.06 %, PT 8.8 % | export + D2, D3 | exporter elision was masking the truth (fixed, #331/#332); the residual near-total Polish failure is a run-level failure indistinguishable from refusal in the artifact | Polish Step 6 batch logs on Roihu |
| 7 DNA | FI 27.1 % of in-frame records; PL collection absent; PT 4 rows | children table + manifest `children_detail` | grain mismatch verified and now stated in the manifest; the per-record skip reason is not in the export | Finnish Step 7 logs; keep PL "absent" as a finding, not a zero |
| 8 SNA | FI 12.3 %, PL/PT collection absent | export + manifest | consistent with a DNA dependency; no status field exists to confirm ordering | list the collections live; Step 8 stays paused by default (#346) |
| 9 RDF | no columns, file-based | export schema | design gap in the contract, not a failed stage | emit `rdf_exported` per record or manifest coverage from the output directory |

Nothing in this table is a claim that a live failure was repaired. Four of the five need Roihu data that
this session cannot reach; the fifth (Step 9) is a contract change this repository can make without it.

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
