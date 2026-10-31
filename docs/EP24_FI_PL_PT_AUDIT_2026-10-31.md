# EP24 FI/PL/PT artifact audit — 2026-10-31

**Source:** the deployed `mongo_export_210` artifact set (`record_unit: video`), read on this host.
**Method:** `scripts/ep24/audit_ep24_results.py <export-dir>` — counts only, no transcripts, quotes or
identifiers.

This audit answers *what a consumer of the delivered export can see*, which is not the same question as
*what the pipeline produced*. Where the two disagree, the disagreement is the finding.

## Coverage matrix (artifact view)

| step | finland (2125) | poland (1781) | portugal (960) |
|---|---|---|---|
| 0_ingest | 100 % · 544 not-OK | 100 % · 239 | 100 % · 255 |
| 1_preprocess | 100 % · 19 | 100 % · 639 | 100 % · 300 |
| 2_frame | 100 % · 0 | 100 % · 16 | 100 % · 66 |
| 3_video | **0 %** · 1275 `off` | **0 %** · — | **0 %** · — |
| 4_summary | 0.8 % · 337 | 1.0 % · 139 | 0 % · 153 |
| 5_postprocess | 46.7 % | 97.1 % | 80.4 % |
| 6_discourse | 3.8 % · 31 error | 0.1 % | 8.8 % |
| 7_dna | **0 % in index** · child rows **1583** | 0 · **0** | 0 % · child **4** |
| 8_sna | 6.8 % · child rows **3359** | 0 · 0 | 0 · 0 |
| 9_rdf | columns absent | absent | absent |

## What is real, and what is an artefact of the export

1. **Step 3 produced nothing in this export.** `vllm_structured_status` is `off` for 1,275 Finnish rows
   and empty for the remaining 850; `vllm_structured_output` is non-empty for **zero** rows. That is a
   genuine empty stage, not a missing column — the columns are present.
2. **Steps 7 and 6 look empty because of the >1,000-character split, not because the models failed.**
   517 Finnish rows carry `dna_statement_count > 0`, but `dna_statements_json` is absent from the index
   for every one of them — the values were moved to `finland_text.jsonl.gz` with no sentinel and no
   manifest echo (#331.3). Likewise `laclau_status = ok` on 671 rows while only 80 show output in the
   index.
3. **Step 7 is genuinely absent for Poland** (`children/poland_dna_statements.csv` does not exist) and
   **4 statements for Portugal** — matching the manifest. Step 8 is absent for Poland and Portugal.
4. **Step 9 is file-based**, so it cannot be measured from the CSVs at all; reporting it as 0 % would be
   the exact error this exporter's docs warn about.
5. **`status not OK` is not "failed".** `processing_status` carries `OK` (1,581), `REPROCESS` (518) and
   `DELETE` (26) — a queue state, not an error count.

## Consequences for a consumer

A dashboard reading this export with the documented contract would report: no DNA evidence for Finland
(wrong — 1,583 child statements), no discourse output on 88 % of rows (wrong — 671 `ok`), and no video
analysis at all (correct).

Fixes #331/#332 make the same export self-describing: sentinel instead of a deleted cell, `heavy_fields`
in the manifest, coverage for child-table steps, and `children_unit` plus per-country `children_qa`.

## What this audit does not claim

- It reads **artifacts**, not the database. A stage that failed at source looks the same as one whose
  output was never exported; distinguishing them needs the run logs and collections on Roihu.
- It says nothing about research quality — codebook coverage, prompt versions, label reliability — which
  needs sampling transcripts and reviewing outputs (#333's second half).
- It measures one export. `mongo_export_20261028` and earlier runs were not compared.

## Reproduce

```bash
python scripts/ep24/audit_ep24_results.py <export-dir>          # markdown
python scripts/ep24/audit_ep24_results.py <export-dir> --json   # machine-readable
```
