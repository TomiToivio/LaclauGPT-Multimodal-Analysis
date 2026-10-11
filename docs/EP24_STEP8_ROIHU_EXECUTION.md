# EP24 Step 8 Roihu execution: evidence-based SNA pilot (#238)

Step 8 extracts observable actor–actor relationships and builds SNA node/edge
tables and metrics. A separate Castells interpretation describes the empirical
graph; it must not change topology. Unlike Step 7 Leifeld DNA actor–concept
statements, Step 8 needs **explicit evidence of a relationship**.

## Deployment preflight

1. On Roihu, pull the latest merged public checkout and record Git SHA.
   The Step 8 SLURM launcher has been brought onto the shared
   `roihu_job_common.sh` PyTorch/Ollama initialization, consistent with
   Steps 4–7; common preflight now accepts Steps 7 and 8.
2. Confirm the private `.env`, MongoDB dataset/database and country
   collection match Steps 0–7. Never print credentials.
3. Confirm `_pipeline.step_07.status=complete` for each prospective
   record. Inspect DNA statement completeness and source quotations before
   continuing; upstream 'complete' alone does not validate research quality.
4. Confirm the nested-object SQLite checkpoint repair (#267) is present
   in the deployed checkout. Step 8 uses the same writer.
5. Validate configured Ollama model, GPU runtime and prompt controls.
   `LACLAUGPT_STEP8_MAX_EVIDENCE_CHARS` defaults to 12000; extraction
   options in the current code specify temperature 0 and 8192 context.
   Record model/version, prompt fingerprint and packet truncation.
6. Run CI/static checks and inspect any unrelated baseline failures before
   allocating the GH200. Do not change network schema or invent relations
   to make a graph nonempty.

## Bounded pilot: ten Finland records

After Step 7 pilot review, on a Roihu login node:

```bash
cd /scratch/project_2009497/LaclauGPT-Multimodal-Analysis
git pull --ff-only
export LACLAUGPT_COUNTRY=finland
export LACLAUGPT_MAX_ROWS=10
export LACLAUGPT_STEP8_MAX_EVIDENCE_CHARS=12000
bash scripts/roihu/run_step.sh 8 --country finland --limit 10
```

This submits a **real** GPU job via
`scripts/roihu/step_8_roihu_social_network_analysis.sbatch`.
Never use `--force` for the pilot. `--limit 10` selects the next eligible
MongoDB rows, not necessarily the same source IDs as the Step 4–7 pilots.
Capture actual `_storage_id`/`video_id` and compare **matched** records.
A deterministic FI/PL/PT 10-record benchmark requires an isolated
ID-scoped selection instead of a production force/reset.

Inspect `/scratch/project_2009497/logs/ep24_s8_<jobid>.out` and
`.err`, MongoDB `_pipeline.step_08`, SNA node/edge exports,
private stage logs, cumulative CSV/SQLite manifests, and row-level errors.
A successful process exit does not establish the validity of extracted ties.

## Manual edge-quality gate

For each sampled video, evaluate source ASR and visible words, Step 4
fusion, Step 6 analysis, Step 7 DNA claims and the Step 8 empirical output.

- Every SNA edge must have two distinct identifiable actors, an accurately
  typed relation and a **verbatim evidence_quote within the model-visible
  source/DNA packet**. The quote-presence filter is necessary, but not
  sufficient proof of the proposed relation.
- Reject self-edges, empty or very short unsupported quotations, presumed
  hidden coordination, friendship, ideological alliances, influence or
  causation drawn only from co-mentions.
- If only an actor–concept stance exists, keep it as DNA. Do not turn it into
  an actor–actor SNA tie without independently evidenced contact/relation.
- Confirm directionality, canonical actor identifiers, unique/stable IDs,
  `network_layer=social`, node and edge schema and graph metrics.
- With no supported explicit ties, **zero edges is a valid result**.
  Do not fabricate a network for the visualization.
- Castellsian communication/network power must be explicitly marked as
  interpretation *after* topology construction; no unobserved programmer,
  switcher, hidden network or strategic control claims.
- Keep prior Mongo/CSV fields, source data and researcher interpretations
  intact; do not delete media based on a processing disposition.

Log observed false-positive and false-negative edges, unsupported relation
quotes, malformed JSON, multilingual evidence loss, graph invariants,
model/settings, sample identities and comparisons to prior outputs where
matching IDs exist. After Finland passes, repeat the same controlled
ten-record validation for Poland and Portugal before full-country reruns.
Keep #238 open until the real Roihu/manual quality gate is complete.
