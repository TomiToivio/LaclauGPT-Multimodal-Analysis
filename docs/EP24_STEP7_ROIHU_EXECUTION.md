# EP24 Step 7 Roihu execution: Leifeld DNA pilot (#240)

Step 7 extracts **observable actor–concept statements** for Philip Leifeld's
Discourse Network Analysis (DNA), using the existing `ep24_dna` compatibility
schema. It is not Step 6's Laclaudian interpretation and not Step 8's SNA
actor–actor relationship inference. No theory or statement schema change is
authorized by this execution handoff.

## Preflight and dependencies

- Pull an updated checkout on Roihu login node; note Git SHA. This branch
  fixes the older Step 7 sbatch runtime path, replacing it with
  `roihu_job_common.sh`, and extends preflight to Step 7.
- Confirm the shared private `.env`, MongoDB database/dataset and per-country
  collections match Steps 0–6. Do not print connection credentials.
- Confirm sample records have `_pipeline.step_06.status=complete`, usable
  original speech/transcripts and Step 6 outputs. The generic orchestrator
  checks Stage 6 completion, **not** statement evidence quality.
- Confirm the nested-value SQLite checkpoint repair (#267) exists in the
  deployed checkout. Step 7's orchestrator uses the same checkpoint writer.
- Step 7 defaults to a source-first 20,000-character evidence packet
  (`LACLAUGPT_STEP7_MAX_CONTEXT_CHARS`); check
  `LACLAUGPT_STEP7_NUM_CTX` and `LACLAUGPT_STEP7_NUM_PREDICT`
  and the installed Ollama model. Memory and RAG are not direct evidence.
- Check CI and syntax before launching on a GH200. The runtime script loads
  PyTorch ARM64 environment, runs Step 7 preflight and starts Ollama.

## First pilot (10 eligible Finland records)

Run **on the Roihu login node** after Step 6 pilot quality review:

```bash
cd /scratch/project_2009497/LaclauGPT-Multimodal-Analysis
git pull --ff-only
export LACLAUGPT_COUNTRY=finland
export LACLAUGPT_MAX_ROWS=10
export LACLAUGPT_STEP7_MAX_CONTEXT_CHARS=20000
bash scripts/roihu/run_step.sh 7 --country finland --limit 10
```

This submits a real job. **Do not pass `--force`.** The ten rows are
the next eligible MongoDB claims, not necessarily the same IDs selected
for the Step 4–6 review. Record actual `_storage_id`/`video_id` values,
then compare matched source IDs; use an isolated ID-filtered sample for
a reproducible 10-per-country FI/PL/PT benchmark.

Inspect `/scratch/project_2009497/logs/ep24_s7_<jobid>.out` and
`.err`, private logs, `_pipeline.step_07` Mongo statuses, exports,
statement JSON and CSV/SQLite checkpoint manifest. A job exit code alone
does not certify statement-level fidelity.

## DNA evidence-quality gate

Review each proposed statement against original-language source material:
- One identifiable actor, **one** claimed concept/policy position, stated
  time/source context and verbatim `evidence_quote`.
- Qualifier `support` or `oppose` only when explicitly supported;
  otherwise `uncertain` with uncertainty notes. Do not infer agreement
  from an actor merely mentioning a topic.
- Preserve observed actor and concept labels alongside canonical versions.
  Codebook/RAG/memory may normalize, but cannot manufacture statements.
- Source quotations must be present in actual evidence; uncertain OCR/ASR
  translations and contradictory claims are not smoothed away.
- No source claim means **empty statements**, not invented edges.
- Confirm Leifeld DNA/rDNA schema and stable statement IDs, network layer,
  statement time, provenance, and actor–concept direction/qualifier.
- Keep DNA ties distinct from empirically supported Step 8 social relations.
  Preserve all upstream rows, researcher annotations and source fields.

Log sample IDs, model and effective settings, prompt/context hash, observed
false positives, missing statements and comparison with previous output for
matching IDs. Review Finland, Poland and Portugal samples (ten each) before
any full-country rerun. Leave issue #240 open pending manual validation.
