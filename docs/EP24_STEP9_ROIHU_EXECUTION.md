# Step 9 RDF export: merge and Roihu readiness

Step 9 is a deterministic, **CPU-only** RDF/N-Triples/Turtle projection of
completed EP24 cumulative records. It does not perform new model inference or
alter the preceding analytical stages. The authoritative records remain in
MongoDB/CSV; graph statements must preserve source/derived provenance.

## Merge checklist

- Review the Step 9 `small` partition (x86_64, 4 CPUs, 16G, four hours).
- The sbatch launcher must use
  `/scratch/project_2009497/LaclauGPT-Private/analysis/ep24_reprocess`,
  not the legacy `LaclauGPT-Private/ep24_reprocess` directory.
- It must activate exactly the **Step 9 CPU** virtual environment generated
  by `scripts/roihu/setup_step_9_rdf.sh`, not the ARM64/GPU .venv.
- Keep the preflight lightweight and strictly CPU: no GH200, Ollama or vLLM.
- Validate against the existing RDF tests before merging (for example
  `pytest -q tests/test_roihu_csv_rdf.py`), plus shell syntax
  `bash -n scripts/roihu/step_9_roihu_rdf.sbatch`.
- Do not change the RDF schema, graph identities, quote provenance, or
  previous stage outputs just to prepare this deployment.

## Execution gate

1. On Roihu, pull the merged checkout and confirm the private `.env`
   points to the same MongoDB dataset/country namespace used for Steps 0–8.
   Never paste credentials into issues.
2. Verify `_pipeline.step_08.status=complete` for the selected records.
   Stage 8 SNA quote quality should be reviewed before broad export.
   The orchestrator uses upstream completion rather than automatically
   validating research fidelity.
3. Install the x86_64 environment from the login node:
   `bash scripts/roihu/setup_step_9_rdf.sh`. On ARM64, the helper
   schedules setup on a CPU compute node.
4. Pilot from the login node, **without `--force`**:

   ```bash
   cd /scratch/project_2009497/LaclauGPT-Multimodal-Analysis
   git pull --ff-only
   bash scripts/roihu/setup_step_9_rdf.sh
   export LACLAUGPT_COUNTRY=finland
   export LACLAUGPT_MAX_ROWS=10
   bash scripts/roihu/run_step.sh 9 --country finland --limit 10
   ```

5. Check `/scratch/project_2009497/logs/ep24_s9_<jobid>.out` and `.err`,
   the unique RDF output directory, `manifest.json`, `warnings.jsonl`,
   `graph.nt`, `graph.ttl`, and the cumulative checkpoint. Check that the
   RDF stage returns the orchestrator's `LACLAUGPT_OUTPUT_CSV` so the
   projected `rdf_* ` fields reach MongoDB without losing originals.
6. Compare record IDs, RDF source and provenance links, actor/concept
   statement and social-network edges to the original documents.
   An empty SNA graph is valid; RDF must not manufacture ties.
7. Repeat a small review for Poland and Portugal before full export.
   `--limit 10` chooses next eligible Mongo records, not necessarily the
   same IDs used in prior pilot runs. Compare actual source IDs.

## Status

This PR prepares Step 9 for **merge**, not for immediate unreviewed
full-country execution. The dedicated CPU setup and RDF exporter existed
already; the corrected launcher brings them together. Real Roihu tests
and source-grounded manual graph validation remain operational gates.
