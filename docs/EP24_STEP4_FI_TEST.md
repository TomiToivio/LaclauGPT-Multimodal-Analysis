# Finnish Step 4 Roihu test (issue #236)

This is a **small manual quality gate**, not a force rerun. PRs #244 and #262 implement bounded and auditable model-visible evidence, but the human review on real FI videos is still required.

## On Roihu, from the repository root

1. Update your checkout to a reviewed current revision, activate the project's Roihu Python environment, and ensure the configured Ollama model is running/reachable. The Step 4 default is specified in `ep24_models.py`; override it only if needed with `LACLAUGPT_MULTIMODAL_MODEL`. Do not substitute a different model unnoticed when comparing outputs.
2. Identify the **existing Finland Step 3 CSV checkpoint** (normally `/scratch/project_2009497/LaclauGPT-Private/analysis/ep24_reprocess/outputs/finland/step_03_video.csv`). Confirm it contains a few real Finnish records with Step 3 narrative plus ASR/frame/OCR where available. The script will fail if the file does not exist.
3. Run:
   ```bash
   export LACLAUGPT_MULTIMODAL_MODEL=qwen3.8:27b # only if this model is installed; otherwise set your actual model
   bash scripts/ep24/test_step4_fi.sh /scratch/project_2009497/LaclauGPT-Private/analysis/ep24_reprocess/outputs/finland/step_03_video.csv 10
   ```
4. The script creates a **timestamped isolated folder** under `.local/step4-fi-test-*/` with `step3_fi_sample.csv`, `step4_fi_output.csv`, and a private SQLite cache. It disables Mongo writes and retrieval/RAG, unsets Redis URL, and leaves the input CSV unchanged. It selects the first 10 eligible records in the checkpoint; this is not a random or representative sample. It *does not* automatically run Steps 1–3, download video, or perform automatic scientific validation.
5. Manually compare `summary_analysis` in `step4_fi_output.csv` with the source evidence and any previous `summary_analysis` values for the **same video_id/source IDs**. Score original Finnish wording, English translation fidelity, source vs inferred claims, ASR/OCR accuracy, cross-modal contradictions, political descriptions, unsupported statements, output completeness, and any truncation reported by the `context_budget` log. Review `logs/summary.log` for `row_failed`, `cache_hit`, and model errors.

**Do not** run the normal `--country finland` production command as your first sample test: its default output is the canonical cumulative checkpoint. Do not overwrite previous Step 4 results or enable Mongo until the ten-video comparison is approved. The script disables the Mongo connection and corpus retrieval, but other local shared runtime logs may still be updated.

## Acceptance record

Paste a short scorecard into [issue #236](https://github.com/TomiToivio/LaclauGPT-Multimodal-Analysis/issues/236): source revision, sample IDs, model and settings, number succeeded/failed, truncation examples, unsupported claims, and verdict on whether to widen the sample. Leave the issue open until the actual Roihu review is complete.
