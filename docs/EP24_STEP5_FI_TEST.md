# Finnish Step 5 smoke test

Run **after** completing the isolated Finnish Step 4 test, using its real output CSV path.

```bash
bash scripts/ep24/test_step5_fi.sh /absolute/path/to/step4_fi_output.csv 10
```

This script copies up to ten Finnish records with valid Step 4 summaries into a new isolated directory under `.local/`, sets explicit input/output paths, disables Mongo, and isolates the memory database and private root. The configured Ollama model must be available on Roihu.

**Important safeguard:** Step 5 quality-routing can unlink local files for rows marked DELETE. The runner deliberately clears `frame_file`, `vllm_video_local_path`, and `vllm_video_analysis_path` **only in the temporary input CSV** so original media artifacts are never deletion targets. The copied paths intentionally differ from the source. This tests analysis/routing, not destructive cleanup.

Inspect `step5_fi_output.csv` for OK records and separately inspect the REPROCESS and DELETE audit CSVs created next to it. The main output contains only OK records. Check model-generated `postprocess_entities`, `postprocess_themes`, and sentiment lists against original Finnish content and the Step 4 summary. Confirm human `entities` and `themes` are preserved. Review `logs/postprocess.log` for model/normalization failures. Isolated private-root settings may prevent canonical enrichment if necessary private seed files are not present; distinguish extraction results from enrichment success.

The script has not been executed on real Roihu data. Do not promote it to production or overwrite previous outputs before reviewing the sample.
