# EP24 issue #306: public/private reporting safeguard

The EP24 reporter in `ep24_result_reporting.py` is **disabled by default**. Generated text can reproduce source transcripts and researcher materials, so output to stdout or batch logs is not automatically safe merely because original input fields are excluded.

On an **authorized private** compute environment only, enable human-readable result logs explicitly:

```bash
export LACLAUGPT_PRINT_ANALYSIS=1
export LACLAUGPT_EP24_REVIEW_DIR="$PRIVATE_REVIEW_DIR"
export LACLAUGPT_ANALYSIS_LOG_DIR="$PRIVATE_LOG_DIR"
```

Never enable this switch in public CI, public GitHub Actions artifacts, or externally archived Slurm logs. Paths are deliberately supplied at runtime, not committed. Keep real export bundles and all review output in private storage.

## Reuse the existing canonical export contract

Do **not** add a competing producer exporter. The existing read-only FI/PL/PT export is maintained in `LaclauGPT-Data-Visualization/scripts/ep24/export_final_results.py` and documented in `docs/EP24_FINAL_EXPORT.md` in that repository (issue #195). It supports explicit country checkpoints or a private MongoDB URI via environment configuration, deduplication and quarantines, stage coverage, per-country and combined exports, and manifest provenance. Before presenting an export as current, check the actual source collection and run IDs on an authorized host; the exporter alone does not prove that it selected the most recent reanalysis.

## Unmet empirical acceptance gates

No authorized shared MongoDB/CSC runtime has been inspected as part of this safeguard. FI/PL/PT per-stage error rates, actual latest-run provenance, quarantined records, codebook review, pipeline reruns, deployed dashboard and before/after empirical quality **remain unverified**. Keep issue #306 open until real-data checks, stage 0–9 audits, privacy reviews and dashboard smoke tests are complete.

Regression test: `python -m pytest -q tests/test_ep24_result_reporting.py`.
