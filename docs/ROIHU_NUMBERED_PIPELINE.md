# Numbered CSC Roihu pipeline

The active EP24 workflow is intentionally boring to operate: **one Python file, one sbatch file, one stage at a time**. This makes failures, logs, intermediate CSVs and model changes easy to inspect.

## Stable numbering

| Step | Python | Slurm | Purpose |
|---|---|---|---|
| 1 | `step_1_roihu_preprocess.py` | `scripts/roihu/step_1_roihu_preprocess.sbatch` | frames/OCR/ASR |
| 2 | `step_2_roihu_frame.py` | `scripts/roihu/step_2_roihu_frame.sbatch` | sampled-frame VLM |
| 3 | `step_3_roihu_video.py` | `scripts/roihu/step_3_roihu_video.sbatch` | optional whole-video vLLM |
| 4 | `step_4_roihu_summary.py` | `scripts/roihu/step_4_roihu_summary.sbatch` | multimodal summary/fusion |
| 5 | `step_5_roihu_postprocess.py` | `scripts/roihu/step_5_roihu_postprocess.sbatch` | structured legacy postprocess |
| 6 | `step_6_roihu_discourse_analysis.py` | `scripts/roihu/step_6_roihu_discourse_analysis.sbatch` | Laclau/Palonen discourse analysis |
| 7 | `step_7_roihu_discourse_network_analysis.py` | `scripts/roihu/step_7_roihu_discourse_network_analysis.sbatch` | DNA actor-concept statements |
| 8 | `step_8_roihu_social_network_analysis.py` | `scripts/roihu/step_8_roihu_social_network_analysis.sbatch` | SNA actor-actor relations |
| 9 | `step_9_roihu_rdf.py` | `scripts/roihu/step_9_roihu_rdf.sbatch` | RDF export |

Step 3 is reserved now so Steps 4-9 never need renaming. Until the vLLM experiment is accepted, Step 3 is optional.

## First demo: 100 videos/rows

The numbered jobs default to:

```bash
export LACLAUGPT_MAX_ROWS=100
```

Run the full corpus later by changing/removing the limit as appropriate.

## Submission pattern

From the public checkout:

```bash
sbatch scripts/roihu/step_1_roihu_preprocess.sbatch
# inspect output/logs
sbatch scripts/roihu/step_2_roihu_frame.sbatch
# optional while experimental
sbatch scripts/roihu/step_3_roihu_video.sbatch
sbatch scripts/roihu/step_4_roihu_summary.sbatch
sbatch scripts/roihu/step_5_roihu_postprocess.sbatch
sbatch scripts/roihu/step_6_roihu_discourse_analysis.sbatch
sbatch scripts/roihu/step_7_roihu_discourse_network_analysis.sbatch
sbatch scripts/roihu/step_8_roihu_social_network_analysis.sbatch
sbatch scripts/roihu/step_9_roihu_rdf.sbatch --input ep24_fi.csv
```

All model-backed batch files target Roihu `gpumedium` with one GH200. The deterministic RDF export uses a CPU partition.

## Why DNA and SNA are separate

DNA is a discourse-coding layer: it emits evidence-linked actor-concept/proposition statements with stance/agreement and uncertainty.

SNA is an actor-network layer: it emits evidence-supported actor-to-actor relations. It must not invent friendship, coordination, ideology, influence or hidden ties.

Keeping them separate makes the methodological boundary inspectable and allows later aggregate graph metrics to be computed without rerunning the LLM extraction.

## RDF placement

RDF belongs after DNA/SNA because it is a serialization/export layer, not another interpretive model call. The existing `roihu_csv_rdf.py` exporter remains the implementation.

## Legacy compatibility

The historical `roihu_*.py` implementations remain in the repository. The numbered files are the canonical batch-facing interface, not a license to remove legacy columns, prompts, caches or outputs.

## Restartable execution

The preferred operational interface is now the one-command launcher for each stage:

```bash
python scripts/ep24/bootstrap_ep24_mongodb.py
bash scripts/roihu/run_step_1.sh
bash scripts/roihu/run_step_2.sh
# ...
bash scripts/roihu/run_step_9.sh
```

The sbatch jobs call `ep24_stage_orchestrator.py`, which claims only eligible MongoDB records, runs the existing numbered Python stage in small batches, preserves every cumulative column, checkpoints each successful batch to CSV + SQLite, and marks it complete before claiming more work. A later step therefore does not wait for the entire previous country/corpus to finish.

Country priority is Finland, Poland, Portugal, then the remaining `to_reprocess` files alphabetically. GPU jobs remain on `gpumedium` at no more than 36 hours. The orchestrator uses a 35-hour soft deadline by default so it can exit cleanly before Slurm kills the job.

See [EP24_RESTARTABLE_PIPELINE.md](EP24_RESTARTABLE_PIPELINE.md).
