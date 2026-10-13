# CSC Roihu per-step environments and execution

Issue #209 makes every EP24 numbered stage independently installable and independently submit-able. Shared shell helpers implement the mechanics, but the public operator interface is one setup script and one submit script per stage.

## Step matrix

| Step | Python entry point | Setup | Submit | Runtime / modules | Resources | Main dependencies |
|---|---|---|---|---|---|---|
| 1 preprocess | `step_1_roihu_preprocess.py` | `setup_step_1_preprocess.sh` | `submit_step_1_preprocess.sh` | `python-pytorch`, GCC 13.4, CUDA FFmpeg, Allas | GH200, 72 CPU, 36h | Pandas, OpenCV, EasyOCR, NeMo/Canary, optional Mongo/Redis |
| 2 frame | `step_2_roihu_frame.py` | `setup_step_2_frame.sh` | `submit_step_2_frame.sh` | `python-pytorch` + Ollama | GH200, 72 CPU, 36h | Pandas, OpenCV, Ollama |
| 3 video | `step_3_roihu_video.py` | `setup_step_3_video.sh` | `submit_step_3_video.sh` | `python-vllm`, Allas, GCC 14.3 + FFmpeg | GH200, 72 CPU, 36h | vLLM/Transformers from CSC module, Pandas, Allas/rclone |
| 4 summary | `step_4_roihu_summary.py` | `setup_step_4_summary.sh` | `submit_step_4_summary.sh` | `python-pytorch` + Ollama | GH200, 72 CPU, 36h | Pandas, Ollama, optional Mongo/Redis |
| 5 postprocess | `step_5_roihu_postprocess.py` | `setup_step_5_postprocess.sh` | `submit_step_5_postprocess.sh` | `python-pytorch` + Ollama | GH200, 72 CPU, 36h | Pandas, Pydantic, Ollama |
| 6 discourse | `step_6_roihu_discourse_analysis.py` | `setup_step_6_discourse_analysis.sh` | `submit_step_6_discourse_analysis.sh` | `python-pytorch` + Ollama | GH200, 72 CPU, 36h | Pandas, Pydantic, Ollama |
| 7 DNA | `step_7_roihu_discourse_network_analysis.py` | `setup_step_7_discourse_network_analysis.sh` | `submit_step_7_discourse_network_analysis.sh` | `python-pytorch` + Ollama | GH200, 72 CPU, 36h | Pandas, Pydantic, Ollama; deterministic DNA helpers are stdlib |
| 8 SNA | `step_8_roihu_social_network_analysis.py` | `setup_step_8_social_network_analysis.sh` | `submit_step_8_social_network_analysis.sh` | `python-pytorch` + Ollama | GH200, 72 CPU, 36h | Pandas, Pydantic, Ollama; graph metrics are deterministic stdlib |
| 9 RDF | `step_9_roihu_rdf.py` | `setup_step_9_rdf.sh` | `submit_step_9_rdf.sh` | CPU Python venv, no model stack | `small`, 4 CPU, 16G, 4h | RDF is stdlib; lightweight Pandas/PyMongo/Redis support optional orchestration |

Each venv lives under the private EP24 scratch root at `.venvs/step-N`. Requirements are reviewable under `requirements/roihu-step*.txt`. Set up Steps 1–8 on `roihu-gpu.csc.fi` (ARM64). You can also run `bash scripts/roihu/setup_step_9_rdf.sh` from that **same GPU login terminal**: it submits a CPU `small` Slurm setup job and waits for completion, creating the Step 9 x86_64 venv on the correct architecture. On an x86_64 CPU terminal, it installs directly. Step 9's execution continues to use the CPU `small` partition; no GH200 is allocated for RDF. Step 3 deliberately inherits vLLM and Transformers from CSC's `python-vllm` module rather than pip-replacing that stack. Step 9 deliberately avoids GPU/model dependencies.

## Setup and individual submission

From the repository checkout on Roihu:

```bash
bash scripts/roihu/setup_step_1_preprocess.sh
bash scripts/roihu/submit_step_1_preprocess.sh --country finland --limit 1
```

Repeat with the corresponding setup/submit pair for later steps. Setup scripts are idempotent and should be rerun when their requirements manifest changes, the CSC module stack changes, a model/runtime changes, or a venv is damaged.

Resource defaults can be overridden without editing scripts:

```bash
LACLAUGPT_SBATCH_TIME=08:00:00 \
  bash scripts/roihu/submit_step_4_summary.sh --country finland
```

Available overrides are `LACLAUGPT_SBATCH_PARTITION`, `LACLAUGPT_SBATCH_CPUS`, `LACLAUGPT_SBATCH_MEM`, `LACLAUGPT_SBATCH_GRES`, and `LACLAUGPT_SBATCH_TIME`.

## Optional Mongo/Redis

The generic runner has two paths:

- with `LACLAUGPT_MONGO_ENABLED=1`, use `ep24_stage_orchestrator.py` for restartable Mongo claims/checkpoints and optional Redis coordination;
- otherwise invoke the numbered Python step directly against the cumulative CSV chain.

This keeps MongoDB and Redis optional rather than making them accidental prerequisites for the per-step interface.

## End-to-end test run

The first integration run is exactly 30 records:

- 10 random Finland videos;
- 10 random Poland videos;
- 10 random Portugal videos.

```bash
bash scripts/roihu/run_pipeline.sh --test
```

The default seed is `209`; override it with `--seed N`. The selection is written once to `manifest.json` and materialized as three sampled input CSVs. The same records then propagate through Steps 1–9. Test mode disables Mongo orchestration so a pre-existing full Mongo collection cannot silently widen the sample.

Each country's Step 1–9 jobs form an `afterok` dependency chain. A final `afterany` summary job writes `summary.json` and `summary.md` with per-country/per-step Slurm state and exit code.

## Full production run

Only an explicit `--full` starts an all-country run:

```bash
bash scripts/roihu/run_pipeline.sh --full
```

The controller discovers every `ep24_*.csv` under the authoritative input root and submits all videos with no row limit through Steps 1–9. `--test` and `--full` are mutually exclusive; absence of a mode is an error.

Each run gets a stable run directory containing:

- `manifest.json`: mode, seed/sample (for test), countries and source root;
- `jobs.tsv`: country, step and Slurm job ID;
- `run.json`: manifest plus git SHA and submitted jobs;
- `outputs/`: cumulative per-country stage artifacts;
- `summary.json` and `summary.md`: final Slurm status report.

The pipeline fails forward safely: a failed required stage prevents later stages for that country via `afterok`, but the final `afterany` summary still runs.
