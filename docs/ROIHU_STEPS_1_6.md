# CSC Roihu operator runbook: EP24 steps 1-6

This is the operator path for the first real EP24 runs. Steps 7-9 are intentionally out of scope here.

## Verified Roihu resource contract

As of 2 October 2026, CSC documents `gpumedium` as a one-node Roihu-GPU partition with a **36 hour** time limit and 1-4 full NVIDIA GH200 GPUs. One reserved GH200 grants up to 72 CPU cores and 212 GiB allocatable memory. Build/install GPU-side Python software on `roihu-gpu.csc.fi`, because Roihu-GPU uses NVIDIA Grace ARM CPUs and is binary-incompatible with the x86 CPU side.

The jobs in this repository request one GH200, 72 CPU cores, one node and 36 hours. Step 3 uses CSC's `python-vllm` module; the other steps use `python-pytorch`.

## 1. First-time setup

On your workstation:

```bash
ssh roihu-gpu.csc.fi
```

On Roihu-GPU:

```bash
cd /scratch/project_2009497
git clone https://github.com/TomiToivio/LaclauGPT-Multimodal-Analysis.git
git clone git@github.com:TomiToivio/LaclauGPT-Private.git

cd LaclauGPT-Multimodal-Analysis
export CSC_PROJECT=project_2009497
export LACLAUGPT_PRIVATE_ROOT=/scratch/project_2009497/LaclauGPT-Private
export LACLAUGPT_MULTIMODAL_PRIVATE_ROOT=$LACLAUGPT_PRIVATE_ROOT/analysis/ep24_reprocess

bash scripts/roihu/install_roihu.sh
```

Roihu uses the verified multimedia module pair:

```bash
module load gcc/13.4.0
module load ffmpeg/7.1-cuda12.4
```

### One-time Allas setup

The jobs load CSC\'s `allas` module automatically, but authentication must be configured interactively once for your CSC account. On Roihu run:

```bash
module load allas
allas-conf project_2009497
check-allas-connections
```

Roihu\'s `allas-conf` defaults to S3. It writes user-level S3/rclone/AWS configuration used by later sessions and batch jobs. Do not put the CSC password or generated credentials in the public repository.

The installer creates/reuses an ARM64 venv at `.venv-roihu-gpu`, uses CSC's CUDA-enabled `python-pytorch` packages through `--system-site-packages`, installs the steps 1-6 Python dependencies, prepares project-scratch caches, and installs a local ARM64 Ollama under the private EP24 tree.

### Private configuration

Keep secrets and machine-specific settings in:

```text
/scratch/project_2009497/LaclauGPT-Private/analysis/ep24_reprocess/.env
```

At minimum for the restartable sbatch path:

```bash
LACLAUGPT_MONGO_ENABLED=1
LACLAUGPT_MONGO_URI='mongodb://...'
LACLAUGPT_MONGO_DATABASE=laclaugpt

# Optional Redis coordination
LACLAUGPT_REDIS_URL='redis://...'

# Roihu defaults; override only after testing
LACLAUGPT_ASR_ENGINE=canary
LACLAUGPT_OCR_ENGINE=easyocr
LACLAUGPT_MULTIMODAL_MODEL=qwen3.8:27b
```

Do not put these values in the public repository.

## 2. Update both repositories

```bash
cd /scratch/project_2009497/LaclauGPT-Multimodal-Analysis
git pull --ff-only

cd /scratch/project_2009497/LaclauGPT-Private
git pull --ff-only

cd /scratch/project_2009497/LaclauGPT-Multimodal-Analysis
```

Re-run the installer after dependency changes:

```bash
bash scripts/roihu/install_roihu.sh
```

## 3. First smoke run: Finland

Run one video per step, inspect the output, then continue:

```bash
sbatch scripts/roihu/step_1_roihu_preprocess.sbatch --country finland --limit 1
sbatch scripts/roihu/step_2_roihu_frame.sbatch --country finland --limit 1
sbatch scripts/roihu/step_3_roihu_video.sbatch --country finland --limit 1
sbatch scripts/roihu/step_4_roihu_summary.sbatch --country finland --limit 1
sbatch scripts/roihu/step_5_roihu_postprocess.sbatch --country finland --limit 1
sbatch scripts/roihu/step_6_roihu_discourse_analysis.sbatch --country finland --limit 1
```

Submit the next step only after the previous one has completed successfully.

For a slightly larger test:

```bash
sbatch scripts/roihu/step_1_roihu_preprocess.sbatch --country finland --limit 10
# then steps 2-6 with the same country/limit
```

Poland and Portugal use the identical commands:

```bash
sbatch scripts/roihu/step_1_roihu_preprocess.sbatch --country poland --limit 10
sbatch scripts/roihu/step_1_roihu_preprocess.sbatch --country portugal --limit 10
```

Repeat steps 2-6 for each country after Step 1 succeeds.

## 4. What each step consumes and produces

| Step | Runtime | Main input | Appended output |
|---|---|---|---|
| 1 preprocess | PyTorch + Canary/EasyOCR | private country CSV + staged video | one t=1s frame, OCR, ASR transcript/translation, provenance |
| 2 frame | Ollama multimodal | complete Step 1 record | one-frame visual analysis |
| 3 video | CSC `python-vllm` | complete Step 2 record + video | whole-video temporal analysis |
| 4 summary | Ollama | complete Step 3 record | evidence-preserving fused summary |
| 5 postprocess | Ollama | complete Step 4 record | entities, themes, sentiment targets, normalization context |
| 6 discourse | Ollama | complete Step 5 record | Laclau/Palonen structured discourse analysis |

The sbatch path uses MongoDB as durable cumulative state and writes human-readable checkpoint CSVs after each durable batch under:

```text
$LACLAUGPT_EP24_OUTPUT_ROOT/<country>/
```

Default:

```text
/scratch/project_2009497/LaclauGPT-Private/analysis/ep24_reprocess/outputs/<country>/
```

Local stage logs/caches live under the private EP24 tree. Slurm stdout/stderr uses `/scratch/project_2009497/logs/`.

## 5. Monitor

```bash
squeue -u "$USER"
sacct -j JOBID --format=JobID,JobName,Partition,State,Elapsed,ExitCode,AllocTRES
tail -f /scratch/project_2009497/logs/ep24_s1_JOBID.out
tail -f /scratch/project_2009497/logs/ep24_s1_JOBID.err
```

GPU diagnostics are printed at job start with `nvidia-smi`, together with hostname, Python version and git SHA.

## 6. Restart after failure

The orchestrator claims records in MongoDB and marks failed claims as errors. Jobs are safe to resubmit. By default, completed records are not reprocessed.

To retry rows marked as errors, use the orchestrator's retry option when applicable or reset the affected stage state deliberately. Do not delete cumulative CSVs or Mongo collections merely to retry a job.

Step-local SQLite caches are restart aids, not the durable source of truth.

## 7. Preflight without an expensive run

Inside a GPU allocation or job environment:

```bash
python scripts/roihu/preflight_steps_1_6.py --step 1 --country finland --limit 1
```

The sbatch files run preflight automatically. It catches a missing private root, bad country/limit, missing Mongo configuration, unavailable GPU, missing Python imports, unwritable output root, and a missing Step 1 country CSV.

## 8. Direct Python execution remains supported

For debugging outside the restartable sbatch wrapper:

```bash
python step_1_roihu_preprocess.py --country finland --limit 1
python step_2_roihu_frame.py --country finland --limit 1
```

The numbered scripts still preserve the existing CLI contract. The Roihu infrastructure changes do not rewrite the research prompts.
