# Laskin vLLM native-video feasibility

Issue: #32. This is an isolated experiment that reuses `experiments/vllm_video_test.py`; it does not replace the Roihu pipeline.

## Measured result (2026-10-01)

Laskin exposed three Tesla V100-PCIE-32GB GPUs (Volta, compute capability 7.0), driver 570.211.01 and CUDA 12.8. Current vLLM/Qwen3-VL releases require a newer GPU architecture, while the last tested vLLM line that runs on Volta does not register Qwen3-VL. Therefore the reference `Qwen/Qwen3-VL-8B-Instruct` is not portable to Laskin through a supported prebuilt vLLM stack.

A clean Python 3.10 environment with vLLM 0.8.5.post1, torch 2.6.0+cu124 and transformers 4.51.3 completed native whole-video inference after the mandatory 1.0-second trim:

| Model | Result | Inference | Peak GPU memory |
| --- | --- | ---: | ---: |
| Qwen2.5-VL-3B-Instruct | succeeded | 21.9 s | 14.54 GB |
| Qwen2.5-VL-7B-Instruct | succeeded | 21.0 s | 23.32 GB |
| Qwen3-VL-8B-Instruct | not viable with a prebuilt Volta-compatible vLLM | not run | not measured |

The successful clips were synthetic red-screen diagnostics, not private EP24 videos. These figures establish fallback-model feasibility, not Roihu parity or full private-sample acceptance. A CUDA 12.6 source build that restores sm_70 may be investigated separately, but it is not the reproducible default here.

## Install and inventory

```bash
cd /path/to/LaclauGPT-Multimodal-Analysis
bash scripts/laskin/install_vllm_video_test.sh
source .venv-laskin-vllm/bin/activate
bash scripts/laskin/check_vllm_environment.sh | tee /private/path/laskin-environment.txt
```

Do not use `--system-site-packages`: the observed user-site vLLM 0.15.1 installation was incompatible with the installed torch.

## Run

Prepare or update the private sample in `LaclauGPT-Private` first:

```bash
python analysis/ep24_reprocess/scripts/build_vllm_video_test_input.py
```

Then, on Laskin:

```bash
export LACLAUGPT_VLLM_TEST_INPUT_CSV=/private/path/analysis/ep24_reprocess/experiments/vllm_video_test_input.csv
export LACLAUGPT_VLLM_TEST_WORK_ROOT=/scratch/$USER/laclaugpt-vllm-video-test
# Configure Allas/rclone outside Git, or set FETCH_BACKEND=local and ALLAS_LOCAL_ROOT.
bash scripts/laskin/vllm_video_test.sh
```

The wrapper uses Qwen2.5-VL-7B and the older `direct` video API by default. It creates separate result, log, cache and download directories. The shared harness preserves source objects, trims a derived clip, processes every prepared row by default, isolates row failures, and records model/runtime/GPU provenance in the additive output CSV.

## Comparison boundary

A Roihu-vs-Laskin comparison is valid only when the same source sample, prompt hash, trim rule and model are used. Since Qwen3-VL-8B cannot run in the tested Laskin wheel stack, compare operational behavior separately from model-output quality. Do not label Qwen2.5-VL-7B output as a same-model benchmark.

Allas access and the full private ten-video run still require execution on the authorized host. Local CPU/stub tests do not prove either operational acceptance condition.