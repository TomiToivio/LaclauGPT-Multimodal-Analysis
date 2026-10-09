#!/usr/bin/env bash
# Canonical per-step Roihu execution profiles for EP24 issue #209.
set -euo pipefail

roihu_step_name() {
  case "${1}" in
    1) echo preprocess ;;
    2) echo frame ;;
    3) echo video ;;
    4) echo summary ;;
    5) echo postprocess ;;
    6) echo discourse_analysis ;;
    7) echo discourse_network_analysis ;;
    8) echo social_network_analysis ;;
    9) echo rdf ;;
    *) echo "invalid EP24 step: ${1}" >&2; return 2 ;;
  esac
}

roihu_step_entrypoint() {
  local step="${1}" name
  name="$(roihu_step_name "${step}")"
  echo "step_${step}_roihu_${name}.py"
}

roihu_step_requirements() {
  local step="${1}" name
  name="$(roihu_step_name "${step}")"
  echo "requirements/roihu-step${step}-${name//_/-}.txt"
}

roihu_step_venv() {
  local step="${1}"
  local project="${CSC_PROJECT:-project_2009497}"
  local private="${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT:-/scratch/${project}/LaclauGPT-Private/analysis/ep24_reprocess}"
  echo "${private}/.venvs/step-${step}"
}

roihu_step_runtime() {
  case "${1}" in
    3) echo vllm ;;
    9) echo cpu ;;
    *) echo pytorch ;;
  esac
}

roihu_step_needs_ollama() {
  case "${1}" in
    2|4|5|6|7|8) return 0 ;;
    *) return 1 ;;
  esac
}

roihu_step_needs_gpu() {
  case "${1}" in
    1|2|3|4|5|6|7|8) return 0 ;;
    *) return 1 ;;
  esac
}

roihu_step_resources() {
  case "${1}" in
    1) echo "gpumedium|72|0|gpu:gh200:1|36:00:00" ;;
    2) echo "gpumedium|72|0|gpu:gh200:1|36:00:00" ;;
    3) echo "gpumedium|72|0|gpu:gh200:1|36:00:00" ;;
    4) echo "gpumedium|72|0|gpu:gh200:1|36:00:00" ;;
    5) echo "gpumedium|72|0|gpu:gh200:1|36:00:00" ;;
    6) echo "gpumedium|72|0|gpu:gh200:1|36:00:00" ;;
    7) echo "gpumedium|72|0|gpu:gh200:1|36:00:00" ;;
    8) echo "gpumedium|72|0|gpu:gh200:1|36:00:00" ;;
    9) echo "small|4|16G||04:00:00" ;;
    *) return 2 ;;
  esac
}

roihu_load_step_modules() {
  local step="${1}"
  module --force purge
  case "${step}" in
    1)
      module load python-pytorch
      module load gcc/13.4.0
      module load ffmpeg/7.1-cuda12.4
      ;;
    2|4|5|6|7|8)
      module load python-pytorch
      ;;
    3)
      module load python-vllm
      module load gcc/14.3.0 ffmpeg
      ;;
    9)
      # RDF runs on x86_64 CPU; do not load GPU or model modules.
      ;;
  esac
  # Required on every Roihu step for consistent Allas/S3 utilities. Load
  # after the module purge and runtime-specific dependencies.
  module load allas
  echo "Step ${step} modules loaded (Allas + runtime profile)" >&2
  unset PYTHONPATH PYTHONHOME
}
