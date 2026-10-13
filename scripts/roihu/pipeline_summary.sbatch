#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
set -euo pipefail

RUN_DIR="${LACLAUGPT_PIPELINE_RUN_DIR:?}"
JOBS_TSV="${LACLAUGPT_PIPELINE_JOBS_TSV:?}"
PUBLIC_ROOT="${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT:-/scratch/${CSC_PROJECT:-project_2009497}/LaclauGPT-Multimodal-Analysis}"

python3 "${PUBLIC_ROOT}/scripts/roihu/summarize_pipeline_run.py" "${RUN_DIR}" "${JOBS_TSV}"
