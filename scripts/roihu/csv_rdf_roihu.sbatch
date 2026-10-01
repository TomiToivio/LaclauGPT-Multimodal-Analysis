#!/usr/bin/env bash
#SBATCH --job-name=ep24-rdf
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=ep24-rdf-%j.log
set -euo pipefail

# Select a valid CPU partition and allocation at submission time. No GPU/Ollama
# is needed. Submit from private storage so the Slurm log stays private too.
: "${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT:?Set public checkout path}"
: "${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT:?Set private runtime path}"
cd "${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}"
python3 "${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT}/roihu_csv_rdf.py" "$@"
