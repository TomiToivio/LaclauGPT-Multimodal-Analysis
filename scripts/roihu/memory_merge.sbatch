#!/usr/bin/env bash
#SBATCH --job-name=ep24-memory-merge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00

set -euo pipefail

PUBLIC_ROOT=${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
: "${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT:?Set LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}"
: "${LACLAUGPT_PROPOSAL_SHARDS:?Space-separated private proposal shard paths required}"
PRIVATE_ROOT=${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}
VENV=${LACLAUGPT_MULTIMODAL_VENV:-${PUBLIC_ROOT}/.venv-roihu}
CANONICAL_DB=${LACLAUGPT_MEMORY_CANONICAL_DB:-${PRIVATE_ROOT}/database/accepted-memory.sqlite3}
EXPORT_DIR=${PRIVATE_ROOT}/provenance/memory-merge-${SLURM_JOB_ID}
mkdir -p "${EXPORT_DIR}" "${PRIVATE_ROOT}/database/backups"

[[ -x "${VENV}/bin/python" ]] || {
  echo "Missing Roihu-native venv: ${VENV}" >&2
  exit 2
}
source "${VENV}/bin/activate"

BACKUP="${PRIVATE_ROOT}/database/backups/pre-merge-${SLURM_JOB_ID}.sqlite3"
python "${PUBLIC_ROOT}/roihu_memory.py" --db "${CANONICAL_DB}" snapshot "${BACKUP}"

read -r -a SHARDS <<< "${LACLAUGPT_PROPOSAL_SHARDS}"
for shard in "${SHARDS[@]}"; do
  [[ -f "${shard}" ]] || { echo "Missing proposal shard: ${shard}" >&2; exit 3; }
done

python "${PUBLIC_ROOT}/roihu_memory.py" --db "${CANONICAL_DB}" \
  merge-proposals "${SHARDS[@]}" | tee "${EXPORT_DIR}/merge-result.json"
python "${PUBLIC_ROOT}/roihu_memory.py" --db "${CANONICAL_DB}" \
  export-csv "${EXPORT_DIR}/csv"

sha256sum "${CANONICAL_DB}" > "${EXPORT_DIR}/canonical-db.sha256"
printf 'job=%s\ncanonical_db=%s\nbackup=%s\nshards=%s\n' \
  "${SLURM_JOB_ID}" "${CANONICAL_DB}" "${BACKUP}" "${LACLAUGPT_PROPOSAL_SHARDS}" \
  > "${EXPORT_DIR}/manifest.txt"
