#!/usr/bin/env bash
#SBATCH --job-name=ep24-memory-prepare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00

set -euo pipefail

PUBLIC_ROOT=${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
: "${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT:?Set LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}"
PRIVATE_ROOT=${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT}
VENV=${LACLAUGPT_MULTIMODAL_VENV:-${PUBLIC_ROOT}/.venv-roihu}
CANONICAL_DB=${LACLAUGPT_MEMORY_CANONICAL_DB:-${PRIVATE_ROOT}/database/accepted-memory.sqlite3}
SNAPSHOT_DIR=${LACLAUGPT_MEMORY_SNAPSHOT_DIR:-${PRIVATE_ROOT}/database/snapshots}
JOB_LOCAL=${TMPDIR:-${LOCAL_SCRATCH:-/tmp}}/laclaugpt-memory-${SLURM_JOB_ID}
mkdir -p "${JOB_LOCAL}" "${SNAPSHOT_DIR}" "${PRIVATE_ROOT}/logs"

[[ -x "${VENV}/bin/python" ]] || {
  echo "Missing Roihu-native venv: ${VENV}" >&2
  exit 2
}
source "${VENV}/bin/activate"

python "${PUBLIC_ROOT}/roihu_enrich.py" \
  --private-root "${PRIVATE_ROOT}" \
  --memory-db "${CANONICAL_DB}" \
  --prepare-memory

LOCAL_SNAPSHOT="${JOB_LOCAL}/accepted-memory.sqlite3"
python "${PUBLIC_ROOT}/roihu_memory.py" --db "${CANONICAL_DB}" snapshot "${LOCAL_SNAPSHOT}"
sha256sum "${LOCAL_SNAPSHOT}" > "${JOB_LOCAL}/accepted-memory.sqlite3.sha256"

PROMOTE_TMP="${SNAPSHOT_DIR}/.accepted-memory-${SLURM_JOB_ID}.sqlite3.tmp"
PROMOTE="${SNAPSHOT_DIR}/accepted-memory-${SLURM_JOB_ID}.sqlite3"
cp "${LOCAL_SNAPSHOT}" "${PROMOTE_TMP}"
mv "${PROMOTE_TMP}" "${PROMOTE}"
cp "${JOB_LOCAL}/accepted-memory.sqlite3.sha256" "${PROMOTE}.sha256"

printf 'snapshot=%s\nsha256=%s\n' "${PROMOTE}" "$(cut -d' ' -f1 "${PROMOTE}.sha256")" \
  > "${PRIVATE_ROOT}/logs/memory-prepare-${SLURM_JOB_ID}.manifest"
