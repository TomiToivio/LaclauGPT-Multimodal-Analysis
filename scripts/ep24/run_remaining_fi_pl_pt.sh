#!/usr/bin/env bash
# Continue Finland, Poland and Portugal after the 500-per-country sample
#
# This is a thin front door to `rollout.py`, which plans from the records themselves (see
# `ep24_stage_orchestrator.eligible_query`) and submits through `scripts/roihu/run_country_pipeline.sh`.
# Nothing here re-implements a stage, a query or a state machine.
#
# Dry-run is the default and is always allowed. Submitting requires the sample to be approved:
#   EP24_SAMPLE_APPROVED=1 bash run_remaining_fi_pl_pt.sh --submit
# or:  bash run_remaining_fi_pl_pt.sh --submit --approval-file /path/to/approval
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUBLIC_ROOT="${LACLAUGPT_MULTIMODAL_PUBLIC_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

exec python3 "${PUBLIC_ROOT}/scripts/ep24/rollout.py" --mode resume-sample "$@"
