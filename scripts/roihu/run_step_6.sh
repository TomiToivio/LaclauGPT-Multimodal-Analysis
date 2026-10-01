#!/usr/bin/env bash
# EP24 Roihu step 6 (discourse_analysis) launcher -- issue #64.
#
# Submits scripts/roihu/step_6_roihu_discourse_analysis.sbatch and returns immediately.
# Run this after `scripts/ep24/bootstrap_ep24_mongodb.py` has created the
# canonical `entities`/`themes` fields; every analysis step receives them.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/launcher_lib.sh"
launch_step 6 discourse_analysis
