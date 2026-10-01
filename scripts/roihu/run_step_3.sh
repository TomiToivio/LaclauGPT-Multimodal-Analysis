#!/usr/bin/env bash
# EP24 Roihu step 3 (video) launcher -- issue #64.
#
# Submits scripts/roihu/step_3_roihu_video.sbatch and returns immediately.
# Run this after `scripts/ep24/bootstrap_ep24_mongodb.py` has created the
# canonical `entities`/`themes` fields; every analysis step receives them.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/launcher_lib.sh"
launch_step 3 video
