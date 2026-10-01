#!/usr/bin/env bash
# Historical Laskin vLLM video experiment runner (issue #32).
#
# SECURITY STATUS: retired. See docs/LASKIN_VLLM_VIDEO_TEST.md.
# The old Volta-compatible environment contains dependencies with published
# security advisories and must not be used for new analysis.

set -euo pipefail

cat >&2 <<'EOF'
SECURITY: the Laskin legacy vLLM video experiment is retired.

Do not run an old ~/.venvs/laskin-vllm-video environment. The only stack proven
on Laskin's V100/Volta GPUs relies on legacy dependencies with known security
vulnerabilities.

Run the active native-video experiment on CSC Roihu instead:
  docs/VLLM_VIDEO_TEST.md
  scripts/roihu/vllm_video_test.sbatch

The Laskin document is retained only as a historical measurement record:
  docs/LASKIN_VLLM_VIDEO_TEST.md
EOF

exit 78
