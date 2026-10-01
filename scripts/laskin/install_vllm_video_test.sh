#!/usr/bin/env bash
# Historical Laskin vLLM experiment installer (issue #32).
#
# SECURITY STATUS: retired.
#
# The only stack proven to run native-video vLLM on Laskin's Tesla V100
# (Volta/sm_70) depended on versions that are now covered by security
# advisories. Modern supported vLLM releases no longer support compute
# capability 7.0, so there is no safe drop-in replacement for this installer.
#
# Use CSC Roihu for the active native-video pipeline. The measured Laskin
# results are retained in docs/LASKIN_VLLM_VIDEO_TEST.md for reproducibility,
# but this repository intentionally no longer recreates the vulnerable venv.

set -euo pipefail

cat >&2 <<'EOF'
SECURITY: the Laskin legacy vLLM environment has been retired.

The previously proven Volta stack used legacy Transformers/PyTorch versions
that now have published security advisories. Recreating that environment would
restore known-vulnerable dependencies.

Use the CSC Roihu vLLM workflow instead:
  docs/VLLM_VIDEO_TEST.md
  scripts/roihu/vllm_video_test.sbatch

Historical measurements remain in:
  docs/LASKIN_VLLM_VIDEO_TEST.md
EOF

exit 78
