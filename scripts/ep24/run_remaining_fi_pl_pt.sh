#!/usr/bin/env bash
# EP24 #373: verified, approval-gated rollout; --submit remains a compatibility alias.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${DIR}/run_ep24_rollout.sh" continue "$@"
