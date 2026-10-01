#!/usr/bin/env bash
# Laskin environment inventory for the vLLM video experiment (issue #32).
#
# Read-only. Prints everything the experiment needs to explain its own result,
# and never prints secrets: no rclone config contents, no credential files, no
# complete environment dumps.
#
#   bash scripts/laskin/check_vllm_environment.sh
#
# The decisive lines are the compute capability and the torch arch list: vLLM's
# prebuilt wheels require compute capability >= 7.5, so a Volta (7.0) host must
# use the pinned legacy stack documented in docs/LASKIN_VLLM_VIDEO_TEST.md.

set -uo pipefail

section() { printf '\n=== %s ===\n' "$1"; }
have()    { command -v "$1" >/dev/null 2>&1; }

section "host"
echo "hostname      : $(hostname)"
echo "date_utc      : $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "date_local    : $(date '+%Y-%m-%dT%H:%M:%S%z')"
echo "kernel        : $(uname -srmo)"
if [ -r /etc/os-release ]; then
  # shellcheck disable=SC1091
  echo "distribution  : $(. /etc/os-release && printf '%s %s' "$NAME" "$VERSION_ID")"
fi
echo "git_head      : $(git rev-parse --short HEAD 2>/dev/null || echo '<not a git checkout>')"

section "cpu / memory"
if have nproc; then echo "cores         : $(nproc)"; fi
if [ -r /proc/cpuinfo ]; then
  echo "model         : $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //')"
fi
if have free; then free -h | sed 's/^/  /'; fi

section "gpu (the deciding factor)"
if have nvidia-smi; then
  nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap \
             --format=csv,noheader 2>/dev/null | sed 's/^/  /' \
    || nvidia-smi --query-gpu=index,name,memory.total,driver_version \
                  --format=csv,noheader 2>/dev/null | sed 's/^/  /'
  echo "driver_cuda   : $(nvidia-smi | grep -oE 'CUDA Version: [0-9.]+' | head -1)"
else
  echo "  nvidia-smi not found — no NVIDIA GPU visible"
fi
echo
echo "NOTE: vLLM prebuilt wheels need compute capability >= 7.5."
echo "      Volta (7.0) requires the pinned legacy stack (vLLM 0.8.x)."

section "cuda toolkit"
echo "nvcc          : $(nvcc --version 2>/dev/null | grep -oE 'release [0-9.]+' || echo '<not found>')"

section "python"
for py in python3 python3.10 python3.11 python3.12; do
  if have "$py"; then
    printf '%-14s: %s\n' "$py" "$("$py" --version 2>&1)"
  fi
done

section "python packages (system interpreters)"
for py in python3 python3.11 python3.10; do
  have "$py" || continue
  printf -- '-- %s --\n' "$py"
  "$py" - <<'PY' 2>/dev/null | sed 's/^/  /'
mods = ["torch", "vllm", "transformers", "qwen_vl_utils", "av", "pandas"]
for m in mods:
    try:
        mod = __import__(m)
        print(f"{m:16}: {getattr(mod, '__version__', 'installed')}")
    except Exception as exc:
        print(f"{m:16}: NOT AVAILABLE ({type(exc).__name__})")
try:
    import torch
    print(f"{'cuda_available':16}: {torch.cuda.is_available()}")
    print(f"{'device_count':16}: {torch.cuda.device_count()}")
    print(f"{'arch_list':16}: {torch.cuda.get_arch_list()}")
    if torch.cuda.is_available():
        print(f"{'cc_device0':16}: {torch.cuda.get_device_capability(0)}")
        print(f"{'sm_70_in_build':16}: {any('70' in a for a in torch.cuda.get_arch_list())}")
except Exception as exc:
    print(f"torch probe failed: {type(exc).__name__}: {exc}")
PY
done

section "media tooling"
for tool in ffmpeg ffprobe; do
  if have "$tool"; then echo "$tool          : $($tool -version 2>/dev/null | head -1)"; else echo "$tool          : NOT FOUND"; fi
done

section "allas / object storage"
if have rclone; then
  echo "rclone        : $(rclone version 2>/dev/null | head -1)"
  echo "remotes       : $(rclone listremotes 2>/dev/null | tr '\n' ' ')"
  echo "  (remote names only; config contents are never printed)"
else
  echo "rclone        : NOT FOUND"
fi
[ -f "$HOME/allas_conf" ] && echo "allas_conf    : present" || echo "allas_conf    : not found"

section "containers"
for c in docker podman apptainer singularity; do
  have "$c" && echo "$c : $(command -v "$c")"
done

section "existing gpu workloads"
if have ollama; then
  echo "ollama        : running (models resident, GPU is shared)"
  ollama list 2>/dev/null | head -5 | sed 's/^/  /'
else
  echo "ollama        : not found"
fi

section "disk"
df -h "${HOME:-/}" 2>/dev/null | sed 's/^/  /'

printf '\n=== done ===\n'
