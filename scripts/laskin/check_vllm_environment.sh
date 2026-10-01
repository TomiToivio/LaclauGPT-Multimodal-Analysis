#!/usr/bin/env bash
set -euo pipefail

echo "utc=$(date -u --iso-8601=seconds)"
echo "local=$(date --iso-8601=seconds)"
echo "hostname=$(hostname)"
uname -a
command -v lscpu >/dev/null && lscpu
command -v free >/dev/null && free -h
command -v nvidia-smi >/dev/null && nvidia-smi || echo "nvidia-smi unavailable"
command -v ffmpeg >/dev/null && ffmpeg -version | head -n 1 || echo "ffmpeg unavailable"
command -v ffprobe >/dev/null && ffprobe -version | head -n 1 || echo "ffprobe unavailable"
command -v rclone >/dev/null && rclone version | head -n 2 || echo "rclone unavailable"
python --version
python - <<'PY'
from importlib.metadata import PackageNotFoundError, version
for name in ("vllm", "torch", "transformers", "qwen-vl-utils", "av"):
    try:
        print(name, version(name))
    except PackageNotFoundError:
        print(name, "unavailable")
try:
    import torch
    print("cuda_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            print("gpu", index, torch.cuda.get_device_name(index), torch.cuda.get_device_capability(index))
except Exception as exc:
    print("torch_diagnostics_error", type(exc).__name__, str(exc))
PY