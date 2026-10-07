#!/usr/bin/env python3
from __future__ import annotations
import argparse
import importlib
import os
import shutil
import sys
from pathlib import Path

IMPORTS = {
    1: ("pandas", "cv2"),
    2: ("pandas", "cv2", "ollama"),
    3: ("pandas", "vllm", "transformers"),
    4: ("pandas", "ollama"),
    5: ("pandas", "ollama", "pydantic"),
    6: ("pandas", "ollama", "pydantic"),
    7: ("pandas", "ollama", "pydantic"),
    8: ("pandas", "ollama", "pydantic"),
    9: (),
}
GPU_STEPS = set(range(1, 9))
OLLAMA_STEPS = {2, 4, 5, 6, 7, 8}

def fail(msg: str) -> None:
    print(f"STEP PREFLIGHT ERROR: {msg}", file=sys.stderr)
    raise SystemExit(2)

def require_import(name: str) -> None:
    try:
        importlib.import_module(name)
    except Exception as exc:
        fail(f"cannot import {name}: {type(exc).__name__}: {exc}")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, choices=range(1, 10), required=True)
    ap.add_argument("--setup", action="store_true")
    args = ap.parse_args()
    step = args.step

    for name in IMPORTS[step]:
        require_import(name)

    if step in GPU_STEPS and not args.setup and shutil.which("nvidia-smi") is None:
        fail("nvidia-smi is unavailable on a GPU step")

    if step == 1:
        asr = os.getenv("LACLAUGPT_ASR_ENGINE", "canary").strip().lower()
        ocr = os.getenv("LACLAUGPT_OCR_ENGINE", "easyocr").strip().lower()
        asr_import = {
            "canary": "nemo.collections.asr",
            "parakeet": "nemo.collections.asr",
            "qwen3-asr": "qwen_asr",
            "whisper": "whisper",
            "faster-whisper": "faster_whisper",
        }.get(asr)
        ocr_import = {"easyocr": "easyocr", "paddleocr": "paddleocr"}.get(ocr)
        if not asr_import or not ocr_import:
            fail(f"unsupported Step 1 backend asr={asr} ocr={ocr}")
        require_import(asr_import)
        require_import(ocr_import)
        if shutil.which("ffmpeg") is None:
            fail("ffmpeg is required by Step 1")
        # Setup can optionally instantiate backends. This catches missing model/runtime
        # dependencies while allowing sites to defer large model downloads explicitly.
        if args.setup and os.getenv("LACLAUGPT_SETUP_VALIDATE_MODELS", "1").lower() in {"1","true","yes","on"}:
            try:
                from asr_backend import load_asr_model
                from ocr_backend import load_ocr_backend
                load_ocr_backend()
                load_asr_model()
            except Exception as exc:
                fail(f"Step 1 backend initialization failed: {type(exc).__name__}: {exc}")

    if step == 3:
        for exe in ("ffmpeg", "ffprobe", "rclone"):
            if shutil.which(exe) is None:
                fail(f"{exe} is required by Step 3")

    if step in OLLAMA_STEPS:
        install_root = Path(os.getenv(
            "OLLAMA_INSTALL_ROOT",
            Path(os.getenv("LACLAUGPT_MULTIMODAL_PRIVATE_ROOT", ".")) / ".ollama",
        ))
        if shutil.which("ollama") is None and not (install_root / "bin/ollama").is_file():
            fail(f"Ollama missing for Step {step}: {install_root}")

    print(f"step_environment_ok step={step} python={sys.executable}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
