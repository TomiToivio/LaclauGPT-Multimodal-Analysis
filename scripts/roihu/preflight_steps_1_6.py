#!/usr/bin/env python3
"""Cheap fail-fast checks before EP24 steps 1-6 consume a GH200 allocation."""
from __future__ import annotations

import argparse
import importlib
import os
import shutil
import sys
from pathlib import Path

CORE_IMPORTS = ("pandas", "pymongo", "redis")
STEP_IMPORTS = {
    1: ("cv2",),
    2: ("cv2", "ollama"),
    3: (),
    4: ("ollama",),
    5: ("ollama", "pydantic"),
    6: ("ollama", "pydantic"),
}
COUNTRIES = {"finland", "poland", "portugal", "germany", "spain", "hungary", "croatia", "france", "bulgaria", "sweden"}

def fail(message: str) -> None:
    print(f"PREFLIGHT ERROR: {message}", file=sys.stderr)
    raise SystemExit(2)

def require_imports(names: tuple[str, ...]) -> None:
    missing = []
    for name in names:
        try:
            importlib.import_module(name)
        except Exception:
            missing.append(name)
    if missing:
        fail("missing Python imports: " + ", ".join(sorted(set(missing))))

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, choices=range(1, 7), required=True)
    parser.add_argument("--country")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    private_root = Path(os.environ.get("LACLAUGPT_EP24_PRIVATE_ROOT") or os.environ.get("LACLAUGPT_MULTIMODAL_PRIVATE_ROOT", ""))
    if not private_root.is_dir():
        fail(f"private EP24 root not found: {private_root}")
    output_root = Path(os.environ.get("LACLAUGPT_EP24_OUTPUT_ROOT", private_root / "outputs"))
    output_root.mkdir(parents=True, exist_ok=True)
    if not os.access(output_root, os.W_OK):
        fail(f"output root is not writable: {output_root}")

    country = (args.country or os.environ.get("LACLAUGPT_COUNTRY") or "").strip().casefold()
    if country and country not in COUNTRIES:
        fail(f"unsupported country: {country}")
    limit = args.limit
    if limit is None and os.environ.get("LACLAUGPT_MAX_ROWS"):
        try:
            limit = int(os.environ["LACLAUGPT_MAX_ROWS"])
        except ValueError:
            fail("LACLAUGPT_MAX_ROWS must be an integer")
    if limit is not None and limit < 0:
        fail("sample limit must be >= 0")

    if os.environ.get("LACLAUGPT_MONGO_ENABLED", "0").lower() in {"1", "true", "yes", "on"}:
        if not os.environ.get("LACLAUGPT_MONGO_URI"):
            fail("LACLAUGPT_MONGO_URI is required by the restartable sbatch orchestrator")
    else:
        fail("restartable sbatch jobs require LACLAUGPT_MONGO_ENABLED=1")

    if shutil.which("nvidia-smi") is None:
        fail("nvidia-smi not found; submit on Roihu-GPU")

    require_imports(CORE_IMPORTS + STEP_IMPORTS[args.step])

    if args.step == 1:
        asr = os.environ.get("LACLAUGPT_ASR_ENGINE", "canary").strip().lower()
        ocr = os.environ.get("LACLAUGPT_OCR_ENGINE", "easyocr").strip().lower()
        asr_import = {"canary": "nemo.collections.asr", "parakeet": "nemo.collections.asr",
                      "qwen3-asr": "qwen_asr", "whisper": "whisper",
                      "faster-whisper": "faster_whisper"}.get(asr)
        ocr_import = {"easyocr": "easyocr", "paddleocr": "paddleocr"}.get(ocr)
        if not asr_import:
            fail(f"unknown ASR engine: {asr}")
        if not ocr_import:
            fail(f"unknown OCR engine: {ocr}")
        require_imports((asr_import, ocr_import))
        if country:
            input_root = Path(os.environ.get("LACLAUGPT_EP24_INPUT_ROOT", private_root / "data/to_reprocess"))
            source = input_root / f"ep24_{country}.csv"
            if not source.is_file():
                fail(f"Step 1 country input missing: {source}")

    print(f"preflight_ok step={args.step} country={country or '<orchestrator>'} limit={limit if limit is not None else '<default>'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
