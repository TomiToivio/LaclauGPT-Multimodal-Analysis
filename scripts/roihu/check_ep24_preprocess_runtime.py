#!/usr/bin/env python3
"""Fail fast for the ASR/OCR engines selected for EP24 Step 1 (#128).

This checks importability only. It deliberately does not download or load model
weights on the login/startup path; the actual model load remains inside the
allocated GH200 job.
"""
from __future__ import annotations

import importlib
import os
import sys

ASR_IMPORTS = {
    "canary": ("nemo.collections.asr", "nemo_toolkit[asr]"),
    "parakeet": ("nemo.collections.asr", "nemo_toolkit[asr]"),
    "qwen3-asr": ("qwen_asr", "qwen-asr"),
    "whisper": ("whisper", "openai-whisper"),
    "faster-whisper": ("faster_whisper", "faster-whisper"),
}

OCR_IMPORTS = {
    "paddleocr": ("paddleocr", "paddleocr + a Roihu-compatible PaddlePaddle GPU build"),
    "easyocr": ("easyocr", "easyocr"),
}


def require(selected: str, mapping: dict[str, tuple[str, str]], kind: str) -> None:
    try:
        module, install_hint = mapping[selected]
    except KeyError:
        choices = ", ".join(sorted(mapping))
        raise SystemExit(f"Unknown {kind} engine {selected!r}; expected one of: {choices}") from None

    try:
        importlib.import_module(module)
    except ImportError as exc:
        print(
            f"Selected EP24 {kind} engine {selected!r} is not importable: {module!r}. "
            f"Install {install_hint} in the selected Roihu venv before submitting Step 1.",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc


def main() -> int:
    asr = os.getenv("LACLAUGPT_ASR_ENGINE", "canary").strip().lower()
    ocr = os.getenv("LACLAUGPT_OCR_ENGINE", "paddleocr").strip().lower()

    require(asr, ASR_IMPORTS, "ASR")
    require(ocr, OCR_IMPORTS, "OCR")

    for module in ("cv2", "pandas"):
        try:
            importlib.import_module(module)
        except ImportError as exc:
            print(f"Missing EP24 Step-1 dependency: {module}", file=sys.stderr)
            raise SystemExit(2) from exc

    print(f"EP24 Step-1 runtime imports OK: ASR={asr} OCR={ocr}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
