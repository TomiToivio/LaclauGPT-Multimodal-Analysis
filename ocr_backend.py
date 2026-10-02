"""OCR backends for the EP24 one-frame preprocess stage.

The active contract is deliberately tiny: one image in, one text result out.
Backend/model identity is returned separately for provenance.  PaddleOCR is the
modern candidate; EasyOCR remains available for benchmarking/fallback.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Callable

LOG = logging.getLogger(__name__)

DEFAULT_ENGINE = os.getenv("LACLAUGPT_OCR_ENGINE", "paddleocr").strip().lower()
DEFAULT_PADDLE_VERSION = os.getenv("LACLAUGPT_OCR_MODEL", "PP-OCRv5")
EASYOCR_LANGS = ["en", "fr", "pl", "sv", "pt", "de", "es", "hu", "hr", "bg"]


@dataclass(frozen=True)
class OCRBackend:
    engine: str
    model: str
    read: Callable[[str], tuple[str, int]]


def _extract_paddle_text(result) -> list[str]:
    """Best-effort extraction across PaddleOCR 3.x result wrappers."""
    texts: list[str] = []
    for item in result or []:
        payload = None
        if isinstance(item, dict):
            payload = item.get("res", item)
        else:
            for attr in ("json", "res"):
                value = getattr(item, attr, None)
                if callable(value):
                    try:
                        value = value()
                    except TypeError:
                        value = None
                if isinstance(value, dict):
                    payload = value.get("res", value)
                    break
        if not isinstance(payload, dict):
            continue
        values = payload.get("rec_texts") or payload.get("texts") or []
        if isinstance(values, str):
            values = [values]
        texts.extend(str(v).strip() for v in values if str(v).strip())
    return texts


def load_ocr_backend() -> OCRBackend:
    engine = os.getenv("LACLAUGPT_OCR_ENGINE", DEFAULT_ENGINE).strip().lower()

    if engine == "paddleocr":
        from paddleocr import PaddleOCR

        version = os.getenv("LACLAUGPT_OCR_MODEL", DEFAULT_PADDLE_VERSION)
        device = os.getenv("LACLAUGPT_OCR_DEVICE", "gpu")
        LOG.info("Loading OCR backend=paddleocr model=%s device=%s", version, device)
        pipeline = PaddleOCR(
            ocr_version=version,
            device=device,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

        def _read(path: str) -> tuple[str, int]:
            texts = _extract_paddle_text(pipeline.predict(path))
            return "\n".join(texts), len(texts)

        return OCRBackend("paddleocr", version, _read)

    if engine == "easyocr":
        import easyocr

        LOG.info("Loading OCR backend=easyocr languages=%s", EASYOCR_LANGS)
        reader = easyocr.Reader(EASYOCR_LANGS)

        def _read(path: str) -> tuple[str, int]:
            results = reader.readtext(path)
            texts = [str(result[1]).strip() for result in results if str(result[1]).strip()]
            return "\n".join(texts), len(texts)

        return OCRBackend("easyocr", "easyocr-runtime", _read)

    raise ValueError(
        f"Unknown LACLAUGPT_OCR_ENGINE={engine!r}; expected 'paddleocr' or 'easyocr'"
    )


def describe_ocr_backend() -> dict[str, str]:
    engine = os.getenv("LACLAUGPT_OCR_ENGINE", DEFAULT_ENGINE).strip().lower()
    if engine == "paddleocr":
        return {
            "engine": engine,
            "model": os.getenv("LACLAUGPT_OCR_MODEL", DEFAULT_PADDLE_VERSION),
        }
    return {"engine": engine, "model": "easyocr-runtime"}
