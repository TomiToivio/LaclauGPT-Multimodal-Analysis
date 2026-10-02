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

#: EasyOCR groups languages by script and refuses a Reader that spans groups.
#: The single flat list this replaced was
#:
#:     ["en", "fr", "pl", "sv", "pt", "de", "es", "hu", "hr", "bg"]
#:
#: which mixes Latin and Cyrillic, so `easyocr.Reader(...)` raised before doing any
#: work at all:
#:
#:     ValueError: Cyrillic is only compatible with English, try lang_list=
#:     ["ru","rs_cyrillic","be","bg","uk","mn","en"]
#:
#: `raise ValueError(language.capitalize() + ' is only compatible with English...')`
#: fires on the *first* group that does not contain every requested language — and
#: no single group contains all ten EP24 languages. So the EasyOCR adapter was
#: unusable for the whole EP24 set, and no test caught it because none constructed
#: a real Reader (the module is imported lazily and tests inject a fake backend).
#:
#: Bulgarian is the only Cyrillic-script EP24 language; the other nine are Latin
#: (Croatian and Hungarian included). The fix is one Reader per script group,
#: selected from the canonical country, which is also cheaper and more accurate
#: than asking one reader to cover two scripts.
#:
#: EasyOCR language codes with a recognition model, established by constructing a
#: real Reader per code (the model tables are not reliably introspectable across
#: versions). Verified on easyocr 1.7.2:
#:
#:     bg pl pt de es hu hr fr sv en   -> Reader constructs
#:     fi                              -> ValueError: ({'fi'}, 'is not supported')
#:
#: Nine of the ten EP24 languages are covered. **Finnish has no EasyOCR model**,
#: so Finnish on-screen text is not recognised natively. Finland therefore maps to
#: the Latin group without `fi`: the Reader still builds (previously it crashed for
#: *every* country), but a Finnish-only overlay will come back empty rather than
#: wrong-text. That limitation is pinned by
#: `tests/test_ep24_ocr_script_groups.py::test_easyocr_has_no_finnish_model` so it
#: is a recorded gap rather than an assumption.
EASYOCR_UNSUPPORTED_EP24_LANGS = ["fi"]

EASYOCR_LATIN_LANGS = ["en", "fr", "pl", "sv", "pt", "de", "es", "hu", "hr"]
EASYOCR_CYRILLIC_LANGS = ["bg", "en"]

#: Script group per EP24 country token, matching `asr_backend.COUNTRY_LANGUAGE_HINTS`.
EASYOCR_COUNTRY_GROUPS: dict[str, list[str]] = {
    "bulgaria": EASYOCR_CYRILLIC_LANGS,
    "finland": EASYOCR_LATIN_LANGS,
    "poland": EASYOCR_LATIN_LANGS,
    "portugal": EASYOCR_LATIN_LANGS,
    "germany": EASYOCR_LATIN_LANGS,
    "spain": EASYOCR_LATIN_LANGS,
    "hungary": EASYOCR_LATIN_LANGS,
    "croatia": EASYOCR_LATIN_LANGS,
    "france": EASYOCR_LATIN_LANGS,
    "sweden": EASYOCR_LATIN_LANGS,
}


def easyocr_languages(country: str | None = None) -> list[str]:
    """EasyOCR language list for one EP24 country, within a single script group.

    Defaults to the Latin group when the country is unknown, since nine of the ten
    EP24 languages are Latin. Raises for a country that is not in the EP24 set, so
    a typo cannot silently pick a script.

    Note Finland: EasyOCR has no Finnish (`fi`) model, so Finnish is absent from
    every group by necessity, not oversight. Finland's Reader still constructs and
    recognises the Latin languages it does have. See
    `EASYOCR_UNSUPPORTED_EP24_LANGS`.
    """
    token = str(country or "").strip().casefold()
    if not token:
        return list(EASYOCR_LATIN_LANGS)
    if token not in EASYOCR_COUNTRY_GROUPS:
        raise ValueError(
            f"unknown EP24 country for OCR script selection: {country!r}; "
            f"expected one of {sorted(EASYOCR_COUNTRY_GROUPS)}"
        )
    return list(EASYOCR_COUNTRY_GROUPS[token])


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

        country = os.getenv("LACLAUGPT_COUNTRY", "").strip()
        languages = easyocr_languages(country)
        LOG.info(
            "Loading OCR backend=easyocr country=%s languages=%s", country or "(default)", languages
        )
        reader = easyocr.Reader(languages)

        def _read(path: str) -> tuple[str, int]:
            results = reader.readtext(path)
            texts = [str(result[1]).strip() for result in results if str(result[1]).strip()]
            return "\n".join(texts), len(texts)

        return OCRBackend("easyocr", f"easyocr-runtime:{'+'.join(languages)}", _read)

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
