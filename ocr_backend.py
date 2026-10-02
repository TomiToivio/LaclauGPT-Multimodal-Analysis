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

# EasyOCR's recognition models are grouped by script, and `Reader` refuses a
# language list that spans incompatible groups: passing all ten EP24 languages at
# once raises
#   ValueError: Cyrillic is only compatible with English, try lang_list=[...]
# because "bg" needs the cyrillic model while the latin languages need latin_g2.
# That made `LACLAUGPT_OCR_ENGINE=easyocr` fail on load, in the benchmark and in
# production Step 1 alike. Latin and Cyrillic cannot share one recognition model,
# so group the languages and use the group's model instead of the raw list.
EASYOCR_SCRIPT_GROUPS = {
    "latin": ["en", "fr", "pl", "sv", "pt", "de", "es", "hu", "hr"],
    "cyrillic": ["bg"],
}

# EasyOCR language codes that actually have a recognition model, measured by
# constructing a real Reader per code rather than read off the docs (the model
# tables are not reliably introspectable across easyocr versions). On 1.7.2:
#
#     bg pl pt de es hu hr fr sv en  -> Reader constructs
#     fi                             -> ValueError: ({'fi'}, 'is not supported')
#
# Nine of the ten EP24 languages are covered; Finnish is the exception. It is
# therefore absent from EASYOCR_SCRIPT_GROUPS by necessity, not oversight, and
# asking for it raises rather than silently dropping it. Consequence worth
# knowing before choosing engines for Finland: EasyOCR will not read Finnish
# on-screen text, so a Finnish-only overlay comes back empty. This is recorded
# here so the natural future edit -- "EP24 has ten languages, add the missing
# one" -- fails loudly instead of rediscovering the unloadable-list error.
EASYOCR_UNSUPPORTED_LANGS = ["fi"]


def easyocr_script_groups(languages: list[str] | None = None) -> list[str]:
    """Group a language set by script, in a deterministic order.

    Latin first: nine of the ten EP24 languages are latin, so that reader is the
    common case. Raises on an unknown language rather than silently dropping it --
    a language that is not supported must not look like one that ran and found no
    text.
    """
    wanted = [str(lang).strip().lower() for lang in (languages or EASYOCR_LANGS)]
    unknown = sorted(set(wanted) - {
        lang for members in EASYOCR_SCRIPT_GROUPS.values() for lang in members
    })
    if unknown:
        raise ValueError(f"unsupported EasyOCR language(s): {unknown}")
    present = [
        name for name, members in EASYOCR_SCRIPT_GROUPS.items()
        if set(wanted) & set(members)
    ]
    return sorted(present, key=lambda name: (name != "latin", name))


def easyocr_language_group(languages: list[str] | None = None) -> str:
    """Return the single script group a language set belongs to.

    Raises when the set spans groups: EasyOCR loads one recognition model per
    reader and cannot mix scripts, so such a set is unloadable and must be split
    with `easyocr_script_groups` instead.
    """
    groups = easyocr_script_groups(languages)
    if len(groups) != 1:
        raise ValueError(
            "EasyOCR cannot combine scripts in one reader; requested "
            f"{sorted(set(str(x).strip().lower() for x in (languages or EASYOCR_LANGS)))} "
            f"spans {groups}. Run one reader per group."
        )
    return groups[0]


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

        # EasyOCR cannot load latin and cyrillic languages into one recognition
        # model, so a reader per script group is the only way to cover the EP24
        # language set at all (Bulgarian is the lone cyrillic member).
        configured = os.getenv("LACLAUGPT_OCR_LANGS")
        languages = (
            [x.strip() for x in configured.split(",") if x.strip()]
            if configured
            else list(EASYOCR_LANGS)
        )
        groups = easyocr_script_groups(languages)
        readers = {
            group: easyocr.Reader(
                [l for l in languages if l in EASYOCR_SCRIPT_GROUPS[group]]
            )
            for group in groups
        }
        LOG.info(
            "Loading OCR backend=easyocr groups=%s languages=%s",
            sorted(readers),
            sorted(languages),
        )

        def _read(path: str) -> tuple[str, int]:
            # Latin script is the default: every EP24 language except Bulgarian
            # uses it, and the EP24 set is 9 latin to 1 cyrillic.
            order = ["latin", "cyrillic"]
            texts: list[str] = []
            for group in (g for g in order if g in readers):
                for result in readers[group].readtext(path):
                    text = str(result[1]).strip()
                    if text:
                        texts.append(text)
            return "\n".join(texts), len(texts)

        return OCRBackend("easyocr", f"easyocr-runtime+{'+'.join(sorted(readers))}", _read)

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
