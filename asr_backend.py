"""Configurable speech-to-text backend for the EP24 multimodal preprocess stage.

Purpose
-------
``puhti_preprocess.py`` historically hard-coded::

    model = whisper.load_model('large', download_root='./whisper/')
    ...
    result = model.transcribe(video_filename, temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

This module lets the ASR engine and checkpoint be selected by environment
variable, in the same spirit as the existing ``LACLAUGPT_MULTIMODAL_MODEL``
convention used by the LLM-backed stages.

The historical behaviour is the default
---------------------------------------
Calling :func:`transcribe_audio` with no environment overrides reproduces the
legacy call exactly: the ``openai-whisper`` engine, the ``large`` checkpoint, the
same temperature fallback ladder, and the same three returned strings
(``transcript``, ``language``, ``translated`` are assembled by the caller — this
module returns transcript and language only, as the legacy code did).

Nothing here changes prompts, schemas, stage order or CSV fields. Selecting a
different engine is an *opt-in* infrastructure change, not a methodology change.

Environment variables
---------------------
``LACLAUGPT_ASR_ENGINE``
    ``whisper`` (default, historical) or ``faster-whisper``.
``LACLAUGPT_ASR_MODEL``
    Checkpoint name. Default ``large`` for the historical engine; use
    ``large-v3`` / ``large-v3-turbo`` etc. with ``faster-whisper``.
``LACLAUGPT_ASR_DEVICE``
    ``cuda`` (default) or ``cpu``. Only used by ``faster-whisper``.
``LACLAUGPT_ASR_COMPUTE_TYPE``
    CTranslate2 compute type for ``faster-whisper``; default ``int8_float16``.
``LACLAUGPT_ASR_DOWNLOAD_ROOT``
    Model cache directory for the historical engine; default ``./whisper/``.

Why this is additive rather than a replacement
----------------------------------------------
Different ASR engines produce different transcripts, and transcripts propagate
into the summary, postprocess and populism stages. The engine is therefore
configurable with the legacy path preserved as the default, so an existing
result set remains reproducible and a switch is an explicit, reviewable choice.
See ``docs/ASR_EVALUATION.md`` for the measured comparison.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# The historical temperature fallback ladder, preserved verbatim.
# Stored as a tuple because faster-whisper types ``temperature`` as
# ``float | tuple[float, ...]`` while openai-whisper accepts either; a tuple is
# valid for both and keeps the legacy values unchanged.
LEGACY_TEMPERATURE = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

DEFAULT_ENGINE = "whisper"
DEFAULT_MODEL = "large"
DEFAULT_DOWNLOAD_ROOT = "./whisper/"
DEFAULT_COMPUTE_TYPE = "int8_float16"

_ENGINES = ("whisper", "faster-whisper")


def _engine() -> str:
    name = os.getenv("LACLAUGPT_ASR_ENGINE", DEFAULT_ENGINE).strip().lower()
    if name not in _ENGINES:
        raise ValueError(
            f"Unknown LACLAUGPT_ASR_ENGINE={name!r}; expected one of {_ENGINES}"
        )
    return name


def load_asr_model():
    """Load the configured ASR model and return a uniform wrapper.

    The returned object exposes ``transcribe(path) -> (transcript, language)``
    so the caller does not need to branch on engine.
    """
    engine = _engine()

    if engine == "whisper":
        import whisper

        model_name = os.getenv("LACLAUGPT_ASR_MODEL", DEFAULT_MODEL)
        download_root = os.getenv("LACLAUGPT_ASR_DOWNLOAD_ROOT", DEFAULT_DOWNLOAD_ROOT)
        logger.info("Loading ASR: engine=whisper model=%s root=%s", model_name, download_root)
        model = whisper.load_model(model_name, download_root=download_root)

        def _transcribe(path: str):
            result = model.transcribe(path, temperature=LEGACY_TEMPERATURE)
            text = result.get("text", "")
            if isinstance(text, list):
                text = " ".join(text)
            return str(text), str(result.get("language", ""))

        return _transcribe

    # faster-whisper
    from faster_whisper import WhisperModel

    model_name = os.getenv("LACLAUGPT_ASR_MODEL", "large-v3")
    device = os.getenv("LACLAUGPT_ASR_DEVICE", "cuda")
    compute_type = os.getenv("LACLAUGPT_ASR_COMPUTE_TYPE", DEFAULT_COMPUTE_TYPE)
    logger.info(
        "Loading ASR: engine=faster-whisper model=%s device=%s compute=%s",
        model_name, device, compute_type,
    )
    model = WhisperModel(model_name, device=device, compute_type=compute_type)

    def _transcribe(path: str):
        segments, info = model.transcribe(path, temperature=LEGACY_TEMPERATURE)
        text = " ".join(segment.text for segment in segments)
        return str(text).strip(), str(getattr(info, "language", "") or "")

    return _transcribe


def describe_backend() -> dict:
    """Return the effective ASR configuration (for logging/provenance)."""
    engine = _engine()
    if engine == "whisper":
        return {
            "engine": engine,
            "model": os.getenv("LACLAUGPT_ASR_MODEL", DEFAULT_MODEL),
            "download_root": os.getenv("LACLAUGPT_ASR_DOWNLOAD_ROOT", DEFAULT_DOWNLOAD_ROOT),
        }
    return {
        "engine": engine,
        "model": os.getenv("LACLAUGPT_ASR_MODEL", "large-v3"),
        "device": os.getenv("LACLAUGPT_ASR_DEVICE", "cuda"),
        "compute_type": os.getenv("LACLAUGPT_ASR_COMPUTE_TYPE", DEFAULT_COMPUTE_TYPE),
    }
