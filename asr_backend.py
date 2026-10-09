"""Backend-neutral ASR adapters for EP24 on CSC Roihu.

Issue #128 deliberately removes Whisper from the dataframe/schema identity.
The default is NVIDIA Canary v2 because it covers all ten EP24 country-language
families (including Bulgarian and Croatian) and can translate speech directly
to English. Qwen3-ASR, Parakeet-TDT V3 and Whisper/faster-whisper remain
selectable for the required Roihu benchmark.
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Callable

LOG = logging.getLogger(__name__)

DEFAULT_ENGINE = os.getenv("LACLAUGPT_ASR_ENGINE", "canary").strip().lower()
DEFAULT_CANARY_MODEL = "nvidia/canary-1b-v2"
DEFAULT_PARAKEET_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
DEFAULT_QWEN_MODEL = "Qwen/Qwen3-ASR-1.7B"
LEGACY_TEMPERATURE = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

COUNTRY_LANGUAGE_HINTS = {
    "finland": "fi",
    "poland": "pl",
    "portugal": "pt",
    "germany": "de",
    "spain": "es",
    "hungary": "hu",
    "croatia": "hr",
    "france": "fr",
    "bulgaria": "bg",
    "sweden": "sv",
}


@dataclass(frozen=True)
class ASRResult:
    transcript: str
    language: str
    translated: str


@dataclass(frozen=True)
class ASRBackend:
    engine: str
    model: str
    transcribe: Callable[[str, str | None], ASRResult]


def language_hint(country: str | None) -> str:
    return COUNTRY_LANGUAGE_HINTS.get(str(country or "").strip().casefold(), "")


def _nemo_audio_path(path: str, directory: str) -> str:
    """Decode a trimmed video once to PCM WAV for Lhotse's soundfile backend.

    Recent torchaudio releases have no torchaudio.io.StreamReader, which the
    FFmpeg-torchaudio Lhotse backend still attempts to import.
    """
    destination = Path(directory) / "speech.wav"
    completed = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
         "-y", "-i", path, "-vn", "-ac", "1", "-ar", "16000",
         "-c:a", "pcm_s16le", str(destination)],
        check=False, capture_output=True, text=True,
    )
    if completed.returncode or not destination.is_file() or destination.stat().st_size <= 44:
        raise RuntimeError("Cannot decode analysis audio to PCM WAV: " +
                           (completed.stderr or "").strip()[:500])
    return str(destination)


def _configure_nemo_audio_backend() -> None:
    import soundfile
    from lhotse.audio import set_current_audio_backend

    # Explicitly avoid the removed torchaudio.io.StreamReader API.
    set_current_audio_backend("LibsndfileBackend")
    LOG.info("NeMo/Lhotse audio backend=LibsndfileBackend soundfile=%s",
             soundfile.__version__)


def load_asr_model() -> ASRBackend:
    engine = os.getenv("LACLAUGPT_ASR_ENGINE", DEFAULT_ENGINE).strip().lower()

    if engine == "canary":
        _configure_nemo_audio_backend()
        from nemo.collections.asr.models import ASRModel

        model_name = os.getenv("LACLAUGPT_ASR_MODEL", DEFAULT_CANARY_MODEL)
        LOG.info("Loading ASR backend=canary model=%s", model_name)
        model = ASRModel.from_pretrained(model_name=model_name)

        def _transcribe(path: str, hint: str | None = None) -> ASRResult:
            source = (hint or "").strip().lower()
            if not source:
                raise ValueError(
                    "Canary requires a source language hint; pass the canonical country "
                    "so EP24 can select its language."
                )
            with tempfile.TemporaryDirectory(prefix="ep24_canary_") as tmpdir:
                wav = _nemo_audio_path(path, tmpdir)
                same = model.transcribe([wav], source_lang=source, target_lang=source)[0]
                transcript = str(getattr(same, "text", same) or "").strip()
                translated = transcript
                if source != "en":
                    en = model.transcribe([wav], source_lang=source, target_lang="en")[0]
                    translated = str(getattr(en, "text", en) or "").strip()
                return ASRResult(transcript, source, translated)

        return ASRBackend(engine, model_name, _transcribe)

    if engine == "parakeet":
        _configure_nemo_audio_backend()
        from nemo.collections.asr.models import ASRModel

        model_name = os.getenv("LACLAUGPT_ASR_MODEL", DEFAULT_PARAKEET_MODEL)
        LOG.info("Loading ASR backend=parakeet model=%s", model_name)
        model = ASRModel.from_pretrained(model_name=model_name)

        def _transcribe(path: str, hint: str | None = None) -> ASRResult:
            with tempfile.TemporaryDirectory(prefix="ep24_parakeet_") as tmpdir:
                wav = _nemo_audio_path(path, tmpdir)
                out = model.transcribe([wav])[0]
            transcript = str(getattr(out, "text", out) or "").strip()
            detected = str(
                getattr(out, "language", "")
                or getattr(out, "lang", "")
                or hint
                or ""
            ).strip()
            return ASRResult(transcript, detected, transcript if detected == "en" else "")

        return ASRBackend(engine, model_name, _transcribe)

    if engine == "qwen3-asr":
        import torch
        from qwen_asr import Qwen3ASRModel

        model_name = os.getenv("LACLAUGPT_ASR_MODEL", DEFAULT_QWEN_MODEL)
        LOG.info("Loading ASR backend=qwen3-asr model=%s", model_name)
        model = Qwen3ASRModel.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map=os.getenv("LACLAUGPT_ASR_DEVICE", "cuda:0"),
            max_inference_batch_size=int(os.getenv("LACLAUGPT_ASR_BATCH_SIZE", "8")),
            max_new_tokens=int(os.getenv("LACLAUGPT_ASR_MAX_NEW_TOKENS", "1024")),
        )

        def _transcribe(path: str, hint: str | None = None) -> ASRResult:
            result = model.transcribe(audio=path, language=None)[0]
            transcript = str(getattr(result, "text", "") or "").strip()
            detected = str(getattr(result, "language", "") or hint or "").strip()
            return ASRResult(transcript, detected, transcript if detected.lower() == "english" else "")

        return ASRBackend(engine, model_name, _transcribe)

    if engine in {"whisper", "faster-whisper"}:
        model_name = os.getenv(
            "LACLAUGPT_ASR_MODEL",
            "large-v3" if engine == "faster-whisper" else "large",
        )
        LOG.warning(
            "Using Whisper-family comparison baseline engine=%s model=%s; "
            "this is not the EP24 default.",
            engine,
            model_name,
        )
        if engine == "whisper":
            import whisper

            model = whisper.load_model(
                model_name,
                download_root=os.getenv("LACLAUGPT_ASR_DOWNLOAD_ROOT", "./whisper/"),
            )

            def _transcribe(path: str, hint: str | None = None) -> ASRResult:
                result = model.transcribe(path, temperature=LEGACY_TEMPERATURE)
                text = str(result.get("text", "") or "").strip()
                lang = str(result.get("language", "") or hint or "").strip()
                return ASRResult(text, lang, text if lang == "en" else "")

        else:
            from faster_whisper import WhisperModel

            model = WhisperModel(
                model_name,
                device=os.getenv("LACLAUGPT_ASR_DEVICE", "cuda"),
                compute_type=os.getenv("LACLAUGPT_ASR_COMPUTE_TYPE", "float16"),
            )

            def _transcribe(path: str, hint: str | None = None) -> ASRResult:
                segments, info = model.transcribe(path, temperature=LEGACY_TEMPERATURE)
                text = " ".join(segment.text for segment in segments).strip()
                lang = str(getattr(info, "language", "") or hint or "").strip()
                return ASRResult(text, lang, text if lang == "en" else "")

        return ASRBackend(engine, model_name, _transcribe)

    raise ValueError(
        "Unknown LACLAUGPT_ASR_ENGINE={!r}; expected canary, parakeet, "
        "qwen3-asr, whisper, or faster-whisper".format(engine)
    )


def describe_backend() -> dict[str, str]:
    engine = os.getenv("LACLAUGPT_ASR_ENGINE", DEFAULT_ENGINE).strip().lower()
    defaults = {
        "canary": DEFAULT_CANARY_MODEL,
        "parakeet": DEFAULT_PARAKEET_MODEL,
        "qwen3-asr": DEFAULT_QWEN_MODEL,
        "whisper": "large",
        "faster-whisper": "large-v3",
    }
    if engine not in defaults:
        raise ValueError(
            f"Unknown LACLAUGPT_ASR_ENGINE={engine!r}; expected one of {tuple(defaults)}"
        )
    return {
        "engine": engine,
        "model": os.getenv("LACLAUGPT_ASR_MODEL", defaults.get(engine, "")),
    }
