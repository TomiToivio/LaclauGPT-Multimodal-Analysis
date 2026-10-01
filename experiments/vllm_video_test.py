#!/usr/bin/env python3
"""Standalone CSC Roihu smoke test: native whole-video analysis with vLLM.

This is an **isolated experiment**, not a production stage. It does not touch the
five-stage EP24 pipeline, `roihu_frame.py`, `roihu_summary.py`, the Ollama
backend, or any legacy CSV contract. It answers one question:

    Can we submit a clean sbatch job on CSC Roihu, download prepared EP24
    videos from CSC Allas, analyze each video directly with Qwen3-VL-8B through
    vLLM, and write readable CSV + debug-log output?

Deliberately readable: standard library + pandas, small functions, obvious
sequential control flow, explicit logging. No classes, no queues, no databases,
no shared abstractions. Read it top to bottom and you have seen the whole test.

Video input is **native whole-video**, not six independent stills. The source is
handed to vLLM as a video, and the Qwen preprocessing stack samples frames
internally while preserving temporal order. See `docs/VLLM_VIDEO_TEST.md` for the
exact backend path and the Roihu version caveats.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import random
import re
import shutil
import shlex
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import pandas as pd

# Direct execution sets sys.path[0] to experiments/. Add the repository root so
# the shared EP24 media contract is importable in sbatch and local runs alike.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ep24_video import (
    VIDEO_INITIAL_SKIP_SECONDS,
    analysis_clip_path,
    needs_resplit,
    parse_scroll_metadata,
    prepare_analysis_clip,
)
from ep24_schema import (
    REQUIRED_MEDIA_COLUMNS,
    source_metadata,
    stable_source_id,
    value as ep24_value,
)

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_SAMPLE_SIZE = None

# allas_filename is authoritative for the researcher-feed reprocessing corpus.
# This fallback exists only for backwards-compatible reads of old scraper rows.
DEFAULT_ALLAS_PATH_TEMPLATE = "Scraper/TikTok/Videos/{country}/{author}/{video_id}.mp4"

REQUIRED_COLUMNS = REQUIRED_MEDIA_COLUMNS
REQUIRED_INITIAL_SKIP_SECONDS = 1.0

# Experimental output columns. Existing columns are never renamed or dropped.
OUTPUT_COLUMNS = (
    "vllm_video_model",
    "vllm_video_version",
    "vllm_video_status",
    "vllm_video_analysis",
    "vllm_video_error",
    "vllm_video_source_row_index",
    "vllm_video_source_id",
    "vllm_video_allas_source",
    "vllm_video_remote_path",
    "vllm_video_local_path",
    "vllm_video_analysis_path",
    "vllm_video_remote_path_logged",
    "vllm_video_bytes",
    "vllm_video_sha256",
    "vllm_video_source_duration_seconds",
    "vllm_video_analysis_duration_seconds",
    "vllm_video_trim_command",
    "vllm_video_trim_exit_status",
    "vllm_video_prompt_version",
    "vllm_video_prompt_sha256",
    "vllm_video_markdown_analysis",
    "vllm_video_raw_output",
    "vllm_video_structured_json",
    "vllm_video_structured_output_status",
    "vllm_video_structured_output_error",
    "vllm_video_runtime_seconds",
    "vllm_video_selected_index",
    "vllm_video_prompt",
    "SCROLL",
    "SCROLL_SECONDS",
    "needs_resplit",
    "video_initial_skip_seconds",
    # Added for the Laskin feasibility experiment (issue #32) so the same
    # harness can produce a like-for-like Roihu vs Laskin comparison. These
    # are purely additive: every column above is unchanged.
    "vllm_video_api",
    "vllm_video_analyzed_duration_seconds",
    "vllm_video_inference_seconds",
    "vllm_video_prompt_hash",
    "vllm_version",
    "vllm_torch_version",
    "vllm_cuda_version",
    "vllm_gpu_name",
    "vllm_hostname",
    "vllm_peak_gpu_memory_mb",
    "vllm_structured_status",
    "vllm_structured_output",
)

# EDITABLE RESEARCHER PROMPT SECTION.
# Keep researcher-authored questions here in one place. Record prompt text and
# hash in each result so an output can be tied to the exact prompt used.
# Descriptive, pre-discursive. This stage produces a video *evidence
# representation* only: no Laclau/populism, ideological, partisan, sentiment,
# DNA or SNA analysis. Those belong downstream.
SYSTEM_PROMPT = (
    "You are performing a descriptive multimodal pre-analysis of one short "
    "social-media video from the EP2024 collection.\n\n"
    "This is an upstream observational stage. Do NOT perform political, "
    "ideological, partisan, populism, sentiment, discourse or Laclauian "
    "analysis. Do not classify signifiers, nodal points, chains of equivalence, "
    "antagonisms, hegemony or political camps. Do not guess the identity of any "
    "person. Those tasks belong to later stages.\n\n"
    "Describe only what is observable in the video, and mark uncertainty "
    "explicitly rather than filling gaps with plausible invention.\n\n"
    "EP24 COLLECTION RULE: the original split clip always begins with a known "
    "feed-scroll artifact. The application removes the first 1.0 second before "
    "you see the video. Do not count that known initial transition as an "
    "additional scroll. Inspect the remaining content for later TikTok/Instagram "
    "feed scrolls that indicate the original splitter failed."
)

VIDEO_PROMPT = (
    "Watch this video from beginning to end and produce an in-depth description "
    "of its whole temporal narrative. Cover, wherever observable:\n"
    "1. Major scenes and scene changes, in the order they occur.\n"
    "2. Actions and events over time.\n"
    "3. People and participants (describe them without guessing unknown "
    "identities).\n"
    "4. Spoken words and visible text, when you can perceive them.\n"
    "5. Gestures, facial expressions and interactions.\n"
    "6. Camera work and editing changes (cuts, zooms, transitions).\n"
    "7. Graphics, captions, memes, screenshots and platform interface elements.\n"
    "8. Important visual signs, symbols and logos.\n"
    "9. Temporal relationships between events (what happens before, during and "
    "after what).\n"
    "10. A concise beginning -> middle -> end narrative summary.\n\n"
    "Finish with a short 'Uncertainty' note listing what you could not determine "
    "or are unsure about. Then append exactly one JSON object on its own line with "
    "keys SCROLL and SCROLL_SECONDS. SCROLL must be true only when an additional "
    "feed-scroll transition separates distinct TikTok/Instagram items after the "
    "known initial artifact. Distinguish a feed scroll from normal camera motion, "
    "cuts, pans, zooms, in-post scrolling, or animation. SCROLL_SECONDS must list "
    "approximate timestamps in seconds on the ORIGINAL source timeline. Because "
    "the visible analysis clip begins at original t=1.0s, add 1.0 second to visible "
    "timestamps. If no additional feed scroll exists, output "
    "{\"SCROLL\": false, \"SCROLL_SECONDS\": []}. Write in English. Do not identify "
    "unknown individuals."
)
PROMPT_VERSION = "ep24-vllm-video-description-v1"
PROMPT_TEXT = f"{SYSTEM_PROMPT}\n\n{VIDEO_PROMPT}"
PROMPT_SHA256 = hashlib.sha256(PROMPT_TEXT.encode("utf-8")).hexdigest()

STRUCTURED_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "analysis_markdown": {"type": "string"},
        "SCROLL": {"type": "boolean"},
        "SCROLL_SECONDS": {"type": "array", "items": {"type": "number"}},
    },
    "required": ["analysis_markdown", "SCROLL", "SCROLL_SECONDS"],
    "additionalProperties": False,
}


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

def setup_logging(log_path: Path) -> logging.Logger:
    """Log everything operationally important to the file and to stderr."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("vllm_video_test")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
    )
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


_SECRET_KEY_VALUE_RE = re.compile(
    r"(?i)\b(aws_access_key_id|aws_secret_access_key|access_key|secret_key|"
    r"api_key|token|password|passwd)\s*[:=]\s*\S+"
)
_BASIC_AUTH_URL_RE = re.compile(r"://[^/\s:@]+:[^/\s:@]+@")


def _package_version(name: str) -> str:
    """Version of an installed package, or a marker. Never raises.

    A missing package on one host must not hide the versions that are available
    (this harness runs on both Roihu and Laskin).
    """
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version(name)
    except PackageNotFoundError:
        return "<not installed>"
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        return f"<unavailable: {type(exc).__name__}>"


def parse_structured_output(raw: str) -> tuple[str, str, str, str]:
    """Split a structured-output reply into (analysis, json, status, error).

    ``status`` is ``ok`` when a JSON object was parsed, otherwise ``parse_failed``.
    The caller always keeps the raw text, so a parse failure is a diagnostic and
    never a lost result.
    """
    if not raw:
        return raw, "", "empty", "no model output"
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return raw, "", "parse_failed", "no JSON object in model output"
    candidate = match.group(0)
    try:
        parsed = json.loads(candidate)
    except Exception as exc:  # noqa: BLE001 - the fallback is the point
        return raw, candidate, "parse_failed", f"{type(exc).__name__}: {exc}"
    if not isinstance(parsed, dict):
        return raw, candidate, "parse_failed", "JSON was not an object"
    analysis = parsed.get("analysis_markdown")
    if not isinstance(analysis, str) or not analysis.strip():
        return raw, candidate, "parse_failed", "analysis_markdown missing or not a non-empty string"
    return analysis, candidate, "ok", ""


def redact_secret_like(text: object) -> str:
    """Mask obvious credential-shaped substrings before they reach a log line.

    This is a best-effort backstop over free-form subprocess output (rclone,
    ffprobe), never a substitute for not reading/logging secrets in the first
    place. The harness itself never reads credentials.
    """
    if text is None:
        return ""
    text = str(text)
    if not text:
        return text
    redacted = _SECRET_KEY_VALUE_RE.sub(
        lambda m: f"{m.group(1)}=<redacted>", text
    )
    redacted = _BASIC_AUTH_URL_RE.sub("://<redacted>@", redacted)
    return redacted


# Backwards-compatible internal name retained for the older Roihu harness
# call sites. Both hosts use the same redaction implementation.
redact_sensitive = redact_secret_like


def _run_capture(command: list[str]) -> str:
    """Run a helper command and return its output, or a note on why it failed."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    except Exception as exc:  # noqa: BLE001 - diagnostics must never be fatal
        return redact_sensitive(f"<{command[0]} unavailable: {exc}>")
    output = (result.stdout or "") + (result.stderr or "")
    return redact_secret_like(output.strip()) or f"<{command[0]} produced no output>"


def log_environment(logger: logging.Logger) -> None:
    """Record the environment once, up front, so a failure is diagnosable."""
    logger.info("=== environment ===")
    logger.info("utc_now            : %s", datetime.now(timezone.utc).isoformat())
    logger.info("local_now          : %s", datetime.now().isoformat())
    logger.info("hostname           : %s", socket.gethostname())
    logger.info("slurm_job_id       : %s", os.environ.get("SLURM_JOB_ID", "<unset>"))
    logger.info("slurm_job_name     : %s", os.environ.get("SLURM_JOB_NAME", "<unset>"))
    logger.info("slurm_submit_dir   : %s", os.environ.get("SLURM_SUBMIT_DIR", "<unset>"))
    logger.info("slurm_job_partition: %s", os.environ.get("SLURM_JOB_PARTITION", "<unset>"))
    logger.info("python_version     : %s", platform.python_version())
    logger.info("platform           : %s", platform.platform())
    logger.info("machine            : %s", platform.machine())
    logger.info("cwd                : %s", Path.cwd())
    logger.info("git_commit_sha     : %s", git_commit_sha())
    logger.info("gpu_query          : %s", _run_capture([
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,driver_version",
        "--format=csv,noheader",
    ]))
    logger.info("gpu_details        : %s", _run_capture(["nvidia-smi"]))
    logger.info("=== end environment ===")


def git_commit_sha() -> str:
    """Best-effort commit SHA for reproducibility provenance. Never fatal."""
    try:
        result = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:  # noqa: BLE001 - diagnostics must never be fatal
        return "<unknown>"
    sha = (result.stdout or "").strip()
    return sha or "<unknown>"


def gpu_name_from_nvidia_smi() -> str:
    """First GPU name reported by nvidia-smi, or a placeholder if unavailable."""
    output = _run_capture(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    first_line = output.splitlines()[0].strip() if output.strip() else ""
    return first_line or "<unavailable>"


def collect_runtime_versions(logger: logging.Logger) -> dict[str, str]:
    """Collect the version fields needed for the Roihu vs Laskin comparison.

    Every lookup is independently guarded: a missing package on one host must
    not hide the versions that *are* available, and must never raise.
    """
    info: dict[str, str] = {
        "vllm_version": "<unavailable>",
        "torch_version": "<unavailable>",
        "cuda_version": "<unavailable>",
        "gpu_name": gpu_name_from_nvidia_smi(),
        "hostname": socket.gethostname(),
    }
    try:
        import vllm

        info["vllm_version"] = getattr(vllm, "__version__", "<unknown>")
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        logger.debug("vllm introspection unavailable: %s", exc)
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_version"] = getattr(torch.version, "cuda", "<unknown>") or "<unknown>"
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        logger.debug("torch introspection unavailable: %s", exc)
    return info


def peak_gpu_memory_mb() -> str:
    """Peak allocated CUDA memory across visible devices, in MB. Never fatal."""
    try:
        import torch

        if not torch.cuda.is_available():
            return ""
        total_bytes = sum(
            torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())
        )
        return f"{total_bytes / (1024 * 1024):.1f}"
    except Exception:  # noqa: BLE001 - diagnostics must never be fatal
        return ""


def prompt_hash(prompt: str) -> str:
    """Short stable hash identifying the exact researcher prompt used."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# Input CSV and sample selection
# --------------------------------------------------------------------------- #

def load_input_csv(input_csv: Path, logger: logging.Logger) -> pd.DataFrame:
    """Read the EP24 CSV without altering it."""
    if not input_csv.is_file():
        raise SystemExit(f"Input CSV not found: {input_csv}")
    # dtype=str keeps identifiers byte-for-byte as stored.
    df = pd.read_csv(input_csv, dtype=str, keep_default_na=False)
    logger.info("read input CSV %s: %d rows x %d columns", input_csv, len(df), len(df.columns))
    logger.info("EP24 source metadata fields present: %s", list(df.columns))
    if "allas_filename" not in df.columns:
        legacy_fallback = {"scrapedCountry", "authorUniqueId", "videoId"}
        if not legacy_fallback.issubset(df.columns):
            raise SystemExit(
                "Input CSV must contain canonical allas_filename (preferred), or the "
                "legacy scraper fallback fields scrapedCountry/authorUniqueId/videoId. "
                f"Found columns: {list(df.columns)}"
            )
    return df


def row_has_fetch_identifier(row: pd.Series) -> bool:
    """Whether the row has either a historical Allas key or fallback identifiers."""
    if str(row.get("allas_filename", "")).strip():
        return True
    return all(ep24_value(row, column) for column in REQUIRED_COLUMNS)


def select_sample(
    df: pd.DataFrame, size: int | None, seed: int, logger: logging.Logger
) -> list[int]:
    """Choose `size` usable rows reproducibly for a given seed.

    The prepared private sample is authoritative: with no explicit size, every
    input row is processed in CSV order. Sampling is an opt-in compatibility
    feature and only considers rows with enough source information to fetch.
    """
    if size is None:
        chosen = list(df.index)
        logger.info("processing all %d prepared input rows in CSV order", len(chosen))
        return chosen
    usable = [
        i
        for i, row in df.iterrows()
        if row_has_fetch_identifier(row)
    ]
    logger.info("usable rows for selection: %d of %d", len(usable), len(df))
    if len(usable) < size:
        raise SystemExit(
            f"Requested {size} videos but only {len(usable)} usable row(s) exist "
            f"(need non-empty allas_filename/video_id metadata)."
        )
    chosen = sorted(random.Random(seed).sample(usable, size))
    logger.info("selected row indices (seed=%d): %s", seed, chosen)
    return chosen


# --------------------------------------------------------------------------- #
# CSC Allas
# --------------------------------------------------------------------------- #

def derive_remote_path(row: pd.Series, template: str) -> str:
    """Use canonical allas_filename; legacy scraper path is compatibility-only."""
    allas_filename = ep24_value(row, "allas_filename")
    if allas_filename:
        return allas_filename
    return template.format(
        country=ep24_value(row, "country"),
        author=ep24_value(row, "author_username"),
        video_id=ep24_value(row, "video_id"),
    )


def build_rclone_source(remote: str, bucket: str, object_path: str) -> str:
    """Compose an rclone source spec such as `allas:my-bucket/path/to.mp4`."""
    parsed = urlsplit(object_path)
    if parsed.scheme in {"http", "https"}:
        return object_path
    if parsed.scheme == "s3":
        bucket = parsed.netloc
        object_path = parsed.path.lstrip("/")
    clean = f"{bucket}/{object_path}".lstrip("/")
    return f"{remote}:{clean}" if remote else clean


def fetch_video(
    object_path: str,
    local_dir: Path,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> Path:
    """Bring one video into job-local storage. Raises on failure.

    Three backends, all reading only:

    * ``rclone``  - CSC's recommended Allas client, against the S3-compatible
      endpoint configured in the job environment. Downloads exactly one object.
    * ``local``   - copy from an already-staged Allas mirror (the production
      pipeline keeps videos under ``./Allas/...``), which is also how this test
      is exercised offline.
    * ``none``    - do not fetch; fail so the run is obviously incomplete.

    No object is ever written back to Allas.
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / Path(object_path).name

    if args.fetch_backend == "none":
        raise RuntimeError("fetch disabled (--fetch-backend none)")

    if args.fetch_backend == "local":
        mirror_key = urlsplit(object_path).path.lstrip("/")
        source = Path(args.allas_local_root) / mirror_key
        command = ["copy", str(source), str(local_path)]
        logger.info("download_backend=local command=%s", shlex.join(command))
        logger.debug("copying local Allas mirror %s -> %s", source, local_path)
        if not source.is_file():
            raise FileNotFoundError(f"not found in local Allas mirror: {source}")
        shutil.copy2(source, local_path)
        logger.info("download_exit_status=0 stdout=<local copy> stderr=<empty>")
        return local_path

    remote_source = build_rclone_source(args.rclone_remote, args.allas_bucket, object_path)
    operation = "copyurl" if urlsplit(remote_source).scheme in {"http", "https"} else "copyto"
    rclone_bin = os.environ.get("RCLONE_BIN", "rclone")
    command = [rclone_bin, operation, remote_source, str(local_path)]
    logger.info("download_backend=rclone command=%s", redact_sensitive(shlex.join(command)))
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=args.download_timeout,
    )
    logger.info("download_exit_status=%s stdout=%s stderr=%s",
                result.returncode,
                redact_sensitive((result.stdout or "").strip()),
                redact_sensitive((result.stderr or "").strip()))
    if result.returncode != 0:
        raise RuntimeError(
            f"rclone failed for {redact_sensitive(remote_source)}: "
            f"{redact_sensitive((result.stderr or result.stdout or '').strip())}"
        )
    if not local_path.is_file():
        raise RuntimeError(f"rclone reported success but {local_path} is absent")
    return local_path


def probe_video_metadata(local_path: Path, logger: logging.Logger) -> dict:
    """Read and log ffprobe metadata; real inference requires a valid duration."""
    command = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration:stream=codec_type,codec_name,width,height,r_frame_rate,duration",
        "-of", "json",
        str(local_path),
    ]
    logger.info("ffprobe_command=%s", shlex.join(command))
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    logger.info("ffprobe_exit_status=%d stdout=%s stderr=%s",
                result.returncode,
                redact_sensitive((result.stdout or "").strip()),
                redact_sensitive((result.stderr or "").strip()))
    if result.returncode:
        raise RuntimeError(
            f"ffprobe failed for {local_path}: "
            f"{redact_sensitive((result.stderr or result.stdout or '').strip())}"
        )
    metadata = json.loads(result.stdout)
    duration = metadata.get("format", {}).get("duration")
    if duration is None:
        duration = next(
            (stream.get("duration") for stream in metadata.get("streams", [])
             if stream.get("codec_type") == "video" and stream.get("duration") is not None),
            None,
        )
    if duration is None:
        raise RuntimeError(f"ffprobe did not report a duration for {local_path}")
    metadata["duration_seconds"] = float(duration)
    logger.info("ffprobe_metadata path=%s metadata=%s", local_path, json.dumps(metadata, sort_keys=True))
    return metadata


def sha256_file(path: Path) -> str:
    """Compute a streaming checksum for download/source-integrity provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as video_file:
        for block in iter(lambda: video_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe_duration_seconds(local_path: Path) -> str:
    """Container duration in seconds via ffprobe, or '' if unavailable.

    Used for both the source and the trimmed analysis clip so the CSV carries
    both durations for the Roihu vs Laskin comparison (issue #32).
    """
    if shutil.which("ffprobe") is None or not local_path.is_file():
        return ""
    output = _run_capture([
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(local_path),
    ])
    try:
        return f"{float(output.strip()):.3f}"
    except (TypeError, ValueError):
        return ""


# --------------------------------------------------------------------------- #
# vLLM / Qwen3-VL
# --------------------------------------------------------------------------- #

def load_model(args: argparse.Namespace, logger: logging.Logger):
    """Load Qwen3-VL with vLLM. Returns model, sampling parameters and status.

    Imports happen here, not at module import time, so the rest of this file
    stays importable and testable on a machine without vLLM installed.
    """
    if args.model_backend == "stub":
        logger.warning("model backend is STUB: no real inference will be performed")
        return None, None

    import vllm
    from vllm import LLM, SamplingParams

    logger.info("vllm_version      : %s", getattr(vllm, "__version__", "<unknown>"))
    try:
        import torch

        logger.info("torch_version     : %s", torch.__version__)
        logger.info("cuda_available    : %s", torch.cuda.is_available())
        logger.info("cuda_version      : %s", getattr(torch.version, "cuda", "<unknown>"))
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        logger.warning("torch introspection failed: %s", exc)

    logger.info("loading model %s (this can take minutes)", args.model)
    llm = LLM(
        model=args.model,
        # One video per request is all this test needs.
        limit_mm_per_prompt={"video": 1},
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )
    sampling_kwargs = dict(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    structured_status = "off"
    if args.structured_output:
        try:
            from vllm.sampling_params import StructuredOutputsParams

            structured_params = StructuredOutputsParams(json=STRUCTURED_OUTPUT_SCHEMA)
            sampling_params = SamplingParams(
                **sampling_kwargs, structured_outputs=structured_params
            )
            structured_status = "requested"
            logger.info("structured_output_schema=%s", json.dumps(STRUCTURED_OUTPUT_SCHEMA, sort_keys=True))
        except Exception as exc:  # noqa: BLE001 - constrained decoding is optional
            logger.warning(
                "structured output unsupported by installed vLLM path; using raw text: %s",
                redact_sensitive(f"{type(exc).__name__}: {exc}"),
            )
            sampling_params = SamplingParams(**sampling_kwargs)
            structured_status = "unsupported"
    else:
        sampling_params = SamplingParams(**sampling_kwargs)
    logger.info("model loaded: %s", args.model)
    return llm, sampling_params, structured_status


def ep24_metadata_context(row: pd.Series | None) -> str:
    """Render real researcher-feed metadata for the model without inventing scraper fields."""
    if row is None:
        return ""
    lines = [f"- {column}: {value}" for column, value in source_metadata(row)]
    if not lines:
        return ""
    return (
        "\n\nEP24 SOURCE METADATA (researcher-recorded feed clip; not scraper metadata):\n"
        + "\n".join(lines)
        + "\nTreat researcher_* fields and researcher_note as human annotation, not model output."
    )


def build_video_messages(
    local_path: Path,
    args: argparse.Namespace,
    row: pd.Series | None = None,
) -> list[dict]:
    """Build the Qwen chat messages with the video as a first-class video part.

    The application API treats the source as a *video*; temporal ordering is
    preserved and any frame sampling happens inside the Qwen preprocessing
    stack, not as six unrelated image prompts.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(local_path),
                    "min_pixels": args.video_min_pixels,
                    "max_pixels": args.video_max_pixels,
                    "total_pixels": args.video_total_pixels,
                },
                {"type": "text", "text": VIDEO_PROMPT + ep24_metadata_context(row)},
            ],
        },
    ]


DEFAULT_VIDEO_API = "mm_processor_kwargs"


def _vllm_version_tuple(logger: logging.Logger) -> tuple[int, int]:
    """Best-effort (major, minor) of the installed vLLM, or (0, 0) if unknown."""
    try:
        import re as _re

        import vllm

        match = _re.match(r"(\d+)\.(\d+)", str(getattr(vllm, "__version__", "") or ""))
        if match:
            return int(match.group(1)), int(match.group(2))
    except Exception as exc:  # noqa: BLE001 - never fatal, only selects an API
        logger.debug("could not determine vLLM version: %s", exc)
    return 0, 0


def resolve_video_api(video_api: str, logger: logging.Logger) -> str:
    """Resolve ``auto`` to the request shape the installed vLLM accepts.

    The two shapes are not interchangeable: ``mm_processor_kwargs`` is fatal on
    vLLM 0.8.5 (the processor cache hashes the kwargs dict and raises
    ``TypeError: unhashable type: 'dict'``). Guessing wrong therefore kills the
    whole run on the Laskin/Volta stack, so ``auto`` exists to prevent an
    operator from having to remember which host needs which shape.

    Rule: vLLM >= 0.9 supports the metadata path; 0.8.x and anything unreadable
    fall back to ``direct``, the shape that cannot crash.
    """
    aliases = {"legacy": "direct", "modern": "mm_processor_kwargs"}
    video_api = aliases.get(video_api, video_api)
    if video_api in ("direct", "mm_processor_kwargs"):
        return video_api
    major, minor = _vllm_version_tuple(logger)
    resolved = "mm_processor_kwargs" if (major, minor) >= (0, 9) else "direct"
    logger.info("video_api auto-resolved to %s for vLLM %s.%s", resolved, major, minor)
    return resolved


def prepare_vllm_request(
    messages: list[dict],
    processor,
    logger: logging.Logger,
    video_api: str = DEFAULT_VIDEO_API,
) -> dict:
    """Turn chat messages into a vLLM generate input with video mm_data.

    Two API variants are supported, selected by ``video_api``. The analysis
    contract (prompt, messages, output schema) is identical either way; only
    the shape handed to vLLM differs, matching the measured feasibility result
    from issue TomiToivio/LaclauGPT-Multimodal-Analysis#32.

    * ``mm_processor_kwargs`` (default, Roihu/current vLLM): Qwen3-VL wants
      ``image_patch_size=16`` and ``return_video_metadata=True``.
      `process_vision_info` returns ``(video, metadata)`` pairs; current
      vLLM requires those pairs to remain intact inside
      ``multi_modal_data["video"]``. Processor kwargs remain separate.
    * ``direct`` (Laskin/vLLM 0.8.5-era, Qwen2.5-VL): the older
      `process_vision_info` does not accept ``return_video_metadata`` and
      passing ``video_metadata`` inside ``mm_processor_kwargs`` raises
      ``TypeError: unhashable type: 'dict'``. The video tensor is passed to
      vLLM directly instead.
    """
    from qwen_vl_utils import process_vision_info

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    video_api = resolve_video_api(video_api, logger)

    if video_api == "direct":
        # Do NOT ask for video kwargs here. vLLM 0.8.5's processor cache does
        # ``hash(tuple(...))`` on mm_processor_kwargs and raises
        # ``TypeError: unhashable type: 'dict'`` for the mapping this returns,
        # so any non-empty dict is fatal on Laskin. The video tensors go to
        # vLLM directly instead, which is what 0.8.x supports.
        vision_result = process_vision_info(
            messages,
            image_patch_size=16,
        )
        # qwen-vl-utils releases differ here: the legacy path returns either
        # (image_inputs, video_inputs) or a three-tuple with an extra kwargs
        # value. Never forward that kwargs value to vLLM 0.8.x.
        image_inputs, video_inputs = vision_result[:2]
        mm_data: dict = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            mm_data["video"] = video_inputs
        video_kwargs = None
    else:
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=16,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs
        if video_inputs is not None:
            # Current vLLM/Qwen3-VL requires video metadata in multi_modal_data.
            # qwen-vl-utils already returns (video, metadata) pairs, so preserve
            # those pairs intact. Splitting metadata into mm_processor_kwargs
            # causes: "Video metadata is required but not found in mm input."
            mm_data["video"] = list(video_inputs)

    logger.debug(
        "prepared request: video_api=%s video_parts=%s mm_processor_kwargs_keys=%s",
        video_api,
        len(mm_data.get("video", [])),
        sorted((video_kwargs or {}).keys()),
    )
    request = {"prompt": prompt, "multi_modal_data": mm_data}
    if video_kwargs:
        request["mm_processor_kwargs"] = video_kwargs
    return request


def generate_stub(local_path: Path, model: str) -> str:
    """Deterministic placeholder used only with --model-backend stub.

    It exists so the harness (selection, fetch, logging, CSV contract, failure
    isolation) can be verified offline without a GPU. It performs no inference
    and says so in its own output.
    """
    size = local_path.stat().st_size if local_path.is_file() else -1
    return (
        f"[STUB OUTPUT - no model was run] model={model} file={local_path.name} "
        f"bytes={size}. This confirms the experiment harness works end to end "
        f"without a GPU; replace --model-backend vllm for real analysis."
    )


def guided_decoding_schema() -> dict:
    """JSON schema for complete structured analysis plus scroll metadata."""
    return STRUCTURED_OUTPUT_SCHEMA

def attempt_structured_output(
    analysis_path: Path,
    args: argparse.Namespace,
    llm,
    processor,
    logger: logging.Logger,
) -> tuple[str, str]:
    """Best-effort guided-JSON re-generation. Never fatal; mirrors issue #32 section 11.

    Structured decoding is optional and must never block the experiment: the
    free-text analysis is already captured by `analyze_one_video` regardless of
    this function's outcome.

    Returns ``(status, output_text)`` where status is one of ``disabled``
    (flag not passed), ``skipped_stub`` (no real model loaded), ``ok``, or
    ``unsupported: <reason>``.
    """
    if args.model_backend == "stub":
        return "skipped_stub", ""
    if not args.structured_output:
        return "disabled", ""
    try:
        from vllm import SamplingParams as VllmSamplingParams
        from vllm.sampling_params import GuidedDecodingParams

        messages = build_video_messages(analysis_path, args)
        request = prepare_vllm_request(messages, processor, logger, video_api=args.video_api)
        structured_params = VllmSamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            guided_decoding=GuidedDecodingParams(json=guided_decoding_schema()),
        )
        outputs = llm.generate([request], sampling_params=structured_params)
        return "ok", outputs[0].outputs[0].text
    except Exception as exc:  # noqa: BLE001 - optional path, never fatal
        logger.warning("structured output unsupported: %s: %s", type(exc).__name__, exc)
        return f"unsupported: {type(exc).__name__}: {exc}", ""


def analyze_one_video(
    local_path: Path,
    args: argparse.Namespace,
    llm,
    sampling_params,
    processor,
    logger: logging.Logger,
    row: pd.Series | None = None,
) -> tuple[str, str, float]:
    """Return analysis text, structured-output status, and inference seconds."""
    if args.model_backend == "stub":
        return generate_stub(local_path, args.model), "off", 0.0

    messages = build_video_messages(local_path, args, row=row)
    request = prepare_vllm_request(messages, processor, logger, video_api=args.video_api)
    inference_started = time.monotonic()
    logger.info("inference_start_utc=%s", datetime.now(timezone.utc).isoformat())
    outputs = llm.generate([request], sampling_params=sampling_params)
    inference_seconds = time.monotonic() - inference_started
    logger.info("inference_end_utc=%s inference_runtime_seconds=%.3f",
                datetime.now(timezone.utc).isoformat(), inference_seconds)
    completion = outputs[0].outputs[0]
    logger.info("vllm_completion_metadata=%s", redact_sensitive({
        "finish_reason": getattr(completion, "finish_reason", None),
        "stop_reason": getattr(completion, "stop_reason", None),
        "prompt_tokens": len(getattr(outputs[0], "prompt_token_ids", []) or []),
        "generated_tokens": len(getattr(completion, "token_ids", []) or []),
    }))
    return completion.text, getattr(args, "structured_output_status", "off"), inference_seconds


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone vLLM whole-video smoke test (Qwen3-VL-8B) on CSC Roihu."
    )
    # Paths. Environment variables provide defaults so the sbatch file stays thin.
    parser.add_argument("--input-csv", default=os.environ.get("LACLAUGPT_VLLM_TEST_INPUT_CSV"))
    parser.add_argument("--output-csv", default=os.environ.get("LACLAUGPT_VLLM_TEST_OUTPUT_CSV"))
    parser.add_argument("--log-path", default=os.environ.get("LACLAUGPT_VLLM_TEST_LOG"))
    parser.add_argument("--download-dir", default=os.environ.get("LACLAUGPT_VLLM_TEST_DOWNLOAD_DIR"))

    # The prepared sample is processed as-is unless a caller explicitly asks
    # for a smaller reproducible subset.
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=20261001)

    # Allas.
    parser.add_argument("--allas-bucket", default=os.environ.get("ALLAS_BUCKET", ""))
    parser.add_argument("--allas-path-template", default=DEFAULT_ALLAS_PATH_TEMPLATE)
    parser.add_argument("--fetch-backend", choices=("rclone", "local", "none"), default="rclone")
    parser.add_argument("--rclone-remote", default=os.environ.get("RCLONE_REMOTE", "allas"))
    parser.add_argument("--allas-local-root", default=os.environ.get("ALLAS_LOCAL_ROOT", "./Allas"))
    parser.add_argument("--download-timeout", type=int, default=900)

    # Model.
    parser.add_argument("--model", default=os.environ.get("LACLAUGPT_VLLM_TEST_MODEL", DEFAULT_MODEL))
    parser.add_argument("--model-backend", choices=("vllm", "stub"), default="vllm")
    parser.add_argument(
        "--video-api",
        choices=("auto", "modern", "legacy", "mm_processor_kwargs", "direct"),
        default=os.environ.get("LACLAUGPT_VLLM_TEST_VIDEO_API", "auto"),
        help=(
            "vLLM multi-modal request shape. 'mm_processor_kwargs' is the "
            "Roihu/current-vLLM default (Qwen3-VL). 'direct' is the Laskin/"
            "vLLM-0.8.5-era path (Qwen2.5-VL): see issue #32."
        ),
    )
    parser.add_argument(
        "--structured-output",
        action="store_true",
        help="Attempt vLLM guided/structured JSON decoding in addition to free text.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--video-min-pixels", type=int, default=4 * 32 * 32)
    parser.add_argument("--video-max-pixels", type=int, default=256 * 32 * 32)
    parser.add_argument("--video-total-pixels", type=int, default=20480 * 32 * 32)
    parser.add_argument("--keep-downloads", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if VIDEO_INITIAL_SKIP_SECONDS != REQUIRED_INITIAL_SKIP_SECONDS:
        raise SystemExit(
            "EP24 vLLM experiment requires video_initial_skip_seconds=1.0; "
            f"the shared contract is configured as {VIDEO_INITIAL_SKIP_SECONDS!r}."
        )
    if args.sample_size is not None and args.sample_size < 1:
        raise SystemExit("--sample-size must be positive when explicitly provided.")
    if not args.input_csv:
        raise SystemExit("Set --input-csv or LACLAUGPT_VLLM_TEST_INPUT_CSV.")
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv or (input_csv.with_suffix("").as_posix() + "_vllm_test.csv"))
    log_path = Path(args.log_path or "./logs/vllm_video_test.log")
    download_dir = Path(args.download_dir or "./vllm_video_downloads")

    logger = setup_logging(log_path)
    log_environment(logger)

    # Resolve ``auto`` once, before the value is logged or used to build a
    # request. Both request shapes are not interchangeable: ``mm_processor_kwargs``
    # is fatal on vLLM 0.8.5, so the resolved value must be the one the run
    # actually uses -- not the operator's unexpanded ``auto``.
    args.video_api = resolve_video_api(args.video_api, logger)

    logger.info("=== configuration ===")
    for key, value in sorted(vars(args).items()):
        logger.info("  %-24s %s", key, redact_secret_like(str(value)))
    logger.info("input_csv  : %s", input_csv)
    logger.info("output_csv : %s", output_csv)
    logger.info("log_path   : %s", log_path)
    logger.info("prompt_version : %s", PROMPT_VERSION)
    logger.info("prompt_sha256  : %s", PROMPT_SHA256)
    logger.info("prompt_text    : %s", redact_sensitive(PROMPT_TEXT))
    logger.info("=== end configuration ===")

    df = load_input_csv(input_csv, logger)
    selected = select_sample(df, args.sample_size, args.seed, logger)

    runtime_versions = collect_runtime_versions(logger)
    logger.info("=== runtime versions (for Roihu/Laskin comparison) ===")
    for key, value in sorted(runtime_versions.items()):
        logger.info("  %-14s %s", key, value)
    prompt_version_hash = prompt_hash(VIDEO_PROMPT)
    logger.info("  %-14s %s", "prompt_hash", prompt_version_hash)
    logger.info("=== end runtime versions ===")

    # The processor is only needed for real inference; load it with the model so
    # a stub run needs neither transformers nor a GPU.
    llm, sampling_params, processor = None, None, None
    args.structured_output_status = "off"
    if args.model_backend == "vllm":
        from transformers import AutoProcessor

        logger.info("loading processor for %s", args.model)
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        llm, sampling_params, args.structured_output_status = load_model(args, logger)

    results = []
    succeeded = 0
    failed = 0

    for position, index in enumerate(selected, start=1):
        row = df.loc[index]
        author = ep24_value(row, "new_id") or ep24_value(row, "video_filename")
        video_id = ep24_value(row, "video_id")
        object_path = derive_remote_path(row, args.allas_path_template)
        safe_object_path = redact_sensitive(object_path)
        source_id = stable_source_id(row)

        logger.info("--- video %d/%d: %s / %s ---", position, len(selected), author, video_id)
        logger.info("  source_row_index  : %s", index)
        logger.info("  source_id         : %s", redact_sensitive(source_id))
        logger.info("  remote_object     : %s", safe_object_path)
        for metadata_key, metadata_value in source_metadata(row):
            logger.info(
                "  metadata.%-20s %s",
                metadata_key + ":",
                redact_sensitive(metadata_value[:500]),
            )

        record = {column: "" for column in OUTPUT_COLUMNS}
        record["vllm_video_model"] = args.model
        record["vllm_video_version"] = runtime_versions["vllm_version"]
        record["vllm_video_api"] = resolve_video_api(args.video_api, logger)
        record["vllm_version"] = runtime_versions["vllm_version"]
        record["vllm_torch_version"] = runtime_versions["torch_version"]
        record["vllm_cuda_version"] = runtime_versions["cuda_version"]
        record["vllm_gpu_name"] = runtime_versions["gpu_name"]
        record["vllm_hostname"] = runtime_versions["hostname"]
        record["vllm_video_prompt_hash"] = prompt_version_hash
        record["vllm_video_selected_index"] = index
        record["vllm_video_source_row_index"] = index
        record["vllm_video_source_id"] = redact_sensitive(source_id)
        record["vllm_video_allas_source"] = safe_object_path
        record["vllm_video_remote_path"] = safe_object_path
        record["vllm_video_remote_path_logged"] = safe_object_path
        record["vllm_video_prompt"] = PROMPT_TEXT
        record["vllm_video_prompt_version"] = PROMPT_VERSION
        record["vllm_video_prompt_sha256"] = PROMPT_SHA256
        record["video_initial_skip_seconds"] = str(VIDEO_INITIAL_SKIP_SECONDS)
        record["SCROLL"] = "FALSE"
        record["SCROLL_SECONDS"] = "[]"
        record["needs_resplit"] = "FALSE"

        started = time.monotonic()
        local_path = None
        analysis_path = None
        try:
            local_path = fetch_video(object_path, download_dir, args, logger)
            record["vllm_video_local_path"] = str(local_path)
            record["vllm_video_bytes"] = str(local_path.stat().st_size)
            source_checksum = sha256_file(local_path)
            record["vllm_video_sha256"] = source_checksum
            logger.info("  local_path        : %s", local_path)
            logger.info("  bytes             : %s", record["vllm_video_bytes"])

            if args.model_backend == "stub":
                # Synthetic harness fixtures are not real media and CI does not
                # require ffmpeg. Stub output is never submitted to a model.
                analysis_path = local_path
                logger.info("  analysis_path     : %s (stub; trim not executed)", analysis_path)
                record["vllm_video_analysis_path"] = str(analysis_path)
                record["vllm_video_structured_output_status"] = "off"
            else:
                source_metadata = probe_video_metadata(local_path, logger)
                source_duration = float(source_metadata["duration_seconds"])
                record["vllm_video_source_duration_seconds"] = str(source_duration)
                if source_duration <= VIDEO_INITIAL_SKIP_SECONDS:
                    raise ValueError(
                        f"source duration {source_duration:.3f}s does not exceed mandatory "
                        f"{VIDEO_INITIAL_SKIP_SECONDS:.1f}s initial skip"
                    )

                analysis_path = analysis_clip_path(local_path, download_dir / "analysis-clips")
                if analysis_path.exists():
                    analysis_path.unlink()
                    logger.info("removed stale derived clip before trimming: %s", analysis_path)

                def logged_trim_runner(command, **kwargs):
                    record["vllm_video_trim_command"] = shlex.join(command)
                    logger.info("trim_command=%s", shlex.join(command))
                    result = subprocess.run(command, **kwargs)
                    record["vllm_video_trim_exit_status"] = str(result.returncode)
                    logger.info(
                        "trim_exit_status=%d stdout=%s stderr=%s",
                        result.returncode,
                        redact_sensitive((result.stdout or "").strip()),
                        redact_sensitive((result.stderr or "").strip()),
                    )
                    return result

                analysis_path = prepare_analysis_clip(
                    local_path,
                    download_dir / "analysis-clips",
                    runner=logged_trim_runner,
                )
                analysis_metadata = probe_video_metadata(analysis_path, logger)
                analysis_duration = float(analysis_metadata["duration_seconds"])
                record["vllm_video_analysis_duration_seconds"] = str(analysis_duration)
                record["vllm_video_analyzed_duration_seconds"] = str(analysis_duration)
                expected_duration = source_duration - VIDEO_INITIAL_SKIP_SECONDS
                tolerance = max(0.15, source_duration * 0.01)
                if analysis_duration <= 0 or abs(analysis_duration - expected_duration) > tolerance:
                    raise ValueError(
                        f"derived clip duration {analysis_duration:.3f}s differs from the "
                        f"expected {expected_duration:.3f}s after mandatory 1.0s trim"
                    )
                logger.info("  source_duration   : %.3f", source_duration)
                logger.info("  analysis_duration : %.3f", analysis_duration)
                logger.info("  analysis_path     : %s", analysis_path)
                record["vllm_video_analysis_path"] = str(analysis_path)
            logger.info("  initial_skip_s    : %.1f", VIDEO_INITIAL_SKIP_SECONDS)
            raw_output, structured_status, inference_seconds = analyze_one_video(
                analysis_path, args, llm, sampling_params, processor, logger, row=row
            )
            record["vllm_video_inference_seconds"] = f"{inference_seconds:.3f}"
            record["vllm_peak_gpu_memory_mb"] = peak_gpu_memory_mb()
            record["vllm_structured_status"] = structured_status
            record["vllm_video_raw_output"] = redact_sensitive(raw_output)
            analysis = raw_output
            record["vllm_video_structured_output_status"] = structured_status
            if structured_status == "requested":
                analysis, structured_json, parse_status, parse_error = parse_structured_output(raw_output)
                record["vllm_video_structured_json"] = structured_json
                record["vllm_structured_output"] = structured_json
                record["vllm_video_structured_output_status"] = parse_status
                record["vllm_video_structured_output_error"] = parse_error
                logger.info("structured_parse_status=%s error=%s json=%s",
                            parse_status, redact_sensitive(parse_error), structured_json)
            elif structured_status == "unsupported":
                record["vllm_video_structured_output_error"] = (
                    "Installed vLLM offline generate path does not support JSON Schema decoding."
                )
            scroll_meta = parse_scroll_metadata(
                record["vllm_video_structured_json"] or raw_output
            )
            record["SCROLL"] = "TRUE" if scroll_meta["SCROLL"] else "FALSE"
            record["SCROLL_SECONDS"] = json.dumps(scroll_meta["SCROLL_SECONDS"])
            record["needs_resplit"] = "TRUE" if needs_resplit(scroll_meta) else "FALSE"
            record["vllm_video_analysis"] = analysis
            record["vllm_video_markdown_analysis"] = analysis
            record["vllm_video_status"] = "ok"
            logger.info("  analysis_chars    : %d", len(analysis))
            logger.debug("  raw_response      : %s", redact_sensitive(raw_output))
            if sha256_file(local_path) != source_checksum:
                raise RuntimeError("downloaded source changed during analysis")
            logger.info("source_integrity=unchanged sha256=%s", source_checksum)
            succeeded += 1
        except Exception as exc:  # noqa: BLE001 - one bad video must not stop the run
            failed += 1
            record["vllm_video_status"] = "error"
            record["vllm_video_error"] = redact_sensitive(f"{type(exc).__name__}: {exc}")
            logger.error("  FAILED: %s", redact_sensitive(f"{type(exc).__name__}: {exc}"))
            logger.error("  traceback:\n%s", redact_sensitive(traceback.format_exc()))
        finally:
            record["vllm_video_runtime_seconds"] = f"{time.monotonic() - started:.1f}"
            logger.info("  runtime_seconds   : %s", record["vllm_video_runtime_seconds"])
            if not args.keep_downloads:
                for cleanup_path in (analysis_path, local_path):
                    if cleanup_path is not None and cleanup_path != local_path and cleanup_path.exists():
                        cleanup_path.unlink()
                        logger.info("cleanup removed derived clip: %s", cleanup_path)
                if local_path is not None and local_path.exists():
                    local_path.unlink()
                    logger.info("cleanup removed downloaded source copy: %s", local_path)
            else:
                logger.info("cleanup retained source and analysis files (--keep-downloads)")

        results.append(record)

    # Build the output from the SELECTED source rows, then append the
    # experimental columns. No original column is renamed, reordered or dropped,
    # and the source CSV on disk is never written to.
    out_df = df.loc[selected].reset_index(drop=True).copy()
    for column in OUTPUT_COLUMNS:
        out_df[column] = [result[column] for result in results]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False, encoding="utf-8")

    logger.info("=== summary ===")
    logger.info("  selected          : %d", len(selected))
    logger.info("  succeeded         : %d", succeeded)
    logger.info("  failed            : %d", failed)
    logger.info("  output_csv        : %s", output_csv)
    logger.info("  source_csv_untouched: %s", input_csv)
    logger.info("=== end summary ===")

    print(f"succeeded {succeeded}/{len(selected)}; output CSV: {output_csv}; log: {log_path}")
    return 0 if succeeded else 1


if __name__ == "__main__":
    sys.exit(main())
