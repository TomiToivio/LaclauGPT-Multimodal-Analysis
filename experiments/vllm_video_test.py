#!/usr/bin/env python3
"""Standalone CSC Roihu smoke test: native whole-video analysis with vLLM.

This is an **isolated experiment**, not a production stage. It does not touch the
five-stage EP24 pipeline, `roihu_frame.py`, `roihu_summary.py`, the Ollama
backend, or any legacy CSV contract. It answers one question:

    Can we submit a clean sbatch job on CSC Roihu, download five real EP24
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
import logging
import os
import platform
import random
import shutil
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_SAMPLE_SIZE = 5

# The repository already fixes this object convention; see
# docs/LEGACY_PIPELINE_CONTRACT.md section 1 and roihu_preprocess.py.
DEFAULT_ALLAS_PATH_TEMPLATE = "Scraper/TikTok/Videos/{country}/{author}/{video_id}.mp4"

# Columns needed to derive a remote object path. A row without these cannot be
# fetched and is therefore not "usable" for this test.
REQUIRED_COLUMNS = ("authorUniqueId", "videoId", "scrapedCountry")

# Experimental output columns. Existing columns are never renamed or dropped.
OUTPUT_COLUMNS = (
    "vllm_video_model",
    "vllm_video_status",
    "vllm_video_analysis",
    "vllm_video_error",
    "vllm_video_remote_path",
    "vllm_video_local_path",
    "vllm_video_remote_path_logged",
    "vllm_video_bytes",
    "vllm_video_runtime_seconds",
    "vllm_video_selected_index",
    "vllm_video_prompt",
)

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
    "explicitly rather than filling gaps with plausible invention."
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
    "or are unsure about. Write in English. Do not identify unknown individuals."
)


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
        "%(asctime)sZ %(levelname)-8s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
    )
    # logging's asctime uses local time; the explicit UTC banner below removes
    # any ambiguity about the timezone of the run as a whole.
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def _run_capture(command: list[str]) -> str:
    """Run a helper command and return its output, or a note on why it failed."""
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    except Exception as exc:  # noqa: BLE001 - diagnostics must never be fatal
        return f"<{command[0]} unavailable: {exc}>"
    output = (result.stdout or "") + (result.stderr or "")
    return output.strip() or f"<{command[0]} produced no output>"


def log_environment(logger: logging.Logger) -> None:
    """Record the environment once, up front, so a failure is diagnosable."""
    logger.info("=== environment ===")
    logger.info("utc_now            : %s", datetime.now(timezone.utc).isoformat())
    logger.info("local_now          : %s", datetime.now().isoformat())
    logger.info("hostname           : %s", socket.gethostname())
    logger.info("slurm_job_id       : %s", os.environ.get("SLURM_JOB_ID", "<unset>"))
    logger.info("slurm_job_name     : %s", os.environ.get("SLURM_JOB_NAME", "<unset>"))
    logger.info("slurm_submit_dir   : %s", os.environ.get("SLURM_SUBMIT_DIR", "<unset>"))
    logger.info("python_version     : %s", platform.python_version())
    logger.info("platform           : %s", platform.platform())
    logger.info("machine            : %s", platform.machine())
    logger.info("cwd                : %s", Path.cwd())
    logger.info("gpu_query          : %s", _run_capture([
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,driver_version",
        "--format=csv,noheader",
    ]))
    logger.info("=== end environment ===")


# --------------------------------------------------------------------------- #
# Input CSV and sample selection
# --------------------------------------------------------------------------- #

def load_input_csv(input_csv: Path, logger: logging.Logger) -> pd.DataFrame:
    """Read the EP24 CSV without altering it."""
    if not input_csv.is_file():
        raise SystemExit(f"Input CSV not found: {input_csv}")
    # dtype=str keeps IDs exactly as stored; the pipeline treats them as opaque
    # strings, so casting videoId to a number would corrupt the join.
    df = pd.read_csv(input_csv, dtype=str, keep_default_na=False)
    logger.info("read input CSV %s: %d rows x %d columns", input_csv, len(df), len(df.columns))
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(
            f"Input CSV is missing required column(s) {missing}; "
            f"found columns: {list(df.columns)}"
        )
    return df


def select_sample(df: pd.DataFrame, size: int, seed: int, logger: logging.Logger) -> list[int]:
    """Choose `size` usable rows reproducibly for a given seed.

    "Usable" means every column needed to derive a remote object path is present
    and non-empty. The selection is deterministic: the same seed and the same
    CSV always yield the same rows, in the same order.
    """
    usable = [
        i
        for i, row in df.iterrows()
        if all(str(row[c]).strip() for c in REQUIRED_COLUMNS)
    ]
    logger.info("usable rows for selection: %d of %d", len(usable), len(df))
    if len(usable) < size:
        raise SystemExit(
            f"Requested {size} videos but only {len(usable)} usable row(s) exist "
            f"(need non-empty {list(REQUIRED_COLUMNS)})."
        )
    chosen = sorted(random.Random(seed).sample(usable, size))
    logger.info("selected row indices (seed=%d): %s", seed, chosen)
    return chosen


# --------------------------------------------------------------------------- #
# CSC Allas
# --------------------------------------------------------------------------- #

def derive_remote_path(row: pd.Series, template: str) -> str:
    """Build the object path inside the Allas bucket from EP24 metadata."""
    return template.format(
        country=str(row["scrapedCountry"]).strip(),
        author=str(row["authorUniqueId"]).strip(),
        video_id=str(row["videoId"]).strip(),
    )


def build_rclone_source(remote: str, bucket: str, object_path: str) -> str:
    """Compose an rclone source spec such as `allas:my-bucket/path/to.mp4`."""
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
        source = Path(args.allas_local_root) / object_path
        logger.debug("copying local Allas mirror %s -> %s", source, local_path)
        if not source.is_file():
            raise FileNotFoundError(f"not found in local Allas mirror: {source}")
        shutil.copy2(source, local_path)
        return local_path

    remote_source = build_rclone_source(args.rclone_remote, args.allas_bucket, object_path)
    logger.debug("rclone copyto %s -> %s", remote_source, local_path)
    result = subprocess.run(
        ["rclone", "copyto", remote_source, str(local_path)],
        capture_output=True,
        text=True,
        timeout=args.download_timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"rclone failed for {remote_source}: "
            f"{(result.stderr or result.stdout or '').strip()}"
        )
    if not local_path.is_file():
        raise RuntimeError(f"rclone reported success but {local_path} is absent")
    return local_path


def probe_video_metadata(local_path: Path, logger: logging.Logger) -> dict[str, str]:
    """Basic container metadata when ffprobe is available. Never fatal."""
    if shutil.which("ffprobe") is None:
        return {"probe": "ffprobe not available"}
    output = _run_capture([
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration,codec_name",
        "-of", "json",
        str(local_path),
    ])
    return {"probe": output[:1000]}


# --------------------------------------------------------------------------- #
# vLLM / Qwen3-VL
# --------------------------------------------------------------------------- #

def load_model(args: argparse.Namespace, logger: logging.Logger):
    """Load Qwen3-VL with vLLM. Returns (llm, sampling_params).

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
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )
    logger.info("model loaded: %s", args.model)
    return llm, sampling_params


def build_video_messages(local_path: Path, args: argparse.Namespace) -> list[dict]:
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
                {"type": "text", "text": VIDEO_PROMPT},
            ],
        },
    ]


def prepare_vllm_request(messages: list[dict], processor, logger: logging.Logger) -> dict:
    """Turn chat messages into a vLLM generate input with video mm_data.

    Qwen3-VL wants ``image_patch_size=16`` and ``return_video_metadata=True``;
    `process_vision_info` then returns per-video metadata that must travel with
    the request as ``mm_processor_kwargs``. The observed shape is logged so a
    Roihu/vLLM difference is visible rather than hidden.
    """
    from qwen_vl_utils import process_vision_info

    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    mm_data: dict = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        # Qwen3-VL yields (video, metadata) pairs; split them for vLLM.
        videos, video_metadatas = zip(*video_inputs)
        mm_data["video"] = list(videos)
        video_kwargs = dict(video_kwargs or {})
        video_kwargs["video_metadata"] = list(video_metadatas)

    logger.debug(
        "prepared request: video_parts=%s mm_processor_kwargs_keys=%s",
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


def analyze_one_video(
    local_path: Path,
    args: argparse.Namespace,
    llm,
    sampling_params,
    processor,
    logger: logging.Logger,
) -> str:
    """Run whole-video analysis for one downloaded file and return the text."""
    if args.model_backend == "stub":
        return generate_stub(local_path, args.model)

    messages = build_video_messages(local_path, args)
    request = prepare_vllm_request(messages, processor, logger)
    outputs = llm.generate([request], sampling_params=sampling_params)
    return outputs[0].outputs[0].text


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

    # Sample selection.
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

    if not args.input_csv:
        raise SystemExit("Set --input-csv or LACLAUGPT_VLLM_TEST_INPUT_CSV.")
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv or (input_csv.with_suffix("").as_posix() + "_vllm_test.csv"))
    log_path = Path(args.log_path or "./logs/vllm_video_test.log")
    download_dir = Path(args.download_dir or "./vllm_video_downloads")

    logger = setup_logging(log_path)
    log_environment(logger)
    logger.info("=== configuration ===")
    for key, value in sorted(vars(args).items()):
        logger.info("  %-24s %s", key, value)
    logger.info("input_csv  : %s", input_csv)
    logger.info("output_csv : %s", output_csv)
    logger.info("log_path   : %s", log_path)
    logger.info("=== end configuration ===")

    df = load_input_csv(input_csv, logger)
    selected = select_sample(df, args.sample_size, args.seed, logger)

    # The processor is only needed for real inference; load it with the model so
    # a stub run needs neither transformers nor a GPU.
    llm, sampling_params, processor = None, None, None
    if args.model_backend == "vllm":
        from transformers import AutoProcessor

        logger.info("loading processor for %s", args.model)
        processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
        llm, sampling_params = load_model(args, logger)

    results = []
    succeeded = 0
    failed = 0

    for position, index in enumerate(selected, start=1):
        row = df.loc[index]
        author = str(row["authorUniqueId"])
        video_id = str(row["videoId"])
        object_path = derive_remote_path(row, args.allas_path_template)

        logger.info("--- video %d/%d: %s / %s ---", position, len(selected), author, video_id)
        logger.info("  remote_object     : %s", object_path)
        logger.info("  country           : %s", row.get("scrapedCountry", ""))
        logger.info("  language          : %s", row.get("language", ""))
        logger.info("  description       : %s", str(row.get("videoDescription", ""))[:300])

        record = {column: "" for column in OUTPUT_COLUMNS}
        record["vllm_video_model"] = args.model
        record["vllm_video_selected_index"] = index
        record["vllm_video_remote_path"] = object_path
        record["vllm_video_remote_path_logged"] = object_path
        record["vllm_video_prompt"] = VIDEO_PROMPT

        started = time.monotonic()
        try:
            local_path = fetch_video(object_path, download_dir, args, logger)
            record["vllm_video_local_path"] = str(local_path)
            record["vllm_video_bytes"] = str(local_path.stat().st_size)
            logger.info("  local_path        : %s", local_path)
            logger.info("  bytes             : %s", record["vllm_video_bytes"])
            logger.info("  metadata          : %s", probe_video_metadata(local_path, logger))

            analysis = analyze_one_video(local_path, args, llm, sampling_params, processor, logger)
            record["vllm_video_analysis"] = analysis
            record["vllm_video_status"] = "ok"
            succeeded += 1
            logger.info("  analysis_chars    : %d", len(analysis))
            logger.debug("  raw_response      : %s", analysis)
        except Exception as exc:  # noqa: BLE001 - one bad video must not stop the run
            failed += 1
            record["vllm_video_status"] = "error"
            record["vllm_video_error"] = f"{type(exc).__name__}: {exc}"
            logger.error("  FAILED: %s: %s", type(exc).__name__, exc)
            logger.error("  traceback:\n%s", traceback.format_exc())
        finally:
            record["vllm_video_runtime_seconds"] = f"{time.monotonic() - started:.1f}"
            logger.info("  runtime_seconds   : %s", record["vllm_video_runtime_seconds"])

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
