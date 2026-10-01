#!/usr/bin/env python3
"""Standalone Roihu vLLM/Qwen3-VL native-video smoke test.

This experiment is intentionally isolated from the EP24 production pipeline.
It reads an EP24 CSV, selects a reproducible random sample, downloads only those
videos from CSC Allas with rclone, analyzes each video as a temporally ordered
video input with Qwen3-VL through vLLM, and writes a new CSV plus a debug log.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
REMOTE_COLUMNS = (
    "allas_path",
    "allasPath",
    "video_path",
    "videoPath",
    "video_file",
    "videoFile",
    "object_path",
    "objectPath",
)

VIDEO_PROMPT = """Describe this whole social-media video in depth, preserving its temporal narrative.

Stay descriptive and evidence-focused. Do NOT perform political, ideological,
partisan, populism, sentiment, Laclau, DNA, or SNA analysis.

Where observable, cover:
- major scenes and scene changes;
- actions and events over time;
- people/participants without guessing unknown identities;
- spoken or visible text when perceptible;
- gestures, expressions, and interactions;
- camera movement, cuts, editing, and transitions;
- graphics, captions, memes, screenshots, logos, symbols, and interface elements;
- temporal relationships between events;
- ambiguity, uncertainty, illegible text, and details that cannot be established.

End with a concise beginning -> middle -> end narrative description. Distinguish
direct observation from uncertain interpretation.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--tmp-dir", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument(
        "--remote-column",
        default=os.getenv("LACLAUGPT_VLLM_VIDEO_REMOTE_COLUMN", ""),
        help="CSV column containing the Allas object path/URI.",
    )
    parser.add_argument(
        "--remote-template",
        default=os.getenv("LACLAUGPT_VLLM_VIDEO_REMOTE_TEMPLATE", ""),
        help=(
            "Python format template built from row fields, e.g. "
            "'ep24/{scrapedCountry}/{authorUniqueId}/{videoId}.mp4'."
        ),
    )
    parser.add_argument(
        "--allas-root",
        default=os.getenv("LACLAUGPT_VLLM_ALLAS_ROOT", "s3allas:"),
        help="rclone Allas root/remote, e.g. s3allas:bucket/prefix",
    )
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--keep-videos", action="store_true")
    return parser.parse_args()


def setup_logging(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("vllm_video_test")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def run_capture(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return (result.stdout + result.stderr).strip()
    except Exception as exc:
        return f"<failed: {exc}>"


def log_environment(logger: logging.Logger, args: argparse.Namespace) -> None:
    logger.info("Standalone vLLM native-video smoke test starting")
    logger.info("hostname=%s", socket.gethostname())
    logger.info("platform=%s", platform.platform())
    logger.info("python=%s", sys.version.replace("\n", " "))
    logger.info("slurm_job_id=%s", os.getenv("SLURM_JOB_ID", ""))
    logger.info("slurm_submit_dir=%s", os.getenv("SLURM_SUBMIT_DIR", ""))
    logger.info("cuda_visible_devices=%s", os.getenv("CUDA_VISIBLE_DEVICES", ""))
    logger.info("model=%s", args.model)
    logger.info("input_csv=%s", args.input_csv)
    logger.info("output_csv=%s", args.output_csv)
    logger.info("tmp_dir=%s", args.tmp_dir)
    logger.info("seed=%d sample_size=%d fps=%s", args.seed, args.sample_size, args.fps)
    logger.info("allas_root=%s remote_column=%s", args.allas_root, args.remote_column)
    logger.info("remote_template=%s", args.remote_template or "<unset>")
    logger.debug("nvidia-smi:\n%s", run_capture(["nvidia-smi"]))
    logger.debug("ffprobe version:\n%s", run_capture(["ffprobe", "-version"]))

    try:
        import torch

        logger.info("torch=%s cuda=%s", torch.__version__, torch.version.cuda)
    except Exception:
        logger.exception("Could not inspect PyTorch")

    try:
        import vllm

        logger.info("vllm=%s", getattr(vllm, "__version__", "unknown"))
    except Exception:
        logger.exception("Could not inspect vLLM")


def clean_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def normalize_remote(value: str, allas_root: str) -> str:
    value = value.strip()
    if not value:
        return ""
    if value.startswith("s3://"):
        value = value[5:]
    if ":" in value.split("/", 1)[0]:
        return value
    return f"{allas_root.rstrip('/')}/{value.lstrip('/')}"


def resolve_remote(row: pd.Series, args: argparse.Namespace) -> str:
    if args.remote_column:
        if args.remote_column not in row.index:
            return ""
        return normalize_remote(clean_value(row[args.remote_column]), args.allas_root)

    for column in REMOTE_COLUMNS:
        if column in row.index:
            value = clean_value(row[column])
            if value:
                return normalize_remote(value, args.allas_root)

    if args.remote_template:
        values = {key: clean_value(value) for key, value in row.items()}
        try:
            rendered = args.remote_template.format_map(values)
        except KeyError:
            return ""
        return normalize_remote(rendered, args.allas_root)

    return ""


def local_filename(row: pd.Series, remote: str, position: int) -> str:
    video_id = clean_value(row.get("videoId", "")) or f"sample_{position}"
    suffix = Path(remote.split("?", 1)[0]).suffix or ".mp4"
    safe_id = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in video_id)
    return f"{position:02d}_{safe_id}{suffix}"


def download_video(remote: str, local_path: Path, logger: logging.Logger) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    command = ["rclone", "copyto", "--no-traverse", remote, str(local_path)]
    logger.info("Allas download: %s -> %s", remote, local_path)
    logger.debug("download command=%s", command)
    subprocess.run(command, check=True)


def video_metadata(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration,size,format_name:stream=index,codec_name,codec_type,width,height,r_frame_rate",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
        return json.loads(result.stdout)
    except Exception as exc:
        return {"ffprobe_error": str(exc), "size_bytes": path.stat().st_size}


def initialize_model(args: argparse.Namespace, logger: logging.Logger):
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    logger.info("Loading processor: %s", args.model)
    processor = AutoProcessor.from_pretrained(args.model)

    logger.info("Initializing vLLM model: %s", args.model)
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        limit_mm_per_prompt={"video": 1},
        seed=args.seed,
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
        top_k=-1,
    )
    logger.info("Model initialization complete")
    return processor, llm, sampling


def prepare_video_input(
    path: Path,
    processor: Any,
    fps: float,
    logger: logging.Logger,
) -> dict[str, Any]:
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(path),
                    "fps": fps,
                },
                {"type": "text", "text": VIDEO_PROMPT},
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    multimodal_data: dict[str, Any] = {}
    if image_inputs is not None:
        multimodal_data["image"] = image_inputs
    if video_inputs is not None:
        multimodal_data["video"] = video_inputs

    logger.debug("prompt=%s", VIDEO_PROMPT)
    logger.debug("qwen video kwargs=%s", video_kwargs)
    return {
        "prompt": prompt,
        "multi_modal_data": multimodal_data,
        "mm_processor_kwargs": video_kwargs,
    }


def remote_exists(remote: str, logger: logging.Logger) -> bool:
    """Check that an Allas object exists without downloading it."""
    command = ["rclone", "lsjson", "--stat", remote]
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception:
        logger.debug("remote stat failed for %s\n%s", remote, traceback.format_exc())
        return False

    if result.returncode == 0:
        logger.debug("remote exists: %s", remote)
        return True

    logger.debug(
        "remote missing/unavailable: %s returncode=%s stderr=%s",
        remote,
        result.returncode,
        result.stderr.strip(),
    )
    return False


def select_usable_rows(
    df: pd.DataFrame,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Choose a reproducible random sample of rows whose Allas objects exist."""
    resolvable_mask = df.apply(lambda row: bool(resolve_remote(row, args)), axis=1)
    candidates = df.loc[resolvable_mask].copy()
    logger.info(
        "rows_total=%d rows_with_resolvable_video=%d",
        len(df),
        len(candidates),
    )

    if len(candidates) < args.sample_size:
        raise RuntimeError(
            f"Need exactly {args.sample_size} resolvable rows, found {len(candidates)}. "
            "Set --remote-column or --remote-template to match the EP24 CSV."
        )

    randomized = candidates.sample(frac=1.0, random_state=args.seed)
    selected_indices: list[Any] = []
    for index, row in randomized.iterrows():
        remote = resolve_remote(row, args)
        if remote_exists(remote, logger):
            selected_indices.append(index)
            logger.info(
                "usable remote %d/%d: index=%s remote=%s",
                len(selected_indices),
                args.sample_size,
                index,
                remote,
            )
        if len(selected_indices) == args.sample_size:
            break

    if len(selected_indices) != args.sample_size:
        raise RuntimeError(
            f"Could not find exactly {args.sample_size} existing Allas video objects; "
            f"found {len(selected_indices)} after checking {len(candidates)} resolvable rows."
        )

    return df.loc[selected_indices].copy()


def main() -> int:
    args = parse_args()
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")

    log_path = Path(args.log).expanduser().resolve()
    logger = setup_logging(log_path)
    log_environment(logger, args)

    input_path = Path(args.input_csv).expanduser().resolve()
    output_path = Path(args.output_csv).expanduser().resolve()
    tmp_dir = Path(args.tmp_dir).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Reading EP24 CSV")
    source = pd.read_csv(input_path)
    selected = select_usable_rows(source, args, logger)
    selected["vllm_video_model"] = args.model
    selected["vllm_video_analysis"] = ""
    selected["vllm_video_status"] = "pending"
    selected["vllm_video_error"] = ""
    selected["vllm_video_remote_path"] = ""
    selected["vllm_video_local_path"] = ""
    selected["vllm_video_runtime_seconds"] = pd.NA
    selected["vllm_video_metadata"] = ""

    id_columns = [c for c in ("authorUniqueId", "videoId", "scrapedCountry", "language") if c in selected.columns]
    logger.info(
        "selected rows=%s",
        selected[id_columns].to_dict(orient="records") if id_columns else list(selected.index),
    )

    processor, llm, sampling = initialize_model(args, logger)
    successes = 0
    failures = 0

    for position, (index, row) in enumerate(selected.iterrows(), start=1):
        started = time.monotonic()
        remote = resolve_remote(row, args)
        local_path = tmp_dir / local_filename(row, remote, position)
        logger.info(
            "video %d/%d start index=%s author=%s videoId=%s",
            position,
            args.sample_size,
            index,
            clean_value(row.get("authorUniqueId", "")),
            clean_value(row.get("videoId", "")),
        )
        selected.at[index, "vllm_video_remote_path"] = remote
        selected.at[index, "vllm_video_local_path"] = str(local_path)

        try:
            download_video(remote, local_path, logger)
            size_bytes = local_path.stat().st_size
            logger.info("downloaded size_bytes=%d", size_bytes)
            metadata = video_metadata(local_path)
            selected.at[index, "vllm_video_metadata"] = json.dumps(metadata, ensure_ascii=False)
            logger.debug("video metadata=%s", metadata)

            request = prepare_video_input(local_path, processor, args.fps, logger)
            logger.info("vLLM generation start")
            outputs = llm.generate([request], sampling_params=sampling)
            response = outputs[0].outputs[0].text.strip()
            logger.debug("raw model response=%s", response)

            selected.at[index, "vllm_video_analysis"] = response
            selected.at[index, "vllm_video_status"] = "ok"
            successes += 1
        except Exception as exc:
            failures += 1
            error = f"{type(exc).__name__}: {exc}"
            selected.at[index, "vllm_video_status"] = "error"
            selected.at[index, "vllm_video_error"] = error
            logger.error("video failed: %s", error)
            logger.debug("traceback:\n%s", traceback.format_exc())
        finally:
            runtime = time.monotonic() - started
            selected.at[index, "vllm_video_runtime_seconds"] = round(runtime, 3)
            logger.info("video %d/%d end runtime_seconds=%.3f", position, args.sample_size, runtime)
            selected.to_csv(output_path, index=False)
            logger.info("checkpoint CSV written: %s", output_path)

            if local_path.exists() and not args.keep_videos:
                try:
                    local_path.unlink()
                    logger.debug("deleted temporary video: %s", local_path)
                    selected.at[index, "vllm_video_local_path"] = f"{local_path} (deleted after inference)"
                    selected.to_csv(output_path, index=False)
                except Exception:
                    logger.exception("Could not delete temporary video: %s", local_path)

    logger.info(
        "finished successes=%d failures=%d output_csv=%s debug_log=%s",
        successes,
        failures,
        output_path,
        log_path,
    )
    if failures:
        logger.warning("Smoke test completed with per-video failures; inspect status/error columns and debug log.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
