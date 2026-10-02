#!/usr/bin/env python3
"""Issue #128 real-media benchmark harness for CSC Roihu GH200.

Input is a PRIVATE manifest CSV. Required columns:
  sample_id,country,video_path,reference_transcript,reference_ocr

The script never prints transcript/reference text. Detailed per-sample text stays
in the private results CSV; the Markdown summary contains aggregate metrics only.
"""
from __future__ import annotations

import csv
import os
import statistics
import time
from pathlib import Path

import cv2

from asr_backend import language_hint, load_asr_model
from ep24_video import analysis_start_seconds, prepare_analysis_clip
from ocr_backend import load_ocr_backend

REQUIRED = ("sample_id", "country", "video_path", "reference_transcript", "reference_ocr")
DEFAULT_ASR = ("canary", "parakeet", "qwen3-asr", "faster-whisper")
DEFAULT_OCR = ("paddleocr", "easyocr")


def _tokens(text: str) -> list[str]:
    return " ".join(str(text or "").casefold().split()).split()


def _distance(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        cur = [i]
        for j, y in enumerate(b, start=1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def wer(reference: str, hypothesis: str) -> float:
    ref = _tokens(reference)
    hyp = _tokens(hypothesis)
    return _distance(ref, hyp) / max(1, len(ref))


def cer(reference: str, hypothesis: str) -> float:
    ref = list(" ".join(str(reference or "").casefold().split()))
    hyp = list(" ".join(str(hypothesis or "").casefold().split()))
    return _distance(ref, hyp) / max(1, len(ref))


def gpu_peak_mb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated()) / (1024 * 1024)
    except Exception:
        pass
    return 0.0


def reset_gpu_peak() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def frame_at_analysis_start(video: Path, output: Path) -> Path:
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError(f"cannot open video: {video}")
        capture.set(cv2.CAP_PROP_POS_MSEC, analysis_start_seconds() * 1000.0)
        ok, image = capture.read()
        if not ok or image is None:
            raise ValueError(f"cannot read frame at t={analysis_start_seconds():g}s: {video}")
    finally:
        capture.release()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        raise ValueError(f"cannot write benchmark frame: {output}")
    return output


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    missing = [name for name in REQUIRED if not rows or name not in rows[0]]
    if missing:
        raise ValueError(f"benchmark manifest missing columns: {missing}")
    countries = {r["country"].strip().casefold() for r in rows}
    required_countries = {"finland", "poland", "portugal"}
    if not required_countries.issubset(countries):
        raise ValueError(
            "benchmark must include Finland, Poland and Portugal; "
            f"present={sorted(countries)}"
        )
    return rows


def main() -> int:
    manifest = Path(os.environ["LACLAUGPT_BENCH_MANIFEST"])
    output_root = Path(os.getenv("LACLAUGPT_BENCH_OUTPUT", "./benchmark_issue128"))
    output_root.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(manifest)
    asr_engines = tuple(
        x.strip() for x in os.getenv(
            "LACLAUGPT_BENCH_ASR_ENGINES", ",".join(DEFAULT_ASR)
        ).split(",") if x.strip()
    )
    ocr_engines = tuple(
        x.strip() for x in os.getenv(
            "LACLAUGPT_BENCH_OCR_ENGINES", ",".join(DEFAULT_OCR)
        ).split(",") if x.strip()
    )

    prepared = []
    for row in rows:
        video = Path(row["video_path"]).expanduser()
        if not video.is_file():
            raise FileNotFoundError(video)
        sample_dir = output_root / "derived" / row["sample_id"]
        frame = frame_at_analysis_start(video, sample_dir / "frame_t1.0s.jpg")
        clip = prepare_analysis_clip(video, sample_dir)
        prepared.append((row, video, frame, clip))

    results: list[dict[str, object]] = []

    for engine in asr_engines:
        os.environ["LACLAUGPT_ASR_ENGINE"] = engine
        os.environ.pop("LACLAUGPT_ASR_MODEL", None)
        backend = load_asr_model()
        for row, _video, _frame, clip in prepared:
            reset_gpu_peak()
            started = time.perf_counter()
            result = backend.transcribe(str(clip), language_hint(row["country"]))
            elapsed = time.perf_counter() - started
            results.append({
                "kind": "asr",
                "sample_id": row["sample_id"],
                "country": row["country"],
                "backend": backend.engine,
                "model": backend.model,
                "runtime_seconds": f"{elapsed:.6f}",
                "gpu_peak_mb": f"{gpu_peak_mb():.1f}",
                "wer": f"{wer(row['reference_transcript'], result.transcript):.6f}",
                "cer": "",
                "detected_language": result.language,
                "reference_text": row["reference_transcript"],
                "generated_text": result.transcript,
            })
            print(
                f"ASR sample={row['sample_id']} country={row['country']} "
                f"backend={backend.engine} runtime={elapsed:.2f}s"
            )

    for engine in ocr_engines:
        os.environ["LACLAUGPT_OCR_ENGINE"] = engine
        os.environ.pop("LACLAUGPT_OCR_MODEL", None)
        backend = load_ocr_backend()
        for row, _video, frame, _clip in prepared:
            reset_gpu_peak()
            started = time.perf_counter()
            text, _count = backend.read(str(frame))
            elapsed = time.perf_counter() - started
            results.append({
                "kind": "ocr",
                "sample_id": row["sample_id"],
                "country": row["country"],
                "backend": backend.engine,
                "model": backend.model,
                "runtime_seconds": f"{elapsed:.6f}",
                "gpu_peak_mb": f"{gpu_peak_mb():.1f}",
                "wer": "",
                "cer": f"{cer(row['reference_ocr'], text):.6f}",
                "detected_language": "",
                "reference_text": row["reference_ocr"],
                "generated_text": text,
            })
            print(
                f"OCR sample={row['sample_id']} country={row['country']} "
                f"backend={backend.engine} runtime={elapsed:.2f}s"
            )

    fieldnames = list(results[0])
    result_csv = output_root / "issue128_backend_benchmark.csv"
    with result_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    summary = output_root / "issue128_backend_benchmark.md"
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for result in results:
        key = (str(result["kind"]), str(result["backend"]), str(result["country"]))
        groups.setdefault(key, []).append(result)

    lines = [
        "# Issue #128 Roihu real-media benchmark",
        "",
        f"Samples: {len(rows)}. Analysis skip: {analysis_start_seconds():g}s.",
        "",
        "| kind | backend | country | n | mean error | mean runtime s | max GPU MiB |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for (kind, backend, country), values in sorted(groups.items()):
        metric = "wer" if kind == "asr" else "cer"
        errors = [float(v[metric]) for v in values if str(v[metric])]
        runtimes = [float(v["runtime_seconds"]) for v in values]
        peaks = [float(v["gpu_peak_mb"]) for v in values]
        lines.append(
            f"| {kind} | {backend} | {country} | {len(values)} | "
            f"{statistics.mean(errors):.4f} | {statistics.mean(runtimes):.3f} | "
            f"{max(peaks):.1f} |"
        )
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote private benchmark results: {result_csv}")
    print(f"Wrote aggregate benchmark summary: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
