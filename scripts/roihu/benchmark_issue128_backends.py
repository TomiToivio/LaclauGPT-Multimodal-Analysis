#!/usr/bin/env python3
"""Issue #128 real-media benchmark harness for CSC Roihu GH200.

Input is a PRIVATE manifest CSV. Required columns:
  sample_id,country,video_path,reference_transcript,reference_transcript_provenance,
  reference_ocr,reference_ocr_provenance

Any non-empty accuracy reference must be explicitly human/researcher/manual/gold
verified. Model-generated text (including old Whisper output) is not ground truth.

The script never prints transcript/reference text. Detailed per-sample text stays
in the private results CSV; the Markdown summary contains aggregate metrics only.
"""
from __future__ import annotations

import csv
import os
import statistics
import sys
import time
from pathlib import Path

import cv2

# This script lives in scripts/roihu/ but imports the pipeline modules that sit at
# the repository root. Python puts the SCRIPT's directory on sys.path (not the
# cwd), so without this the job dies on the first import no matter where it is
# launched from -- verified by running it as the sbatch does.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asr_backend import language_hint, load_asr_model
from ep24_video import analysis_start_seconds, prepare_analysis_clip
from ocr_backend import load_ocr_backend

REQUIRED = (
    "sample_id",
    "country",
    "video_path",
    "reference_transcript",
    "reference_transcript_provenance",
    "reference_ocr",
    "reference_ocr_provenance",
)
HUMAN_REFERENCE_PREFIXES = ("human", "researcher", "manual", "gold")
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


def wer(reference: str, hypothesis: str) -> float | None:
    """Word error rate, or None when there is no reference to compare against.

    An empty reference must not silently become a number: dividing the edit
    distance by ``max(1, len(ref))`` turns a missing reference into the
    hypothesis length, which then enters the aggregate as if it were a very
    wrong transcript. A missing reference means the metric is undefined.
    """
    ref = _tokens(reference)
    if not ref:
        return None
    return _distance(ref, _tokens(hypothesis)) / len(ref)


def cer(reference: str, hypothesis: str) -> float | None:
    """Character error rate, or None when there is no reference."""
    ref = list(" ".join(str(reference or "").casefold().split()))
    if not ref:
        return None
    return _distance(ref, list(" ".join(str(hypothesis or "").casefold().split()))) / len(ref)


def _metric(value: float | None) -> str:
    """Serialise a metric, keeping "no reference" distinguishable from 0.0."""
    return "" if value is None else f"{value:.6f}"


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


def _require_human_reference(
    row: dict[str, str],
    *,
    reference_field: str,
    provenance_field: str,
) -> None:
    """Reject pseudo-ground-truth before any expensive model load.

    #128 selects a production backend by measured accuracy. Earlier EP24
    "reference" transcripts were themselves Whisper outputs, so scoring a new
    ASR against them rewards reproducing Whisper rather than matching speech.
    Empty references remain allowed (the group is reported unscored), but a
    non-empty reference must carry explicit human/researcher/manual/gold
    provenance.
    """
    reference = str(row.get(reference_field, "") or "").strip()
    if not reference:
        return
    provenance = str(row.get(provenance_field, "") or "").strip().casefold()
    normalized = provenance.replace("-", "_").replace(" ", "_")
    if not any(normalized.startswith(prefix) for prefix in HUMAN_REFERENCE_PREFIXES):
        sample_id = str(row.get("sample_id", "") or "<unknown>")
        raise ValueError(
            f"sample {sample_id}: non-empty {reference_field} requires human-verified "
            f"{provenance_field}; got {provenance!r}. Model-generated references "
            "(including Whisper output) must not be used to rank #128 backends."
        )


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    missing = [name for name in REQUIRED if not rows or name not in rows[0]]
    if missing:
        raise ValueError(f"benchmark manifest missing columns: {missing}")
    for row in rows:
        _require_human_reference(
            row,
            reference_field="reference_transcript",
            provenance_field="reference_transcript_provenance",
        )
        _require_human_reference(
            row,
            reference_field="reference_ocr",
            provenance_field="reference_ocr_provenance",
        )
    countries = {r["country"].strip().casefold() for r in rows}
    required_countries = {"finland", "poland", "portugal"}
    if not required_countries.issubset(countries):
        raise ValueError(
            "benchmark must include Finland, Poland and Portugal; "
            f"present={sorted(countries)}"
        )
    return rows


def render_summary(sample_count: int, results: list[dict[str, object]]) -> list[str]:
    """Aggregate per-sample results into the Markdown summary lines.

    Split out of ``main`` so the aggregation can be tested without loading any
    model: a country with no reference material must still appear, must be
    reported as unscored, and must not crash the run.
    """
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for result in results:
        key = (str(result["kind"]), str(result["backend"]), str(result["country"]))
        groups.setdefault(key, []).append(result)

    lines = [
        "# Issue #128 Roihu real-media benchmark",
        "",
        f"Samples: {sample_count}. Analysis skip: {analysis_start_seconds():g}s.",
        "",
        "| kind | backend | country | n | n scored | mean error | mean runtime s | max GPU MiB |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for (kind, backend, country), values in sorted(groups.items()):
        metric = "wer" if kind == "asr" else "cer"
        # A country can legitimately have no reference (no ground truth exists for
        # it yet). Report the group, but never let a missing reference contribute a
        # fabricated error value, and never crash the whole run on `mean([])`.
        errors = [float(str(v[metric])) for v in values if v[metric] not in ("", None)]
        runtimes = [float(str(v["runtime_seconds"])) for v in values]
        peaks = [float(str(v["gpu_peak_mb"])) for v in values]
        mean_error = f"{statistics.mean(errors):.4f}" if errors else "n/a"
        lines.append(
            f"| {kind} | {backend} | {country} | {len(values)} | {len(errors)} | "
            f"{mean_error} | {statistics.mean(runtimes):.3f} | {max(peaks):.1f} |"
        )
    return lines


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
                "wer": _metric(wer(row["reference_transcript"], result.transcript)),
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
                "cer": _metric(cer(row["reference_ocr"], text)),
                "detected_language": "",
                "reference_text": row["reference_ocr"],
                "generated_text": text,
            })
            print(
                f"OCR sample={row['sample_id']} country={row['country']} "
                f"backend={backend.engine} runtime={elapsed:.2f}s"
            )

    fieldnames = [
        "kind", "sample_id", "country", "backend", "model", "runtime_seconds",
        "gpu_peak_mb", "wer", "cer", "detected_language", "reference_text",
        "generated_text",
    ]
    result_csv = output_root / "issue128_backend_benchmark.csv"
    with result_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    summary = output_root / "issue128_backend_benchmark.md"
    lines = render_summary(len(rows), results)
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote private benchmark results: {result_csv}")
    print(f"Wrote aggregate benchmark summary: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
