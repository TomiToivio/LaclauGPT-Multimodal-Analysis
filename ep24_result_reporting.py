"""Human-readable, private, per-stage analysis reporting.

Stage CSVs are the authoritative full-fidelity artifacts. This companion log
prints only generated result fields, not private input records or prompt context.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

LOG = logging.getLogger("ep24.results")
OUTPUT_PREFIXES = {
    1: ("ocr_1", "asr_transcript", "asr_translated", "preprocess_status"),
    2: ("frame_analysis", "frame_response", "frame_quality"),
    3: ("vllm_video_analysis", "vllm_video_markdown_analysis", "vllm_video_status"),
    4: ("summary_analysis", "summary_",),
    5: ("postprocess_", "positive", "neutral", "negative"),
    6: ("laclau_",),
    7: ("dna_",),
    8: ("sna_",),
    9: ("rdf_",),
}

def report_stage_rows(step: int, country: str, frame: pd.DataFrame, csv_path: str | Path) -> None:
    """Emit completed model/results fields to stdout and a private per-stage log.

    Never print all dataframe fields; those include original research inputs.
    Control with LACLAUGPT_PRINT_ANALYSIS=0 or LACLAUGPT_ANALYSIS_MAX_CHARS.
    """
    if os.getenv("LACLAUGPT_PRINT_ANALYSIS", "1").lower() in {"0", "false", "no"}:
        return
    maximum = max(100, int(os.getenv("LACLAUGPT_ANALYSIS_MAX_CHARS", "12000")))
    prefixes = OUTPUT_PREFIXES.get(step, ())
    selected = [name for name in frame.columns if any(name.startswith(p) for p in prefixes)]
    lines = [f"=== EP24 step={step} country={country} rows={len(frame)} csv={csv_path} ==="]
    for index, row in frame.iterrows():
        lines.append(f"--- result row={index} ---")
        for column in selected:
            value = str(row[column] if pd.notna(row[column]) else "").strip()
            if value:
                lines.append(f"{column}: {value[:maximum]}{' [TRUNCATED; FULL CSV RETAINS VALUE]' if len(value)>maximum else ''}")
    lines.append(f"=== EP24 step={step} end ===")
    content = "\n".join(lines) + "\n"
    log_dir = Path(os.getenv("LACLAUGPT_ANALYSIS_LOG_DIR") or
                   Path(os.getenv("LACLAUGPT_MULTIMODAL_PRIVATE_ROOT", ".")) / "logs" / "analysis")
    log_dir.mkdir(parents=True, exist_ok=True)
    with (log_dir / f"step_{step:02d}_{country}.log").open("a", encoding="utf-8") as handle:
        handle.write(content)
    print(content, flush=True)

def main(argv: list[str] | None = None) -> int:
    import argparse
    from ep24_cli import checkpoint_path, normalize_country
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("-c", "--country")
    parser.add_argument("-n", "--limit", type=int)
    args, _ = parser.parse_known_args(argv)
    country = normalize_country(args.country or os.getenv("LACLAUGPT_COUNTRY"))
    if not country:
        print("EP24 reporting: no country supplied; cumulative stage CSV remains available", flush=True)
        return 0
    root = Path(os.getenv("LACLAUGPT_EP24_OUTPUT_ROOT") or
                Path(os.getenv("LACLAUGPT_MULTIMODAL_PRIVATE_ROOT", ".")) / "outputs")
    path = Path(os.getenv("LACLAUGPT_OUTPUT_CSV") or checkpoint_path(args.step, country, root))
    if not path.exists():
        print(f"EP24 reporting: no stage CSV at {path}", flush=True)
        return 0
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    if args.limit and args.limit > 0:
        frame = frame.head(args.limit)
    report_stage_rows(args.step, country, frame, path)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
