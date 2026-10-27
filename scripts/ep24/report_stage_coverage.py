#!/usr/bin/env python3
"""Country x step 0-9 coverage report for an EP24 Mongo export (#306, deliverable 2).

Reads the exporter's per-country CSV (or an export directory) and reports, for each
country and each pipeline step, how many records carry substantive output, how many
carry an error, and how many are simply absent. Denominators are the country's record
count, printed with every figure so a share is never read without its base.

The distinction this report exists to make is the one the issue asks for:
**None/null vs not-started vs failed vs completed** are four different states, and a
single "has a value" test collapses all four into one misleading number.

Step 9 (RDF) is not a column in the canonical CSV; the report says so explicitly
rather than reporting zero coverage, which would read as a pipeline failure.

Usage:
    python scripts/ep24/report_stage_coverage.py --export-dir DIR [--json OUT] [--md OUT]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

COUNTRIES = ("finland", "poland", "portugal")

#: step -> (label, status columns, content columns). A step is substantive when a
#: content column is non-empty; the status column explains a non-substantive row.
STEPS: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {
    "1": ("preprocess / ASR / OCR", ("preprocess_status",), ("asr_transcript", "ocr_1")),
    "2": ("frames", ("frame_analysis_status", "frame_quality_status"), ("frame_analysis_1", "frame_file")),
    "3": ("video analysis", ("vllm_video_status", "vllm_structured_status"), ("vllm_video_analysis", "vllm_structured_output")),
    "4": ("summary", ("summary_quality_status", "summary_quality_reason"), ("summary_analysis", "summary_summary_md")),
    "5": ("postprocess", (), ("postprocess_summary_md", "postprocess_entities", "postprocess_themes")),
    "6": ("discourse", ("laclau_status",), ("laclau_structured_json", "laclau_summary_md")),
    "7": ("DNA", ("dna_status",), ("dna_statements_json", "dna_analysis_markdown")),
    "8": ("SNA", ("sna_status",), ("sna_nodes_json", "sna_report_markdown")),
}

ABSENT = {"", "none", "null", "nan", "pending", "not_started", "not started", "missing"}
ERRORISH = ("error", "fail")


def _blank(value: str | None) -> bool:
    if value is None or value.strip().lower() in ABSENT:
        return True
    # Serialized empty containers/nulls are not substantive analysis results.
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        return False
    return parsed is None or parsed == [] or parsed == {} or parsed == ""


def _errored(status: str | None) -> bool:
    if status is None:
        return False
    lowered = status.strip().lower()
    return any(token in lowered for token in ERRORISH)


def report_country(path: Path) -> dict[str, Any]:
    """Per-step coverage from the canonical CSV.

    Two different measurements are kept apart because they disagree for real reasons.
    ``status_present`` counts rows where a step's status column carries a value, which is
    the pipeline's own statement that it ran. ``text_present`` counts rows where the
    projected text column is non-empty - and the CSV projection drops the largest text
    fields, so a text-bearing step can read 0% here while the ``*_text.jsonl.gz`` bundle
    is full. Reporting the second as if it were the first would invent a pipeline failure.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    total = len(rows)
    steps: dict[str, dict[str, Any]] = {}
    for step, (label, status_columns, content_columns) in STEPS.items():
        status_present = text_present = errored = 0
        for row in rows:
            is_error = any(_errored(row.get(col)) for col in status_columns)
            if is_error:
                errored += 1
            has_status = any(not _blank(row.get(col)) for col in status_columns) and not is_error
            has_text = any(col in row and not _blank(row.get(col)) for col in content_columns)
            if has_status:
                status_present += 1
            if has_text:
                text_present += 1
        if not status_columns:
            # A step with no status column is measured by its text alone; saying it
            # "never ran" because it has no status field would be a category error.
            status_present = text_present
        steps[step] = {
            "label": label,
            "records": total,
            "status_present": status_present,
            "measured_by": "status" if status_columns else "text",
            "text_present": text_present,
            "errored": errored,
            "absent": total - status_present - errored,
            "share": round(status_present / total, 4) if total else None,
        }

    steps["9"] = {
        "label": "RDF",
        "records": total,
        "status_present": None,
        "text_present": None,
        "errored": None,
        "absent": None,
        "share": None,
        "note": "not carried by the canonical CSV; see the rdf child collection",
    }

    identity = sum(1 for row in rows if not _blank(row.get("record_id")))
    return {
        "records": total,
        "with_record_id": identity,
        "without_record_id": total - identity,
        "steps": steps,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# EP24 step coverage by country", "",
             "Denominators are the country's canonical record count. Absent, errored and",
             "projected text are measured separately. Status presence does not establish",
             "completion; projected text presence does not validate analytic quality.",
             "Missing CSV columns are unobserved, not evidence of missing source results.", ""]
    for country, data in report.items():
        lines.append(f"## {country} — {data['records']} records "
                     f"({data['with_record_id']} with record_id)")
        lines.append("")
        lines.append("| step | surface | status present | text projected | errored | absent | share |")
        lines.append("|---|---|---|---|---|---|---|")
        for step in sorted(data["steps"], key=int):
            s = data["steps"][step]
            if s.get("status_present") is None:
                lines.append(f"| {step} | {s['label']} | — | — | — | — | {s.get('note','')} |")
            else:
                share = "—" if s["share"] is None else f"{s['share']:.1%}"
                lines.append(f"| {step} | {s['label']} | {s['status_present']} | {s['text_present']} "
                             f"| {s['errored']} | {s['absent']} | {share} |")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", required=True, type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--md", type=Path)
    args = parser.parse_args(argv)

    report: dict[str, Any] = {}
    for country in COUNTRIES:
        path = args.export_dir / f"{country}.csv"
        if not path.exists():
            continue
        report[country] = report_country(path)

    if not report:
        raise SystemExit(f"no per-country CSV found in {args.export_dir}")

    markdown = render_markdown(report)
    print(markdown)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.md:
        args.md.write_text(markdown, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
