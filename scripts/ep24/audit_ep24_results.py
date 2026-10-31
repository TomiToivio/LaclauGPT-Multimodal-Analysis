"""Audit an EP24 export artifact set: per-country, per-stage coverage with real denominators.

Reads only the delivered artifacts (``MANIFEST.json``, the country CSVs, the child tables) so the audit
answers a different question from "what does the database say": *what can a consumer of this export
actually see?* Both answers matter and they disagree (see #331), which is why this reports the artifact
view rather than merging them.

    python scripts/ep24/audit_ep24_results.py <export-dir> [--json]

Privacy: the output contains counts only — no transcripts, quotes, identifiers or paths.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

STEPS = [
    ("0_ingest", "processing_status", ["processing_status_reason"]),
    ("1_preprocess", "preprocess_status", ["asr_transcript", "ocr_1", "frame_file"]),
    ("2_frame", "frame_analysis_status", ["frame_analysis_1", "frame_quality_status"]),
    ("3_video", "vllm_structured_status", ["vllm_structured_output", "vllm_video_analysis",
                                           "vllm_video_markdown_analysis"]),
    ("4_summary", "summary_quality_status", ["summary_analysis", "summary_summary_md"]),
    ("5_postprocess", "", ["postprocess_entities", "postprocess_themes"]),
    ("6_discourse", "laclau_status", ["laclau_structured_json", "laclau_summary_md"]),
    ("7_dna", "", ["dna_statements_json"]),
    ("8_sna", "", ["sna_nodes_json", "sna_edges_json"]),
    ("9_rdf", "", ["rdf_ttl_path"]),
]

CHILD_ROLES = {"7_dna": "dna_statements", "8_sna": "sna"}


def _present(row: dict, fields: list[str]) -> bool:
    for field in fields:
        value = row.get(field)
        if value is None:
            continue
        text = str(value).strip()
        if text and text not in ("[]", "{}", "null", "None"):
            return True
    return False


def audit(export_dir: Path) -> dict:
    manifest_path = export_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    countries = sorted(p.stem for p in export_dir.glob("*.csv") if p.stem in
                       {"finland", "poland", "portugal"})
    report: dict = {"export": export_dir.name, "record_unit": manifest.get("record_unit"),
                    "children_unit": manifest.get("children_unit"), "countries": {}}

    for country in countries:
        with (export_dir / f"{country}.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        total = len(rows)
        columns = set(rows[0].keys()) if rows else set()
        stages = {}
        for step, status_field, outputs in STEPS:
            # A step whose columns are absent from this artifact is not "0 % complete" — it is not
            # measurable here at all. Collapsing those two is how an absent stage becomes a null finding.
            absent_columns = not any(field in columns for field in outputs)
            present = 0 if absent_columns else sum(1 for row in rows if _present(row, outputs))
            status_other = 0
            if status_field and status_field in columns:
                status_other = sum(1 for row in rows if (row.get(status_field) or "").strip().lower()
                                   not in ("", "done", "ok", "success", "complete", "completed",
                                           "available"))
            child = 0
            role = CHILD_ROLES.get(step)
            if role:
                path = export_dir / "children" / f"{country}_{role}.csv"
                if path.exists():
                    with path.open(newline="", encoding="utf-8") as handle:
                        child = sum(1 for _ in csv.DictReader(handle))
            stages[step] = {
                "rows_with_output": present,
                "pct": round(100 * present / total, 1) if total and not absent_columns else None,
                "status_other": status_other,
                "child_rows": child,
                "measurable": not absent_columns and (present > 0 or child > 0),
                "absent_columns": absent_columns,
            }
        report["countries"][country] = {"records": total, "stages": stages}
    return report


def render(report: dict) -> str:
    lines = [f"# EP24 artifact audit — {report['export']}", "",
             f"record_unit: {report['record_unit']} · children_unit: {report.get('children_unit')}", ""]
    for country, data in report["countries"].items():
        lines.append(f"## {country} — {data['records']} records")
        lines.append("")
        lines.append("| step | rows with output | % | status not OK | child rows |")
        lines.append("|---|---|---|---|---|")
        for step, stage in data["stages"].items():
            pct = "n/a" if stage["absent_columns"] else f"{stage['pct']}%"
            rows_text = "columns absent" if stage["absent_columns"] else str(stage["rows_with_output"])
            lines.append(f"| {step} | {rows_text} | {pct} | {stage['status_other']} | "
                         f"{stage['child_rows'] or '—'} |")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if not args.export_dir.is_dir():
        print(f"not a directory: {args.export_dir}", file=sys.stderr)
        return 2
    report = audit(args.export_dir)
    print(json.dumps(report, indent=2) if args.json else render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
