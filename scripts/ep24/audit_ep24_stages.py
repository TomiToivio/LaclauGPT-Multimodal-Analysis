#!/usr/bin/env python3
"""Audit a pinned EP24 export into a per-country, per-stage coverage matrix (#333).

This does not re-implement the coverage rules. It calls the exporter's own ``stage_coverage()`` so the
audit cannot disagree with the export it is auditing — a second opinion would be a second contract, and
the whole point of #332 was that one contract was already one too many.

What is added on top of the exporter's numbers:

* **per country**, because a pooled percentage hides that Poland has no SNA at all;
* **child-table counts**, which live outside the row frame;
* **the grain check** — manifest child statements vs the sum of the row-level counters. When those two
  disagree the export is holding two different populations (ingest vs video), and saying so is more
  useful than picking one.

Usage:
    python scripts/ep24/audit_ep24_stages.py --export /path/to/export --out docs/EP24_STAGE_AUDIT.md
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import ep24_mongo_export as exporter

STAGE_LABELS = {
    "0_ingest": "Ingest",
    "1_preprocess": "Preprocess",
    "2_frame": "Frame analysis",
    "3_video": "Video analysis",
    "4_summary": "Summary",
    "5_postprocess": "Postprocess",
    "6_discourse": "Discourse",
    "7_dna": "DNA statements",
    "8_sna": "Social network",
    "9_rdf": "RDF export",
}

CHILD_TABLES = ("dna_statements", "entities", "sna", "themes")


def read_rows(export_dir: Path) -> dict[str, list[dict]]:
    """Rows grouped by country, from the canonical frame."""
    path = export_dir / "combined.csv"
    if not path.is_file():
        raise SystemExit(f"no combined.csv in {export_dir}")
    by_country: dict[str, list[dict]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="", errors="ignore") as handle:
        for row in csv.DictReader(handle):
            country = (row.get("country") or "unknown").strip().lower()
            by_country[country].append(row)
    return dict(by_country)


def child_counts(export_dir: Path) -> dict[str, dict[str, int]]:
    """Row counts per country for each child table, read from the files themselves."""
    out: dict[str, dict[str, int]] = defaultdict(dict)
    children = export_dir / "children"
    if not children.is_dir():
        return {}
    for table in CHILD_TABLES:
        for path in sorted(children.glob(f"*_{table}.csv")):
            country = path.stem[: -(len(table) + 1)]
            with path.open(encoding="utf-8", newline="", errors="ignore") as handle:
                out[country][table] = sum(1 for _ in csv.DictReader(handle))
    return dict(out)


def manifest_children(export_dir: Path) -> dict:
    path = export_dir / "MANIFEST.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("children", {})
    except json.JSONDecodeError:
        return {}


def grain_check(rows: list[dict], manifest: dict, country: str) -> str:
    """Compare the manifest's child count against the frame's own counter."""
    declared = manifest.get(country, {}).get("dna_statements")
    if declared is None:
        return "—"
    counted = sum(int(row.get("dna_statement_count") or 0) for row in rows)
    if counted == declared:
        return f"**{declared}** (frame and children agree)"
    return f"**{declared}** in children vs **{counted}** in frame — different grains, do not mix"


def build_report(export_dir: Path) -> str:
    by_country = read_rows(export_dir)
    children = child_counts(export_dir)
    manifest = manifest_children(export_dir)

    lines = [
        "# EP24 stage audit (#333)",
        "",
        f"**Export:** `{export_dir}`  ",
        "**Method:** the exporter's own `stage_coverage()`, per country. No second opinion.",
        "",
        "## Records",
        "",
        "| Country | Rows in frame | " + " | ".join(CHILD_TABLES) + " |",
        "|---|---|" + "---|" * len(CHILD_TABLES),
    ]
    for country in sorted(by_country):
        counts = children.get(country, {})
        lines.append(
            f"| {country} | {len(by_country[country])} | "
            + " | ".join(str(counts.get(t, 0)) for t in CHILD_TABLES)
            + " |"
        )

    for country in sorted(by_country):
        rows = by_country[country]
        coverage = exporter.stage_coverage(rows)
        lines += ["", f"## {country} ({len(rows)} records)", "",
                  "| Step | Stage | With output | % | Failed / other | Available |",
                  "|---|---|---|---|---|---|"]
        for step, label in STAGE_LABELS.items():
            entry = coverage.get(step, {})
            lines.append(
                f"| {step.split('_')[0]} | {label} | {entry.get('records', 0)} | "
                f"{entry.get('pct_of_exported', 0.0)}% | {entry.get('failed_or_other', 0)} | "
                f"{'yes' if entry.get('available') else '**no**'} |"
            )
        lines += ["", f"**DNA grain:** {grain_check(rows, manifest, country)}", ""]

    lines += [
        "## How to read this",
        "",
        "* **Available = no** means no row carries any output field for that step: the export cannot express",
        "  it. That is different from `0` with output, which means the step ran and produced nothing.",
        "* Child-table counts are file rows, not collapsed records. They are shown separately on purpose.",
        "* The grain line exists because a manifest number and a frame number can both be true and still",
        "  describe different populations (#331).",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit an EP24 export per country and stage")
    parser.add_argument("--export", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = build_report(args.export)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
