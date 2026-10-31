#!/usr/bin/env python3
"""Plan a selective EP24 retry: only the records a step actually failed or never produced (#342).

The audit in #333 showed why "rerun everything" is the wrong default: Step 1 reports 100% coverage for
Poland while 639 of 1,781 records failed or are pending, and Discourse produced output for exactly one
Polish record. Rerunning all of it wastes cluster time and buries the real gap; rerunning nothing leaves
the gap. This produces the third option — a per-country, per-step list of the records that need work.

It reuses the exporter's ``COVERAGE_STEPS`` and its ``_has_output`` semantics rather than defining a
second notion of "done". If the contract changes, this planner changes with it.

A record is selected for a step when either:

* the step's status field exists and is not a success status (an explicit failure), or
* none of the step's output fields carries output (the step never produced anything for that record).

Records already holding output are never selected, even if a status field looks odd: output is evidence,
and re-running evidence risks overwriting a good result with a worse one.

Usage:
    python scripts/ep24/plan_ep24_retry.py --export <dir> --out retry_plan.json
    python scripts/ep24/plan_ep24_retry.py --export <dir> --steps 1,6,7 --country poland
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import ep24_mongo_export as exporter

STEPS = tuple(exporter.COVERAGE_STEPS)


def _status_failed(row: dict, status_field: str) -> bool:
    if not status_field:
        return False
    raw = (row.get(status_field) or "").strip().lower()
    if not raw:
        return False
    return raw not in exporter.SUCCESS_STATUSES


def _has_any_output(row: dict, outputs: tuple[str, ...]) -> bool:
    return any(exporter._has_output(row.get(field)) for field in outputs)


def select(rows: list[dict], steps: tuple[str, ...],
           elided: set[str] | None = None) -> dict[str, dict[str, list[str]]]:
    """Record ids needing work, split by *why* they need it.

    ``missing`` — the step produced no output for the record and the export can *prove* that absence.
    Retry these by default.

    ``unverifiable`` — the step's output fields are known to live outside the index CSV (the >1,000
    character cells moved to ``<country>_text.jsonl.gz``, #331/#333). A blank column here is not
    evidence of missing work: Step 3's whole output is elided on the pinned export, so treating the
    blank as absence would schedule all 4,866 records for a rerun that changes nothing. These are
    reported so the operator can decide from live status instead.

    ``failed_with_output`` — a status reports failure but output exists. **Reported, not retried by
    default**: output is evidence, and a blind rerun can overwrite a usable result with a worse one.
    """
    elided = elided or set()
    plan: dict[str, dict[str, list[str]]] = {}
    for step in steps:
        status_field, outputs = exporter.COVERAGE_STEPS[step]
        missing: list[str] = []
        partial: list[str] = []
        unverifiable: list[str] = []
        carries_remotely = bool(set(outputs) & elided)
        for row in rows:
            record_id = row.get("record_id") or row.get("video_id") or ""
            if not record_id:
                continue
            has_output = _has_any_output(row, outputs)
            if not has_output:
                if carries_remotely:
                    unverifiable.append(record_id)
                else:
                    missing.append(record_id)
            elif _status_failed(row, status_field):
                partial.append(record_id)
        plan[step] = {
            "missing": sorted(set(missing)),
            "failed_with_output": sorted(set(partial)),
            "unverifiable": sorted(set(unverifiable)),
        }
    return plan


def elided_fields(export_dir: Path) -> set[str]:
    """Fields the export moved out of the row CSV, from the manifest or from the bundle's schema.

    Reading one line of each bundle is a schema probe, not a corpus read: the field names are what
    matters, and they are the same on every line.
    """
    fields: set[str] = set()
    manifest_path = export_dir / "MANIFEST.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for country_fields in (manifest.get("heavy_fields") or {}).values():
            fields.update(country_fields or {})
    for path in sorted(export_dir.glob("*_text.jsonl.gz")):
        import gzip

        with gzip.open(path, "rt", encoding="utf-8") as handle:
            line = handle.readline()
        if line:
            fields.update(name for name in json.loads(line) if name != "record_id")
    return fields


def plan_for_export(export_dir: Path, steps: tuple[str, ...], countries: list[str] | None) -> dict:
    by_country: dict[str, list[dict]] = defaultdict(list)
    with (export_dir / "combined.csv").open(encoding="utf-8", newline="", errors="ignore") as handle:
        for row in csv.DictReader(handle):
            country = (row.get("country") or "unknown").strip().lower()
            by_country[country].append(row)

    out = {"export": str(export_dir), "steps": list(steps), "countries": {}}
    for country in sorted(by_country):
        if countries and country not in countries:
            continue
        rows = by_country[country]
        plan = select(rows, steps, elided=elided_fields(export_dir))
        out["countries"][country] = {
            "records": len(rows),
            "steps": {
                step: {
                    "missing": len(buckets["missing"]),
                    "unverifiable": len(buckets["unverifiable"]),
                    "failed_with_output": len(buckets["failed_with_output"]),
                    "of": len(rows),
                    "record_ids": buckets["missing"],
                    "unverifiable_ids": buckets["unverifiable"],
                    "failed_with_output_ids": buckets["failed_with_output"],
                }
                for step, buckets in plan.items()
            },
        }
    return out


def summarise(plan: dict) -> str:
    lines = ["# EP24 retry plan (#342)", "", f"Export: `{plan['export']}`", ""]
    lines.append("| Country | Records | " + " | ".join(plan["steps"]) + " |")
    lines.append("|---|---|" + "---|" * len(plan["steps"]))
    for country, data in plan["countries"].items():
        cells = []
        for s in plan["steps"]:
            entry = data["steps"][s]
            cell = str(entry["missing"])
            if entry.get("unverifiable"):
                cell += f" (+{entry['unverifiable']} unverifiable)"
            if entry["failed_with_output"]:
                cell += f" (+{entry['failed_with_output']} partial)"
            cells.append(cell)
        lines.append(f"| {country} | {data['records']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Counts are records needing work. A record is selected when its step status reports failure or",
        "when the step produced no output for it; records that already hold output are never selected.",
        "",
        "`unverifiable` is not work: those steps' cells live in the text bundle (#331/#333), so the",
        "index cannot prove absence. Decide them from live status, not from a blank CSV column.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan a selective EP24 retry")
    parser.add_argument("--export", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--steps", default=None, help="comma-separated step keys, e.g. 1_preprocess,6_discourse")
    parser.add_argument("--country", default=None, help="comma-separated countries to include")
    parser.add_argument("--include-paused", action="store_true",
                        help="also plan the paused Steps 8-9 (#346); they are excluded by default")
    args = parser.parse_args()

    # #346: a normal run ends at Step 7. Steps 8-9 are paused, so a plan must not schedule them by
    # default; asking for them explicitly is how reactivation starts.
    normal = tuple(step for step in STEPS
                   if not step.startswith(("8_", "9_")) or args.include_paused)
    steps = tuple(s.strip() for s in args.steps.split(",")) if args.steps else normal
    unknown = [s for s in steps if s not in exporter.COVERAGE_STEPS]
    if unknown:
        raise SystemExit(f"unknown steps: {unknown}; known: {list(exporter.COVERAGE_STEPS)}")
    countries = [c.strip().lower() for c in args.country.split(",")] if args.country else None

    plan = plan_for_export(args.export, steps, countries)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        print(summarise(plan))
        print(f"\nwrote {args.out}")
    else:
        print(summarise(plan))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
