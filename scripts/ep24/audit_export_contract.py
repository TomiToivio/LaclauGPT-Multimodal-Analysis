#!/usr/bin/env python3
"""Audit a delivered EP24 export against its own manifest (issues #330, #331, #332, #333).

Re-derives the child-table counts, join status, duplicate statement ids and the Step 7 coverage
answer from the files on disk, so a mismatch between MANIFEST.json and the CSVs is a failure, not a
surprise for the next consumer. Needs no database and no research data.

    python scripts/ep24/audit_export_contract.py <export-dir> [--json]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

HEAVY_STEP = "7_dna"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _child_qa(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Child join QA per ``country/role``, under either manifest spelling.

    The delivered manifest states it as ``children_detail`` (``records``/``matched``/``orphan``);
    exports written before that key existed carried ``children_contract.countries`` with
    ``rows``/``joined_rows``/``unjoined_rows``. The auditor re-derives the same four numbers, so it
    reads whichever is present and compares like with like instead of failing on a rename.
    """
    out: dict[str, dict[str, Any]] = {}
    for country, roles in (manifest.get("children_detail") or {}).items():
        for role, entry in (roles or {}).items():
            entry = entry or {}
            out[f"{country}/{role}"] = {
                "rows": entry.get("records"),
                "joined_rows": entry.get("matched"),
                "unjoined_rows": entry.get("orphan"),
                "duplicate_statement_ids": entry.get("duplicate_statement_ids"),
            }
    legacy = ((manifest.get("children_contract") or {}).get("countries") or {})
    for country, roles in legacy.items():
        for role, entry in (roles or {}).items():
            out.setdefault(f"{country}/{role}", dict(entry or {}))
    return out


def _heavy_fields(manifest: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Which CSV cells were elided into the text bundle, per country.

    ``heavy_fields`` is the current statement; ``heavy_field_splits`` wrapped the same counts one
    level deeper. Either way the auditor needs ``country -> field -> rows affected``.
    """
    current = manifest.get("heavy_fields") or {}
    if current:
        return {str(country): {str(f): int(n or 0) for f, n in (fields or {}).items()}
                for country, fields in current.items() if isinstance(fields, dict)}
    legacy = manifest.get("heavy_field_splits") or {}
    return {str(country): {str(f): int(n or 0) for f, n in ((split or {}).get("fields") or {}).items()}
            for country, split in legacy.items()}


def _truthy_count(value: str | None) -> bool:
    try:
        return int(float(value or 0)) > 0
    except ValueError:
        return False


def audit(export_dir: Path) -> dict[str, Any]:
    problems: list[str] = []
    manifest_path = export_dir / "MANIFEST.json"
    if not manifest_path.is_file():
        return {"ok": False, "problems": [f"missing {manifest_path}"], "children": {}}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = _child_qa(manifest)
    children: dict[str, Any] = {}

    parent_rows = _rows(export_dir / "combined.csv") if (export_dir / "combined.csv").is_file() else []
    parent_ids = {str(row.get("record_id") or "") for row in parent_rows}
    counts = manifest.get("children", {})
    for country, roles in counts.items():
        for role, declared in roles.items():
            path = export_dir / "children" / f"{country}_{role}.csv"
            if int(declared) == 0:
                if path.is_file() and _rows(path):
                    problems.append(f"{country}/{role}: manifest says 0 but {path.name} has rows")
                continue
            if not path.is_file():
                problems.append(f"{country}/{role}: manifest declares {declared} rows, file missing")
                continue
            rows = _rows(path)
            entry: dict[str, Any] = {"rows": len(rows), "declared": int(declared)}
            if len(rows) != int(declared):
                problems.append(
                    f"{country}/{role}: manifest {declared} rows, file has {len(rows)}")
            ids = [str(row.get("_storage_id") or "") for row in rows]
            joined = [row for row in rows if str(row.get("parent_record_id") or "")]
            entry.update(
                joined_rows=len(joined),
                unjoined_rows=len(rows) - len(joined),
                duplicate_statement_ids=len([i for i in ids if i]) - len({i for i in ids if i}),
            )
            reported = contract.get(f"{country}/{role}", {})
            for field in ("rows", "joined_rows", "unjoined_rows", "duplicate_statement_ids"):
                # A manifest that does not state this number is silent, not wrong.
                if reported.get(field) is not None and reported[field] != entry[field]:
                    problems.append(
                        f"{country}/{role}: {field} manifest={reported[field]} derived={entry[field]}")
            unknown = sorted({str(row.get("parent_record_id")) for row in joined} - parent_ids)
            if unknown:
                problems.append(
                    f"{country}/{role}: {len(unknown)} parent_record_id(s) absent from combined.csv")
            children[f"{country}/{role}"] = entry

    coverage = manifest.get("stage_coverage", {})
    for country, steps in coverage.items():
        step = steps.get(HEAVY_STEP)
        if not step:
            continue
        country_csv = export_dir / f"{country}.csv"
        if not country_csv.is_file():
            continue
        rows = _rows(country_csv)
        derived = sum(1 for row in rows if _truthy_count(row.get("dna_statement_count")))
        if "records" in step and step["records"] != derived:
            problems.append(
                f"{country}: stage_coverage.{HEAVY_STEP}.records={step['records']} "
                f"but {derived} rows report dna_statement_count > 0")

    for country, fields in _heavy_fields(manifest).items():
        country_csv = export_dir / f"{country}.csv"
        if not country_csv.is_file():
            continue
        rows = _rows(country_csv)
        for field, declared in fields.items():
            if field not in (rows[0].keys() if rows else ()):
                continue
            blank = sum(1 for row in rows if row.get(field, "") in ("", None))
            if declared and blank == 0:
                problems.append(
                    f"{country}: heavy field {field} is blanked in the bundle but not in the CSV")
    return {"ok": not problems, "problems": problems, "children": children}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_dir", type=Path)
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    args = parser.parse_args(argv)
    report = audit(args.export_dir.expanduser().resolve())
    if args.json:
        print(json.dumps(report, indent=2))
    elif report["ok"]:
        print(f"contract OK: {args.export_dir} ({len(report['children'])} child tables)")
    else:
        print(f"contract FAILED: {len(report['problems'])} problem(s) in {args.export_dir}")
        for problem in report["problems"]:
            print("  " + problem)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
