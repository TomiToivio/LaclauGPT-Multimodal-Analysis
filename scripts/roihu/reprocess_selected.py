"""Selective EP24 reprocessing on Roihu (#342).

Preview first, always. The default is a **dry run** that reads an authorized export, classifies every
record and stage, and prints the manifest of what would run — by reason, country and stage — plus the
Slurm commands it would submit. This command is a PREVIEW and manifest writer only. ``--execute`` is deliberately rejected until a
live-collection-backed, per-record Slurm submission path exists. ``--apply`` writes a local manifest;
it does not submit work or mutate MongoDB.

Eligibility comes from `ep24_eligibility`: a record with usable output is never selected, even if an old
failure flag or a blank index cell says otherwise; a record with usable input but no output always is.

    # what would run?
    python scripts/roihu/reprocess_selected.py --export <export-dir> --steps 3,4,6 --countries poland

    # write the manifest for an audit trail
    python scripts/roihu/reprocess_selected.py --export <export-dir> --steps 3 --apply --manifest out.json

    # preview is the only supported operating mode; no jobs are submitted

Privacy: prints record ids and counts only — no transcripts, no quotes, no paths beyond the export
directory name.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import ep24_eligibility as eligibility  # noqa: E402

COUNTRIES = ("finland", "poland", "portugal")


def load_rows(export_dir: Path, countries: list[str] | None) -> list[dict]:
    wanted = countries or list(COUNTRIES)
    rows: list[dict] = []
    for country in wanted:
        path = export_dir / f"{country}.csv"
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row.setdefault("country", country)
                rows.append(row)
    return rows


def commands_for(manifest: dict, steps: list[int]) -> list[str]:
    """One sbatch line per country and stage that has selected work."""
    lines: list[str] = []
    for step in steps:
        for country in COUNTRIES:
            count = sum(1 for entry in manifest["selected"]
                        if entry["stage"] == step and entry["country"] == country)
            if not count:
                continue
            lines.append(
                f"sbatch scripts/roihu/submit_step_{step}_*.sh --country {country} "
                f"--retry-errors  # {count} record-stage pairs selected"
            )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--steps", required=True, help="comma-separated stage numbers, e.g. 3,4,6")
    parser.add_argument("--countries", nargs="*", choices=COUNTRIES)
    parser.add_argument("--include-empty", action="store_true",
                        help="also reselect stages whose status says success but produced no output")
    parser.add_argument("--apply", action="store_true", help="write the manifest to disk")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--commands", action="store_true", help="unsupported: placeholder sbatch commands are unsafe")
    parser.add_argument("--execute", action="store_true",
                        help="unsupported: live submission not implemented; fails closed")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.execute or args.commands:
        print("Live submission and sbatch command generation are not implemented. "
              "Use --apply for a local selection manifest only; verify against live "
              "Mongo/Slurm before constructing any job.", file=sys.stderr)
        return 2

    if not args.export.is_dir():
        print(f"not a directory: {args.export}", file=sys.stderr)
        return 2
    steps = [int(item) for item in args.steps.replace(" ", "").split(",") if item]
    rows = load_rows(args.export, args.countries)
    if not rows:
        print("no rows found for the requested countries", file=sys.stderr)
        return 2

    manifest = eligibility.select_reprocess(
        rows, steps, countries=args.countries, include_empty=args.include_empty)
    manifest["source"] = {"export": args.export.name, "steps": steps,
                          "countries": args.countries or list(COUNTRIES),
                          "include_empty": args.include_empty}

    if args.json:
        print(json.dumps(manifest, indent=2))
    else:
        print(f"# EP24 selective reprocess — {eligibility.summary_line(manifest)}")
        print()
        print("| reason | count |")
        print("|---|---|")
        for reason, count in manifest["counts"]["by_reason"].items():
            print(f"| {reason} | {count} |")
        print()
        print("Preview only. No jobs submitted or database records changed.")

    if args.apply:
        target = args.manifest or (args.export.parent / f"reprocess_manifest_{args.export.name}.json")
        target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"manifest written: {target.name}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
