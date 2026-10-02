#!/usr/bin/env python3
"""Report english_label coverage for one or all EP24 country codebooks.

Default mode is read-only and returns 0 even when review is required.
Use --strict for QA/CI: any required missing English label returns exit code 1.
Exit code 2 means the private input/configuration is unavailable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roihu_codebooks import COUNTRY_PROFILES, load_profile  # noqa: E402


def audit(private_root: Path, countries: list[str]) -> dict:
    rows = []
    for code in countries:
        try:
            _entries, meta = load_profile(private_root, code)
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
            rows.append({"country": code, "state": "UNAVAILABLE", "error": str(exc)})
            continue
        qa = meta["english_label_coverage"]
        rows.append(
            {
                "country": code,
                "entry_count": meta["entry_count"],
                "required": qa["required_count"],
                "present": qa["present_count"],
                "missing": qa["missing_count"],
                "exempt": qa["exempt_count"],
                "coverage_pct": qa["coverage_pct"],
                "state": qa["state"],
            }
        )
    return {"policy": "explicit English label required for non-English entries unless a documented exemption exists", "countries": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="LaclauGPT-Private/analysis/ep24 directory")
    parser.add_argument("--country", action="append", help="ISO2 code; repeatable. Default: all EP24 countries")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true", help="return 1 if any country needs English-label review")
    args = parser.parse_args(argv)

    root = Path(args.root).expanduser()
    if not root.is_dir():
        print(f"ERROR: root is not a directory: {root}", file=sys.stderr)
        return 2

    countries = [c.upper() for c in args.country] if args.country else sorted(COUNTRY_PROFILES)
    report = audit(root, countries)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("EP24 bilingual english_label coverage")
        print("country entries required present missing exempt coverage state")
        for row in report["countries"]:
            if row["state"] == "UNAVAILABLE":
                print(f"{row['country']:>2} UNAVAILABLE {row['error']}")
            else:
                print(
                    f"{row['country']:>2} {row['entry_count']:>7} {row['required']:>8} "
                    f"{row['present']:>7} {row['missing']:>7} {row['exempt']:>6} "
                    f"{row['coverage_pct']:>7.1f}% {row['state']}"
                )

    unavailable = any(row["state"] == "UNAVAILABLE" for row in report["countries"])
    if unavailable:
        return 2
    if args.strict and any(row["state"] == "REVIEW_REQUIRED" for row in report["countries"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
