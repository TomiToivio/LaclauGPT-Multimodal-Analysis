#!/usr/bin/env python3
"""Report and gate bilingual ``english_label`` coverage for EP24 codebooks (#101).

Answers the question the country QA passes could not: *how much of each country's
codebook is missing an English label, and is that gap allowed to stay silent?*

Run against the private codebook root:

    python3 scripts/ep24/check_bilingual_coverage.py --root <private codebook root>

    # QA gate: fail if any country is still above the agreed gap
    python3 scripts/ep24/check_bilingual_coverage.py --root <root> --max-missing-pct 5

Read-only. It never writes to a codebook and never merges, renames or translates
an entry. Ambiguity is reported, never resolved.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_loader():
    spec = importlib.util.spec_from_file_location(
        "roihu_codebooks", ROOT / "roihu_codebooks.py"
    )
    if spec is None or spec.loader is None:
        raise SystemExit("cannot load roihu_codebooks.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["roihu_codebooks"] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", required=True, help="private EP24 root holding codebooks/")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    parser.add_argument(
        "--max-missing-pct",
        type=float,
        default=None,
        help="fail (exit 1) when any country exceeds this missing-English percentage",
    )
    parser.add_argument(
        "--top", type=int, default=12, help="example entry ids to show per country"
    )
    args = parser.parse_args(argv)

    cb = _load_loader()
    report = cb.bilingual_coverage_report(args.root)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        t = report["totals"]
        print("EP24 bilingual (english_label) coverage report")
        print(f"policy: {report['policy']}")
        print()
        print(f"{'ctry':5} {'entries':>8} {'needs_en':>9} {'missing':>8} {'missing%':>9}  by layer")
        for c in sorted(report["countries"], key=lambda x: -x["missing_pct"]):
            print(
                f"{c['country_code']:5} {c['entries']:8d} {c['needs_english']:9d} "
                f"{c['missing_english']:8d} {c['missing_pct']:8.1f}%  {c['missing_by_layer']}"
            )
        print()
        print(
            f"{'TOTAL':5} {t['entries']:8d} {t['needs_english']:9d} "
            f"{t['missing_english']:8d} {t['missing_pct']:8.1f}%"
        )
        print()
        print("NOTE: 'needs_en' counts only entries whose canonical label is not already")
        print("English. Entries that are already English require no gloss and are not")
        print("counted as gaps. Ambiguity is reported, never resolved.")

        worst = max(report["countries"], key=lambda x: x["missing_pct"])
        if worst["missing_entry_ids"]:
            print()
            print(f"Example gaps in {worst['country_code']} (first {args.top}):")
            for entry_id in worst["missing_entry_ids"][: args.top]:
                print(f"  {entry_id}")

    if args.max_missing_pct is not None:
        try:
            cb.assert_bilingual_coverage(report, max_missing_pct=args.max_missing_pct)
        except ValueError as exc:
            print(f"\nFAIL: {exc}", file=sys.stderr)
            return 1
        print(f"\nOK: every country is at or below {args.max_missing_pct}% missing English.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
