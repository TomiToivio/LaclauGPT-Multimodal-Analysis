#!/usr/bin/env python3
"""Verify that docs/EP24_CODEBOOK_COVERAGE_CROSS_COUNTRY.md still matches reality.

Why this exists
---------------
That document promises that *"every number below is reproducible with one command
per country"*. It was accurate when published (#86): all ten rows reproduce
exactly against the auditor as it stood then.

It then silently went stale. #100 wired `fold_fix` into the auditor's `_fold`,
which changed what counts as a single token, so every fragmentation count moved
(599 -> 623 in total) while the document went on claiming reproducibility. Nobody
noticed, because re-running a ten-country table is not part of any other task.

A promise of reproducibility that nothing checks is worse than no promise: it
makes a stale number look authoritative. This script turns the promise into a
test.

Usage:
    python3 scripts/ep24/check_coverage_table.py --root <private codebook root>

    # report the diff without failing (useful when intentionally refreshing)
    python3 scripts/ep24/check_coverage_table.py --root <root> --report

Exit codes: 0 = table matches, 1 = table is stale (prints the corrected rows),
2 = input/config problem.
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs" / "EP24_CODEBOOK_COVERAGE_CROSS_COUNTRY.md"

#: The country books the table claims to cover, in the order it lists them.
COUNTRIES = ("PL", "HU", "SE", "FR", "BG", "PT", "ES", "FI", "DE", "HR")


def _load_auditor():
    spec = importlib.util.spec_from_file_location(
        "codebook_coverage", ROOT / "scripts" / "ep24" / "codebook_coverage.py"
    )
    if not spec or not spec.loader:  # pragma: no cover - defensive
        raise SystemExit("cannot load scripts/ep24/codebook_coverage.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_table(text: str) -> dict[str, tuple[int, int, float, int, float, int, int]]:
    """Pull the country rows out of the published table."""
    out: dict[str, tuple[int, int, float, int, float, int, int]] = {}
    # Values may be wrapped in markdown emphasis (`**91.3%**`), which the
    # original table uses for the two extreme columns. Tolerate it rather than
    # failing to parse a row the maintainer may legitimately embolden.
    row = re.compile(
        r"^\|\s*(?P<c>[A-Z]{2})\s*\|\s*\**\s*(?P<e>\d+)\s*\**\s*\|\s*\**\s*(?P<na>\d+)\s*\**\s*\|\s*"
        r"\**\s*(?P<pct>[\d.]+)%\s*\**\s*\|\s*\**\s*(?P<al>\d+)\s*\**\s*\|\s*\**\s*(?P<ae>[\d.]+)\s*\**\s*\|\s*"
        r"\**\s*(?P<fr>\d+)\s*\**\s*\|\s*\**\s*(?P<td>\d+)\s*\**\s*\|$"
    )
    for line in text.splitlines():
        m = row.match(line.strip())
        if m:
            out[m["c"]] = (
                int(m["e"]), int(m["na"]), float(m["pct"]), int(m["al"]),
                float(m["ae"]), int(m["fr"]), int(m["td"]),
            )
    return out


def measure(auditor, root: Path, country: str) -> tuple[int, int, float, int, float, int, int]:
    report = auditor.audit(root, country)
    layers = [layer for name, layer in report["layers"].items() if name.startswith("ep24_")]
    if not layers:
        raise SystemExit(f"no country book layer for {country} under {root}")
    layer = layers[0]
    cov = layer["alias_coverage"]
    return (
        cov["entries"],
        cov["entries_without_aliases"],
        cov["entries_without_aliases_pct"],
        cov["aliases_total"],
        cov["aliases_per_entry"],
        len(layer["entity_fragmentation"]),
        len(layer["theme_near_duplicates"]),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True, help="directory holding the per-country codebooks")
    parser.add_argument("--report", action="store_true", help="print the diff but exit 0")
    args = parser.parse_args(argv)

    root = Path(args.root).expanduser()
    if not root.is_dir():
        print(f"ERROR: root is not a directory: {root}", file=sys.stderr)
        return 2
    if not DOC.exists():
        print(f"ERROR: {DOC} is missing", file=sys.stderr)
        return 2

    published = _parse_table(DOC.read_text(encoding="utf-8"))
    if not published:
        print("ERROR: no country rows parsed from the table", file=sys.stderr)
        return 2

    auditor = _load_auditor()
    stale: list[str] = []
    missing: list[str] = []

    for country in COUNTRIES:
        if country not in published:
            missing.append(country)
            continue
        now = measure(auditor, root, country)
        was = published[country]
        # Compare the integer/ratio columns; a 0.01 drift in a mean is not a
        # stale table, a changed fragmentation count is.
        changed = [
            f"{label} {w:g}->{n:g}"
            for label, w, n in zip(
                ("entries", "noalias", "pct", "aliases", "alias/entry", "frag", "themedup"),
                was, now,
            )
            if abs(w - n) > 0.011
        ]
        if changed:
            stale.append(f"  {country}: " + "; ".join(changed))

    if missing:
        print(f"ERROR: table is missing rows for {missing}", file=sys.stderr)
        return 2

    if not stale:
        print(f"EP24 coverage table matches the auditor for all {len(COUNTRIES)} countries.")
        return 0

    print("EP24 coverage table is STALE. The document promises these numbers are")
    print("reproducible, so it must be refreshed or the auditor change reverted:")
    print()
    print("\n".join(stale))
    print()
    print("Refresh the table's Fragmented groups / Theme near-dups columns (and any")
    print("prose that cites them) in docs/EP24_CODEBOOK_COVERAGE_CROSS_COUNTRY.md.")
    return 0 if args.report else 1


if __name__ == "__main__":
    raise SystemExit(main())
