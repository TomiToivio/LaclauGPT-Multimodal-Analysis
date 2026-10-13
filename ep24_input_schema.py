#!/usr/bin/env python3
"""Real incoming-schema inspection for the EP24 Roihu preprocess stage (#128 §5).

The active EP24 reprocess inputs are the researcher-recorded/split feed CSVs under
``analysis/ep24_reprocess/data/to_reprocess/ep24_<country>.csv`` in
LaclauGPT-Private. This module reads those files *as stored* and reports what is
actually there, instead of projecting the input into a guessed legacy schema.

Design rules, from the issue:

* the real CSV is authoritative -- never infer columns from old preprocessing code;
* refuse Git LFS pointer stubs (a pointer has no rows to read);
* preserve every incoming column: this module is an *inspector*, it never writes;
* log filename, SHA256, row count, the full ordered column list, required media
  fields found/missing, and the schema signature;
* validate every country rather than assuming identical headers.

Run it::

    python3 ep24_input_schema.py --root /path/to/LaclauGPT-Private
    python3 ep24_input_schema.py --root ... --json

Nothing here is specific to private data: it prints column names, counts and
hashes only, never cell values.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ep24_schema import EP24_REPROCESS_COLUMNS, REQUIRED_MEDIA_COLUMNS
from ep24_stage_contract import COUNTRY_PRIORITY, COUNTRY_TOKENS

logger = logging.getLogger("ep24_input_schema")

#: Relative location of the canonical reprocess inputs inside the private repo.
TO_REPROCESS_RELPATH = Path("analysis/ep24_reprocess/data/to_reprocess")

#: Countries in the deterministic processing order: the three smoke-test
#: countries first, then the remainder in the documented order.
COUNTRIES: tuple[str, ...] = (*COUNTRY_PRIORITY, *(
    name for name in COUNTRY_TOKENS if name not in COUNTRY_PRIORITY
))

_LFS_POINTER_MAGIC = "version https://git-lfs.github.com/spec/v1"


class LfsPointerError(RuntimeError):
    """Raised when a codebook/CSV is still a Git LFS pointer stub."""


def is_lfs_pointer_bytes(data: bytes) -> bool:
    """True when ``data`` is a Git LFS pointer rather than real content."""
    return data[: len(_LFS_POINTER_MAGIC)] == _LFS_POINTER_MAGIC.encode("ascii")


def country_csv_path(root: str | Path, country: str) -> Path:
    """Return the canonical input path for one country."""
    token = COUNTRY_TOKENS.get(country, country.lower())
    return Path(root) / TO_REPROCESS_RELPATH / f"ep24_{token}.csv"


def read_rows(path: str | Path) -> tuple[list[str], list[dict[str, str]], str]:
    """Read a materialized CSV, refusing LFS pointers.

    Returns ``(columns, rows, sha256)``. Column order is preserved exactly as
    stored; no renaming, no projection, no dtype coercion.
    """
    path = Path(path)
    raw = path.read_bytes()
    if is_lfs_pointer_bytes(raw):
        raise LfsPointerError(
            f"{path} is only a Git LFS pointer. Materialize it first "
            "(git lfs pull / use a smudged checkout); it has no rows to inspect."
        )
    digest = hashlib.sha256(raw).hexdigest()
    text = raw.decode("utf-8-sig", errors="replace")
    reader = csv.DictReader(text.splitlines())
    columns = list(reader.fieldnames or [])
    rows = [dict(row) for row in reader]
    return columns, rows, digest


@dataclass
class CountrySchema:
    """What one country's real incoming CSV actually looks like."""

    country: str
    path: str
    exists: bool
    is_lfs_pointer: bool = False
    sha256: str = ""
    row_count: int = 0
    columns: list[str] = field(default_factory=list)
    column_count: int = 0
    extra_columns: list[str] = field(default_factory=list)
    missing_canonical_columns: list[str] = field(default_factory=list)
    required_media_present: list[str] = field(default_factory=list)
    required_media_missing: list[str] = field(default_factory=list)
    empty_columns: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def signature(self) -> str:
        """A stable key grouping files that share an identical header."""
        return "|".join(self.columns)

    @property
    def matches_canonical(self) -> bool:
        return self.columns == list(EP24_REPROCESS_COLUMNS)


def inspect_country(root: str | Path, country: str) -> CountrySchema:
    """Inspect one country's CSV. Never raises for missing/pointer files."""
    path = country_csv_path(root, country)
    result = CountrySchema(country=country, path=str(path), exists=path.exists())
    if not path.exists():
        result.error = "input CSV not found"
        logger.error("%s: input CSV not found at %s", country, path)
        return result

    try:
        columns, rows, digest = read_rows(path)
    except LfsPointerError as exc:
        result.is_lfs_pointer = True
        result.error = str(exc)
        logger.error("%s: %s", country, exc)
        return result
    except OSError as exc:
        result.error = str(exc)
        logger.error("%s: cannot read %s: %s", country, path, exc)
        return result

    result.sha256 = digest
    result.row_count = len(rows)
    result.columns = columns
    result.column_count = len(columns)
    canonical = list(EP24_REPROCESS_COLUMNS)
    result.extra_columns = [c for c in columns if c not in canonical]
    result.missing_canonical_columns = [c for c in canonical if c not in columns]
    result.required_media_present = [c for c in REQUIRED_MEDIA_COLUMNS if c in columns]
    result.required_media_missing = [c for c in REQUIRED_MEDIA_COLUMNS if c not in columns]
    result.empty_columns = sorted(
        c for c in columns if rows and not any(str(r.get(c, "")).strip() for r in rows)
    )

    logger.info(
        "%s: file=%s sha256=%s rows=%d cols=%d required_media_ok=%s",
        country, path.name, digest[:16], len(rows), len(columns),
        not result.required_media_missing,
    )
    logger.debug("%s: columns=%s", country, columns)
    if result.extra_columns:
        logger.info("%s: extra (non-canonical) columns=%s", country, result.extra_columns)
    if result.missing_canonical_columns:
        logger.warning(
            "%s: MISSING canonical columns=%s", country, result.missing_canonical_columns
        )
    if result.required_media_missing:
        logger.error(
            "%s: MISSING required media columns=%s -- media stage cannot run",
            country, result.required_media_missing,
        )
    if result.empty_columns:
        logger.debug("%s: always-empty columns=%s", country, result.empty_columns)

    return result


def inspect_all(root: str | Path) -> list[CountrySchema]:
    """Inspect every EP24 country in the deterministic processing order."""
    return [inspect_country(root, country) for country in COUNTRIES]


def summary(rows: list[CountrySchema]) -> dict[str, Any]:
    """Aggregate the per-country results, including distinct schema signatures."""
    by_signature: dict[str, list[str]] = {}
    for row in rows:
        if row.columns:
            by_signature.setdefault(row.signature, []).append(row.country)
    usable = [r for r in rows if r.columns and not r.required_media_missing]
    return {
        "countries": len(rows),
        "usable_for_media": [r.country for r in usable],
        "not_usable": [r.country for r in rows if r not in usable],
        "missing_files": [r.country for r in rows if not r.exists],
        "lfs_pointers": [r.country for r in rows if r.is_lfs_pointer],
        "distinct_signatures": len(by_signature),
        "signature_groups": {
            f"{len(sig.split('|'))}cols": names for sig, names in by_signature.items()
        },
        "extra_columns_union": sorted({c for r in rows for c in r.extra_columns}),
        "total_rows": sum(r.row_count for r in rows),
    }


def render(rows: list[CountrySchema]) -> str:
    """Human-readable table plus the per-signature column listing."""
    lines = [
        "EP24 incoming schema inspection (#128)",
        "",
        f"{'country':10} {'rows':>6} {'cols':>5} {'canonical':>9}  sha256[:12]",
    ]
    for r in rows:
        state = "-"
        if r.error:
            state = "POINTER" if r.is_lfs_pointer else "MISSING"
        else:
            state = "yes" if r.matches_canonical else "no"
        lines.append(
            f"{r.country:10} {r.row_count:6} {r.column_count:5} {state:>9}  {r.sha256[:12]}"
        )

    groups: dict[str, list[CountrySchema]] = {}
    for r in rows:
        if r.columns:
            groups.setdefault(r.signature, []).append(r)
    for i, (sig, members) in enumerate(groups.items(), 1):
        cols = sig.split("|")
        names = ", ".join(m.country for m in members)
        lines += ["", f"--- signature {i}: {len(cols)} columns ({names})"]
        lines += [f"      {c}" for c in cols]
        extras = [c for c in cols if c not in EP24_REPROCESS_COLUMNS]
        if extras:
            lines.append("    not in the canonical keep-schema:")
            lines += [f"      + {c}" for c in extras]

    agg = summary(rows)
    lines += [
        "",
        "--- aggregate",
        f"    countries inspected      : {agg['countries']}",
        f"    total data rows          : {agg['total_rows']}",
        f"    distinct header signatures: {agg['distinct_signatures']}",
        f"    usable for media stages  : {', '.join(agg['usable_for_media']) or 'none'}",
        f"    missing files            : {', '.join(agg['missing_files']) or 'none'}",
        f"    LFS pointer stubs        : {', '.join(agg['lfs_pointers']) or 'none'}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, required=True,
        help="LaclauGPT-Private checkout containing analysis/ep24_reprocess/...",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    parser.add_argument(
        "--log-level", default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    rows = inspect_all(args.root)
    if args.json:
        print(json.dumps(
            {"countries": [asdict(r) | {"signature": r.signature} for r in rows],
             "aggregate": summary(rows)},
            indent=2,
        ))
    else:
        print(render(rows))

    # Non-zero when any input cannot be read as a real CSV with media identity.
    blocked = [r for r in rows if r.required_media_missing or r.error]
    return 1 if blocked else 0


if __name__ == "__main__":
    sys.exit(main())
