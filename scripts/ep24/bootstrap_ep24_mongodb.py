#!/usr/bin/env python3
"""Bootstrap the EP24 reprocessing run into MongoDB (issue #64).

One explicit command, per the issue:

    python scripts/ep24/bootstrap_ep24_mongodb.py

What it does, in order:

1. discovers the country input files under the private `to_reprocess` directory;
2. imports the per-country rows into MongoDB **without changing the private
   source files**;
3. applies the canonical merges before any analysis step can run:
   `entities` = `new_entity` + `researcher_new_persons`,
   `themes`   = `new_theme` + `researcher_new_themes`,
   keeping the four source columns for provenance;
4. generates stable record ids and initializes per-step state;
5. writes CSV/SQLite backup snapshots and an import manifest;
6. logs every important decision.

Safety:
- the private source CSVs are opened read-only and never written;
- `--dry-run` does everything except write to MongoDB, so the parse and the
  merges can be checked before touching the database;
- no credential is read from or written to the repository.

Country order is Finland -> Poland -> Portugal, then the rest alphabetically
(``ep24_stage_contract.country_order``).
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import ep24_bootstrap as bootstrap  # noqa: E402
import ep24_stage_contract as contract  # noqa: E402

LOG = logging.getLogger("ep24.bootstrap")

#: Default location of the private inputs. Overridable by env var so the script
#: never hard-codes a private absolute path into a committed default.
DEFAULT_PRIVATE_ROOT = "LaclauGPT-Private"
PRIVATE_ROOT_ENV = "LACLAUGPT_PRIVATE_ROOT"
TO_REPROCESS = Path("analysis/ep24_reprocess/data/to_reprocess")


def private_root(explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit)
    return Path(os.getenv(PRIVATE_ROOT_ENV, DEFAULT_PRIVATE_ROOT))


def discover_inputs(root: Path) -> dict[str, Path]:
    """Find `ep24_<token>.csv` inputs and map them to country names."""
    directory = root / TO_REPROCESS
    if not directory.is_dir():
        raise SystemExit(
            f"no EP24 input directory at {directory}. Pass --private-root, or set "
            f"{PRIVATE_ROOT_ENV}. The authoritative inputs are private and are not "
            "committed to this repository."
        )
    by_token = {token: country for country, token in contract.COUNTRY_TOKENS.items()}
    found: dict[str, Path] = {}
    for path in sorted(directory.glob("ep24_*.csv")):
        token = path.stem[len("ep24_"):]
        country = by_token.get(token)
        if country is None:
            LOG.warning("input %s has no known country token; skipping", path.name)
            continue
        found[country] = path
    return {country: found[country] for country in contract.country_order(list(found))}


def read_rows(path: Path) -> tuple[list[str], list[dict]]:
    """Read a country CSV read-only, preserving blank vs value exactly."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        return columns, [dict(row) for row in reader]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bootstrap EP24 reprocessing into MongoDB.")
    parser.add_argument("--private-root", help="path to the LaclauGPT-Private checkout")
    parser.add_argument("--country", action="append", help="only this country (repeatable)")
    parser.add_argument("--limit", type=int, default=0, help="max rows per country (0 = all)")
    parser.add_argument("--dry-run", action="store_true", help="parse and merge, write nothing")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    root = private_root(args.private_root)
    inputs = discover_inputs(root)
    if args.country:
        wanted = {c.lower() for c in args.country}
        inputs = {c: p for c, p in inputs.items() if c.lower() in wanted}
    if not inputs:
        raise SystemExit("no matching country inputs found")

    LOG.info("private_root=%s", root)
    LOG.info("countries in priority order: %s", list(inputs))
    LOG.info("merge contract: %s", bootstrap.provenance_note())

    summary: list[dict] = []
    for country, path in inputs.items():
        columns, rows = read_rows(path)
        if args.limit:
            rows = rows[: args.limit]

        missing = [c for c in contract.SOURCE_COLUMNS if c not in columns]
        if missing:
            # Loud rather than silent: the issue forbids silently mapping or
            # dropping fields, and a schema change must be a decision.
            LOG.error("%s is missing contract columns: %s", path.name, missing)

        bootstrapped = bootstrap.bootstrap_rows(rows, country=country)
        with_entities = sum(1 for row in bootstrapped if row["entities"])
        with_themes = sum(1 for row in bootstrapped if row["themes"])

        LOG.info(
            "%s: %d rows, %d columns, entities on %d rows, themes on %d rows",
            country, len(bootstrapped), len(columns), with_entities, with_themes,
        )
        LOG.debug("%s first record_id=%s", country, bootstrapped[0]["record_id"] if bootstrapped else "")

        summary.append(
            {
                "country": country,
                "source": str(path),
                "rows": len(bootstrapped),
                "columns": columns,
                "entities_rows": with_entities,
                "themes_rows": with_themes,
                "record_ids": [row["record_id"] for row in bootstrapped[:3]],
            }
        )

        if args.dry_run:
            continue

        import ep24_db as db  # noqa: PLC0415 - only needed when actually writing

        client, database = db.connect()
        try:
            db.ensure_indexes(database)
            written = db.upsert_records(database, bootstrapped)
            LOG.info("%s: upserted %d records into %s", country, written, db.COLLECTIONS["records"])
        finally:
            client.close()

    manifest_path = Path("outputs/ep24_bootstrap_manifest.json")
    payload = {
        "private_root": str(root),
        "dry_run": bool(args.dry_run),
        "merge_contract": bootstrap.provenance_note(),
        "countries": summary,
    }
    if not args.dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        LOG.info("wrote import manifest: %s", manifest_path)
    else:
        LOG.info("dry run: no database write, no manifest written")

    print(json.dumps({c["country"]: c["rows"] for c in summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
