#!/usr/bin/env python3
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

TEST_COUNTRIES = ("finland", "poland", "portugal")

def read_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        return reader.fieldnames or [], list(reader)

def stable_identity(row: dict[str, str], index: int) -> tuple[str, str]:
    for key in ("video_id", "_storage_id", "allas_filename", "url"):
        if row.get(key):
            return key, row[key]
    return "_row_hash", hashlib.sha256(
        json.dumps(row, sort_keys=True).encode()
    ).hexdigest()[:16] + f"-{index}"

def main() -> int:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--test", action="store_true")
    mode.add_argument("--full", action="store_true")
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=209)
    args = ap.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    mode_name = "test" if args.test else "full"
    selected_root = args.run_dir / "inputs"
    selected_root.mkdir(exist_ok=True)
    manifest_rows = []
    all_records = []

    if args.test:
        countries = list(TEST_COUNTRIES)
        rng = random.Random(args.seed)
        for country in countries:
            source = args.input_root / f"ep24_{country}.csv"
            fields, rows = read_rows(source)
            if len(rows) < 10:
                raise SystemExit(f"{source} has only {len(rows)} rows; need 10")
            indexed = list(enumerate(rows))
            chosen = rng.sample(indexed, 10)
            out = selected_root / source.name
            with out.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields)
                writer.writeheader()
                writer.writerows(row for _, row in chosen)
            for idx, row in chosen:
                key, record_id = stable_identity(row, idx)
                item = {
                    "country": country,
                    "source_index": idx,
                    "record_key": key,
                    "record_id": record_id,
                }
                manifest_rows.append(item)
                all_records.append(item)
        effective_input = selected_root
    else:
        paths = sorted(args.input_root.glob("ep24_*.csv"))
        countries = [p.stem.removeprefix("ep24_") for p in paths]
        if not countries:
            raise SystemExit(f"no ep24_*.csv files under {args.input_root}")
        for path, country in zip(paths, countries, strict=True):
            _, rows = read_rows(path)
            for idx, row in enumerate(rows):
                key, record_id = stable_identity(row, idx)
                all_records.append(
                    {
                        "country": country,
                        "source_index": idx,
                        "record_key": key,
                        "record_id": record_id,
                    }
                )
        effective_input = args.input_root

    (args.run_dir / "countries.txt").write_text("\n".join(countries) + "\n", encoding="utf-8")
    manifest = {
        "mode": mode_name,
        "seed": args.seed if args.test else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_input_root": str(args.input_root),
        "effective_input_root": str(effective_input),
        "countries": countries,
        "sample_size": len(manifest_rows) if args.test else None,
        "sample": manifest_rows,
        "records": all_records,
    }
    (args.run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(str(effective_input))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
