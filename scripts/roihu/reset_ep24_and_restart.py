#!/usr/bin/env python3
"""Archive EP24 generated state, reimport canonical CSVs, submit fresh Step 1 jobs.

Dry-run by default. Explicit --execute plus a dataset-specific confirmation
are required. The script NEVER drops a Mongo database, deletes source CSVs,
or modifies researcher-seeded memory/entities collections.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Direct script invocation places scripts/roihu on sys.path, not repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ep24_bootstrap import country_slug, ordered_country_files
from ep24_cli import normalize_country
from ep24_settings import load_private_env, private_root
from roihu_storage import MongoStorage, StorageConfig

PRESERVE_SUFFIXES = {"memory", "entities"}
SQLITE_NAMES = ("preprocess_v2.db", "frame.db", "summary.db")
REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE1 = REPO_ROOT / "scripts/roihu/step_1_roihu_preprocess.sbatch"
IMPORTER = REPO_ROOT / "step_0_roihu_import.py"


def targets(db, config: StorageConfig) -> list[str]:
    """All generated country collections, excluding curated researcher collections."""
    prefix = config.prefix + "_"
    return sorted(name for name in db.list_collection_names()
                  if name.startswith(prefix)
                  and name[len(prefix):] not in PRESERVE_SUFFIXES
                  and "_reset_archive_" not in name)


def archive_name(name: str, stamp: str) -> str:
    return f"{name}_reset_archive_{stamp}"


def choose_sources(root: Path, countries: list[str], all_countries: bool) -> list[Path]:
    files = ordered_country_files(root)
    selected = files if all_countries else [
        path for path in files if country_slug(path) in set(countries)
    ]
    if not selected or (not all_countries and set(map(country_slug, selected)) != set(countries)):
        raise RuntimeError("Missing canonical source CSV for one or more requested countries")
    for path in selected:
        if path.stat().st_size < 20:
            raise RuntimeError(f"Missing/empty source data: {path}")
        with path.open(encoding="utf-8-sig") as stream:
            if stream.readline().startswith("version https://git-lfs.github.com/spec/v1"):
                raise RuntimeError(f"Source still a Git LFS pointer: {path}")
    return selected


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--country", action="append", help="Country to reset; repeat for several")
    group.add_argument("--all-countries", action="store_true")
    parser.add_argument("--execute", action="store_true",
                        help="Archive generated state, reimport Step 0 and submit Step 1")
    parser.add_argument("--confirm-dataset", default="",
                        help="Must exactly equal LACLAUGPT_DATASET when executing")
    parser.add_argument("--no-submit", action="store_true",
                        help="Archive/reimport only, without submitting Step 1")
    args = parser.parse_args(argv)
    load_private_env()
    dataset = os.environ.get("LACLAUGPT_DATASET", "")
    if dataset != "ep2024_reprocess":
        raise RuntimeError("Refusing reset outside dataset ep2024_reprocess")
    config = StorageConfig.from_env()
    if not config.mongo_enabled:
        raise RuntimeError("Reset requires MongoDB enabled")
    if args.execute and args.confirm_dataset != dataset:
        parser.error("--execute requires --confirm-dataset ep2024_reprocess")
    root = Path(os.environ["LACLAUGPT_EP24_INPUT_ROOT"]).resolve()
    output_root = Path(os.environ["LACLAUGPT_EP24_OUTPUT_ROOT"]).resolve()
    paths = choose_sources(root, [normalize_country(c) for c in args.country or []],
                           args.all_countries)
    if not STAGE1.is_file() or not IMPORTER.is_file():
        raise RuntimeError(f"Step 0 importer or Step 1 sbatch script not found: importer={IMPORTER} stage1={STAGE1}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_root = private_root().resolve() / "reset-archives" / stamp
    if args.execute and shutil.which("sbatch") is None and not args.no_submit:
        raise RuntimeError("sbatch unavailable; use --no-submit to reimport without submitting")

    # Refuse to reset while this user's EP24 jobs may be writing the same state.
    if args.execute and shutil.which("squeue"):
        queue = subprocess.run(
            ["squeue", "--me", "--noheader", "--format=%j"],
            check=True, capture_output=True, text=True,
        ).stdout.splitlines()
        active = [name for name in queue if "ep24" in name.casefold()]
        if active:
            raise RuntimeError(f"Active EP24 SLURM jobs must finish/cancel before reset: {active}")

    storage = MongoStorage(config)
    try:
        plans = []
        for source in paths:
            country = country_slug(source)
            cfg = StorageConfig(dataset=config.dataset, country=country)
            names = targets(storage.db, cfg)
            plans.append((country, source, names))
            print(f"country={country} source={source} db={config.mongo_database}")
            print(f"  Mongo generated collections to archive ({len(names)}): {names}")
            print(f"  Mongo preserved collections: {cfg.collection('memory')}, {cfg.collection('entities')}")
            print(f"  output directory to archive: {output_root / country}")
        sqlite_paths = []
        for base in (private_root().resolve(), Path.cwd().resolve()):
            sqlite_paths += [base / "database" / name for name in SQLITE_NAMES]
        sqlite_paths = sorted(set(p for p in sqlite_paths if p.is_file()))
        print(f"SQLite restart caches to archive: {sqlite_paths}")
        print(f"Archive location: {archive_root}")
        if not args.execute:
            print("DRY RUN ONLY. Add --execute --confirm-dataset ep2024_reprocess to proceed.")
            return 0
        archive_root.mkdir(parents=True, exist_ok=False)
        manifest = {"dataset": dataset, "database": config.mongo_database,
                    "countries": [c for c, _, _ in plans], "mongo_archived": [],
                    "files_archived": [], "source_csvs": [str(src) for _, src, _ in plans]}
        def persist_manifest():
            (archive_root / "manifest.json").write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        persist_manifest()
        # Rename, do not drop: an archive remains in MongoDB for restoration.
        for country, _, names in plans:
            for name in names:
                archived = archive_name(name, stamp)
                if len(archived.encode()) > 120:
                    raise RuntimeError(f"Mongo archive name too long: {archived}")
                if archived in storage.db.list_collection_names():
                    raise RuntimeError(f"Mongo archive collision: {archived}")
                storage.db[name].rename(archived, dropTarget=False)
                manifest["mongo_archived"].append({"original": name, "archive": archived})
                persist_manifest()
        for country, _, _ in plans:
            src = output_root / country
            if src.exists():
                dst = archive_root / "outputs" / country
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
                manifest["files_archived"].append({"from": str(src), "to": str(dst)})
                persist_manifest()
        for src in sqlite_paths:
            dst = archive_root / "sqlite" / str(len(manifest["files_archived"])) / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            manifest["files_archived"].append({"from": str(src), "to": str(dst)})
            persist_manifest()
        for country, _, _ in plans:
            print(f"Importing canonical source records: {country}")
            subprocess.run([sys.executable, str(IMPORTER), "--country", country],
                           check=True, cwd=str(IMPORTER.parent))
            manifest.setdefault("imported", []).append(country)
            persist_manifest()
        if not args.no_submit:
            for country, _, _ in plans:
                result = subprocess.run(["sbatch", str(STAGE1), "--country", country],
                                        check=True, capture_output=True, text=True,
                                        cwd=str(STAGE1.parent.parent.parent))
                print(result.stdout.strip())
                manifest.setdefault("step1_submitted", []).append(
                    {"country": country, "response": result.stdout.strip()})
                persist_manifest()
        print(f"Reset/reimport complete. Manifest: {archive_root / 'manifest.json'}")
        return 0
    finally:
        storage.close()


if __name__ == "__main__":
    raise SystemExit(main())
