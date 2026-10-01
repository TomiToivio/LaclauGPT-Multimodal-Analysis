"""Run one numbered EP24 stage on only Mongo-eligible cumulative records.

The numbered analysis scripts remain simple and legacy-compatible. This wrapper
owns country ordering, partial-pipeline eligibility, checkpointing, atomic claims,
Mongo persistence and Redis coordination.
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from ep24_allas import stage_media
from ep24_backups import write_checkpoint
from ep24_redis import RedisCoordinator
from roihu_storage import MongoStorage, StorageConfig

LOG = logging.getLogger("ep24_stage")
PRIORITY = ("finland", "poland", "portugal")


def _countries() -> list[str]:
    explicit = os.getenv("LACLAUGPT_COUNTRIES", "").strip()
    if explicit:
        raw = [x.strip().casefold() for x in explicit.split(",") if x.strip()]
    else:
        root = Path(os.getenv("LACLAUGPT_EP24_INPUT_ROOT",
            "/scratch/project_2009497/LaclauGPT-Private/analysis/ep24_reprocess/data/to_reprocess"))
        raw = [p.stem.removeprefix("ep24_").casefold() for p in root.glob("ep24_*.csv")]
    return sorted(set(raw), key=lambda c: ((PRIORITY.index(c) if c in PRIORITY else len(PRIORITY)), c))


def _status_path(step: int) -> str:
    return f"_pipeline.step_{step:02d}"


def _eligible_query(step: int, *, retry_errors: bool, force: bool) -> dict:
    q: dict = {}
    if step > 1:
        q[f"{_status_path(step - 1)}.status"] = "complete"
    if not force:
        allowed = ["pending", "retry"]
        if retry_errors:
            allowed.append("error")
        q["$or"] = [
            {f"{_status_path(step)}.status": {"$exists": False}},
            {f"{_status_path(step)}.status": {"$in": allowed}},
        ]
    return q


def _claim(storage: MongoStorage, step: int, run_id: str, *, limit: int,
           retry_errors: bool, force: bool, stale_hours: int = 6) -> list[dict]:
    from pymongo import ReturnDocument

    collection = storage.db[storage.collection_name("dataframe")]
    now = datetime.now(timezone.utc)
    stale = now - timedelta(hours=stale_hours)
    collection.update_many(
        {f"{_status_path(step)}.status": "claimed",
         f"{_status_path(step)}.claimed_at": {"$lt": stale.isoformat()}},
        {"$set": {f"{_status_path(step)}.status": "retry",
                  f"{_status_path(step)}.recovered_at": now.isoformat()}}
    )
    claimed: list[dict] = []
    query = _eligible_query(step, retry_errors=retry_errors, force=force)
    while limit <= 0 or len(claimed) < limit:
        doc = collection.find_one_and_update(
            query,
            {"$set": {
                f"{_status_path(step)}.status": "claimed",
                f"{_status_path(step)}.run_id": run_id,
                f"{_status_path(step)}.claimed_at": now.isoformat(),
            }},
            sort=[("_storage_id", 1)],
            return_document=ReturnDocument.AFTER,
        )
        if not doc:
            break
        claimed.append(doc)
        # Exclude the claimed document from this loop regardless of force.
        query = {"$and": [query, {"_storage_id": {"$nin": [d["_storage_id"] for d in claimed]}}]}
    return claimed


def _flat_rows(docs: list[dict]) -> pd.DataFrame:
    rows = []
    for doc in docs:
        row = {k: v for k, v in doc.items() if k not in {"_id", "_pipeline", "_provenance"}}
        rows.append(row)
    return pd.DataFrame(rows)


def _persist(storage: MongoStorage, step: int, run_id: str, output: pd.DataFrame,
             *, source_ids: set[str]) -> None:
    if "_storage_id" not in output.columns:
        raise RuntimeError(f"step {step} output dropped _storage_id")
    if set(output["_storage_id"].astype(str)) != source_ids:
        raise RuntimeError(f"step {step} changed the claimed record set")
    collection = storage.db[storage.collection_name("dataframe")]
    completed_at = datetime.now(timezone.utc).isoformat()
    for _, row in output.iterrows():
        rid = str(row["_storage_id"])
        fields = {str(k): v for k, v in row.to_dict().items()}
        fields.update({
            f"{_status_path(step)}.status": "complete",
            f"{_status_path(step)}.run_id": run_id,
            f"{_status_path(step)}.completed_at": completed_at,
        })
        collection.update_one({"_storage_id": rid}, {"$set": fields}, upsert=False)


def run_country(country: str, *, step: int, script: Path, limit: int,
                retry_errors: bool, force: bool, dry_run: bool) -> int:
    old_country = os.environ.get("LACLAUGPT_COUNTRY")
    os.environ["LACLAUGPT_COUNTRY"] = country
    run_id = os.getenv("SLURM_JOB_ID") or str(uuid.uuid4())
    storage = MongoStorage(StorageConfig.from_env())
    redis = RedisCoordinator(country, step)
    try:
        claimed = _claim(storage, step, run_id, limit=limit,
                         retry_errors=retry_errors, force=force)
        if not claimed:
            LOG.info("country=%s step=%d nothing eligible", country, step)
            return 0
        LOG.info("country=%s step=%d claimed=%d run_id=%s", country, step, len(claimed), run_id)
        if dry_run:
            return len(claimed)

        before = _flat_rows(claimed)
        if step == 1:
            staged = stage_media(before)
            LOG.info("country=%s step=1 Allas staged=%d", country, staged)
        source_ids = set(before["_storage_id"].astype(str))
        output_root = Path(os.getenv("LACLAUGPT_EP24_OUTPUT_ROOT",
            "/scratch/project_2009497/LaclauGPT-Private/analysis/ep24_reprocess/outputs"))
        country_root = output_root / country
        country_root.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix=f"ep24_s{step}_{country}_", dir=country_root) as tmpdir:
            inp = Path(tmpdir) / "input.csv"
            out = Path(tmpdir) / "output.csv"
            before.to_csv(inp, index=False)
            env = os.environ.copy()
            env["LACLAUGPT_INPUT_CSV"] = str(inp)
            env["LACLAUGPT_OUTPUT_CSV"] = str(out)
            env["LACLAUGPT_MAX_ROWS"] = "0"
            env["LACLAUGPT_STAGE_RUN_ID"] = run_id
            LOG.debug("exec country=%s step=%d script=%s input_columns=%d",
                      country, step, script, len(before.columns))
            result = subprocess.run([sys.executable, str(script)], env=env, check=False)
            if result.returncode != 0:
                raise RuntimeError(f"step {step} command failed with exit code {result.returncode}")
            if not out.exists():
                # Some historical steps update input in place.
                out = inp
            after = pd.read_csv(out, dtype=str, keep_default_na=False)
            missing = [c for c in before.columns if c not in after.columns]
            if missing:
                raise RuntimeError(f"step {step} dropped cumulative columns: {missing}")

            _persist(storage, step, run_id, after, source_ids=source_ids)
            checkpoint = country_root / f"step_{step:02d}_{script.stem.removeprefix(f'step_{step}_roihu_')}.csv"
            write_checkpoint(after, checkpoint, stage=f"step_{step:02d}", country=country)
            for rid in source_ids:
                redis.mark(rid, "complete")
            LOG.info("country=%s step=%d complete rows=%d output=%s", country, step, len(after), checkpoint)
            return len(after)
    except Exception:
        LOG.exception("country=%s step=%d failed", country, step)
        try:
            coll = storage.db[storage.collection_name("dataframe")]
            coll.update_many(
                {f"{_status_path(step)}.run_id": run_id,
                 f"{_status_path(step)}.status": "claimed"},
                {"$set": {f"{_status_path(step)}.status": "error",
                          f"{_status_path(step)}.error_at": datetime.now(timezone.utc).isoformat()}}
            )
        finally:
            raise
    finally:
        storage.close()
        if old_country is None:
            os.environ.pop("LACLAUGPT_COUNTRY", None)
        else:
            os.environ["LACLAUGPT_COUNTRY"] = old_country


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--step", type=int, required=True, choices=range(1, 10))
    p.add_argument("--script", type=Path, required=True)
    p.add_argument("--country")
    p.add_argument("--limit", type=int, default=int(os.getenv("LACLAUGPT_MAX_ROWS", "0") or 0))
    p.add_argument("--force", action="store_true")
    p.add_argument("--retry-errors", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(name)s %(message)s")

    countries = [args.country.casefold()] if args.country else _countries()
    total = 0
    for country in countries:
        total += run_country(country, step=args.step, script=args.script, limit=args.limit,
                             retry_errors=args.retry_errors, force=args.force, dry_run=args.dry_run)
    LOG.info("step=%d countries=%d rows=%d", args.step, len(countries), total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
