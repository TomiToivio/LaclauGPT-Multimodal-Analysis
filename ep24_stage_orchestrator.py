"""Run one numbered EP24 stage on only Mongo-eligible cumulative records.

The numbered analysis scripts remain simple and legacy-compatible. This wrapper
owns country ordering, partial-pipeline eligibility, small durable batches,
checkpointing, atomic Mongo claims and optional Redis coordination.
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from ep24_allas import stage_media
from ep24_backups import write_checkpoint
from ep24_context import enrich_dataframe, update_retrieval
from ep24_redis import RedisCoordinator
from ep24_cli import normalize_country
from roihu_storage import MongoStorage, StorageConfig

LOG = logging.getLogger("ep24_stage")
PRIORITY = ("finland", "poland", "portugal")


def countries() -> list[str]:
    explicit = os.getenv("LACLAUGPT_COUNTRIES", "").strip()
    if explicit:
        raw = [x.strip().casefold() for x in explicit.split(",") if x.strip()]
    else:
        root = Path(os.getenv(
            "LACLAUGPT_EP24_INPUT_ROOT",
            "/scratch/project_2009497/LaclauGPT-Private/analysis/ep24_reprocess/data/to_reprocess",
        ))
        raw = [p.stem.removeprefix("ep24_").casefold() for p in root.glob("ep24_*.csv")]
    return sorted(
        set(raw),
        key=lambda c: ((PRIORITY.index(c) if c in PRIORITY else len(PRIORITY)), c),
    )


def status_path(step: int) -> str:
    return f"_pipeline.step_{step:02d}"


def eligible_query(step: int, *, retry_errors: bool, force: bool) -> dict:
    query: dict = {}
    if step > 1:
        query[f"{status_path(step - 1)}.status"] = "complete"
    if step == 2:
        query["preprocess_status"] = {"$in": ["ok", "cached"]}
        query["frame_file"] = {"$exists": True, "$nin": ["", None]}
        query["frame_timestamp_seconds"] = {"$exists": True, "$nin": ["", None]}
    elif step == 3:
        query["frame_analysis_status"] = {"$in": ["ok", "cached"]}
    if not force:
        allowed = ["pending", "retry"]
        if retry_errors:
            allowed.append("error")
        query["$or"] = [
            {f"{status_path(step)}.status": {"$exists": False}},
            {f"{status_path(step)}.status": {"$in": allowed}},
        ]
    return query


def claim_batch(
    storage: MongoStorage,
    step: int,
    run_id: str,
    *,
    batch_size: int,
    retry_errors: bool,
    force: bool,
    stale_hours: int = 6,
) -> list[dict]:
    from pymongo import ReturnDocument

    collection = storage.db[storage.collection_name("dataframe")]
    now = datetime.now(timezone.utc)
    stale = now - timedelta(hours=stale_hours)
    try:
        collection.update_many(
            {
                f"{status_path(step)}.status": "claimed",
                f"{status_path(step)}.claimed_at": {"$lt": stale.isoformat()},
            },
            {
                "$set": {
                    f"{status_path(step)}.status": "retry",
                    f"{status_path(step)}.recovered_at": now.isoformat(),
                }
            },
        )
    except Exception as exc:
        # A MongoDB ping can succeed even when this identity has no collection
        # write privileges. Explain the failed permission without logging the URI.
        from pymongo.errors import OperationFailure
        if isinstance(exc, OperationFailure) and exc.code == 13:
            raise RuntimeError(
                "EP24 MongoDB write authorization failed for "
                f"database={storage.config.mongo_database!r}, "
                f"collection={storage.collection_name('dataframe')!r}. "
                "Check LACLAUGPT_MONGO_URI credentials and authSource, "
                "and grant the service user readWrite on this database "
                "(or equivalent least-privilege collection write permissions). "
                "MongoDB ping only checks connectivity, not write access. "
                "Do not expose credentials in logs."
            ) from exc
        raise

    claimed: list[dict] = []
    query = eligible_query(step, retry_errors=retry_errors, force=force)
    while len(claimed) < batch_size:
        doc = collection.find_one_and_update(
            query,
            {
                "$set": {
                    f"{status_path(step)}.status": "claimed",
                    f"{status_path(step)}.run_id": run_id,
                    f"{status_path(step)}.claimed_at": now.isoformat(),
                }
            },
            sort=[("_storage_id", 1)],
            return_document=ReturnDocument.AFTER,
        )
        if not doc:
            break
        claimed.append(doc)
        query = {
            "$and": [
                query,
                {"_storage_id": {"$nin": [d["_storage_id"] for d in claimed]}},
            ]
        }
    return claimed


def flat_rows(docs: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                key: value
                for key, value in doc.items()
                if key not in {"_id", "_pipeline", "_provenance"}
            }
            for doc in docs
        ]
    )


def persist_batch(
    storage: MongoStorage,
    step: int,
    run_id: str,
    output: pd.DataFrame,
    *,
    source_ids: set[str],
) -> None:
    if "_storage_id" not in output.columns:
        raise RuntimeError(f"step {step} output dropped _storage_id")
    output_ids = set(output["_storage_id"].astype(str))
    if output_ids != source_ids:
        raise RuntimeError(
            f"step {step} changed the claimed record set: "
            f"claimed={len(source_ids)} output={len(output_ids)}"
        )

    collection = storage.db[storage.collection_name("dataframe")]
    completed_at = datetime.now(timezone.utc).isoformat()
    status_fields = {1: "preprocess_status", 2: "frame_analysis_status",
                     3: "vllm_video_status"}
    result_field = status_fields.get(step)
    failures = 0
    for _, row in output.iterrows():
        rid = str(row["_storage_id"])
        fields = {str(k): v for k, v in row.to_dict().items()}
        result = str(row.get(result_field, "")).strip().lower() if result_field else ""
        failed = bool(result_field and result not in {"ok", "cached"})
        pipeline_status = "error" if failed else "complete"
        fields.update(
            {
                f"{status_path(step)}.status": pipeline_status,
                f"{status_path(step)}.run_id": run_id,
                f"{status_path(step)}.completed_at": completed_at,
            }
        )
        if failed:
            failures += 1
            fields[f"{status_path(step)}.error_type"] = "StageRowFailed"
        collection.update_one({"_storage_id": rid}, {"$set": fields}, upsert=False)
    if failures:
        LOG.error("STEP_ROW_FAILURES step=%d failures=%d total=%d; inspect review CSV and logs/frame.log; retry errors with --retry-errors", step, failures, len(output))
        raise RuntimeError(f"Step {step}: {failures}/{len(output)} rows failed; kept as retryable Mongo errors")


def mark_claimed_error(storage: MongoStorage, step: int, run_id: str, exc: Exception) -> None:
    storage.db[storage.collection_name("dataframe")].update_many(
        {
            f"{status_path(step)}.run_id": run_id,
            f"{status_path(step)}.status": "claimed",
        },
        {
            "$set": {
                f"{status_path(step)}.status": "error",
                f"{status_path(step)}.error_at": datetime.now(timezone.utc).isoformat(),
                f"{status_path(step)}.error_type": type(exc).__name__,
            }
        },
    )


def completed_dataframe(storage: MongoStorage, step: int) -> pd.DataFrame:
    docs = list(
        storage.db[storage.collection_name("dataframe")]
        .find({f"{status_path(step)}.status": "complete"})
        .sort("_storage_id", 1)
    )
    return flat_rows(docs)


def run_legacy_batch(
    *,
    country: str,
    step: int,
    script: Path,
    before: pd.DataFrame,
    output_root: Path,
    run_id: str,
) -> pd.DataFrame:
    if step == 1:
        staged = stage_media(before)
        LOG.info("country=%s step=1 Allas staged=%d", country, staged)

    country_root = output_root / country
    country_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"ep24_s{step}_{country}_", dir=country_root
    ) as tmpdir:
        input_csv = Path(tmpdir) / "input.csv"
        output_csv = Path(tmpdir) / "output.csv"
        before.to_csv(input_csv, index=False, encoding="utf-8")

        env = os.environ.copy()
        env["LACLAUGPT_INPUT_CSV"] = str(input_csv)
        env["LACLAUGPT_OUTPUT_CSV"] = str(output_csv)
        env["LACLAUGPT_MAX_ROWS"] = "0"
        env["PYTHONUNBUFFERED"] = "1"
        env["LACLAUGPT_STAGE_RUN_ID"] = run_id
        LOG.debug(
            "exec country=%s step=%d script=%s rows=%d input_columns=%d",
            country,
            step,
            script,
            len(before),
            len(before.columns),
        )
        LOG.info("stage_subprocess_start step=%d country=%s rows=%d", step, country, len(before))
        result = subprocess.run([sys.executable, "-u", str(script)], env=env, check=False)
        LOG.info("stage_subprocess_end step=%d country=%s returncode=%d", step, country, result.returncode)
        if result.returncode != 0:
            raise RuntimeError(
                f"step {step} command failed with exit code {result.returncode}"
            )

        actual_output = output_csv if output_csv.exists() else input_csv
        after = pd.read_csv(actual_output, dtype=str, keep_default_na=False)
        missing = [column for column in before.columns if column not in after.columns]
        if missing:
            raise RuntimeError(f"step {step} dropped cumulative columns: {missing}")
        return after


def run_country(
    country: str,
    *,
    step: int,
    script: Path,
    limit: int,
    retry_errors: bool,
    force: bool,
    dry_run: bool,
    requeue_failed_results: bool = False,
) -> int:
    old_country = os.environ.get("LACLAUGPT_COUNTRY")
    os.environ["LACLAUGPT_COUNTRY"] = country
    storage = MongoStorage(StorageConfig.from_env())
    redis = RedisCoordinator(country, step)
    output_root = Path(
        os.getenv(
            "LACLAUGPT_EP24_OUTPUT_ROOT",
            "/scratch/project_2009497/LaclauGPT-Private/analysis/ep24_reprocess/outputs",
        )
    )
    batch_size = max(1, int(os.getenv("LACLAUGPT_STAGE_BATCH_SIZE", "25")))
    soft_seconds = int(os.getenv("LACLAUGPT_STAGE_SOFT_DEADLINE_SECONDS", "126000"))
    started = time.monotonic()
    processed = 0

    try:
        collection = storage.db[storage.collection_name("dataframe")]
        if dry_run:
            query = eligible_query(step, retry_errors=retry_errors, force=force)
            count = collection.count_documents(query)
            if limit > 0:
                count = min(count, limit)
            LOG.info("country=%s step=%d dry-run eligible=%d", country, step, count)
            return count

        if requeue_failed_results:
            if step == 1:
                historical = {
                    f"{status_path(step)}.status": "complete",
                    "preprocess_status": {"$nin": ["ok", "cached"]},
                }
                reason = "HistoricalPreprocessFailure"
            elif step == 2:
                historical = {
                    f"{status_path(step)}.status": "complete",
                    "frame_analysis_status": {"$nin": ["ok", "cached"]},
                }
                reason = "HistoricalFrameFailure"
            elif step == 3:
                from experiments.vllm_video_test import validate_video_analysis

                repaired = 0
                for doc in collection.find(
                    {f"{status_path(step)}.status": "complete"},
                    {"_storage_id": 1, "vllm_video_analysis": 1},
                ):
                    ok, reason = validate_video_analysis(
                        str(doc.get("vllm_video_analysis", "") or "")
                    )
                    if not ok:
                        result = collection.update_one(
                            {"_storage_id": doc["_storage_id"],
                             f"{status_path(step)}.status": "complete"},
                            {"$set": {
                                f"{status_path(step)}.status": "error",
                                f"{status_path(step)}.error_type": "HistoricalDegenerateVideo",
                                "vllm_video_status": "error",
                                "vllm_video_error": reason,
                            }},
                        )
                        repaired += result.modified_count
                LOG.warning("Requeued %d historical degenerate Step 3 analyses", repaired)
                retry_errors = True
                historical = None
            else:
                raise ValueError("--requeue-failed-results supports Steps 1-3")
            if historical is not None:
                result = collection.update_many(
                    historical,
                    {"$set": {f"{status_path(step)}.status": "error",
                              f"{status_path(step)}.error_type": reason}},
                )
                LOG.warning("Requeued %d historical Step %d failures in %s",
                            result.modified_count, step, storage.collection_name("dataframe"))
            retry_errors = True

        if force:
            # Reset this stage ONCE for the eligible upstream subset. Claims then
            # use normal pending semantics so completed rows cannot be reclaimed forever.
            base = {}
            if step > 1:
                base[f"{status_path(step - 1)}.status"] = "complete"
            collection.update_many(base, {"$unset": {status_path(step): ""}})
            force = False

        while True:
            if soft_seconds > 0 and time.monotonic() - started >= soft_seconds:
                LOG.info(
                    "country=%s step=%d soft deadline reached after %d rows",
                    country,
                    step,
                    processed,
                )
                break
            if limit > 0 and processed >= limit:
                break

            wanted = batch_size if limit <= 0 else min(batch_size, limit - processed)
            run_id = f"{os.getenv('SLURM_JOB_ID') or uuid.uuid4()}-{processed}"
            claimed = claim_batch(
                storage,
                step,
                run_id,
                batch_size=wanted,
                retry_errors=retry_errors,
                force=force,
            )
            if not claimed:
                # An empty claim can mean no import, a namespace mismatch,
                # completed/error rows, or a pending upstream stage.
                # Log counts without exposing private record data or Mongo URI.
                stage_key = status_path(step)
                total_rows = collection.count_documents({})
                stage_counts = {
                    state: collection.count_documents({f"{stage_key}.status": state})
                    for state in ("complete", "claimed", "error", "pending", "retry")
                }
                unstarted = collection.count_documents({stage_key: {"$exists": False}})
                upstream = (collection.count_documents(
                    {f"{status_path(step - 1)}.status": "complete"}
                ) if step > 1 else total_rows)
                LOG.warning(
                    "UPSTREAM_READINESS country=%s step=%d valid_input=%d",
                    country, step, collection.count_documents(
                        eligible_query(step, retry_errors=True, force=True)),
                )
                LOG.warning(
                    "NO_ELIGIBLE_ROWS country=%s step=%d database=%s collection=%s "
                    "total=%d upstream_complete=%d stage_not_started=%d "
                    "stage_counts=%s retry_errors=%s. "
                    "If total=0 run Step 0 non-dry-run with identical "
                    "LACLAUGPT_DATASET/LACLAUGPT_MONGO_DATABASE. "
                    "If error>0, retry with --retry-errors; "
                    "if claimed>0, check concurrent jobs/stale claims; "
                    "if complete>0, existing analyses are preserved.",
                    country, step, storage.config.mongo_database,
                    storage.collection_name("dataframe"), total_rows, upstream,
                    unstarted, stage_counts, retry_errors,
                )
                if total_rows == 0:
                    try:
                        alternatives = [
                            name for name in collection.database.list_collection_names()
                            if name.endswith(f"_{country}_dataframe")
                            and name != storage.collection_name("dataframe")
                        ]
                    except Exception:
                        # Some Mongo identities may read their collection but
                        # lack permission to enumerate the database.
                        alternatives = ["<collection listing unavailable>"]
                    LOG.error(
                        "STEP_%d_EMPTY_INPUT database=%s expected_collection=%s "
                        "dataset=%s other_country_collections=%s. "
                        "Run Step 0 WITHOUT --dry-run against the same private "
                        ".env, and check inserted count before resubmitting.",
                        step, storage.config.mongo_database,
                        storage.collection_name("dataframe"),
                        storage.config.dataset, alternatives,
                    )
                if total_rows == 0:
                    raise RuntimeError(
                        f"Step {step} input collection is empty: "
                        f"{storage.config.mongo_database}."
                        f"{storage.collection_name('dataframe')}. "
                        "Run Step 0 without --dry-run with matching dataset "
                        "and database before requesting a GPU job."
                    )
                LOG.info("country=%s step=%d nothing else eligible", country, step)
                break

            before = enrich_dataframe(storage, flat_rows(claimed))
            source_ids = set(before["_storage_id"].astype(str))
            LOG.info(
                "country=%s step=%d batch_claimed=%d total_before=%d run_id=%s",
                country,
                step,
                len(before),
                processed,
                run_id,
            )
            after = None
            try:
                after = run_legacy_batch(
                    country=country,
                    step=step,
                    script=script,
                    before=before,
                    output_root=output_root,
                    run_id=run_id,
                )
                persist_batch(
                    storage,
                    step,
                    run_id,
                    after,
                    source_ids=source_ids,
                )
                update_retrieval(storage, after, stage=f"step_{step:02d}")
            except Exception as exc:
                if after is not None:
                    try:
                        from ep24_result_reporting import report_stage_rows
                        report_stage_rows(
                            step, country, after,
                            output_root / country / f"step_{step:02d}_failed_batch.csv",
                        )
                    except Exception:
                        LOG.exception("Unable to emit failed batch review")
                mark_claimed_error(storage, step, run_id, exc)
                raise

            processed += len(after)
            for rid in source_ids:
                redis.mark(rid, "complete")

            # Backup the COMPLETE cumulative subset after every durable batch.
            cumulative = completed_dataframe(storage, step)
            checkpoint = (
                output_root
                / country
                / f"step_{step:02d}_{script.stem.removeprefix(f'step_{step}_roihu_')}.csv"
            )
            manifest = write_checkpoint(
                cumulative,
                checkpoint,
                stage=f"step_{step:02d}",
                country=country,
            )
            try:
                from ep24_result_reporting import report_stage_rows
                report_stage_rows(step, country, after, checkpoint)
            except Exception:
                LOG.exception("analysis result reporting failed after durable checkpoint; data is safe")
            LOG.info(
                "country=%s step=%d durable_batch=%d cumulative=%d backup=%s sha256=%s",
                country,
                step,
                len(after),
                len(cumulative),
                checkpoint,
                manifest["csv_sha256"],
            )
        return processed
    finally:
        storage.close()
        if old_country is None:
            os.environ.pop("LACLAUGPT_COUNTRY", None)
        else:
            os.environ["LACLAUGPT_COUNTRY"] = old_country


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, required=True, choices=range(1, 10))
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("-c", "--country")
    parser.add_argument("-n", "--limit", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--requeue-failed-results", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        if args.country is not None:
            selected_country = normalize_country(args.country)
            country_source = "cli"
        elif os.getenv("LACLAUGPT_COUNTRY"):
            selected_country = normalize_country(os.environ["LACLAUGPT_COUNTRY"])
            country_source = "environment"
        else:
            selected_country = None
            country_source = "default"

        if args.limit is not None:
            if args.limit < 0:
                raise ValueError("--limit must be >= 0")
            resolved_limit = args.limit
            limit_source = "cli"
        elif os.getenv("LACLAUGPT_MAX_ROWS") not in (None, ""):
            resolved_limit = int(os.environ["LACLAUGPT_MAX_ROWS"])
            if resolved_limit < 0:
                raise ValueError("LACLAUGPT_MAX_ROWS must be >= 0")
            limit_source = "environment"
        else:
            resolved_limit = 0
            limit_source = "default"
    except ValueError as exc:
        parser.error(str(exc))

    available = countries()
    if selected_country is not None:
        if available and selected_country not in available:
            parser.error(
                f"country {selected_country!r} is not available under LACLAUGPT_EP24_INPUT_ROOT; "
                f"available={available}"
            )
        selected = [selected_country]
    else:
        selected = available

    LOG.info(
        "runtime_selection step=%d country=%s limit=%d selection_source.country=%s "
        "selection_source.limit=%s",
        args.step,
        selected_country or "<all>",
        resolved_limit,
        country_source,
        limit_source,
    )
    total = 0
    for country in selected:
        total += run_country(
            country,
            step=args.step,
            script=args.script,
            limit=resolved_limit,
            retry_errors=args.retry_errors,
            force=args.force,
            dry_run=args.dry_run,
            requeue_failed_results=args.requeue_failed_results,
        )
    LOG.info("step=%d countries=%d processed_rows=%d", args.step, len(selected), total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
