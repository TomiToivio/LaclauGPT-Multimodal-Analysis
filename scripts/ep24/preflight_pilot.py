#!/usr/bin/env python3
"""Preflight the Monday500 pilot before a Roihu resubmission (#379).

The issue's runbook says "HARD STOP before submission" and lists what must be true: exactly 500
distinct pilot records per country, the isolated Monday500 dataset, and no stale claims that would make
work silently ineligible. Until now that check was prose a human had to perform by eye, with
`ep24_status.py` showing totals only — which cannot distinguish 500 records from 500 rows over 480
records.

This tool performs the checks and **exits non-zero** when any of them fails, so a resubmission can be
gated on it instead of on attention:

    python scripts/ep24/preflight_pilot.py --countries finland poland portugal \
        --dataset ep24_monday500_20261031 --expect-unique 500

Read-only: no claim, no reset, no write.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: The pilot namespace. Anything else is a different corpus, which is what this tool exists to catch.
PILOT_DATASET = "ep24_monday500_20261031"
FORBIDDEN_DATASETS = ("ep2024_reprocess",)


@dataclass
class CountryReport:
    country: str
    dataset: str
    total: int = 0
    unique_ids: int = 0
    stages: dict[int, dict[str, int]] = field(default_factory=dict)
    stale_claims: dict[int, int] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def evaluate(
    report: CountryReport,
    *,
    expected: str,
    expect_unique: int | None = 500,
    allow_supersets: bool = False,
) -> CountryReport:
    """All the issue's assertions in one place, so the CLI and the tests agree."""
    problems = report.problems
    if report.dataset in FORBIDDEN_DATASETS:
        problems.append(
            f"{report.country}: dataset {report.dataset!r} is the full-corpus namespace, not the pilot"
        )
    elif report.dataset != expected:
        problems.append(f"{report.country}: dataset {report.dataset!r} != {expected!r}")
    if expect_unique is not None:
        if report.unique_ids < expect_unique:
            problems.append(
                f"{report.country}: {report.unique_ids} distinct records, expected {expect_unique}"
            )
        elif report.unique_ids > expect_unique and not allow_supersets:
            problems.append(
                f"{report.country}: {report.unique_ids} distinct records, expected exactly "
                f"{expect_unique}; pass --allow-supersets only if the extra identities are intended"
            )
    if report.total != report.unique_ids:
        problems.append(
            f"{report.country}: {report.total} rows for {report.unique_ids} distinct records "
            "(duplicate _storage_id rows would make a resume redo work)"
        )
    for step, counts in sorted(report.stages.items()):
        if counts.get("claimed") and report.stale_claims.get(step):
            problems.append(
                f"{report.country}: step {step} has {report.stale_claims[step]} stale claim(s); "
                "they become eligible after the claim-age threshold, so wait rather than using --force"
            )
    return report


def _summarise(report: CountryReport) -> str:
    lines = [
        f"{report.country}: dataset={report.dataset} rows={report.total} distinct={report.unique_ids}"
    ]
    for step in sorted(report.stages):
        counts = report.stages[step]
        rendered = " ".join(f"{state}={counts[state]}" for state in sorted(counts))
        stale = report.stale_claims.get(step, 0)
        suffix = f" stale_claims={stale}" if stale else ""
        lines.append(f"  step {step}: {rendered}{suffix}")
    for problem in report.problems:
        lines.append(f"  PROBLEM: {problem}")
    return "\n".join(lines)


def collect(country: str, dataset: str, *, stale_hours: int = 6, sample: int | None = None) -> CountryReport:
    """Read the pilot collection. Imported lazily so the scoring logic above stays testable offline."""
    from ep24_settings import load_private_env

    load_private_env()
    # The collection under test is the one the caller named, never whatever the env file defaults to.
    os.environ["LACLAUGPT_DATASET"] = dataset

    from ep24_stage_orchestrator import status_path
    from roihu_storage import MongoStorage, StorageConfig

    previous = os.environ.get("LACLAUGPT_COUNTRY")
    os.environ["LACLAUGPT_COUNTRY"] = country
    report = CountryReport(country=country, dataset=dataset)
    try:
        storage = MongoStorage(StorageConfig.from_env())
    except Exception as error:  # a missing URI is a verdict, not a crash
        report.problems.append(f"{country}: cannot connect to MongoDB ({error.__class__.__name__})")
        return report
    try:
        collection = storage.db[storage.collection_name("dataframe")]
        report.total = collection.count_documents({})
        ids = collection.distinct("_storage_id")
        report.unique_ids = len(ids)
        if sample is not None:
            report.total = min(report.total, sample)
        for step in range(1, 10):
            path = status_path(step)
            report.stages[step] = {
                state: collection.count_documents({f"{path}.status": state})
                for state in ("claimed", "complete", "error", "retry", "skipped")
            }
            threshold = (datetime.now(timezone.utc) - timedelta(hours=stale_hours)).isoformat()
            report.stale_claims[step] = collection.count_documents(
                {f"{path}.status": "claimed", f"{path}.claimed_at": {"$lt": threshold}}
            )
    finally:
        storage.close()
        if previous is None:
            os.environ.pop("LACLAUGPT_COUNTRY", None)
        else:
            os.environ["LACLAUGPT_COUNTRY"] = previous
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--countries", nargs="+", default=["finland", "poland", "portugal"])
    parser.add_argument("--dataset", default=PILOT_DATASET)
    parser.add_argument(
        "--expect-unique", type=int, default=500,
        help="distinct _storage_id records expected per country (0 disables the check)",
    )
    parser.add_argument("--allow-supersets", action="store_true")
    parser.add_argument("--stale-hours", type=int, default=6)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    expected_unique = args.expect_unique or None
    reports: list[CountryReport] = []
    for country in args.countries:
        report = collect(country, args.dataset, stale_hours=args.stale_hours)
        reports.append(
            evaluate(
                report,
                expected=args.dataset,
                expect_unique=expected_unique,
                allow_supersets=args.allow_supersets,
            )
        )

    if args.json:
        print(json.dumps([asdict(report) for report in reports], indent=2, sort_keys=True))
    else:
        for report in reports:
            print(_summarise(report))

    failed = [report.country for report in reports if not report.ok]
    if failed:
        print(
            "\nHARD STOP: do not submit. Investigate before resubmitting: " + ", ".join(failed),
            file=sys.stderr,
        )
        return 1
    print("\nPreflight passed: pilot identity, isolation and claim state are as expected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
