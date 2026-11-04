#!/usr/bin/env python3
"""Plan and launch the EP24 full-corpus rollout (#373).

This is the controller the issue asks for, and it is deliberately thin: it reads the same MongoDB
records, uses the same eligibility rules (`ep24_stage_orchestrator.eligible_query`) as the per-stage
runners, and submits through `scripts/roihu/run_country_pipeline.sh`. It does not re-implement a stage, a
query or a state machine.

Two modes:

* ``resume-sample`` — Finland, Poland and Portugal after the 500-per-country sample is approved. Every
  record already complete at a step stays complete: the plan is derived from record identity and stage
  status, never from an offset, so a country that finished 300 of 500 at Step 5 resumes at Step 6 for
  those 300 and at Step 5 for the rest.
* ``remaining-countries`` — every other country in the input manifest, from scratch, same stage code.

``--dry-run`` is the default and prints the plan. Submitting requires an explicit approval marker,
because this issue is preparation, not authorization: ``EP24_SAMPLE_APPROVED=1`` must be set (or the
file named by ``--approval-file`` must exist) before any ``sbatch`` is called.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTECTED = ("fi", "pl", "pt")
DEFAULT_STEPS = tuple(range(1, 8))  # Steps 8-9 are paused (#346); opt in deliberately elsewhere.
#: The documented order of the remaining countries (README), used only as a tie-break; discovery wins.
DOCUMENTED_ORDER = ("de", "es", "hu", "hr", "fr", "bg", "se")


def _fallback_input_root() -> Path:
    return Path(
        os.getenv(
            "LACLAUGPT_EP24_INPUT_ROOT",
            "/scratch/project_2009497/LaclauGPT-Private/analysis/ep24_reprocess/data/to_reprocess",
        )
    )


def discover_countries(input_root: Path | None = None) -> list[str]:
    """Countries present in the private input manifest, in a reproducible order.

    Discovery beats the README list: the manifest is authoritative, and a country that is documented but
    has no input must not be scheduled just because a paragraph names it.
    """
    root = Path(input_root) if input_root is not None else _fallback_input_root()
    found = sorted(
        {p.stem.removeprefix("ep24_").casefold() for p in root.glob("ep24_*.csv")}
    )
    def key(name: str) -> tuple[int, str]:
        return (DOCUMENTED_ORDER.index(name) if name in DOCUMENTED_ORDER else len(DOCUMENTED_ORDER), name)
    return sorted(found, key=key)


def remaining_countries(input_root: Path | None = None) -> list[str]:
    return [c for c in discover_countries(input_root) if c not in PROTECTED]


def _status_path(step: int) -> str:
    return f"_pipeline.step_{step:02d}.status"


def plan_country(collection: Any, country: str, *, steps: Sequence[int] = DEFAULT_STEPS,
                 retry_errors: bool = False) -> dict[str, Any]:
    """Per-step counts for one country, from the records themselves.

    ``collection`` is anything with ``count_documents(query)`` — the Mongo collection in production, a
    fake in the tests. The queries come from ``eligible_query`` so the controller cannot disagree with
    the stage runner about what is outstanding.
    """
    from ep24_stage_orchestrator import eligible_query

    base = {"country": country}
    total = collection.count_documents(base)
    per_step: dict[str, dict[str, int]] = {}
    outstanding_total = 0
    for step in steps:
        complete = collection.count_documents({**base, _status_path(step): "complete"})
        errored = collection.count_documents({**base, _status_path(step): "error"})
        outstanding = collection.count_documents({**base, **eligible_query(step, retry_errors=retry_errors, force=False)})
        per_step[str(step)] = {
            "complete": complete,
            "error": errored,
            "outstanding": outstanding,
            "pending_after_this_step": max(0, total - complete - outstanding),
        }
        outstanding_total += outstanding
    return {
        "country": country,
        "records": total,
        "steps": per_step,
        "outstanding": outstanding_total,
        "resumable": total > 0 and any(v["complete"] for v in per_step.values()),
        "complete_through_final_step": per_step[str(steps[-1])]["complete"] == total and total > 0,
    }


def build_plan(collection: Any, countries: Iterable[str], *, steps: Sequence[int] = DEFAULT_STEPS,
               retry_errors: bool = False) -> dict[str, Any]:
    rows = [plan_country(collection, c, steps=steps, retry_errors=retry_errors) for c in countries]
    return {
        "steps": list(steps),
        "retry_errors": retry_errors,
        "countries": rows,
        "totals": {
            "countries": len(rows),
            "records": sum(r["records"] for r in rows),
            "outstanding": sum(r["outstanding"] for r in rows),
            "finished": sum(1 for r in rows if r["complete_through_final_step"]),
        },
    }


def approval_granted(*, approval_file: Path | None = None) -> bool:
    if os.getenv("EP24_SAMPLE_APPROVED", "").strip() in {"1", "true", "yes", "on"}:
        return True
    return bool(approval_file and Path(approval_file).is_file())


def submit_country(country: str, *, runner: Path, extra: Sequence[str] = ()) -> int:
    """Hand one country to the existing per-country SLURM chain."""
    command = ["bash", str(runner), country, "--to-step", "7", *extra]
    return subprocess.run(command, check=False).returncode


def _collection_from_settings() -> Any:
    from ep24_db import MongoStorage, StorageConfig

    storage = MongoStorage(StorageConfig.from_env())
    return storage.db[storage.collection_name("dataframe")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("resume-sample", "remaining-countries", "audit"),
                        default="audit")
    parser.add_argument("--submit", action="store_true",
                        help="actually call sbatch; without it this is a read-only plan")
    parser.add_argument("--input-root", type=Path, default=None)
    parser.add_argument("--approval-file", type=Path, default=None)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--country", action="append", default=[])
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--runner", type=Path,
                        default=ROOT / "scripts" / "roihu" / "run_country_pipeline.sh")
    args = parser.parse_args(argv)

    if args.mode == "resume-sample":
        countries = list(PROTECTED)
    elif args.mode == "remaining-countries":
        countries = list(remaining_countries(args.input_root))
    else:
        countries = [c for c in args.country] or list(remaining_countries(args.input_root))
    if args.country:
        countries = list(dict.fromkeys(args.country))

    plan = build_plan(_collection_from_settings(), countries, retry_errors=args.retry_errors)
    print(json.dumps(plan, indent=2) if args.json else _render(plan))

    if not args.submit:
        return 0
    if not approval_granted(approval_file=args.approval_file):
        print("refusing to submit: the 500-per-country sample is not approved. "
              "Set EP24_SAMPLE_APPROVED=1 or pass --approval-file once it is.", file=sys.stderr)
        return 3
    status = 0
    for row in plan["countries"]:
        if row["complete_through_final_step"]:
            continue
        status |= submit_country(row["country"], runner=args.runner,
                                 extra=("--retry-errors",) if args.retry_errors else ())
    return status


def _render(plan: Mapping[str, Any]) -> str:
    lines = [f"steps {plan['steps']} · retry-errors={plan['retry_errors']}", ""]
    header = f"{'country':<10} {'records':>8} " + " ".join(f"{'s' + str(s):>7}" for s in plan["steps"])
    lines.append(header)
    for row in plan["countries"]:
        cells = " ".join(f"{row['steps'][str(s)]['outstanding']:>7}" for s in plan["steps"])
        lines.append(f"{row['country']:<10} {row['records']:>8} {cells}")
    totals = plan["totals"]
    lines += ["", f"{totals['countries']} countries · {totals['records']} records · "
                  f"{totals['outstanding']} outstanding · {totals['finished']} finished"]
    lines.append("cells are outstanding records per step (eligible_query), not offsets")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
