#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

STAGE_NAMES = {
    1: "preprocess",
    2: "frame",
    3: "video",
    4: "summary",
    5: "postprocess",
    6: "discourse_analysis",
    7: "discourse_network_analysis",
    8: "social_network_analysis",
    9: "rdf",
}


def slurm_states(jobs: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    ids = ",".join(row["job_id"] for row in jobs)
    states: dict[str, dict[str, str]] = {}
    if not ids:
        return states
    result = subprocess.run(
        ["sacct", "-n", "-P", "-j", ids, "--format=JobIDRaw,State,ExitCode"],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in result.stdout.splitlines():
        parts = line.split("|")
        if len(parts) >= 3 and "." not in parts[0]:
            states[parts[0]] = {"state": parts[1], "exit_code": parts[2]}
    return states


def output_members(path: Path, records: list[dict]) -> dict[str, set[str]]:
    wanted_keys = {
        record["record_key"]
        for record in records
        if record["record_key"] != "_row_hash"
    }
    members: dict[str, set[str]] = {key: set() for key in wanted_keys}
    if not path.is_file():
        return members
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            for key in wanted_keys:
                value = row.get(key)
                if value:
                    members[key].add(value)
    return members


def main() -> int:
    run_dir = Path(sys.argv[1])
    jobs_path = Path(sys.argv[2])
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    jobs = list(csv.DictReader(jobs_path.open(encoding="utf-8"), delimiter="\t"))
    states = slurm_states(jobs)
    for row in jobs:
        row.update(
            states.get(row["job_id"], {"state": "UNKNOWN", "exit_code": "UNKNOWN"})
        )

    output_root = Path(
        os.environ.get("LACLAUGPT_EP24_OUTPUT_ROOT", str(run_dir / "outputs"))
    )
    records = manifest.get("records") or manifest.get("sample") or []
    by_country: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_country[record["country"]].append(record)

    video_steps: list[dict] = []
    for country, country_records in sorted(by_country.items()):
        for step, stage in STAGE_NAMES.items():
            output = output_root / country / f"step_{step:02d}_{stage}.csv"
            members = output_members(output, country_records)
            for record in country_records:
                key = record["record_key"]
                record_id = record["record_id"]
                if key == "_row_hash":
                    status = "unverifiable"
                elif not output.is_file():
                    status = "no_output"
                else:
                    status = (
                        "present" if record_id in members.get(key, set()) else "missing"
                    )
                video_steps.append(
                    {
                        "country": country,
                        "record_key": key,
                        "record_id": record_id,
                        "step": step,
                        "status": status,
                        "output": str(output),
                    }
                )

    job_success = all(row["state"].startswith("COMPLETED") for row in jobs)
    record_success = all(
        item["status"] in {"present", "unverifiable"} for item in video_steps
    )
    success = job_success and record_success
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "success": success,
        "jobs": jobs,
        "video_steps": video_steps,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    overall = "SUCCESS" if success else "FAILED/INCOMPLETE"
    lines = [
        "# EP24 pipeline run summary",
        "",
        f"Overall: {overall}",
        "",
        "| Country | Step | Job | State | Exit |",
        "|---|---:|---:|---|---|",
    ]
    for row in jobs:
        lines.append(
            f"| {row['country']} | {row['step']} | {row['job_id']} | "
            f"{row['state']} | {row['exit_code']} |"
        )

    if video_steps:
        counts: dict[tuple[str, int, str], int] = defaultdict(int)
        for item in video_steps:
            counts[(item["country"], item["step"], item["status"])] += 1
        lines.extend(
            [
                "",
                "## Per-video propagation",
                "",
                "| Country | Step | Status | Videos |",
                "|---|---:|---|---:|",
            ]
        )
        for (country, step, status), count in sorted(counts.items()):
            lines.append(f"| {country} | {step} | {status} | {count} |")
        lines.extend(
            [
                "",
                "The complete per-video/per-step matrix is stored in summary.json.",
            ]
        )

    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
