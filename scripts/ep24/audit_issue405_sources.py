#!/usr/bin/env python3
"""Issue #405: read-only, aggregate QA of exact FI/PL/PT 2026-11-15 Git LFS CSVs.

Never prints source rows, transcripts, researcher notes, actor labels, or quotes.
The files must be locally materialized from Git LFS; pointer text is rejected.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

FIXTURES = {
    "finland": ("finland/finland.csv", 16948981, "486cdcbe3efc64d3922f670171b5413799e28e51e249ff5d706c15e02e7a8fb0"),
    "poland": ("poland/poland.csv", 10847651, "025f351b9c20e65321bc6ec9c4de81701fa7b06ff7bd3ba6264c2319668c8f31"),
    "portugal": ("portugal/portugal.csv", 31519402, "4b825709cb1c109291afe431992be766e7f0ffead45f3098436373aa63080d53"),
}
STAGES = {
    "1": "preprocess_status", "2": "frame_analysis_status",
    "3": "vllm_video_status", "4": "summary_quality_status",
    "6": "laclau_status", "7": "dna_status", "8": "sna_status",
    "9": "rdf_status",
}
STRUCTURED = ("dna_statements_json", "sna_nodes_json", "sna_edges_json",
              "sna_metrics_json", "laclau_structured_json",
              "ep24_entity_resolution_json")
SUCCESS = {"ok", "complete", "completed", "done", "success", "cached"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_country(path: Path, country: str, *, verify_fixture: bool = True) -> dict:
    name, expected_bytes, expected_hash = FIXTURES[country]
    errors = []
    if not path.is_file():
        return {"country": country, "errors": ["missing_file"], "rows": 0}
    with path.open("rb") as stream:
        if stream.read(64).startswith(b"version https://git-lfs.github.com/spec/v1"):
            return {"country": country, "errors": ["git_lfs_pointer_not_dataset"], "rows": 0}
    size = path.stat().st_size
    digest = sha256(path)
    if verify_fixture and (size != expected_bytes or digest != expected_hash):
        errors.append("fixture_checksum_or_size_mismatch")
    statuses = {stage: Counter() for stage in STAGES}
    malformed = Counter()
    identities: set[str] = set()
    duplicates = 0
    missing_identity = 0
    wrong_country = 0
    missing_fields = []
    rows = 0
    # Portugal has very long quoted/newline-bearing cells; never parse by physical line.
    csv.field_size_limit(sys.maxsize)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        for required in ("country", "video_id"):
            if required not in columns:
                missing_fields.append(required)
        for row in reader:
            rows += 1
            if None in row or any(value is None for value in row.values()):
                errors.append("csv_field_count_mismatch")
                break
            value = str(row.get("country") or "").strip().lower()
            if value and value not in (country, {"finland": "fi", "poland": "pl", "portugal": "pt"}[country]):
                wrong_country += 1
            native = str(row.get("_storage_id") or row.get("video_id") or "").strip()
            if not native:
                missing_identity += 1
            elif native in identities:
                duplicates += 1
            else:
                identities.add(native)
            for stage, field in STAGES.items():
                if field in columns:
                    statuses[stage][(row.get(field) or "").strip().lower() or "(missing)"] += 1
            for field in STRUCTURED:
                value = (row.get(field) or "").strip()
                if value and value.lower() not in ("null", "none", "nan"):
                    try:
                        json.loads(value)
                    except ValueError:
                        malformed[field] += 1
    if missing_fields:
        errors.append("required_columns_missing")
    if rows != 500:
        errors.append("not_500_rows")
    if duplicates:
        errors.append("duplicate_storage_ids")
    if missing_identity:
        errors.append("missing_storage_ids")
    if wrong_country:
        errors.append("country_mismatch")
    return {
        "country": country, "file": name, "bytes": size,
        "expected_sha256_match": digest == expected_hash,
        "rows": rows, "column_count": len(columns),
        "missing_required_columns": missing_fields,
        "duplicate_storage_ids": duplicates, "missing_storage_ids": missing_identity,
        "wrong_country_rows": wrong_country,
        "stages": {stage: {
            "status_column": STAGES[stage],
            "present": STAGES[stage] in columns,
            "success": sum(n for key, n in statuses[stage].items() if key in SUCCESS),
            "errors": sum(n for key, n in statuses[stage].items() if "error" in key or "fail" in key),
            "missing": statuses[stage].get("(missing)", 0),
        } for stage in STAGES},
        "invalid_structured_json": dict(malformed),
        "errors": sorted(set(errors + (["invalid_structured_json"] if malformed else []))),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Issue #405 private-source QA; aggregate output only")
    parser.add_argument("--root", type=Path, required=True, help="private checkout .../fiplpt_500_20261115")
    parser.add_argument("--out", type=Path, help="private destination for aggregate JSON report")
    parser.add_argument("--allow-new-snapshot", action="store_true",
                        help="skip exact fixture hash and size checks when auditing a later export")
    args = parser.parse_args(argv)
    results = [audit_country(args.root / relative, country,
                             verify_fixture=not args.allow_new_snapshot)
               for country, (relative, _, _) in FIXTURES.items()]
    payload = {"audit": "ep24-issue405-fi-pl-pt/1", "countries": results,
               "errors": sum(len(item["errors"]) for item in results)}
    output = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding="utf-8")
    print(output, end="")
    return 1 if payload["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
