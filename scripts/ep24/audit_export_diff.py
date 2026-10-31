"""Compare two EP24 export artifact sets and report what changed (#333).

Answers the questions an audit cannot answer from a single export:

* did a stage **regress** — fewer records with output than the previous export?
* is a stage **frozen** — identical output across two runs, which points at stale checkpoint reuse
  rather than fresh computation?
* did records appear or disappear, and did any record id appear under two countries?
* did child tables shrink or grow independently of their parent stages?

Counts and identity only. No cell contents, no paths beyond the export directory names.

    python scripts/ep24/audit_export_diff.py <older-export> <newer-export> [--json]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

COUNTRIES = ("finland", "poland", "portugal")


def _rows(export: Path, country: str) -> dict[str, dict]:
    path = export / f"{country}.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row.get("record_id", ""): row for row in csv.DictReader(handle) if row.get("record_id")}


def _text(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text in ("[]", "{}", "null", "None") else text


def _stage_presence(rows: dict[str, dict], fields: list[str]) -> dict[str, bool]:
    return {rid: any(_text(row.get(field)) for field in fields) for rid, row in rows.items()}


def _stage_digest(rows: dict[str, dict], fields: list[str]) -> dict[str, str]:
    """Per record, a hash of that stage's values.

    Presence alone cannot tell two exports apart: the same 2,125 records can carry the same stages and
    still hold different answers. This makes "did the run change anything" measurable without ever
    printing a cell.
    """
    digests: dict[str, str] = {}
    for rid, row in rows.items():
        payload = "\x00".join(_text(row.get(field)) for field in fields)
        digests[rid] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digests


STAGES = {
    "1_preprocess": ["asr_transcript", "ocr_1", "frame_file"],
    "2_frame": ["frame_analysis_1"],
    "3_video": ["vllm_structured_output", "vllm_video_analysis"],
    "4_summary": ["summary_analysis", "summary_summary_md"],
    "5_postprocess": ["postprocess_entities", "postprocess_themes"],
    "6_discourse": ["laclau_structured_json", "laclau_summary_md"],
    "7_dna": ["dna_statements_json", "dna_statement_count"],
    "8_sna": ["sna_nodes_json", "sna_edges_json"],
}


def compare(older: Path, newer: Path) -> dict:
    report: dict = {"older": older.name, "newer": newer.name, "countries": {}, "findings": []}
    seen_ids: dict[str, list[str]] = {}

    for country in COUNTRIES:
        old_rows, new_rows = _rows(older, country), _rows(newer, country)
        if not old_rows and not new_rows:
            continue
        old_ids, new_ids = set(old_rows), set(new_rows)
        for rid in new_ids:
            seen_ids.setdefault(rid, []).append(country)
        entry = {
            "records_older": len(old_rows),
            "records_newer": len(new_rows),
            "records_added": len(new_ids - old_ids),
            "records_removed": len(old_ids - new_ids),
            "stages": {},
        }
        for stage, fields in STAGES.items():
            old_presence = _stage_presence({k: old_rows[k] for k in old_ids & new_ids}, fields)
            new_presence = _stage_presence({k: new_rows[k] for k in old_ids & new_ids}, fields)
            old_true = {rid for rid, present in old_presence.items() if present}
            new_true = {rid for rid, present in new_presence.items() if present}
            identical = old_true == new_true and len(old_true) > 0
            old_digest = _stage_digest({k: old_rows[k] for k in old_ids & new_ids}, fields)
            new_digest = _stage_digest({k: new_rows[k] for k in old_ids & new_ids}, fields)
            changed = sum(1 for rid in old_digest if old_digest[rid] != new_digest.get(rid))
            entry["stages"][stage] = {
                "with_output_older": len(old_true),
                "with_output_newer": len(new_true),
                "gained": len(new_true - old_true),
                "lost": len(old_true - new_true),
                "unchanged_set": identical,
                "content_changed": changed,
            }
            if len(new_true) < len(old_true):
                report["findings"].append({
                    "severity": "error", "kind": "stage_regression", "country": country,
                    "stage": stage, "older": len(old_true), "newer": len(new_true),
                })
            if identical and len(old_true) > 1 and changed == 0:
                report["findings"].append({
                    "severity": "warning", "kind": "frozen_stage", "country": country,
                    "stage": stage, "records": len(old_true),
                    "detail": ("same records and byte-identical values in both exports — the later "
                               "run recomputed nothing for this stage"),
                })
            elif changed:
                report["findings"].append({
                    "severity": "info", "kind": "content_changed", "country": country,
                    "stage": stage, "records": changed,
                })
        for role in ("dna_statements", "sna", "entities", "themes"):
            old_child = older / "children" / f"{country}_{role}.csv"
            new_child = newer / "children" / f"{country}_{role}.csv"
            old_n = self_count(old_child)
            new_n = self_count(new_child)
            entry.setdefault("children", {})[role] = {"older": old_n, "newer": new_n}
            if old_n and not new_n:
                report["findings"].append({"severity": "error", "kind": "child_table_disappeared",
                                           "country": country, "role": role, "older": old_n})
        report["countries"][country] = entry

    for rid, countries in seen_ids.items():
        if len(set(countries)) > 1:
            report["findings"].append({"severity": "error", "kind": "cross_country_id_collision",
                                       "record_id_sha1_hint": rid[:6], "countries": sorted(set(countries))})
    return report


def self_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def render(report: dict) -> str:
    lines = [f"# Export diff — {report['older']} → {report['newer']}", ""]
    for country, entry in report["countries"].items():
        lines.append(f"## {country}: {entry['records_older']} → {entry['records_newer']} records "
                     f"(+{entry['records_added']} / -{entry['records_removed']})")
        lines.append("")
        lines.append("| stage | older | newer | gained | lost | content changed |")
        lines.append("|---|---|---|---|---|---|")
        for stage, values in entry["stages"].items():
            lines.append(f"| {stage} | {values['with_output_older']} | {values['with_output_newer']} | "
                         f"{values['gained']} | {values['lost']} | {values['content_changed']} |")
        lines.append("")
    if report["findings"]:
        lines.append("## Findings")
        lines.append("")
        for finding in report["findings"]:
            lines.append(f"- **{finding['severity']}** `{finding['kind']}` — "
                         f"{finding.get('country', '')} {finding.get('stage', '')}: "
                         f"{ {k: v for k, v in finding.items() if k not in ('severity', 'kind', 'country', 'stage')} }")
    else:
        lines.append("No regressions or frozen stages detected.")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("older", type=Path)
    parser.add_argument("newer", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    for path in (args.older, args.newer):
        if not path.is_dir():
            print(f"not a directory: {path}", file=sys.stderr)
            return 2
    report = compare(args.older, args.newer)
    print(json.dumps(report, indent=2) if args.json else render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
