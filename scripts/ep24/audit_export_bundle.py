#!/usr/bin/env python3
"""Audit an exported EP24 country bundle without a database (#405, workstream A/B/D).

The MongoDB audit (``scripts/ep24/audit_stage_coverage.py``) answers "is there a result" for a live run.
This one answers the same question for the artifact a researcher actually receives: the three country
CSVs plus their child tables, e.g.
``LaclauGPT-Private/analysis/ep24/pouta_exports/fiplpt_500_20261115/{finland,poland,portugal}/``.

It reuses ``ep24_coverage_audit`` for the stage semantics — *not started* / *failed* / *completed* /
*status says success, output is empty* — and adds the checks that only an export can fail:

* file integrity against the export's own manifest (SHA-256, byte size, row count);
* identity: unique ``_storage_id``, unique ``_storage_id`` per country, duplicate video ids;
* schema: BOM-prefixed headers, contracted columns missing from the frame, unexpected extra columns;
* child-table integrity: every child ``source_key`` resolves to exactly one parent row, and how much of
  each country's population carries children;
* a JSON parseability check over the lossless ``*.jsonl`` sibling when it is present.

**Counts leave this tool. Row references do not**, unless a caller explicitly writes them to a private
sample file — that is what ``--samples`` is for, and it must never be handed to a public artifact.

Usage::

    python scripts/ep24/audit_export_bundle.py \
        --export-dir <private>/analysis/ep24/pouta_exports/fiplpt_500_20261115 \
        --output <private>/audit/export_audit.json \
        --markdown <private>/audit/export_audit.md \
        --samples <private>/audit/export_audit_samples.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import ep24_coverage_audit as audit  # noqa: E402
from ep24_stage_contract import STAGE_CONTRACT  # noqa: E402

COUNTRIES: tuple[str, ...] = ("finland", "poland", "portugal")
#: Columns that identify a row rather than carry a result.
IDENTITY_COLUMNS: tuple[str, ...] = ("_storage_id", "video_id", "source_key")
#: Columns whose emptiness means "the join key is broken", not "the stage produced nothing".
KEY_COLUMNS: tuple[str, ...] = ("source_key",)
#: Which child table carries a stage's one-to-many findings, so "status says done" can be compared with
#: "an artefact exists". A stage can legitimately produce none -- the question the audit answers is how
#: many produced none, not whether zero is allowed.
STAGE_CHILDREN: dict[int, tuple[str, ...]] = {
    6: ("discourse_items",),
    7: ("dna_statements",),
    8: ("sna_nodes", "sna_edges"),
}
CHILD_TABLES: tuple[str, ...] = (
    "dna_statements", "discourse_items", "entities", "sna_edges", "sna_nodes",
    "sentiment_targets", "themes",
)


def read_csv(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Rows plus the schema facts a naive reader would get wrong.

    The exporter writes a UTF-8 BOM (``\\ufeff``) in front of the first header. ``csv.DictReader`` with
    the default ``utf-8`` codec therefore keys the first column as ``"\\ufeffsource_key"``, and every
    ``row["source_key"]`` silently becomes a ``KeyError`` or an empty value downstream. That is a real
    interoperability defect, so it is measured here instead of being worked around silently.
    """
    raw = path.read_bytes()
    has_bom = raw.startswith(b"\xef\xbb\xbf")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = list(reader.fieldnames or ())
        rows = [dict(row) for row in reader]
    return rows, {
        "path": str(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "columns": len(columns),
        "header_bom": has_bom,
        "rows": len(rows),
        "column_names": columns,
    }


def status_columns(stage_appends: Sequence[str]) -> list[str]:
    return [column for column in stage_appends if column.endswith("_status")]


def status_vocabulary(rows: Sequence[Mapping[str, Any]], stage_appends: Sequence[str]) -> dict[str, int]:
    """The status words a stage actually wrote, counted, lowercased (``<empty>`` when none).

    The coverage buckets classify these words; this histogram shows the words themselves, because the
    vocabulary is not the audit's to invent -- for example ``cached`` is a real pipeline status that
    means "reused a previous result", and a reader must be able to see it rather than infer it.
    """
    counts: dict[str, int] = {}
    for row in rows:
        word = audit.status_of(row, status_columns(stage_appends)) or "<empty>"
        counts[word] = counts.get(word, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def stage_audits(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for stage in STAGE_CONTRACT:
        result = audit.audit_stage(
            rows,
            number=stage.number,
            name=stage.name,
            appends=stage.appends,
            status_columns=status_columns(stage.appends),
        )
        result["status_vocabulary"] = status_vocabulary(rows, stage.appends)
        results.append(result)
    return results


def identity_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ids = [str(row.get("_storage_id") or "").strip() for row in rows]
    videos = [str(row.get("video_id") or "").strip() for row in rows]
    present_ids = [value for value in ids if value]
    present_videos = [value for value in videos if value]
    countries = sorted({str(row.get("country") or "").strip().lower() for row in rows})
    return {
        "rows": len(rows),
        "identity_column": "_storage_id" if present_ids else ("video_id" if present_videos else ""),
        "identity_present": len(present_ids) or len(present_videos),
        "identity_missing": len(rows) - (len(present_ids) or len(present_videos)),
        "identity_duplicates": len(present_ids) - len(set(present_ids))
        if present_ids else len(present_videos) - len(set(present_videos)),
        "video_ids_missing": len(rows) - len(present_videos),
        "video_id_duplicates": len(present_videos) - len(set(present_videos)),
        "countries_in_frame": countries,
        "dataset_labels": sorted({str(row.get("dataset") or "").strip() for row in rows} - {""}),
        "missing_key_columns": [c for c in KEY_COLUMNS if c not in rows[0]] if rows else [],
    }


def jsonl_check(path: Path) -> dict[str, Any]:
    """Parseability of the lossless sibling; counts only, never content."""
    if not path.is_file():
        return {"present": False}
    total = parsed = 0
    keys: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            parsed += 1
            if isinstance(record, dict):
                keys.update(record)
    return {"present": True, "records": total, "parseable": parsed,
            "unparseable": total - parsed, "top_level_keys": sorted(keys)}


def child_keys(root: Path) -> dict[str, set[str]]:
    """Per child table, the parent keys it actually references."""
    keys: dict[str, set[str]] = {}
    for table in CHILD_TABLES:
        path = root / "children" / f"{table}.csv"
        if not path.is_file():
            continue
        with path.open(encoding="utf-8-sig", newline="") as stream:
            keys[table] = {
                str(row.get("source_key") or "").strip()
                for row in csv.DictReader(stream)
            } - {""}
    return keys


def evidence_cross_check(
    rows_by_country: Mapping[str, list[dict[str, Any]]],
    keys_by_country: Mapping[str, set[str]],
    children: Mapping[str, set[str]],
) -> list[dict[str, Any]]:
    """Stage completed versus artefact present, per country (#405 workstream B).

    "Completed" is the coverage definition (at least one contracted output column carries something).
    A video counted as completed whose stage produced no child rows is either a legitimately empty
    result or a silent extraction failure; the audit reports the size of that group rather than
    pretending to know which it is.
    """
    results: list[dict[str, Any]] = []
    for country, rows in rows_by_country.items():
        country_keys = keys_by_country.get(country, set())
        for stage_number, tables in STAGE_CHILDREN.items():
            contract = next(item for item in STAGE_CONTRACT if item.number == stage_number)
            outputs = tuple(
                column for column in contract.appends
                if not column.endswith(("_status", "_reason", "_error", "_model", "_version",
                                        "_runtime_ms", "_generated_at", "_prompt_version",
                                        "_context_sha256"))
            )
            if not outputs:
                continue
            completed_keys = {
                str(row.get("source_key") or "").strip()
                for row in rows
                if any(not audit.empty_value(row.get(column)) for column in outputs)
            } - {""}
            with_children = set().union(
                *[children.get(table, set()) for table in tables]
            ).intersection(country_keys)
            results.append({
                "country": country,
                "stage": stage_number,
                "stage_name": contract.name,
                "completed": len(completed_keys),
                "with_child_rows": len(completed_keys.intersection(with_children)),
                "completed_without_child_rows": len(completed_keys.difference(with_children)),
                "child_rows_for_incomplete_videos": len(with_children.difference(completed_keys)),
                "tables": list(tables),
            })
    return results


def child_audit(root: Path, parent_keys: Mapping[str, set[str]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for table in CHILD_TABLES:
        path = root / "children" / f"{table}.csv"
        if not path.is_file():
            results.append({"table": table, "present": False})
            continue
        rows, facts = read_csv(path)
        keys = [str(row.get("source_key") or "").strip() for row in rows]
        populated = [key for key in keys if key]
        unmatched = sorted({key for key in populated
                            if not any(key in group for group in parent_keys.values())})
        per_country = {
            country: len({key for key in populated if key in keys_set})
            for country, keys_set in parent_keys.items()
        }
        results.append({
            "table": table,
            "present": True,
            "rows": facts["rows"],
            "header_bom": facts["header_bom"],
            "distinct_source_keys": len(set(populated)),
            "rows_without_key": len(rows) - len(populated),
            "unmatched_source_keys": len(unmatched),
            "videos_with_children": per_country,
            "parent_rows": {country: len(keys_set) for country, keys_set in parent_keys.items()},
        })
    return results


def manifest_audit(root: Path, facts_by_country: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    path = root / "manifest.json"
    if not path.is_file():
        return {"present": False}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    declared = manifest.get("files") or {}
    comparisons: dict[str, Any] = {}
    for country, facts in facts_by_country.items():
        for name in (f"{country}/{country}.csv", f"{country}.csv"):
            if name in declared:
                comparisons[name] = {
                    "declared_sha256": declared[name],
                    "actual_sha256": facts["sha256"],
                    "match": declared[name] == facts["sha256"],
                    "bytes": facts["bytes"],
                }
                break
    return {
        "present": True,
        "version": manifest.get("version"),
        "exported_at": manifest.get("exported_at") or manifest.get("generated_at"),
        "producer_revision": manifest.get("producer_revision"),
        "files_compared": comparisons,
        "all_match": all(item["match"] for item in comparisons.values()) if comparisons else None,
    }


def audit_export(root: Path, samples: bool = False) -> dict[str, Any]:
    countries: dict[str, Any] = {}
    parent_keys: dict[str, set[str]] = {}
    rows_by_country: dict[str, list[dict[str, Any]]] = {}
    facts_by_country: dict[str, Any] = {}
    sample_rows: dict[str, Any] = {}
    for country in COUNTRIES:
        path = root / country / f"{country}.csv"
        if not path.is_file():
            path = root / f"{country}.csv"
        if not path.is_file():
            countries[country] = {"present": False}
            continue
        rows, facts = read_csv(path)
        identity = identity_audit(rows)
        stages = stage_audits(rows)
        # Child tables join on the parent's ``source_key`` (ep24:country:dataset:platform:video), not on
        # ``_storage_id``: the exporter writes both, and choosing the wrong one reports every child row
        # as an orphan. ``_storage_id`` is only the fallback for frames without a source key.
        key_column = "source_key" if "source_key" in facts["column_names"] else "_storage_id"
        parent_keys[country] = {str(row.get(key_column) or "").strip() for row in rows}
        parent_keys[country].discard("")
        rows_by_country[country] = rows
        facts_by_country[country] = facts
        if samples:
            sample_rows[country] = {
                str(stage["stage"]): [
                    str(row.get("_storage_id") or row.get("video_id") or "")
                    for row in rows
                    if _claimed_but_empty(row, stage)
                ][:20]
                for stage in stages
                if stage.get("claimed_but_empty")
            }
        countries[country] = {
            "present": True,
            "file": facts,
            "identity": identity,
            "jsonl": jsonl_check(path.with_suffix(".jsonl")),
            "stages": stages,
        }
    children_keys = child_keys(root)
    report = {
        "export_dir": str(root),
        "countries": countries,
        "children": child_audit(root, parent_keys),
        "evidence": evidence_cross_check(rows_by_country, parent_keys, children_keys),
        "manifest": manifest_audit(root, facts_by_country),
    }
    if samples:
        report["private_samples"] = sample_rows
    return report


def _claimed_but_empty(row: Mapping[str, Any], stage: Mapping[str, Any]) -> bool:
    """True for a row the stage's own status calls successful while its outputs are all empty."""
    contract = next(item for item in STAGE_CONTRACT if item.number == stage["stage"])
    outputs = tuple(
        column for column in contract.appends
        if not column.endswith(("_status", "_reason", "_error", "_model", "_version", "_runtime_ms",
                                "_generated_at", "_prompt_version", "_context_sha256"))
    )
    if not outputs:
        return False
    status = audit.status_of(row, status_columns(contract.appends))
    return status in audit.SUCCESS_STATUSES and not any(
        not audit.empty_value(row.get(column)) for column in outputs
    )


def markdown(report: Mapping[str, Any]) -> str:
    lines = ["# EP24 export audit", "",
             f"Export: `{report['export_dir']}`", ""]
    manifest = report["manifest"]
    if manifest.get("present"):
        lines += [f"Manifest: `{manifest.get('version')}` exported {manifest.get('exported_at')} · "
                  f"producer `{manifest.get('producer_revision')}` · "
                  f"SHA-256 matches: **{manifest.get('all_match')}**", ""]
    else:
        lines += ["Manifest: **absent** — integrity unverified.", ""]

    lines += ["## Files", "",
              "| country | rows | columns | bytes | SHA-256 match | BOM header | jsonl records |",
              "|---|---|---|---|---|---|---|"]
    for country, data in report["countries"].items():
        if not data.get("present"):
            lines.append(f"| {country} | missing | | | | | |")
            continue
        file = data["file"]
        declared = next((item for name, item in (manifest.get("files_compared") or {}).items()
                         if name.startswith(country)), None)
        match = "—" if declared is None else ("yes" if declared["match"] else "**NO**")
        jsonl = data["jsonl"]
        lines.append(f"| {country} | {file['rows']} | {file['columns']} | {file['bytes']:,} | {match} "
                     f"| {'**yes**' if file['header_bom'] else 'no'} "
                     f"| {jsonl.get('parseable', 0)}/{jsonl.get('records', 0)} |")
    lines.append("")

    lines += ["## Identity", "",
              "| country | rows | id present | id missing | id duplicates | video duplicates | countries in frame |",
              "|---|---|---|---|---|---|---|"]
    for country, data in report["countries"].items():
        if not data.get("present"):
            continue
        identity = data["identity"]
        lines.append(f"| {country} | {identity['rows']} | {identity['identity_present']} "
                     f"| {identity['identity_missing']} | {identity['identity_duplicates']} "
                     f"| {identity['video_id_duplicates']} "
                     f"| {', '.join(identity['countries_in_frame'])} |")
    lines.append("")

    lines += ["## Stage coverage (per country)", "",
              "`claimed/empty` is the defect worth hunting: the status says success and nothing was written.",
              "",
              "| country | stage | rows | completed | valid | failed | not started | claimed/empty | coverage |",
              "|---|---|---|---|---|---|---|---|---|"]
    for country, data in report["countries"].items():
        if not data.get("present"):
            continue
        for stage in data["stages"]:
            if not stage.get("auditable", True):
                lines.append(f"| {country} | {stage['stage']} {stage['name']} | {stage['rows']} | — | — "
                             f"| — | — | — | not auditable from this frame |")
                continue
            coverage = stage.get("coverage_pct")
            lines.append(
                f"| {country} | {stage['stage']} {stage['name']} | {stage['rows']} "
                f"| {stage['completed']} | {stage['valid']} | {stage['failed']} "
                f"| {stage['not_started']} | {stage['claimed_but_empty']} "
                f"| {'—' if coverage is None else f'{coverage}%'} |")
    lines.append("")

    lines += ["## Status vocabulary (words as written, per stage)", "",
              "| country | stage | statuses as recorded |",
              "|---|---|---|"]
    for country, data in report["countries"].items():
        if not data.get("present"):
            continue
        for stage in data["stages"]:
            vocabulary = stage.get("status_vocabulary") or {}
            written = ", ".join(f"`{word}` {count}" for word, count in vocabulary.items()) or "—"
            lines.append(f"| {country} | {stage['stage']} {stage['name']} | {written} |")
    lines.append("")

    if report.get("evidence"):
        lines += ["## Completed stage versus artefact present", "",
                  "A completed stage with no child rows is either a legitimately empty result or a silent",
                  "extraction failure; the report gives the size of that group instead of guessing.",
                  "",
                  "| country | stage | completed | with child rows | completed, no child rows |",
                  "|---|---|---|---|---|"]
        for item in report["evidence"]:
            lines.append(f"| {item['country']} | {item['stage']} {item['stage_name']} "
                         f"| {item['completed']} | {item['with_child_rows']} "
                         f"| {item['completed_without_child_rows']} |")
        lines.append("")

    lines += ["## Child tables", "",
              "| table | rows | distinct source_key | rows without key | unmatched keys |",
              "|---|---|---|---|---|"]
    for child in report["children"]:
        if not child.get("present"):
            lines.append(f"| {child['table']} | missing | | | |")
            continue
        lines.append(f"| {child['table']} | {child['rows']} | {child['distinct_source_keys']} "
                     f"| {child['rows_without_key']} | {child['unmatched_source_keys']} |")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit an exported EP24 FI/PL/PT bundle (#405).")
    parser.add_argument("--export-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="JSON report path")
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--samples", type=Path, default=None,
                        help="PRIVATE ONLY: row references for claimed-but-empty stages")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit_export(args.export_dir, samples=args.samples is not None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown(report), encoding="utf-8")
    if args.samples:
        args.samples.parent.mkdir(parents=True, exist_ok=True)
        args.samples.write_text(json.dumps(report.get("private_samples", {}), indent=2),
                                encoding="utf-8")
    totals = {
        country: sum(stage.get("claimed_but_empty", 0) for stage in data.get("stages", []))
        for country, data in report["countries"].items() if data.get("present")
    }
    print(json.dumps({"claimed_but_empty_by_country": totals,
                      "manifest_all_match": report["manifest"].get("all_match")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
