#!/usr/bin/env python3
"""Offline audit of an EP24 export directory (#333).

`scripts/ep24/export_mongo_results.py` writes an artifact set; this reads it back and answers the
questions a researcher asks before trusting it:

* does every artifact still match the manifest's own sha256?
* for each country and stage 0-9: how many records carry output, is the stage available at all, and
  can the delivered CSV actually reproduce that number?
* do the child tables reconcile — rows in file, rows attached to a parent in this export, orphans,
  and the per-record counters the index carries?
* is anything internally inconsistent: duplicate identities, cross-country contamination, malformed
  JSON, bundle keys with no row?

Read-only, stdlib only, no network, no database. It prints **counts and identifiers of columns**, never
cell contents, so its output is safe to paste into a public issue. Exit status is 1 when an
ERROR-level anomaly is found, so it can gate a pipeline; `--json` writes the same report.

    python scripts/ep24/audit_export.py --export /private/ep24/mongo_export_210
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

try:  # reuse the producer's own step definitions so the audit cannot drift from the exporter
    from ep24_mongo_export import COVERAGE_STEPS, _has_output
except ImportError:  # pragma: no cover - only when run outside the repository
    COVERAGE_STEPS = {}
    def _has_output(value):  # type: ignore[misc]
        return bool(str(value or "").strip() and str(value).strip() not in {"0", "[]", "{}"})

COUNTRIES = ("finland", "poland", "portugal")
CHILD_TABLES = ("entities", "themes", "dna_statements", "sna")
MISSING = {"", "none", "nan", "null", "<na>", "[]", "{}"}
ERROR, WARN, INFO = "error", "warn", "info"


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def present(value: Any) -> bool:
    return bool(str(value or "").strip()) and str(value).strip().lower() not in MISSING


def audit_artifacts(export: Path, manifest: dict[str, Any],
                    anomalies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for entry in manifest.get("outputs") or []:
        path = export / str(entry.get("name") or "")
        if not path.is_file():
            anomalies.append({"severity": ERROR, "code": "missing-output-file",
                              "detail": f"{path.name} is listed in the manifest and is not present"})
            continue
        actual = sha256(path)
        match = actual == entry.get("sha256")
        if not match:
            anomalies.append({"severity": ERROR, "code": "sha256-mismatch",
                              "detail": f"{path.name} differs from the manifest's hash"})
        rows.append({"name": path.name, "bytes": path.stat().st_size, "sha256_match": match})
    for name in ("combined.csv",):
        if not (export / name).is_file():
            anomalies.append({"severity": ERROR, "code": "missing-output-file",
                              "detail": f"{name} is absent; nothing can be audited"})
    return rows


def audit_coverage(frame: list[dict[str, str]], manifest: dict[str, Any],
                   country: str, anomalies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Manifest coverage next to what the delivered CSV can reproduce, per stage."""
    reported = ((manifest.get("stage_coverage") or {}).get(country) or {})
    heavy = ((manifest.get("heavy_fields") or {}).get(country) or {})
    rows = []
    for step, (status_field, outputs) in COVERAGE_STEPS.items():
        entry = reported.get(step) or {}
        reproducible = sum(1 for row in frame
                           if any(_has_output(row.get(field)) for field in outputs))
        carried = [field for field in outputs if field in (frame[0] if frame else {})]
        elided = [field for field in outputs if field in heavy or any(
            field in str(row.get("heavy_fields_in_text_bundle") or "").split(",")
            for row in frame[:50])]
        rows.append({
            "country": country, "step": step,
            "manifest_records": int(entry.get("records") or 0),
            "available": entry.get("available"),
            "failed_or_other": int(entry.get("failed_or_other") or 0),
            "recomputable_from_csv": reproducible,
            "output_columns_present": len(carried),
            "output_columns_elided": elided,
        })
        claimed = int(entry.get("records") or 0)
        if claimed and not reproducible and not elided and entry.get("evidence_source") != "children":
            anomalies.append({
                "severity": ERROR, "code": "coverage-unreproducible", "country": country,
                "step": step,
                "detail": (f"manifest claims {claimed} records with this step's output, but no "
                           f"delivered column carries it and no elision is declared"),
            })
        if elided and reproducible:
            anomalies.append({
                "severity": INFO, "code": "coverage-carried-twice", "country": country, "step": step,
                "detail": "some rows keep the cell and some moved it to the text bundle",
            })
        if claimed and not carried and not elided and entry.get("evidence_source") != "children":
            anomalies.append({
                "severity": WARN, "code": "output-column-absent", "country": country, "step": step,
                "detail": "none of the step's output columns exist in the delivered CSV",
            })
    return rows


def audit_children(export: Path, frame: list[dict[str, str]], manifest: dict[str, Any],
                   country: str, anomalies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    detail = ((manifest.get("children_detail") or {}).get(country) or {})
    report = []
    for table in CHILD_TABLES:
        path = export / "children" / f"{country}_{table}.csv"
        if not path.is_file():
            declared = int(((manifest.get("children") or {}).get(country) or {}).get(table) or 0)
            if declared:
                anomalies.append({
                    "severity": ERROR, "code": "children-file-absent", "country": country,
                    "table": table, "detail": f"manifest declares {declared} rows, file is absent",
                })
            continue
        rows = read_rows(path)
        units = manifest.get("children_unit")
        if isinstance(units, dict) and units.get(table) == "source_collection":
            count = ((manifest.get("children") or {}).get(country) or {}).get(table)
            report.append({"country": country, "table": table, "rows": len(rows),
                           "declared_records": count, "unit": "source_collection"})
            if count not in (None, len(rows)):
                anomalies.append({"severity": ERROR, "code": "children-count-drift",
                                  "country": country, "table": table,
                                  "detail": "source collection row count differs from manifest"})
            continue
        has_parent_key = bool(rows) and "parent_record_id" in rows[0]
        if rows and not has_parent_key:
            anomalies.append({
                "severity": WARN, "code": "children-no-parent-key", "country": country,
                "table": table,
                "detail": (f"{len(rows)} rows predate parent_record_id: a consumer must fall back to "
                           f"source_record_id -> parent _storage_id, and cannot tell an orphan from a "
                           f"matched row"),
            })
        ids = {str(row.get("parent_record_id") or "") for row in rows} - {""}
        parents = {str(row.get("record_id") or "") for row in frame}
        orphan = (sum(1 for row in rows if not str(row.get("parent_record_id") or "").strip())
                  if has_parent_key else None)
        stray = len(ids - parents)
        stray += sum(1 for row in rows if row.get("parent_country")
                     and row["parent_country"].lower() != country)
        if table == "dna_statements":
            entry = ((manifest.get("stage_coverage") or {}).get(country) or {}).get("7_dna") or {}
            if entry.get("evidence_source") == "children" and (
                int(entry.get("records") or 0) != len(ids & parents)
                or int(entry.get("statement_rows") or 0) != len(rows)
            ):
                anomalies.append({"severity": ERROR, "code": "dna-coverage-drift",
                                  "country": country, "detail": "DNA coverage disagrees with child file"})
        declared = detail.get(table) or {}
        report.append({
            "country": country, "table": table, "rows": len(rows),
            "declared_records": declared.get("records"), "declared_matched": declared.get("matched"),
            "declared_orphan": declared.get("orphan"),
            "attached_to_export": len(ids & parents) if ids else None,
            "orphans_by_parent_key": orphan, "parents_not_in_export": stray,
        })
        if declared.get("records") not in (None, len(rows)):
            anomalies.append({
                "severity": ERROR, "code": "children-count-drift", "country": country,
                "table": table,
                "detail": f"manifest says {declared.get('records')} rows, file has {len(rows)}",
            })
        if stray:
            anomalies.append({
                "severity": ERROR, "code": "children-parent-missing", "country": country,
                "table": table,
                "detail": f"{stray} parent ids attached to children are absent from the export",
            })
    # The index's own counters are the second opinion on how much one-to-many output exists.
    counters = {"dna_statements": "dna_statement_count", "entities": "entity_count",
                "themes": "theme_count", "sna": "sna_edge_count"}
    for table, field in counters.items():
        if not frame or field not in frame[0]:
            continue
        total = 0
        for row in frame:
            try:
                total += int(str(row.get(field) or "0").strip() or 0)
            except ValueError:
                anomalies.append({"severity": WARN, "code": "counter-not-integer",
                                  "country": country, "detail": f"{field} holds a non-integer"})
                break
        matched = ((detail.get(table) or {}).get("matched"))
        report.append({"country": country, "table": table, "counter": field,
                       "counter_sum": total, "declared_matched": matched})
        if matched is not None and total and int(matched) != total:
            anomalies.append({
                "severity": WARN, "code": "children-counter-mismatch", "country": country,
                "table": table,
                "detail": (f"per-record {field} sums to {total} but {matched} child rows are "
                           f"attached to an exported parent"),
            })
    return report


def audit_frame(export: Path, anomalies: list[dict[str, Any]]) -> tuple[dict[str, list], list[tuple[str, str]]]:
    """Per-country rows, plus identity/contamination/JSON checks that need the whole frame."""
    by_country: dict[str, list[dict[str, str]]] = defaultdict(list)
    for name in ("combined.csv", *(f"{country}.csv" for country in COUNTRIES)):
        path = export / name
        if not path.is_file():
            continue
        key = "combined" if name == "combined.csv" else name[: -len(".csv")]
        by_country[key] = read_rows(path)
    frame = by_country.get("combined") or []
    keys = [(str(row.get("country") or "").lower(), str(row.get("record_id") or "")) for row in frame]
    duplicates = [key for key, count in Counter(keys).items() if count > 1 and key[1]]
    if duplicates:
        anomalies.append({"severity": ERROR, "code": "duplicate-record-id",
                          "detail": f"{len(duplicates)} record_id values repeat in combined.csv"})
    for row in frame:
        country = str(row.get("country") or "").lower()
        if country not in COUNTRIES:
            anomalies.append({"severity": WARN, "code": "country-unknown",
                              "detail": f"unexpected country value {country!r}"})
            break
    malformed: Counter[str] = Counter()
    for row in frame:
        for field, value in row.items():
            text = str(value or "").strip()
            if text[:1] in ("{", "[") and text[-1:] in ("}", "]"):
                try:
                    json.loads(text)
                except ValueError:
                    malformed[str(field)] += 1
    if malformed:
        top = ", ".join(f"{name}={count}" for name, count in malformed.most_common(5))
        anomalies.append({"severity": WARN, "code": "structured-cell-not-json",
                          "detail": (f"{sum(malformed.values())} cells hold structured text that is "
                                     f"not JSON (Python repr from an upstream step): {top}")})
    return by_country, keys


def audit_bundle(export: Path, manifest: dict[str, Any], keys: list[tuple[str, str]],
                 anomalies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    report = []
    known = set(keys)
    for path in sorted(export.glob("*_text.jsonl.gz")):
        seen = 0
        unknown = 0
        fields: Counter[str] = Counter()
        country = path.name.removesuffix("_text.jsonl.gz")
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                seen += 1
                payload = json.loads(line)
                if (country, str(payload.get("record_id") or "")) not in known:
                    unknown += 1
                fields.update(name for name in payload if name != "record_id")
        report.append({"bundle": path.name, "records": seen, "ids_not_in_index": unknown,
                       "fields": len(fields)})
        if unknown:
            anomalies.append({"severity": ERROR, "code": "bundle-key-not-in-index",
                              "detail": f"{path.name}: {unknown} entries keyed by an unknown record_id"})
        declared = sum(int(count) for count in
                       ((manifest.get("heavy_fields") or {}).get(country) or {}).values())
        if declared and not seen:
            anomalies.append({"severity": ERROR, "code": "bundle-empty-but-declared",
                              "detail": f"{path.name} is empty while heavy_fields declares rows"})
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Offline, privacy-safe audit of an EP24 export directory (#333)")
    parser.add_argument("--export", required=True, type=Path,
                        help="directory holding MANIFEST.json, combined.csv, children/, bundles")
    parser.add_argument("--json", type=Path, help="write the full report here")
    parser.add_argument("--quiet", action="store_true", help="only the anomaly list and summary")
    args = parser.parse_args(argv)

    export = args.export.expanduser().resolve()
    manifest = read_json(export / "MANIFEST.json")
    anomalies: list[dict[str, Any]] = []
    if not manifest:
        anomalies.append({"severity": ERROR, "code": "manifest-missing",
                          "detail": "MANIFEST.json absent or unparseable"})

    artifacts = audit_artifacts(export, manifest, anomalies)
    by_country, keys = audit_frame(export, anomalies)
    frame = by_country.get("combined") or []
    coverage, children = [], []
    for country in COUNTRIES:
        country_frame = ([row for row in frame
                          if str(row.get("country") or "").lower() == country]
                         or by_country.get(country) or [])
        if not country_frame:
            continue
        coverage.extend(audit_coverage(country_frame, manifest, country, anomalies))
        children.extend(audit_children(export, country_frame, manifest, country, anomalies))
    bundles = audit_bundle(export, manifest, keys, anomalies)
    warnings = coverage_reconciliation(manifest, suppressed_countries(export, manifest, anomalies),
                                      anomalies)

    report = {
        "export": str(export), "manifest": {
            "exporter_version": manifest.get("exporter_version"),
            "record_unit": manifest.get("record_unit"),
            "generated_at": manifest.get("generated_at"),
            "records": manifest.get("records"),
            "children_unit": manifest.get("children_unit"),
            "text_bundle": manifest.get("text_bundle"),
        },
        "artifacts": artifacts, "coverage": coverage, "children": children,
        "bundles": bundles, "manifest_warnings": warnings, "anomalies": anomalies,
        "summary": {
            "errors": sum(1 for a in anomalies if a["severity"] == ERROR),
            "warnings": sum(1 for a in anomalies if a["severity"] == WARN),
            "info": sum(1 for a in anomalies if a["severity"] == INFO),
        },
    }
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if not args.quiet:
        print(f"export {report['manifest']['exporter_version']} unit={report['manifest']['record_unit']} "
              f"generated={report['manifest']['generated_at']} records={report['manifest']['records']}")
        print("\ncoverage (manifest vs delivered CSV):")
        for row in coverage:
            elided = ",".join(row["output_columns_elided"]) or "-"
            print(f"  {row['country']:<9} {row['step']:<13} manifest={row['manifest_records']:>5} "
                  f"csv={row['recomputable_from_csv']:>5} available={str(row['available']):<5} "
                  f"elided={elided}")
        print("\nchildren:")
        for row in children:
            print(f"  {row['country']:<9} {row['table']:<15} " + " ".join(
                f"{k}={row[k]}" for k in row if k not in ("country", "table")))
        print("\nbundles:")
        for row in bundles:
            print(f"  {row['bundle']:<28} records={row['records']:>5} "
                  f"unknown_ids={row['ids_not_in_index']} fields={row['fields']}")
    if anomalies:
        print("\nanomalies:")
        for anomaly in anomalies:
            where = " ".join(str(anomaly[key]) for key in ("country", "step", "table") if anomaly.get(key))
            print(f"  {anomaly['severity'].upper():<5} {anomaly['code']:<28} {where:<24} "
                  f"{anomaly['detail']}")
    print(f"\nsummary: {report['summary']['errors']} errors, {report['summary']['warnings']} warnings, "
          f"{report['summary']['info']} info")
    return 1 if report["summary"]["errors"] else 0


def suppressed_countries(export: Path, manifest: dict[str, Any],
                        anomalies: list[dict[str, Any]]) -> set[str]:
    """Countries whose children predate ``children_detail``: one finding, not one per step.

    Without the detail block the exporter's reconciliation cannot tell "no child was exported" from
    "this manifest is older than the contract", and a warning per step reads like seven defects where
    there is one.
    """
    detail = manifest.get("children_detail") or {}
    missing = {country for country in COUNTRIES
               if country not in detail and (export / "children").is_dir()
               and any((export / "children").glob(f"{country}_*.csv"))}
    for country in sorted(missing):
        rows = sum(len(read_rows(path)) for path in (export / "children").glob(f"{country}_*.csv"))
        anomalies.append({
            "severity": WARN, "code": "children-contract-missing", "country": country,
            "detail": (f"{rows} child rows are exported without children_detail/parent_record_id: "
                       f"this export predates the #330 contract, so child counts cannot be "
                       f"reconciled against coverage from the manifest alone"),
        })
    return missing


def coverage_reconciliation(manifest: dict[str, Any], suppressed: set[str] | None = None,
                            anomalies: list[dict[str, Any]] | None = None
                            ) -> list[dict[str, Any]]:
    """Reuse the exporter's own reconciliation, when it is importable."""
    try:
        from ep24_mongo_export import coverage_reconciliation as reconcile
    except ImportError:  # pragma: no cover
        return []
    warnings = [warning for warning in reconcile(manifest)
                if not (warning["kind"] == "coverage-without-children"
                        and warning.get("country") in (suppressed or set()))]
    anomalies = [] if anomalies is None else anomalies
    for warning in warnings:
        anomalies.append({"severity": WARN, "code": warning["kind"],
                          "country": warning.get("country"), "step": warning.get("step"),
                          "detail": warning["detail"]})
    return warnings


if __name__ == "__main__":
    raise SystemExit(main())

