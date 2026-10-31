"""The offline export audit must catch the defects it was written for (#333).

Each fixture below is a tiny artifact set with one planted defect, so a green audit means something:
a healthy export must produce no errors, and each known failure mode must be named rather than
averaged away. Everything is synthetic.
"""
from __future__ import annotations

import csv
import gzip
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = REPO_ROOT / "scripts" / "ep24" / "audit_export.py"
sys.path.insert(0, str(REPO_ROOT))


def load_tool():
    spec = importlib.util.spec_from_file_location("audit_export_333", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    return path


def build(tmp_path: Path, *, orphan: bool = False, counter: str = "1") -> Path:
    """A minimal healthy export: two FI videos, one with a statement in the child table."""
    rows = [
        {"record_id": "video:a.mp4", "country": "Finland", "_storage_id": "h-a",
         "dna_statement_count": counter, "dna_status": "ok", "dna_analysis_markdown": "md"},
        {"record_id": "video:b.mp4", "country": "Finland", "_storage_id": "h-b",
         "dna_statement_count": "0", "dna_status": "", "dna_analysis_markdown": ""},
    ]
    write_csv(tmp_path / "combined.csv", rows)
    write_csv(tmp_path / "finland.csv", rows)
    children = [{"statement_id": "s1", "source_record_id": "h-a", "parent_record_id": "video:a.mp4",
                 "evidence_quote": "quote"}]
    if orphan:
        children.append({"statement_id": "s9", "source_record_id": "h-gone",
                         "parent_record_id": "", "evidence_quote": "quote"})
    write_csv(tmp_path / "children" / "finland_dna_statements.csv", children)
    with gzip.open(tmp_path / "finland_text.jsonl.gz", "wt", encoding="utf-8") as stream:
        stream.write(json.dumps({"record_id": "video:a.mp4", "dna_statements_json": "[]"}) + "\n")
    manifest = {
        "exporter_version": "ep24-mongo-export-1",
        "generated_at": "2026-10-31T00:00:00+00:00",
        "record_unit": "video",
        "children_unit": "video",
        "records": 2,
        "record_counts": {"finland": {"exported": 2}},
        "children": {"finland": {"dna_statements": len(children)}},
        "children_detail": {"finland": {"dna_statements": {
            "records": len(children), "matched": 1, "orphan": len(children) - 1}}},
        "heavy_fields": {"finland": {"dna_statements_json": 1}},
        "text_bundle": {"suffix": "_text.jsonl.gz", "key": "record_id", "threshold": 1000},
        "stage_coverage": {"finland": {
            "7_dna": {"records": 1, "available": True, "failed_or_other": 0}}},
        "outputs": [{"name": "combined.csv", "sha256": None}],
    }
    path = tmp_path / "MANIFEST.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest["outputs"][0]["sha256"] = __import__("hashlib").sha256(
        (tmp_path / "combined.csv").read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def run(export: Path, *extra: str) -> tuple[int, dict]:
    out = export / "audit.json"
    code = load_tool().main(["--export", str(export), "--json", str(out), "--quiet", *extra])
    return code, json.loads(out.read_text(encoding="utf-8"))


def codes(report: dict) -> set[str]:
    return {anomaly["code"] for anomaly in report["anomalies"]}


def test_a_healthy_export_has_no_errors(tmp_path: Path):
    code, report = run(build(tmp_path))
    assert report["summary"]["errors"] == 0
    assert code == 0
    dna = next(row for row in report["coverage"] if row["step"] == "7_dna")
    assert (dna["manifest_records"], dna["recomputable_from_csv"]) == (1, 1)
    assert report["children"][0]["rows"] == 1


def test_an_orphan_child_is_reported_without_failing_the_audit(tmp_path: Path):
    code, report = run(build(tmp_path, orphan=True))
    assert report["children"][0]["orphans_by_parent_key"] == 1
    assert report["summary"]["errors"] == 0
    assert code == 0


def test_a_counter_that_disagrees_with_the_children_is_a_warning(tmp_path: Path):
    code, report = run(build(tmp_path, counter="5"))
    assert "children-counter-mismatch" in codes(report)
    assert report["summary"]["errors"] == 0


def test_a_tampered_artifact_is_an_error(tmp_path: Path):
    export = build(tmp_path)
    with (export / "combined.csv").open("a", encoding="utf-8") as stream:
        stream.write("video:z.mp4,Finland,h-z\n")
    code, report = run(export)
    assert "sha256-mismatch" in codes(report)
    assert code == 1


def test_a_manifest_claim_the_csv_cannot_reproduce_is_an_error(tmp_path: Path):
    export = build(tmp_path)
    manifest = json.loads((export / "MANIFEST.json").read_text(encoding="utf-8"))
    manifest["heavy_fields"] = {}
    manifest["stage_coverage"]["finland"]["7_dna"] = {"records": 2, "available": True}
    rows = [{"record_id": "video:a.mp4", "country": "Finland", "_storage_id": "h-a",
             "dna_status": "ok", "dna_analysis_markdown": ""}]
    write_csv(export / "combined.csv", rows)
    import hashlib
    manifest["outputs"][0]["sha256"] = hashlib.sha256(
        (export / "combined.csv").read_bytes()).hexdigest()
    (export / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    code, report = run(export)
    assert "coverage-unreproducible" in codes(report)
    assert code == 1


def test_coverage_and_children_disagreement_is_surfaced(tmp_path: Path):
    export = build(tmp_path)
    manifest = json.loads((export / "MANIFEST.json").read_text(encoding="utf-8"))
    manifest["children_detail"]["finland"]["dna_statements"]["matched"] = 0
    (export / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    code, report = run(export)
    assert "coverage-without-children" in codes(report)


def test_the_tool_runs_as_a_script(tmp_path: Path):
    export = build(tmp_path)
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT))
    completed = subprocess.run([sys.executable, str(TOOL), "--export", str(export), "--quiet"],
                               capture_output=True, text=True, env=env, check=False)
    assert completed.returncode == 0, completed.stderr
    assert "summary:" in completed.stdout
