"""The delivered-export contract auditor (#330, #332, #333): re-derives counts and fails on drift."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import ep24_mongo_export as export

_SPEC = importlib.util.spec_from_file_location(
    "audit_export_contract",
    Path(__file__).resolve().parents[1] / "scripts" / "ep24" / "audit_export_contract.py",
)
audit_mod = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(audit_mod)


def build_export(root: Path) -> Path:
    rows = [
        {"record_id": "video:a.mp4", "_storage_id": "s1", "dna_statement_count": 2,
         "dna_statements_json": ""},
        {"record_id": "video:b.mp4", "_storage_id": "s2", "dna_statement_count": 0,
         "dna_statements_json": ""},
    ]
    export.write_rows(root / "combined.csv", rows)
    export.write_rows(root / "finland.csv", rows)
    children = [
        {"_storage_id": "st1", "source_record_id": "s1", "parent_record_id": "video:a.mp4",
         "parent_join": "source_record_id"},
        {"_storage_id": "st2", "source_record_id": "s1", "parent_record_id": "video:a.mp4",
         "parent_join": "source_record_id"},
        {"_storage_id": "st3", "source_record_id": "gone", "parent_record_id": "",
         "parent_join": "unjoined"},
    ]
    export.write_rows(root / "children" / "finland_dna_statements.csv", children)
    contract = {
        "finland": {"dna_statements": export.child_contract(
            children, country="finland", role="dna_statements")}
    }
    coverage = export.dashboard_contract(
        {"finland": {"records": 2, "source_documents": 2}},
        {"finland": rows},
    )["stage_coverage"]
    export.write_manifest(
        root,
        record_unit="video",
        countries={"finland": {"records": 2}},
        child_counts={"finland": {"dna_statements": 3}},
        collections_read={"finland": {"dna_statements": "x"}},
        provenance={},
        outputs=[root / "combined.csv"],
        contract={"stage_coverage": coverage},
        children_contract={"unit": "video", "countries": contract},
        heavy_splits={"finland": {"threshold": 1000, "records_with_heavy_fields": 0, "fields": {}}},
    )
    return root


class TestExportContractAudit:
    def test_a_consistent_export_passes(self, tmp_path: Path):
        report = audit_mod.audit(build_export(tmp_path))
        assert report["ok"], report["problems"]
        assert report["children"]["finland/dna_statements"]["unjoined_rows"] == 1

    def test_manifest_child_count_drift_fails(self, tmp_path: Path):
        root = build_export(tmp_path)
        manifest = json.loads((root / "MANIFEST.json").read_text())
        manifest["children"]["finland"]["dna_statements"] = 99
        (root / "MANIFEST.json").write_text(json.dumps(manifest))
        report = audit_mod.audit(root)
        assert not report["ok"]
        assert any("manifest 99 rows" in problem for problem in report["problems"])

    def test_child_join_drift_fails(self, tmp_path: Path):
        root = build_export(tmp_path)
        manifest = json.loads((root / "MANIFEST.json").read_text())
        manifest["children_contract"]["countries"]["finland"]["dna_statements"]["joined_rows"] = 99
        (root / "MANIFEST.json").write_text(json.dumps(manifest))
        report = audit_mod.audit(root)
        assert not report["ok"]
        assert any("joined_rows" in problem for problem in report["problems"])

    def test_coverage_that_contradicts_the_count_fails(self, tmp_path: Path):
        root = build_export(tmp_path)
        manifest = json.loads((root / "MANIFEST.json").read_text())
        manifest["stage_coverage"]["finland"]["7_dna"]["records"] = 2
        (root / "MANIFEST.json").write_text(json.dumps(manifest))
        report = audit_mod.audit(root)
        assert not report["ok"]
        assert any("dna_statement_count" in problem for problem in report["problems"])
