"""An elided cell is not missing work, and a paused stage is not a failed one (#342, #331, #333, #346).

The pinned export moves every cell over 1,000 characters into `<country>_text.jsonl.gz`, so Step 3's
**entire** output is absent from `combined.csv`. A planner reading that blank column as "no output"
schedules all 4,016 records that already have video output — the exact waste #342 exists to prevent.

These tests pin the third bucket (`unverifiable`: reported, never scheduled), the manifest's
`paused_stages` statement, and the audit's refusal to call a paused stage anomalous.
"""
from __future__ import annotations

import csv
import gzip
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import ep24_mongo_export as export  # noqa: E402


def load_planner():
    spec = importlib.util.spec_from_file_location(
        "plan_ep24_retry_elision", REPO_ROOT / "scripts" / "ep24" / "plan_ep24_retry.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def build_export(tmp_path: Path, *, bundled: bool) -> Path:
    rows = [
        {"record_id": "video:a", "country": "Finland", "vllm_video_status": "ok",
         "vllm_video_analysis": "", "laclau_status": "ok", "laclau_structured_json": "{}"},
        {"record_id": "video:b", "country": "Finland", "vllm_video_status": "error",
         "vllm_video_analysis": "", "laclau_status": "", "laclau_structured_json": ""},
    ]
    with (tmp_path / "combined.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader()
        writer.writerows(rows)
    if bundled:
        with gzip.open(tmp_path / "finland_text.jsonl.gz", "wt", encoding="utf-8") as stream:
            stream.write(json.dumps({"record_id": "video:a", "vllm_video_analysis": "analysis"}) + "\n")
    return tmp_path


def test_bundled_output_is_unverifiable_not_missing(tmp_path: Path):
    planner = load_planner()
    plan = planner.plan_for_export(build_export(tmp_path, bundled=True), ("3_video",), ["finland"])
    entry = plan["countries"]["finland"]["steps"]["3_video"]
    assert entry["missing"] == 0, "a bundled cell is not evidence of missing work"
    assert entry["unverifiable"] == 2


def test_without_a_bundle_a_blank_cell_is_missing_work(tmp_path: Path):
    planner = load_planner()
    plan = planner.plan_for_export(build_export(tmp_path, bundled=False), ("3_video",), ["finland"])
    entry = plan["countries"]["finland"]["steps"]["3_video"]
    assert entry["missing"] == 2 and entry["unverifiable"] == 0


def test_a_record_holding_output_is_never_scheduled(tmp_path: Path):
    planner = load_planner()
    plan = planner.plan_for_export(build_export(tmp_path, bundled=False), ("6_discourse",), ["finland"])
    entry = plan["countries"]["finland"]["steps"]["6_discourse"]
    assert entry["missing"] == 1
    assert "video:a" not in entry["record_ids"], "output is evidence: never rerun over it"


def test_failed_status_with_output_is_reported_not_retried():
    planner = load_planner()
    rows = [{"record_id": "video:c", "country": "Finland", "laclau_status": "error",
             "laclau_structured_json": '{"structured": true}'}]
    plan = planner.select(rows, ("6_discourse",))
    assert plan["6_discourse"]["missing"] == []
    assert plan["6_discourse"]["failed_with_output"] == ["video:c"]


def test_the_manifest_states_which_stages_were_paused(tmp_path: Path):
    manifest = export.write_manifest(
        tmp_path, record_unit="video", countries={"finland": {"records": 1}},
        child_counts={}, collections_read={}, provenance={}, outputs=[],
        contract={"stage_coverage": {"finland": {"8_sna": {"records": 3, "available": True}}}})
    assert manifest["paused_stages"] == [8, 9], "paused by default in the active run (#346)"
    # scope never rewrites history: coverage still reports what the data holds
    assert manifest["stage_coverage"]["finland"]["8_sna"]["records"] == 3
    assert json.loads((tmp_path / "MANIFEST.json").read_text())["paused_stages"] == [8, 9]


def test_the_audit_does_not_call_a_paused_stage_anomalous(tmp_path: Path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "audit_elision", REPO_ROOT / "scripts" / "ep24" / "audit_export.py")
    audit = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(audit)
    anomalies: list[dict] = []
    frame = [{"record_id": "video:a", "country": "Finland"}]
    manifest = {"stage_coverage": {"finland": {"8_sna": {"records": 5, "available": True}}},
                "paused_stages": [8, 9]}
    rows = audit.audit_coverage(frame, manifest, "finland", anomalies)
    sna = next(row for row in rows if row["step"] == "8_sna")
    assert sna["paused"] is True
    assert not [a for a in anomalies if a.get("step") == "8_sna"]
