"""A retry plan must not schedule work the export cannot prove is missing (#342, #331, #333).

The pinned export blanks every cell over 1,000 characters into `<country>_text.jsonl.gz`. Step 3's whole
output lives there. A planner that reads the blank column as "no output" therefore schedules all 4,866
records for a Step 3 rerun that would change nothing — and burns the GPU hours #342 exists to protect.

These tests pin the third bucket: `unverifiable`, reported and never scheduled, alongside the invariant
that a record holding output is never scheduled and that paused Steps 8-9 stay out of a default plan.
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


def load_planner():
    spec = importlib.util.spec_from_file_location(
        "plan_ep24_retry_342", REPO_ROOT / "scripts" / "ep24" / "plan_ep24_retry.py")
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
    export = build_export(tmp_path, bundled=True)
    plan = planner.plan_for_export(export, ("3_video",), ["finland"])
    entry = plan["countries"]["finland"]["steps"]["3_video"]
    assert entry["missing"] == 0, "a bundled cell is not evidence of missing work"
    assert entry["unverifiable"] == 2
    assert sorted(entry["unverifiable_ids"]) == ["video:a", "video:b"]


def test_without_a_bundle_a_blank_cell_is_missing_work(tmp_path: Path):
    planner = load_planner()
    export = build_export(tmp_path, bundled=False)
    plan = planner.plan_for_export(export, ("3_video",), ["finland"])
    entry = plan["countries"]["finland"]["steps"]["3_video"]
    assert entry["missing"] == 2 and entry["unverifiable"] == 0


def test_a_record_that_holds_output_is_never_scheduled(tmp_path: Path):
    planner = load_planner()
    export = build_export(tmp_path, bundled=False)
    plan = planner.plan_for_export(export, ("6_discourse",), ["finland"])
    entry = plan["countries"]["finland"]["steps"]["6_discourse"]
    assert entry["missing"] == 1
    assert "video:a" not in entry["record_ids"], "output is evidence: never rerun over it"


def test_failed_status_with_output_is_reported_not_retried(tmp_path: Path):
    planner = load_planner()
    rows = [{"record_id": "video:c", "country": "Finland", "laclau_status": "error",
             "laclau_structured_json": '{"structured": true}'}]
    plan = planner.select(rows, ("6_discourse",))
    assert plan["6_discourse"]["missing"] == []
    assert plan["6_discourse"]["failed_with_output"] == ["video:c"]


def test_paused_steps_are_excluded_from_a_default_plan():
    planner = load_planner()
    normal = tuple(step for step in planner.STEPS if not step.startswith(("8_", "9_")))
    assert normal == tuple(planner.STEPS[:8]), "a normal plan stops at Step 7 (#346)"
