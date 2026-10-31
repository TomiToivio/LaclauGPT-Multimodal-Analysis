"""--records: per-record selection that fails closed (#342).

The contract: a named record narrows the manifest; an unknown or empty id list is refused rather than
silently selecting less; and naming a record never overrides eligibility, so verified output is still
never rescheduled.
"""
from __future__ import annotations

import csv
import json
import sys
from importlib import util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = util.spec_from_file_location("reprocess_selected", ROOT / "scripts" / "roihu" / "reprocess_selected.py")
_mod = util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def _export(tmp_path: Path, rows: list[dict]) -> Path:
    export = tmp_path / "export"
    export.mkdir()
    with (export / "finland.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return export


def test_records_narrows_the_manifest(tmp_path, capsys):
    export = _export(tmp_path, [
        {"record_id": "r1", "country": "finland", "laclau_status": "error"},
        {"record_id": "r2", "country": "finland", "laclau_status": "error"},
    ])
    code = _mod.main(["--export", str(export), "--steps", "6", "--records", "r1", "--json"])
    manifest = json.loads(capsys.readouterr().out)
    assert code == 0
    assert {entry["record_id"] for entry in manifest["selected"]} == {"r1"}
    assert manifest["source"]["records"] == ["r1"]


def test_unknown_record_fails_closed(tmp_path, capsys):
    export = _export(tmp_path, [{"record_id": "r1", "country": "finland", "laclau_status": "error"}])
    code = _mod.main(["--export", str(export), "--steps", "6", "--records", "nope"])
    assert code == 2
    assert "nope" in capsys.readouterr().err


def test_empty_records_list_fails_closed(tmp_path):
    export = _export(tmp_path, [{"record_id": "r1", "country": "finland", "laclau_status": "error"}])
    assert _mod.main(["--export", str(export), "--steps", "6", "--records", " , "]) == 2


def test_naming_a_record_never_reschedules_verified_output(tmp_path, capsys):
    export = _export(tmp_path, [{"record_id": "r1", "country": "finland", "laclau_summary_md": "summary"}])
    code = _mod.main(["--export", str(export), "--steps", "6", "--records", "r1", "--json"])
    manifest = json.loads(capsys.readouterr().out)
    assert code == 0
    assert manifest["selected"] == []


def test_preview_honors_country_specific_heavy_fields(tmp_path, capsys):
    export = _export(tmp_path, [{"record_id": "fi", "country": "finland", "laclau_status": "error"}])
    (export / "poland.csv").write_text("record_id,country,laclau_status\npl,poland,error\n")
    (export / "MANIFEST.json").write_text(json.dumps({
        "heavy_fields": {"poland": ["laclau_structured_json"]}}))
    assert _mod.main(["--export", str(export), "--steps", "6", "--json"]) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert [entry["record_id"] for entry in manifest["selected"]] == ["fi"]
    assert manifest["skipped"][0]["record_id"] == "pl"
    assert manifest["skipped"][0]["reason_code"] == "unverifiable_output"


def test_named_record_with_relocated_video_is_not_rescheduled(tmp_path, capsys):
    export = _export(tmp_path, [{"record_id": "r1", "country": "finland", "vllm_video_status": "error"}])
    (export / "MANIFEST.json").write_text(json.dumps({
        "heavy_fields": {"finland": ["vllm_video_structured_json"]}}))
    assert _mod.main(["--export", str(export), "--steps", "3", "--records", "r1", "--json"]) == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["selected"] == []
    assert manifest["skipped"][0]["reason_code"] == "unverifiable_output"
