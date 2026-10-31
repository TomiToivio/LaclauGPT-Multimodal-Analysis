"""The auditor must recognise an elided cell, not report it as missing work.

`scripts/ep24/audit_export.py` answers "can the delivered CSV reproduce this stage's output?". When a field
is longer than the text threshold it leaves the row for the text bundle, and the lean row says so in
`_text_bundle_fields`. The auditor used to read a comma-joined `heavy_fields_in_text_bundle`; when the
exporter settled on the JSON spelling, reading the old key returned nothing, so **every elided cell was
audited as output the stage never produced** — a false finding that looks authoritative.

These tests pin the reader to the spelling the exporter actually writes.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ep24" / "audit_export.py"


def _audit():
    spec = importlib.util.spec_from_file_location("audit_elision", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_it_reads_the_json_list_the_exporter_writes():
    audit = _audit()
    assert audit._elided_fields({"_text_bundle_fields": '["asr_transcript"]'}) == {"asr_transcript"}
    assert audit._elided_fields({"_text_bundle_fields": '["a_text", "b_text"]'}) == {"a_text", "b_text"}


def test_a_row_with_no_elision_reports_none():
    audit = _audit()
    assert audit._elided_fields({"record_id": "video:a"}) == set()
    assert audit._elided_fields({"_text_bundle_fields": ""}) == set()
    assert audit._elided_fields({"_text_bundle_fields": "[]"}) == set()


def test_it_tolerates_a_legacy_comma_joined_value():
    audit = _audit()
    assert audit._elided_fields({"_text_bundle_fields": "a_text,b_text"}) == {"a_text", "b_text"}


def test_a_list_value_is_accepted_as_is():
    audit = _audit()
    assert audit._elided_fields({"_text_bundle_fields": ["summary"]}) == {"summary"}
