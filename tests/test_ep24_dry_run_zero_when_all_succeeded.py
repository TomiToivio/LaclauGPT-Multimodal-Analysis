"""#342 acceptance: the dry run schedules nothing when nothing needs doing.

The issue's acceptance line is specific — *"Dry-run Roihu selection demonstrates zero successful
record/stage pairs scheduled"* — and it is the one claim a unit test of `select()` cannot make, because
`select()` never sees an export. This drives the same entry point the operator runs
(`plan_for_export`, over a real CSV on disk) and then measures what a retry would actually touch.

The second half matters as much: when exactly one record breaks at exactly one step, one pair must be
scheduled and nothing else. A planner that is safe because it is blind is not safe.
"""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import ep24_mongo_export as exporter
from scripts.ep24 import plan_ep24_retry as plan

COUNTRIES = ("finland", "poland", "portugal")
PREFIX = {"finland": "fi", "poland": "pl", "portugal": "pt"}
ACTIVE_STEPS = tuple(plan.STEPS)


def _successful_row(record_id: str, country: str) -> dict[str, str]:
    """A record every active step succeeded on: success status *and* real output in every field."""
    row: dict[str, str] = {"record_id": record_id, "country": country}
    for step in ACTIVE_STEPS:
        status_field, outputs = exporter.COVERAGE_STEPS[step]
        if status_field:
            row[status_field] = "success"
        for field in outputs:
            row[field] = f"output-for-{field}" if field != "needs_resplit" else "false"
    return row


def build_export(root: Path, rows: list[dict[str, str]], *, manifest: dict | None = None) -> Path:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (root / "combined.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if manifest is not None:
        (root / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def healthy_corpus() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for country in COUNTRIES:
        for index in range(3):
            rows.append(_successful_row(f"{PREFIX[country]}{index}", country))
    return rows


def scheduled(plan_doc: dict) -> int:
    """How many (record, step) pairs a retry would actually rerun."""
    total = 0
    for country in plan_doc["countries"].values():
        for step in country["steps"].values():
            total += step["missing"]
    return total


class TestDryRun:
    def test_a_fully_successful_export_schedules_nothing(self, tmp_path: Path):
        build_export(tmp_path, healthy_corpus())
        doc = plan.plan_for_export(tmp_path, ACTIVE_STEPS, None)
        assert scheduled(doc) == 0, json.dumps(doc, indent=1)
        for country, block in doc["countries"].items():
            for step, entry in block["steps"].items():
                assert entry["missing"] == 0, f"{country}/{step} was scheduled"

    def test_every_country_is_planned_at_every_active_step(self, tmp_path: Path):
        build_export(tmp_path, healthy_corpus())
        doc = plan.plan_for_export(tmp_path, ACTIVE_STEPS, None)
        assert set(doc["countries"]) == set(COUNTRIES)
        for country in COUNTRIES:
            assert set(doc["countries"][country]["steps"]) == set(ACTIVE_STEPS)

    def test_one_broken_pair_schedules_one_pair(self, tmp_path: Path):
        rows = healthy_corpus()
        for row in rows:
            if row["record_id"] == "pl1":
                row["laclau_status"] = "failed"
                row["laclau_structured_json"] = ""
                row["laclau_summary_md"] = ""
        build_export(tmp_path, rows)
        doc = plan.plan_for_export(tmp_path, ACTIVE_STEPS, None)
        assert scheduled(doc) == 1
        entry = doc["countries"]["poland"]["steps"]["6_discourse"]
        assert entry["record_ids"] == ["pl1"]
        # and the other countries are untouched
        assert doc["countries"]["finland"]["steps"]["6_discourse"]["missing"] == 0
        assert doc["countries"]["portugal"]["steps"]["6_discourse"]["missing"] == 0

    def test_a_country_filter_narrows_the_plan_not_the_corpus(self, tmp_path: Path):
        rows = healthy_corpus()
        for row in rows:
            if row["record_id"] in {"pl1", "pt1"}:
                row["laclau_status"] = ""
                row["laclau_structured_json"] = ""
                row["laclau_summary_md"] = ""
        build_export(tmp_path, rows)
        doc = plan.plan_for_export(tmp_path, ACTIVE_STEPS, ["poland"])
        assert set(doc["countries"]) == {"poland"}
        assert scheduled(doc) == 1

    def test_the_plan_is_deterministic(self, tmp_path: Path):
        rows = healthy_corpus()
        rows[0]["laclau_status"] = ""
        rows[0]["laclau_structured_json"] = ""
        build_export(tmp_path, rows)
        first = json.dumps(plan.plan_for_export(tmp_path, ACTIVE_STEPS, None), sort_keys=True)
        second = json.dumps(plan.plan_for_export(tmp_path, ACTIVE_STEPS, None), sort_keys=True)
        assert first == second, "a dry run that changes between invocations cannot be reviewed"

    def test_paused_steps_are_not_planned(self, tmp_path: Path):
        """#346: SNA and RDF are paused by default, so a dry run must not queue them."""
        build_export(tmp_path, healthy_corpus())
        doc = plan.plan_for_export(tmp_path, ACTIVE_STEPS, None)
        for paused in ("8_sna", "9_rdf"):
            assert paused not in doc["steps"], f"{paused} is paused and must not be planned"


class TestElision:
    """#331/#342: a cell that moved to the text bundle is *unverifiable*, never "missing work"."""

    def _elided_export(self, tmp_path: Path) -> Path:
        rows = healthy_corpus()
        for row in rows:
            if row["record_id"] == "fi0":
                row["laclau_status"] = ""
                row["laclau_structured_json"] = ""   # elided below
                row["laclau_summary_md"] = ""        # elided below
        build_export(tmp_path, rows, manifest={
            "heavy_fields": {"finland": {"laclau_structured_json": 1, "laclau_summary_md": 1}},
        })
        with gzip.open(tmp_path / "finland_text.jsonl.gz", "wt", encoding="utf-8") as handle:
            handle.write(json.dumps({"record_id": "fi0", "laclau_structured_json": "x" * 1200}) + "\n")
        return tmp_path

    def test_an_elided_cell_is_unverifiable_not_missing(self, tmp_path: Path):
        root = self._elided_export(tmp_path)
        doc = plan.plan_for_export(root, ACTIVE_STEPS, None)
        entry = doc["countries"]["finland"]["steps"]["6_discourse"]
        assert entry["missing"] == 0, "an elided output cell is not proof the step did no work"
        assert entry["unverifiable"] == 1

    def test_a_genuinely_empty_cell_is_still_missing(self, tmp_path: Path):
        """The elision rule must not become a blanket excuse."""
        rows = healthy_corpus()
        for row in rows:
            if row["record_id"] == "fi1":
                row["laclau_status"] = ""
                row["laclau_structured_json"] = ""
                row["laclau_summary_md"] = ""
        build_export(tmp_path, rows, manifest={"heavy_fields": {"finland": {}}})
        doc = plan.plan_for_export(tmp_path, ACTIVE_STEPS, None)
        entry = doc["countries"]["finland"]["steps"]["6_discourse"]
        assert entry["missing"] == 1
        assert entry["record_ids"] == ["fi1"]


class TestStatusTruth:
    """#342: output is evidence, and a status field is not."""

    def test_a_success_status_without_any_output_is_not_a_success(self, tmp_path: Path):
        rows = healthy_corpus()
        for row in rows:
            if row["record_id"] == "pt2":
                for field in exporter.COVERAGE_STEPS["6_discourse"][1]:
                    row[field] = ""
        build_export(tmp_path, rows)
        doc = plan.plan_for_export(tmp_path, ACTIVE_STEPS, None)
        assert doc["countries"]["portugal"]["steps"]["6_discourse"]["record_ids"] == ["pt2"]

    def test_an_old_failure_flag_with_output_is_not_retried(self, tmp_path: Path):
        rows = healthy_corpus()
        for row in rows:
            if row["record_id"] == "fi2":
                row["laclau_status"] = "failed"   # stale, output is present
        build_export(tmp_path, rows)
        doc = plan.plan_for_export(tmp_path, ACTIVE_STEPS, None)
        entry = doc["countries"]["finland"]["steps"]["6_discourse"]
        assert entry["missing"] == 0
        assert entry["failed_with_output"] == 1, "the plan must still report the contradiction"

