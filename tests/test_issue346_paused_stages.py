"""A paused stage must stay paused — visibly (#346).

Steps 8 (SNA) and 9 (RDF) are out of scope for the active EP24 run. The risk this guards is not that a
job survives; it is that the *artifacts* lose the distinction and a deliberately paused stage is read as a
failure, or worse, that a coverage number quietly changes to match the new scope. History stays true;
scope travels as its own key.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import ep24_mongo_export as export
from ep24_run_scope import (
    NORMAL_LAST_STEP,
    PAUSED_STEPS,
    is_paused,
    normal_steps,
    paused_stages,
)

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (ROOT / "scripts/roihu/run_country_pipeline.sh").read_text(encoding="utf-8")


class TestRunScope:
    def test_a_normal_run_ends_at_step_seven(self):
        assert NORMAL_LAST_STEP == 7
        assert normal_steps({}) == tuple(range(0, 8))

    def test_sna_and_rdf_are_the_paused_stages(self):
        assert PAUSED_STEPS == (8, 9)
        assert is_paused(8) and is_paused(9)
        assert not is_paused(7)

    def test_reactivation_is_an_explicit_environment_decision(self):
        assert paused_stages({"LACLAUGPT_PAUSED_STAGES": ""}) == ()
        assert is_paused(8, {"LACLAUGPT_PAUSED_STAGES": ""}) is False
        assert paused_stages({"LACLAUGPT_PAUSED_STAGES": "9"}) == (9,)


class TestLauncherScope:
    def test_the_default_chain_stops_at_seven(self):
        assert "TO_STEP=7" in LAUNCHER
        assert "TO_STEP=9" not in LAUNCHER.replace("PAUSED_LAST_STEP=9", "")

    def test_paused_steps_need_an_explicit_opt_in(self):
        assert "--include-paused" in LAUNCHER
        assert 'TO_STEP > 7' in LAUNCHER and 'INCLUDE_PAUSED == 0' in LAUNCHER
        assert "Steps 8-9 are paused" in LAUNCHER

    def test_the_scope_is_exported_and_written_beside_the_manifest(self):
        assert 'export LACLAUGPT_PAUSED_STAGES="8,9"' in LAUNCHER
        assert "paused_steps=8,9" in LAUNCHER

    def test_dry_run_of_the_launcher_reports_the_scope(self, tmp_path: Path):
        """The launcher must refuse 8-9 before it touches sbatch, and say why."""
        env = dict(os.environ, LACLAUGPT_EP24_LOG_ROOT=str(tmp_path),
                   LACLAUGPT_MULTIMODAL_PRIVATE_ROOT=str(tmp_path), PATH="/usr/bin:/bin")
        refused = subprocess.run(["bash", str(ROOT / "scripts/roihu/run_country_pipeline.sh"),
                                  "finland", "--to-step", "9"],
                                 capture_output=True, text=True, env=env, check=False)
        assert refused.returncode == 2
        assert "paused" in (refused.stderr or "").lower()
        assert "include-paused" in (refused.stderr or "")


class TestManifestRecordsScope:
    def test_paused_stages_are_written_and_coverage_is_untouched(self, tmp_path: Path):
        os.environ["LACLAUGPT_PAUSED_STAGES"] = "8,9"
        try:
            manifest = export.write_manifest(
                tmp_path, record_unit="video",
                countries={"finland": {"records": 1, "distinct_media": 1}},
                child_counts={}, collections_read={}, provenance={}, outputs=[],
                contract={"stage_coverage": {"finland": {"8_sna": {"records": 3,
                                                                  "available": True}}}})
        finally:
            os.environ.pop("LACLAUGPT_PAUSED_STAGES", None)
        assert manifest["paused_stages"] == [8, 9]
        # Coverage keeps reporting what the data holds: scope does not rewrite history.
        assert manifest["stage_coverage"]["finland"]["8_sna"]["records"] == 3
        assert json.loads((tmp_path / "MANIFEST.json").read_text())["paused_stages"] == [8, 9]

    def test_no_scope_variable_means_no_claim_about_pausing(self, tmp_path: Path):
        os.environ.pop("LACLAUGPT_PAUSED_STAGES", None)
        manifest = export.write_manifest(
            tmp_path, record_unit="video", countries={}, child_counts={}, collections_read={},
            provenance={}, outputs=[])
        assert manifest["paused_stages"] == []
