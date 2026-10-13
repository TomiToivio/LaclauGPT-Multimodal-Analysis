from pathlib import Path

import pandas as pd

from ep24_bootstrap import ordered_country_files, prepare_dataframe
from ep24_stage_orchestrator import PRIORITY, eligible_query


def test_bootstrap_accepts_only_pre_migrated_canonical_annotations():
    source = pd.DataFrame([{
        "video_id": "v1",
        "allas_filename": "clip.mp4",
        "entities": '["Alice", "Bob"]',
        "themes": '["Democracy", "EU"]',
        "researcher_note": "keep me",
    }])
    out = prepare_dataframe(source, country="finland")
    assert out.loc[0, "entities"] == '["Alice", "Bob"]'
    assert out.loc[0, "themes"] == '["Democracy", "EU"]'
    assert out.loc[0, "researcher_note"] == "keep me"
    assert out.loc[0, "_storage_id"]
    for legacy in ("new_entity", "researcher_new_persons", "new_theme", "researcher_new_themes"):
        assert legacy not in out.columns


def test_bootstrap_rejects_unmigrated_legacy_annotations():
    source = pd.DataFrame([{
        "video_id": "v1",
        "allas_filename": "clip.mp4",
        "entities": "[]",
        "themes": "[]",
        "new_entity": "Alice",
    }])
    import pytest
    with pytest.raises(ValueError, match="legacy annotation columns"):
        prepare_dataframe(source, country="finland")


def test_country_order_prioritizes_finland_poland_portugal(tmp_path):
    for name in ("sweden", "portugal", "finland", "bulgaria", "poland"):
        (tmp_path / f"ep24_{name}.csv").write_text("video_id,allas_filename\n", encoding="utf-8")
    ordered = [p.stem.removeprefix("ep24_") for p in ordered_country_files(tmp_path)]
    assert ordered[:3] == list(PRIORITY)
    assert ordered[3:] == ["bulgaria", "sweden"]


def test_later_step_requires_previous_step_complete():
    q = eligible_query(2, retry_errors=False, force=False)
    assert q["_pipeline.step_01.status"] == "complete"
    assert {"_pipeline.step_02.status": {"$exists": False}} in q["$or"]


def test_force_still_requires_previous_stage():
    q = eligible_query(4, retry_errors=False, force=True)
    assert q == {"_pipeline.step_03.status": "complete"}


def test_gpu_sbatch_contract_and_orchestrator_wiring():
    root = Path("scripts/roihu")
    for step, suffix in (
        (1, "preprocess"),
        (2, "frame"),
        (3, "video"),
        (4, "summary"),
        (5, "postprocess"),
        (6, "discourse_analysis"),
        (7, "discourse_network_analysis"),
        (8, "social_network_analysis"),
    ):
        text = (root / f"step_{step}_roihu_{suffix}.sbatch").read_text(encoding="utf-8")
        assert "#SBATCH --partition=gpumedium" in text
        assert "#SBATCH --gres=gpu:gh200:1" in text
        assert "ep24_stage_orchestrator.py" in text
        hours = int(next(line for line in text.splitlines() if line.startswith("#SBATCH --time=")).split("=")[1].split(":")[0])
        assert hours <= 36


def test_every_step_has_one_command_launcher():
    for step in range(1, 10):
        launcher = Path(f"scripts/roihu/run_step_{step}.sh")
        assert launcher.exists()
        assert f"run_step.sh\" {step}" in launcher.read_text(encoding="utf-8")
