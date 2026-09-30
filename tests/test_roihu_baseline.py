from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_roihu_batch_has_no_private_allocation_or_absolute_scratch_path():
    text = (ROOT / "scripts/roihu/multimodal_roihu.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH --account=" not in text
    assert "/scratch/" not in text
    assert "LACLAUGPT_MULTIMODAL_PRIVATE_ROOT" in text
    assert "#SBATCH --gres=gpu:gh200:1" in text
    assert "#SBATCH --partition=gpumedium" in text


def test_runner_preserves_historical_stage_order():
    text = (ROOT / "scripts/roihu/run_pipeline.sh").read_text(encoding="utf-8")
    assert "preprocess frame summary postprocess populism" in text
    assert "puhti_${stage}.py" in text


def test_inference_stages_use_configurable_model():
    for name in ("puhti_frame.py", "puhti_summary.py", "puhti_postprocess.py", "puhti_populism.py"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "LACLAUGPT_MULTIMODAL_MODEL" in text, name
        assert "gemma4:12b" in text, name


def test_readme_marks_legacy_frozen_and_main_roihu():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "`legacy` is the frozen historical CSC Puhti" in text
    assert "CSC Roihu" in text


def test_migration_doc_keeps_private_material_private():
    text = (ROOT / "docs/ROIHU_MIGRATION.md").read_text(encoding="utf-8")
    assert "LaclauGPT-Private" in text
    assert "Private codebooks, settings, researcher notes and source data remain" in text
