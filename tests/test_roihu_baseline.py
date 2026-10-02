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
    assert "roihu_${stage}.py" in text


def test_inference_stages_use_configurable_model():
    """Each stage resolves the model through the shared override chain."""
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
    from ep24_cli import DEFAULT_OLLAMA_MODEL
    for name in ("roihu_frame.py", "roihu_summary.py", "roihu_postprocess.py", "roihu_populism.py"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "LACLAUGPT_MULTIMODAL_MODEL" in text, name
        assert "resolve_model" in text or DEFAULT_OLLAMA_MODEL in text, name


def test_readme_marks_legacy_frozen_and_main_phase2_roihu():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "`legacy` branch" in text
    assert "Phase 2" in text and "CSC Roihu" in text


def test_migration_doc_keeps_private_material_private():
    text = (ROOT / "docs/ROIHU_MIGRATION.md").read_text(encoding="utf-8")
    assert "LaclauGPT-Private" in text
    assert "Private codebooks, settings, source data, researcher notes" in text
