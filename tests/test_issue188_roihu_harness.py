from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    return (ROOT / path).read_text(encoding="utf-8")

def test_roihu_bootstrap_and_runbook_exist():
    assert "requirements-roihu-steps1-6.txt" in read("scripts/roihu/install_roihu.sh")
    runbook = read("docs/ROIHU_STEPS_1_6.md")
    assert "gpumedium" in runbook
    assert "36 hour" in runbook
    assert "finland" in runbook and "poland" in runbook and "portugal" in runbook

def test_all_first_six_jobs_use_current_roihu_contract():
    names = [
        "step_1_roihu_preprocess.sbatch",
        "step_2_roihu_frame.sbatch",
        "step_3_roihu_video.sbatch",
        "step_4_roihu_summary.sbatch",
        "step_5_roihu_postprocess.sbatch",
        "step_6_roihu_discourse_analysis.sbatch",
    ]
    for name in names:
        text = read("scripts/roihu/" + name)
        assert "#SBATCH --partition=gpumedium" in text
        assert "#SBATCH --gres=gpu:gh200:1" in text
        assert "#SBATCH --time=36:00:00" in text
        assert "roihu_job_common.sh" in text
        assert "roihu_preflight" in text
        assert 'srun --ntasks=1 python3' in text

def test_private_configuration_is_not_embedded_in_new_harness():
    for path in [
        "scripts/roihu/install_roihu.sh",
        "scripts/roihu/roihu_job_common.sh",
        "scripts/roihu/preflight_steps_1_6.py",
    ]:
        text = read(path)
        assert "mongodb://" not in text
        assert "redis://" not in text
        assert "password=" not in text.lower()
