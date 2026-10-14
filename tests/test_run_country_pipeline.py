"""Static checks for the restartable single-command country SLURM chain."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_country_runner_submits_ordered_dependency_chain():
    code = (ROOT / "scripts/roihu/run_country_pipeline.sh").read_text()
    assert 'STEP=FROM_STEP; STEP<=TO_STEP; STEP++' in code
    assert 'DEPENDENCY="afterok:${PREVIOUS}"' in code
    assert 'LACLAUGPT_SBATCH_PARSABLE=1' in code
    assert '--country "${COUNTRY}" --limit "${LIMIT}"' in code
    assert '--retry-errors' in code
    assert '--force' not in code


def test_sbatch_dependency_applied_before_stage_script():
    code = (ROOT / "scripts/roihu/run_step.sh").read_text()
    assert 'SBATCH_OPTIONS+=(--dependency="${LACLAUGPT_SBATCH_DEPENDENCY}")' in code
    assert 'SBATCH_OPTIONS+=(--parsable)' in code
    assert 'sbatch --export=ALL "${SBATCH_OPTIONS[@]}"' in code
    assert '"${SBATCH}" "$@"' in code
