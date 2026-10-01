"""Launcher/sbatch tests for the numbered EP24 pipeline (issue #64).

These are static checks plus a no-config smoke run of every launcher. They need no
Slurm, no GPU and no private data -- the point is that a launcher cannot silently
point at a missing sbatch file, target the wrong partition, or request more time
than the partition allows.

The Slurm requirements come straight from the issue: `gpumedium`, at most 36
hours, `project_2009497`, and a GH200 request for GPU-backed steps.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ep24_stage_contract as contract  # noqa: E402

ROIHU = ROOT / "scripts" / "roihu"
SBATCH_FOR = {s.number: ROOT / s.sbatch for s in contract.STAGE_CONTRACT}
GPU_STEPS = [s.number for s in contract.STAGE_CONTRACT if s.gpu]
CPU_STEPS = [s.number for s in contract.STAGE_CONTRACT if not s.gpu]

SLURM_LINE = re.compile(r"^#SBATCH\s+--([a-z-]+)=?(.*)$", re.MULTILINE)


def _slurm(path: Path) -> dict[str, str]:
    return {key: value.strip() for key, value in SLURM_LINE.findall(path.read_text(encoding="utf-8"))}


# --- launchers exist, parse, and point at a real sbatch file ---------------

@pytest.mark.parametrize("number", [s.number for s in contract.STAGE_CONTRACT])
def test_launcher_exists_and_is_executable(number):
    path = ROIHU / f"run_step_{number}.sh"
    assert path.is_file(), f"missing launcher {path}"
    assert path.stat().st_mode & 0o111, f"{path} is not executable"


@pytest.mark.parametrize("number", [s.number for s in contract.STAGE_CONTRACT])
def test_launcher_parses_as_bash(number):
    path = ROIHU / f"run_step_{number}.sh"
    result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("number", [s.number for s in contract.STAGE_CONTRACT])
def test_launcher_names_the_stage_its_sbatch_file_carries(number):
    """The wrapper must launch the same stage the contract names, not a neighbour."""
    text = (ROIHU / f"run_step_{number}.sh").read_text(encoding="utf-8")
    assert f"launch_step {number} {contract.stage(number).name}" in text


def test_shared_library_exists_and_parses():
    lib = ROIHU / "launcher_lib.sh"
    assert lib.is_file()
    result = subprocess.run(["bash", "-n", str(lib)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_launcher_submits_and_does_not_hold_the_session():
    """`sbatch` (submit) must be used, not `srun` (run in this session)."""
    lib = (ROIHU / "launcher_lib.sh").read_text(encoding="utf-8")
    assert "sbatch --parsable" in lib
    assert "launcher_submit" in lib
    assert "srun" not in lib, "the launcher must submit, not run, so a dropped SSH cannot kill the job"


def test_launcher_prints_the_job_id_and_log_locations():
    lib = (ROIHU / "launcher_lib.sh").read_text(encoding="utf-8")
    for needle in ("submitted job", "squeue -j", ".out", ".err"):
        assert needle in lib


# --- sbatch files meet the issue's Slurm requirements ----------------------

@pytest.mark.parametrize("number", GPU_STEPS)
def test_gpu_sbatch_uses_gpumedium_with_a_gh200_and_the_project(number):
    slurm = _slurm(SBATCH_FOR[number])
    assert slurm.get("partition") == "gpumedium"
    assert slurm.get("account") == "project_2009497"
    assert "gh200" in slurm.get("gres", "")


@pytest.mark.parametrize("number", [s.number for s in contract.STAGE_CONTRACT])
def test_no_job_requests_more_than_the_partition_maximum(number):
    """gpumedium is a 36-hour partition; over-requesting is rejected by Slurm."""
    slurm = _slurm(SBATCH_FOR[number])
    hours = int(slurm.get("time", "0").split(":")[0])
    assert 0 < hours <= 36, f"stage {number} requests {hours}h, partition max is 36"


@pytest.mark.parametrize("number", GPU_STEPS)
def test_gpu_sbatch_allocates_cpus_for_the_requested_gpu(number):
    slurm = _slurm(SBATCH_FOR[number])
    assert slurm.get("gres"), f"stage {number} is GPU-backed but requests no GPU"
    assert int(slurm.get("cpus-per-task", "0")) > 0


@pytest.mark.parametrize("number", [s.number for s in contract.STAGE_CONTRACT])
def test_sbatch_requests_are_deterministic_paths(number):
    slurm = _slurm(SBATCH_FOR[number])
    assert slurm.get("output"), f"stage {number} has no deterministic stdout path"
    assert slurm.get("error"), f"stage {number} has no deterministic stderr path"


def test_cpu_only_stage_does_not_request_a_gpu():
    """RDF export wastes a GPU allocation; the issue notes it should not."""
    for number in CPU_STEPS:
        assert not _slurm(SBATCH_FOR[number]).get("gres"), f"stage {number} asks for a GPU"


@pytest.mark.parametrize("number", [s.number for s in contract.STAGE_CONTRACT])
def test_sbatch_scripts_have_no_hardcoded_project_secret(number):
    """No credentials, tokens or URIs in a committed batch script."""
    text = SBATCH_FOR[number].read_text(encoding="utf-8")
    lowered = text.lower()
    for forbidden in ("password", "token", "api_key", "mongodb://", "redis://"):
        assert forbidden not in lowered, f"stage {number} sbatch mentions {forbidden}"


# --- failure behaviour ----------------------------------------------------

def test_launcher_fails_loudly_and_helpfully_without_configuration():
    """A missing config must stop the run, not submit a job that will fail late."""
    result = subprocess.run(
        ["bash", str(ROIHU / "run_step_1.sh")],
        capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "HOME": "/nonexistent"},
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "LACLAUGPT_MULTIMODAL_PRIVATE_ROOT" in combined
    assert "ERROR" in combined


def test_launcher_rejects_a_missing_settings_file_explicitly():
    result = subprocess.run(
        ["bash", str(ROIHU / "run_step_1.sh")],
        capture_output=True, text=True,
        env={
            "PATH": "/usr/bin:/bin", "HOME": "/nonexistent",
            "LACLAUGPT_EP24_SETTINGS": "/nonexistent/settings.env",
        },
    )
    assert result.returncode != 0
    assert "missing file" in (result.stdout + result.stderr)
