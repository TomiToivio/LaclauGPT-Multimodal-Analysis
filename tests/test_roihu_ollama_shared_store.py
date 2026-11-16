"""Regression checks for shared Ollama model-store pull coordination on Roihu."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    ROOT / "scripts/roihu/roihu_job_common.sh",
    ROOT / "scripts/roihu/multimodal_roihu.sbatch",
)


def test_ollama_launchers_are_valid_bash():
    for script in SCRIPTS:
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_ollama_launchers_serialize_model_installs():
    for script in SCRIPTS:
        source = script.read_text(encoding="utf-8")
        assert '.laclaugpt-pull.lock' in source
        assert 'flock -w' in source
        assert 'ollama show' in source
        assert 'ollama pull' in source
        assert '>>"' in source  # progress is redirected away from Slurm output
        assert 'for ' in source and ' in 1 2; do' in source
        assert 'rm -f' not in source  # never remove shared partial blobs
