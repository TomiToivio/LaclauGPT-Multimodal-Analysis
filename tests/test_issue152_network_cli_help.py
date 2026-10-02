from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "script",
    [
        "step_7_roihu_discourse_network_analysis.py",
        "step_8_roihu_social_network_analysis.py",
    ],
)
def test_network_step_help_does_not_import_runtime_only_dependencies(tmp_path, script):
    # Shadow these optional runtime packages with modules that explode if imported.
    # --help must finish during CLI parsing before analysis/runtime imports happen.
    (tmp_path / "ollama.py").write_text("raise RuntimeError('ollama imported during --help')\n")
    (tmp_path / "pydantic.py").write_text("raise RuntimeError('pydantic imported during --help')\n")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(tmp_path), str(ROOT)])
    result = subprocess.run(
        [sys.executable, str(ROOT / script), "--help"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert "imported during --help" not in result.stderr
