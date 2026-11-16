"""Issue #413: compatible, researcher-readable Step 7/8 extraction."""
import importlib
from pathlib import Path
import subprocess
import sys


def test_old_step_imports_alias_the_canonical_implementations():
    for old_name, implementation_name in (
        ("step_7_roihu_discourse_network_analysis", "roihu_discourse_network"),
        ("step_8_roihu_social_network_analysis", "roihu_sna"),
    ):
        old = importlib.import_module(old_name)
        implementation = importlib.import_module(implementation_name)
        assert old is implementation
        assert Path(old.__file__).name == implementation_name + ".py"


def test_cli_entrypoints_still_compile():
    for name in ("step_7_roihu_discourse_network_analysis.py",
                 "step_8_roihu_social_network_analysis.py"):
        compile(Path(name).read_text(encoding="utf-8"), name, "exec")


def test_original_slurm_entrypoint_names_remain():
    root = Path(__file__).resolve().parents[1]
    for step, suffix in ((7, "discourse_network_analysis"), (8, "social_network_analysis")):
        script = root / "scripts" / "roihu" / f"step_{step}_roihu_{suffix}.sbatch"
        assert script.is_file()
        assert f"step_{step}_roihu_{suffix}.py" in script.read_text(encoding="utf-8")
