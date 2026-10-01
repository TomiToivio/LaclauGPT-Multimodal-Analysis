from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# The historical EP24 scripts were named puhti_*.py. On main, the active
# successors were renamed to roihu_*.py while the frozen historical
# implementation remains on the legacy branch. Keep the safety assertions
# expressed in terms of the historical contract, but resolve each script to the
# file that carries that contract on the current branch.
LEGACY_STAGE_FILES = {
    "puhti_preprocess.py": "roihu_preprocess.py",
    "puhti_frame.py": "roihu_frame.py",
    "puhti_summary.py": "roihu_summary.py",
    "puhti_postprocess.py": "roihu_postprocess.py",
    "puhti_populism.py": "roihu_populism.py",
}


def stage_path(name: str) -> Path:
    """Resolve a historical Puhti stage name on either main or legacy."""
    # The restored historical files are what these safety assertions describe
    # (they carry the __main__ and cv2 guards), so they win when present;
    # fall back to the renamed Roihu stage only when they are absent.
    historical = ROOT / name
    if historical.is_file():
        return historical
    current = ROOT / LEGACY_STAGE_FILES.get(name, name)
    return current if current.is_file() else historical


def source(name: str) -> str:
    return stage_path(name).read_text(encoding="utf-8")


SCRIPTS = [stage_path(name) for name in LEGACY_STAGE_FILES]


def test_all_scripts_parse() -> None:
    assert SCRIPTS
    for script in SCRIPTS:
        assert script.is_file(), f"missing pipeline stage: {script.name}"
        ast.parse(script.read_text(encoding="utf-8"), filename=str(script))


def test_batch_scripts_have_main_guards() -> None:
    for script in SCRIPTS:
        tree = ast.parse(script.read_text(encoding="utf-8"))
        assert any(
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__"
            for node in tree.body
        ), f"{script.name} executes without a __main__ guard"


def test_postprocess_country_is_explicit() -> None:
    tree = ast.parse(source("puhti_postprocess.py"))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "analyze_responses"
    )
    expected_scope = "language" if stage_path("puhti_postprocess.py").name.startswith("roihu_") else "country"
    assert [argument.arg for argument in function.args.args] == [expected_scope]
    assert "finland_mobile_fixed.csv" not in source("puhti_postprocess.py")


def test_frame_analysis_has_safe_error_default() -> None:
    text = source("puhti_frame.py")
    assert "frame_analysis = ''" in text
    assert "ast.literal_eval(text)" in text


def test_preprocess_rejects_invalid_video_metadata() -> None:
    text = source("puhti_preprocess.py")
    assert "if not video.isOpened():" in text
    assert "fps <= 0" in text
    assert "frame_count <= 0" in text
    assert "if success and cv2.imwrite" in text
