from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted(ROOT.glob("puhti_*.py"))


def source(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def test_all_scripts_parse() -> None:
    assert SCRIPTS
    for script in SCRIPTS:
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
    assert [argument.arg for argument in function.args.args] == ["country"]
    assert "finland_mobile_fixed.csv" not in source("puhti_postprocess.py")


def test_frame_analysis_has_safe_error_default() -> None:
    text = source("puhti_frame.py")
    assert "frame_analysis = ''" in text
    assert "ast.literal_eval(text)" in text


def test_preprocess_rejects_invalid_video_metadata() -> None:
    text = source("puhti_preprocess.py")
    assert "if not video.isOpened():" in text
    assert "if fps <= 0 or frame_count <= 0:" in text
    assert "if success and cv2.imwrite" in text