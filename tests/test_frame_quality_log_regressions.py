"""Regressions found in EP24 Finland Step 2 log dated 2026-10-10."""
from __future__ import annotations

import ast
from pathlib import Path

from laclaugpt_quality import quality_decision_from_analysis, status_from_text


def test_broken_education_promises_are_not_broken_media():
    analysis = (
        "Political argument about broken education promises. "
        "The video is sharp, well-lit and has legible subtitles. "
        "**(5) Image usability**\nOK — sufficient for coding without reprocessing."
    )
    assert status_from_text(analysis) == "OK"
    assert quality_decision_from_analysis(analysis)[0] == "OK"


def test_explicit_delete_verdict_still_detected():
    assert status_from_text("Image usability: DELETE — the frame is garbage.") == "DELETE"


def test_frame_context_accounts_for_header_and_truncation_suffix():
    source = (Path(__file__).resolve().parents[1] / "roihu_frame.py").read_text()
    tree = ast.parse(source)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_row_context")
    names = {target.id for n in ast.walk(fn) if isinstance(n, ast.Assign)
             for target in n.targets if isinstance(target, ast.Name)}
    assert "prefix" in names
    assert "remaining" in names
    assert "suffix" in names
