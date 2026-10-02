"""Executable regression proof for the single-frame preprocess contract (#128 §2).

#128 §2 requires a test proving, executably, that:

* exactly one frame is generated;
* its timestamp is exactly 1.0 s in *original-video* time;
* OCR runs exactly once, against that frame.

``tests/test_ep24_video_scroll.py`` already pins ``analysis_frame_times`` to
``[1.0]``, and ``ep24_video`` is import-light, so it is tested for real here.

``roihu_preprocess`` imports ``cv2`` / ``easyocr`` at module scope, and those are
*not* in the CI test extra (``pyproject.toml`` ``[test]``). Importing it in a test
module is therefore a collection error that aborts the whole suite. The
preprocess-side assertions below use the repository's established source-contract
style instead of importing the stage. The AST checks are real structural
assertions: they would fail if a second frame request, a second save, or a
second OCR call were reintroduced.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import ep24_video as video  # noqa: E402

PREPROCESS_SOURCE = (REPO_ROOT / "roihu_preprocess.py").read_text(encoding="utf-8")


def test_frame_times_are_exactly_one_timestamp_at_original_t1():
    for duration in (1.001, 2.0, 12.0, 59.9, 600.0):
        times = video.analysis_frame_times(duration)
        assert times == [1.0], f"duration {duration} produced {times!r}"
        assert len(times) == 1


def test_short_clips_yield_no_frame_rather_than_a_fallback_sample():
    for duration in (0.0, 0.5, 1.0):
        assert video.analysis_frame_times(duration) == []


def test_skip_is_applied_in_original_video_time():
    """The 1.0 s boundary must be original-video time, not clip-relative."""
    assert video.VIDEO_INITIAL_SKIP_SECONDS == 1.0
    assert video.analysis_frame_times(100) == [video.analysis_start_seconds()]
    # the trim used for ASR starts at the same boundary
    cmd = video.build_trim_command("source.mp4", "analysis.mp4")
    assert cmd[cmd.index("-ss") + 1] == "1"


def _functions(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def test_keyframe_extraction_requests_frames_from_the_shared_rule_only():
    """get_keyframes must derive its timestamps solely from analysis_frame_times."""
    tree = ast.parse(PREPROCESS_SOURCE)
    fn = _functions(tree)["get_keyframes"]
    calls = [
        n.func.id for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    ]
    assert "analysis_frame_times" in calls, "must use the canonical frame rule"
    assert "save_keyframe" in calls
    # no ad-hoc timestamp arithmetic
    assert not [n for n in ast.walk(fn) if isinstance(n, ast.Constant)
                and isinstance(n.value, float) and n.value != 0.0], \
        "get_keyframes must not hard-code a frame time; ep24_video owns the rule"


def test_save_keyframe_guards_the_scroll_boundary():
    """save_keyframe must refuse any time before the analysis boundary."""
    tree = ast.parse(PREPROCESS_SOURCE)
    fn = _functions(tree)["save_keyframe"]
    src = ast.get_source_segment(PREPROCESS_SOURCE, fn) or ""
    assert "VIDEO_INITIAL_SKIP_SECONDS" in src
    assert "raise ValueError" in src, "must fail loudly, not silently substitute"


def test_get_keyframes_does_not_append_more_than_one_frame():
    """Structurally: exactly one save per loop iteration, no extra appends."""
    tree = ast.parse(PREPROCESS_SOURCE)
    fn = _functions(tree)["get_keyframes"]
    calls = [
        n.func.id for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    ]
    assert calls.count("save_keyframe") == 1, "exactly one frame is saved"
    appends = [
        n for n in ast.walk(fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "append"
    ]
    assert len(appends) == 1, "exactly one append -> at most one frame returned"


def test_ocr_runs_once_on_the_first_frame_only():
    """The stage performs exactly one readtext call, on frame_files[0]."""
    tree = ast.parse(PREPROCESS_SOURCE)
    readtext = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr == "readtext"
    ]
    assert len(readtext) == 1, f"expected exactly one OCR call, found {len(readtext)}"
    arg = ast.get_source_segment(PREPROCESS_SOURCE, readtext[0].args[0])
    assert arg == "frame_files[0]", f"OCR must read the single frame, got {arg!r}"


def test_only_the_first_ocr_column_is_ever_populated():
    """Documents the measured starting point for #128 §1.

    The stage allocates six OCR slots but assigns only the first, so ocr_2..ocr_6
    are always empty. This test records today's behaviour so their removal is a
    deliberate, visible change; it should be replaced when §1 lands.
    """
    tree = ast.parse(PREPROCESS_SOURCE)
    assigned = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (isinstance(target, ast.Subscript)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "ocr_values"):
                    index = target.slice.value if isinstance(target.slice, ast.Constant) \
                        else None
                    if index is not None:
                        assigned[index] = ast.get_source_segment(PREPROCESS_SOURCE, node)
    assert 0 in assigned, "ocr_1 (index 0) must be populated"
    assert set(assigned) == {0}, f"only ocr_values[0] may be written, got {sorted(assigned)}"
