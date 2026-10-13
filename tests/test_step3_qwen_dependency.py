"""Ensure native Step 3 video preprocessing dependency is installed and gated."""
from pathlib import Path


def test_step3_qwen_dependency_is_explicit():
    requirements = Path("requirements/roihu-step3-video.txt").read_text()
    assert "qwen-vl-utils" in requirements


def test_step3_preflight_checks_exact_video_processor_api():
    preflight = Path("scripts/roihu/validate_step_environment.py").read_text()
    assert '3: ("pandas", "vllm", "transformers", "qwen_vl_utils")' in preflight
    assert "from qwen_vl_utils import process_vision_info" in preflight
    assert "setup_step_3_video.sh" in preflight
