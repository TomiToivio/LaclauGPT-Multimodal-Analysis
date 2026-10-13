"""Tests for the retired Laskin vLLM video experiment (issue #32).

The CPU-only harness compatibility tests remain useful because the shared video
request builder still supports historical output interpretation. The Laskin
installer and runner, however, must fail closed: the only proven Volta stack is
now security-retired.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "experiments" / "vllm_video_test.py"
INSTALLER = ROOT / "scripts" / "laskin" / "install_vllm_video_test.sh"
RUNNER = ROOT / "scripts" / "laskin" / "vllm_video_test.sh"


def _load_harness():
    spec = importlib.util.spec_from_file_location("vllm_video_test_laskin", HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["vllm_video_test_laskin"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def harness():
    return _load_harness()


@pytest.fixture()
def quiet_logger():
    logger = logging.getLogger("laskin-test")
    logger.addHandler(logging.NullHandler())
    return logger


def test_explicit_video_api_is_respected(harness, quiet_logger):
    assert harness.resolve_video_api("direct", quiet_logger) == "direct"
    assert harness.resolve_video_api("mm_processor_kwargs", quiet_logger) == "mm_processor_kwargs"


def test_auto_resolves_direct_for_vllm_08(harness, quiet_logger, monkeypatch):
    monkeypatch.setattr(harness, "_vllm_version_tuple", lambda _log: (0, 8))
    assert harness.resolve_video_api("auto", quiet_logger) == "direct"


def test_auto_resolves_metadata_for_newer_vllm(harness, quiet_logger, monkeypatch):
    monkeypatch.setattr(harness, "_vllm_version_tuple", lambda _log: (0, 9))
    assert harness.resolve_video_api("auto", quiet_logger) == "mm_processor_kwargs"
    monkeypatch.setattr(harness, "_vllm_version_tuple", lambda _log: (1, 0))
    assert harness.resolve_video_api("auto", quiet_logger) == "mm_processor_kwargs"


def test_unknown_vllm_version_falls_back_to_the_safe_shape(harness, quiet_logger, monkeypatch):
    monkeypatch.setattr(harness, "_vllm_version_tuple", lambda _log: (0, 0))
    assert harness.resolve_video_api("auto", quiet_logger) == "direct"


def test_version_tuple_tolerates_missing_vllm(harness, quiet_logger, monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace())
    assert harness._vllm_version_tuple(quiet_logger) == (0, 0)


class _StubProcessor:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "PROMPT"


def _stub_qwen_vl_utils(monkeypatch):
    calls = {}

    def process_vision_info(messages, **kwargs):
        calls.update(kwargs)
        if kwargs.get("return_video_metadata"):
            return None, [("VIDEO_TENSOR", {"fps": 2})], {"fps": 2}
        return None, ["VIDEO_TENSOR"], {"fps": 2}

    monkeypatch.setitem(
        sys.modules,
        "qwen_vl_utils",
        types.SimpleNamespace(process_vision_info=process_vision_info),
    )
    return calls


def test_direct_request_sends_bare_video_and_no_metadata(harness, quiet_logger, monkeypatch):
    calls = _stub_qwen_vl_utils(monkeypatch)
    request = harness.prepare_vllm_request(
        [{"role": "user", "content": []}],
        _StubProcessor(),
        quiet_logger,
        video_api="direct",
    )
    assert request["multi_modal_data"]["video"] == ["VIDEO_TENSOR"]
    assert "mm_processor_kwargs" not in request
    assert calls.get("return_video_kwargs") is not True


def test_output_records_the_video_api(harness):
    assert "vllm_video_api" in harness.OUTPUT_COLUMNS


def test_trim_contract_is_still_declared(harness):
    assert harness.VIDEO_INITIAL_SKIP_SECONDS == 1.0
    assert "video_initial_skip_seconds" in harness.OUTPUT_COLUMNS


def test_laskin_scripts_fail_closed_after_security_retirement():
    for path in (INSTALLER, RUNNER):
        text = path.read_text(encoding="utf-8")
        assert "set -euo pipefail" in text
        assert "SECURITY" in text
        assert "retired" in text.lower()
        assert "exit 78" in text


def test_no_installable_laskin_legacy_requirements_manifest():
    assert not (ROOT / "experiments" / "requirements-vllm-video-test-laskin.txt").exists()
