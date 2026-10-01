"""Tests for the Laskin vLLM video experiment (issue #32).

CPU-only, synthetic fixtures, no GPU, no network and no private data. They pin
the properties that make a cross-host result trustworthy.

* the video API is resolved from the installed vLLM version, so the modern
  (Qwen3-VL) shape is used where supported and the legacy (vLLM <= 0.8.x) shape
  where per-video metadata is rejected;
* the resolved API is recorded per row, so a Laskin result cannot be
  misattributed to Roihu;
* the environment diagnostic cannot print secret material.
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
DIAGNOSTIC = ROOT / "scripts/laskin/check_vllm_environment.sh"


def _load_harness():
    """Import the standalone harness without needing a package layout."""
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


# --------------------------------------------------------------------------- #
# video API resolution
# --------------------------------------------------------------------------- #

def test_explicit_video_api_is_respected(harness, quiet_logger):
    assert harness.resolve_video_api("direct", quiet_logger) == "direct"
    assert harness.resolve_video_api("mm_processor_kwargs", quiet_logger) == "mm_processor_kwargs"


def test_auto_resolves_direct_for_vllm_08(harness, quiet_logger, monkeypatch):
    """vLLM 0.8.x rejects video metadata, so auto must pick ``direct``."""
    monkeypatch.setattr(harness, "_vllm_version_tuple", lambda _log: (0, 8))
    assert harness.resolve_video_api("auto", quiet_logger) == "direct"


def test_auto_resolves_metadata_for_newer_vllm(harness, quiet_logger, monkeypatch):
    monkeypatch.setattr(harness, "_vllm_version_tuple", lambda _log: (0, 9))
    assert harness.resolve_video_api("auto", quiet_logger) == "mm_processor_kwargs"
    monkeypatch.setattr(harness, "_vllm_version_tuple", lambda _log: (1, 0))
    assert harness.resolve_video_api("auto", quiet_logger) == "mm_processor_kwargs"


def test_unknown_vllm_version_falls_back_to_the_safe_shape(harness, quiet_logger, monkeypatch):
    """An unreadable version must not select the shape that hard-crashes."""
    monkeypatch.setattr(harness, "_vllm_version_tuple", lambda _log: (0, 0))
    assert harness.resolve_video_api("auto", quiet_logger) == "direct"


def test_version_tuple_tolerates_missing_vllm(harness, quiet_logger, monkeypatch):
    """A host without vLLM must not raise while resolving provenance."""
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace())
    assert harness._vllm_version_tuple(quiet_logger) == (0, 0)


# --------------------------------------------------------------------------- #
# request construction and provenance
# --------------------------------------------------------------------------- #

class _StubProcessor:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "PROMPT"


def _stub_qwen_vl_utils(monkeypatch, *, modern):
    """Faithful stand-in for qwen_vl_utils.process_vision_info.

    The installed library always returns a 3-tuple
    ``(images, videos, video_kwargs)``; ``videos`` is a list of
    (tensor, metadata) pairs only when ``return_video_metadata=True``.
    """
    calls = {}

    def process_vision_info(messages, **kwargs):
        calls.update(kwargs)
        if kwargs.get("return_video_metadata"):
            return None, [("VIDEO_TENSOR", {"fps": 2})], {"fps": 2}
        # Without metadata the videos are bare tensors. The library still
        # reports a kwargs dict when asked; the harness must ignore it.
        return None, ["VIDEO_TENSOR"], {"fps": 2}

    monkeypatch.setitem(sys.modules, "qwen_vl_utils", types.SimpleNamespace(
        process_vision_info=process_vision_info))
    return calls


def test_direct_request_sends_bare_video_and_no_metadata(harness, quiet_logger, monkeypatch):
    """vLLM 0.8.5 crashes on mm_processor_kwargs, so ``direct`` must omit it.

    This is the exact production failure: the processor cache does
    ``hash(tuple(...))`` on the kwargs dict and raises
    ``TypeError: unhashable type: 'dict'``.
    """
    calls = _stub_qwen_vl_utils(monkeypatch, modern=False)
    request = harness.prepare_vllm_request(
        [{"role": "user", "content": []}], _StubProcessor(), quiet_logger, video_api="direct"
    )
    assert request["multi_modal_data"]["video"] == ["VIDEO_TENSOR"]
    assert "mm_processor_kwargs" not in request
    assert calls.get("return_video_kwargs") is not True


def test_auto_uses_direct_on_a_vllm_08_host(harness, quiet_logger, monkeypatch):
    """End to end: auto must not hand vLLM 0.8.x the metadata shape."""
    _stub_qwen_vl_utils(monkeypatch, modern=False)
    monkeypatch.setattr(harness, "_vllm_version_tuple", lambda _log: (0, 8))
    request = harness.prepare_vllm_request(
        [{"role": "user", "content": []}], _StubProcessor(), quiet_logger, video_api="auto"
    )
    assert "mm_processor_kwargs" not in request


def test_output_records_the_video_api(harness):
    """A Laskin row must be distinguishable from a Roihu row."""
    assert "vllm_video_api" in harness.OUTPUT_COLUMNS


def test_trim_contract_is_still_declared(harness):
    """The 1.0 s EP24 rule is unchanged by this branch."""
    assert harness.VIDEO_INITIAL_SKIP_SECONDS == 1.0
    assert "video_initial_skip_seconds" in harness.OUTPUT_COLUMNS


# --------------------------------------------------------------------------- #
# Laskin scripts
# --------------------------------------------------------------------------- #

def test_laskin_scripts_exist_and_are_shell_strict():
    for name in ("install_vllm_video_test.sh", "vllm_video_test.sh"):
        path = ROOT / "scripts/laskin" / name
        assert path.is_file(), name
        assert "set -euo pipefail" in path.read_text(encoding="utf-8"), name


def test_laskin_run_script_does_not_assume_cwd_or_leak_secrets():
    text = (ROOT / "scripts/laskin/vllm_video_test.sh").read_text(encoding="utf-8")
    assert "BASH_SOURCE" in text  # resolves repo root, not the caller's cwd
    for forbidden in ("password", "SECRET", "TOKEN=", "project_200"):
        assert forbidden not in text, forbidden


def test_laskin_requirements_pin_the_volta_constraint():
    text = (ROOT / "experiments/requirements-vllm-video-test-laskin.txt").read_text(
        encoding="utf-8"
    )
    # The pin exists because current vLLM dropped sm_70; the file must say so.
    assert "vllm==0.8.5.post1" in text
    assert "sm_70" in text or "Volta" in text
    assert "transformers==4.51.3" in text


def test_environment_diagnostic_does_not_print_secret_material():
    text = DIAGNOSTIC.read_text(encoding="utf-8")
    assert "rclone listremotes" in text  # names are fine
    for forbidden in ("rclone config show", "cat ~/allas_conf", "cat $HOME/allas_conf", "printenv"):
        assert forbidden not in text, forbidden
