from __future__ import annotations

from pathlib import Path

import ep24_models

ROOT = Path(__file__).resolve().parents[1]

OLLAMA_ACTIVE = (
    "roihu_frame.py",
    "roihu_summary.py",
    "roihu_postprocess.py",
    "roihu_populism.py",
    "step_7_roihu_discourse_network_analysis.py",
    "step_8_roihu_social_network_analysis.py",
    "ep24_entities.py",
)


def test_shared_ollama_default_and_override(monkeypatch):
    monkeypatch.delenv("LACLAUGPT_MULTIMODAL_MODEL", raising=False)
    assert ep24_models.ollama_model() == "qwen3.8:27b"
    monkeypatch.setenv("LACLAUGPT_MULTIMODAL_MODEL", "custom:model")
    assert ep24_models.ollama_model() == "custom:model"


def test_specific_entity_override_wins(monkeypatch):
    monkeypatch.setenv("LACLAUGPT_MULTIMODAL_MODEL", "shared:model")
    monkeypatch.setenv("LACLAUGPT_ENTITY_ADJUDICATOR_MODEL", "entity:model")
    assert (
        ep24_models.ollama_model(specific_env="LACLAUGPT_ENTITY_ADJUDICATOR_MODEL")
        == "entity:model"
    )


def test_active_ollama_paths_have_no_old_default():
    for name in OLLAMA_ACTIVE:
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "gemma4:12b" not in text, name
    assert "qwen3.8:27b" in (ROOT / "ep24_models.py").read_text(encoding="utf-8")


def test_video_step_uses_vllm_qwen3_vl_32b():
    expected = "Qwen/Qwen3-VL-32B-Instruct"
    harness = (ROOT / "experiments/vllm_video_test.py").read_text(encoding="utf-8")
    env_example = (ROOT / "config/vllm_video_test.env.example").read_text(encoding="utf-8")
    sbatch = (ROOT / "scripts/roihu/vllm_video_test.sbatch").read_text(encoding="utf-8")
    wrapper = (ROOT / "step_3_roihu_video.py").read_text(encoding="utf-8")
    assert f'DEFAULT_MODEL = "{expected}"' in harness
    assert expected in env_example
    assert expected in sbatch
    assert "experiments.vllm_video_test" in wrapper
    assert "ollama" not in wrapper.lower()
    assert ep24_models.DEFAULT_VLLM_VIDEO_MODEL == expected
