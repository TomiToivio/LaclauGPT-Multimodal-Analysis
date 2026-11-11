"""Model routing must agree between workers, cache keys and the Slurm server."""
import os
import subprocess
from pathlib import Path

import pytest

from ep24_models import STAGE_MODELS, ollama_model, ollama_model_source

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def clean_models(monkeypatch):
    for key in tuple(os.environ):
        if key.startswith('LACLAUGPT_STEP_') and key.endswith('_MODEL'):
            monkeypatch.delenv(key)
    monkeypatch.delenv('LACLAUGPT_MODEL_PROFILE', raising=False)
    monkeypatch.delenv('LACLAUGPT_MULTIMODAL_MODEL', raising=False)


@pytest.mark.parametrize('stage', STAGE_MODELS)
def test_candidate_worker_and_launcher_agree(stage, monkeypatch):
    monkeypatch.setenv('LACLAUGPT_MODEL_PROFILE', 'small-candidate')
    monkeypatch.setenv('LACLAUGPT_MULTIMODAL_MODEL', 'wrong-shared:4b')
    expected = STAGE_MODELS[stage]
    assert ollama_model(stage=stage) == expected
    result = subprocess.run(
        ['python3', str(ROOT / 'ep24_models.py'), '--stage', str(stage)],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == expected


def test_preserve_profile_keeps_pilot_selection(monkeypatch):
    monkeypatch.setenv('LACLAUGPT_MULTIMODAL_MODEL', 'pilot:model')
    assert ollama_model(stage=7) == 'pilot:model'
    assert ollama_model(stage=6) == 'qwen3.8:27b'


def test_stage_override_is_local_and_reported(monkeypatch):
    monkeypatch.setenv('LACLAUGPT_MODEL_PROFILE', 'small-candidate')
    monkeypatch.setenv('LACLAUGPT_STEP_5_MODEL', 'qwen3:8b-q4_K_M')
    assert ollama_model(stage=5) == 'qwen3:8b-q4_K_M'
    assert ollama_model_source(stage=5) == 'LACLAUGPT_STEP_5_MODEL'
    assert ollama_model(stage=8) == 'qwen3:4b-q4_K_M'
    assert ollama_model(stage=6) == 'qwen3.8:27b'


def test_step6_rejects_accidental_small_override(monkeypatch):
    monkeypatch.setenv('LACLAUGPT_STEP_6_MODEL', 'qwen3:8b-q4_K_M')
    with pytest.raises(ValueError, match='pinned'):
        ollama_model(stage=6)


def test_unknown_profile_and_non_llm_stages_fail_closed(monkeypatch):
    monkeypatch.setenv('LACLAUGPT_MODEL_PROFILE', 'typo')
    with pytest.raises(ValueError, match='Unknown model profile'):
        ollama_model(stage=7)
    for stage in (1, 3, 9):
        with pytest.raises(ValueError, match='not an Ollama stage'):
            ollama_model(stage=stage)
