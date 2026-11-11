"""Model routing must agree between workers, cache keys and the Slurm server."""
import os
import subprocess
from pathlib import Path

import pytest

from ep24_models import STAGE_MODELS, stage_ollama_model, stage_ollama_model_source

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def clean_models(monkeypatch):
    for key in tuple(os.environ):
        if key.startswith('LACLAUGPT_STEP') and key.endswith('_MODEL'):
            monkeypatch.delenv(key)
    monkeypatch.delenv('LACLAUGPT_MODEL_PROFILE', raising=False)
    monkeypatch.delenv('LACLAUGPT_MULTIMODAL_MODEL', raising=False)


@pytest.mark.parametrize('stage', STAGE_MODELS)
def test_candidate_worker_and_launcher_agree(stage, monkeypatch):
    monkeypatch.setenv('LACLAUGPT_MODEL_PROFILE', 'small-candidate')
    monkeypatch.setenv('LACLAUGPT_MULTIMODAL_MODEL', 'wrong-shared:4b')
    expected = STAGE_MODELS[stage]
    assert stage_ollama_model(stage) == expected
    result = subprocess.run(
        ['python3', str(ROOT / 'ep24_models.py'), '--stage', str(stage)],
        capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == expected


def test_preserve_profile_keeps_pilot_selection(monkeypatch):
    monkeypatch.setenv('LACLAUGPT_MULTIMODAL_MODEL', 'pilot:model')
    assert stage_ollama_model(7) == 'pilot:model'
    assert stage_ollama_model(6) == 'qwen3.8:27b'


def test_stage_override_is_local_and_reported(monkeypatch):
    monkeypatch.setenv('LACLAUGPT_MODEL_PROFILE', 'small-candidate')
    monkeypatch.setenv('LACLAUGPT_STEP5_MODEL', 'qwen3:8b-q4_K_M')
    assert stage_ollama_model(5) == 'qwen3:8b-q4_K_M'
    assert stage_ollama_model_source(5) == 'LACLAUGPT_STEP5_MODEL'
    assert stage_ollama_model(8) == 'qwen3:4b-q4_K_M'
    assert stage_ollama_model(6) == 'qwen3.8:27b'


def test_step6_rejects_accidental_small_override(monkeypatch):
    monkeypatch.setenv('LACLAUGPT_STEP6_MODEL', 'qwen3:8b-q4_K_M')
    with pytest.raises(ValueError, match='pinned'):
        stage_ollama_model(6)


def test_unknown_profile_and_non_llm_stages_fail_closed(monkeypatch):
    monkeypatch.setenv('LACLAUGPT_MODEL_PROFILE', 'typo')
    with pytest.raises(ValueError, match='Unknown model profile'):
        stage_ollama_model(7)
    for stage in (1, 3, 9):
        with pytest.raises(ValueError, match='Unsupported Ollama'):
            stage_ollama_model(stage)
