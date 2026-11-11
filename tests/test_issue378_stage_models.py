"""#378: every stage resolves its own model, and Step 6 cannot be dragged off 27B by a shared variable.

The failure this guards against is a specific, plausible accident: someone exports
``LACLAUGPT_MULTIMODAL_MODEL=qwen3:8b`` to speed up frame analysis, and the discourse analysis quietly
starts running on an 8B model. The pipeline would keep producing well-formed JSON, and the methodology
would have changed without a single error in the log.
"""

from __future__ import annotations

import json
import subprocess

# The module under test lives at the repository root, next to the stage scripts.
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ep24_models  # noqa: E402


def clear(monkeypatch):
    monkeypatch.delenv(ep24_models.SHARED_ENV, raising=False)
    for stage in ep24_models.STAGES:
        monkeypatch.delenv(ep24_models.STAGE_ENV.format(stage=stage), raising=False)
        monkeypatch.delenv(f"LACLAUGPT_STEP{stage}_MODEL", raising=False)
    monkeypatch.delenv("LACLAUGPT_MODEL_PROFILE", raising=False)


def test_each_stage_has_its_own_default(monkeypatch):
    clear(monkeypatch)
    expected = {
        2: "qwen3-vl:8b",
        3: "Qwen/Qwen3-VL-8B-Instruct",
        4: "qwen3:8b",
        5: "qwen3:4b",
        6: "qwen3.8:27b",
        7: "qwen3:8b",
    }
    for stage, model in expected.items():
        assert ep24_models.stage_model(stage).name == model, stage


def test_step_six_ignores_the_shared_override(monkeypatch):
    """The whole point of the issue: a shared 8B must not change discourse analysis."""
    clear(monkeypatch)
    monkeypatch.setenv(ep24_models.SHARED_ENV, "qwen3:8b")
    assert ep24_models.stage_model(2).name == "qwen3:8b"          # shared applies
    assert ep24_models.stage_model(6).name == "qwen3.8:27b"       # pinned does not move
    assert "ignored" in ep24_models.stage_model(6).note


def test_step_six_can_still_be_changed_deliberately(monkeypatch):
    clear(monkeypatch)
    monkeypatch.setenv(ep24_models.STAGE_ENV.format(stage=6), "qwen3.8:32b")
    resolved = ep24_models.stage_model(6)
    assert resolved.name == "qwen3.8:32b"
    assert resolved.source == "LACLAUGPT_EP24_MODEL_STEP_6"
    assert resolved.pinned is True


def test_a_stage_override_beats_the_shared_override(monkeypatch):
    clear(monkeypatch)
    monkeypatch.setenv(ep24_models.SHARED_ENV, "qwen3:8b")
    monkeypatch.setenv(ep24_models.STAGE_ENV.format(stage=5), "qwen3:14b")
    assert ep24_models.stage_model(5).name == "qwen3:14b"
    assert ep24_models.stage_model(4).name == "qwen3:8b"


def test_asr_and_ocr_are_not_language_models(monkeypatch):
    clear(monkeypatch)
    step_one = ep24_models.stage_model(1)
    assert step_one.backend == "specialist"
    assert "canary" in step_one.name.lower()
    assert ep24_models.stage_model(9).backend == "deterministic"


def test_an_unknown_stage_is_refused_rather_than_defaulted():
    try:
        ep24_models.stage_model(42)
    except ValueError as exc:
        assert "42" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("an unknown stage must not resolve to a default model")


def test_describe_reports_the_source_of_every_choice(monkeypatch):
    clear(monkeypatch)
    monkeypatch.setenv(ep24_models.SHARED_ENV, "qwen3:8b")
    table = {row["stage"]: row for row in ep24_models.describe()}
    assert table[4]["source"] == ep24_models.SHARED_ENV
    assert table[6]["source"] == "repository_default"      # shared ignored
    assert table[6]["pinned"] is True


def test_the_verification_gate_names_missing_models(monkeypatch, tmp_path):
    clear(monkeypatch)
    available = tmp_path / "tags.txt"
    available.write_text("qwen3:8b\n", encoding="utf-8")
    specs = ep24_models.describe((4, 5))
    missing = ep24_models.verify_available(specs, {"qwen3:8b"})
    assert any("stage 5" in item for item in missing)
    assert not any("stage 4" in item for item in missing)


def test_the_cli_prints_the_table_and_fails_on_a_gap(monkeypatch, tmp_path):
    clear(monkeypatch)
    tags = tmp_path / "tags.txt"
    tags.write_text("qwen3:8b\n", encoding="utf-8")
    assert ep24_models.main(["--stage", "4", "--check", str(tags)]) == 0
    assert ep24_models.main(["--stage", "5", "--check", str(tags)]) == 1


def test_json_output_is_machine_readable(monkeypatch, capsys):
    clear(monkeypatch)
    assert ep24_models.main(["--stage", "6", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["model"] == "qwen3.8:27b"
    assert payload[0]["pinned"] is True


def test_deployed_stage_variable_reaches_runtime_and_provenance(monkeypatch):
    clear(monkeypatch)
    monkeypatch.setenv("LACLAUGPT_STEP5_MODEL", "deployed:8b")
    assert ep24_models.stage_model(5).name == "deployed:8b"
    assert ep24_models.stage_model(5).source == "LACLAUGPT_STEP5_MODEL"
    assert ep24_models.stage_ollama_model(5) == "deployed:8b"
    monkeypatch.setenv("LACLAUGPT_EP24_MODEL_STEP_5", "explicit:4b")
    assert ep24_models.stage_model(5).name == "explicit:4b"
    assert ep24_models.stage_ollama_model(5) == "explicit:4b"


# --------------------------------------------------------------------------- #
# The deployed candidate profile (#396): the worker, the launcher and the cache key must agree on
# which model a stage resolves to. The profile is a *named* choice, so it outranks a stray
# LACLAUGPT_MULTIMODAL_MODEL and loses to an explicit per-stage variable.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("stage", sorted(ep24_models.STAGE_MODELS))
def test_candidate_worker_and_launcher_agree(stage, monkeypatch):
    clear(monkeypatch)
    monkeypatch.setenv("LACLAUGPT_MODEL_PROFILE", "small-candidate")
    monkeypatch.setenv("LACLAUGPT_MULTIMODAL_MODEL", "wrong-shared:4b")
    expected = ep24_models.STAGE_MODELS[stage]
    assert ep24_models.stage_ollama_model(stage) == expected
    result = subprocess.run(
        ["python3", str(ROOT / "ep24_models.py"), "--stage", str(stage), "--json"],
        capture_output=True, text=True, check=True,
    )
    assert json.loads(result.stdout)[0]["model"] == expected


def test_preserve_profile_keeps_pilot_selection(monkeypatch):
    clear(monkeypatch)
    monkeypatch.setenv("LACLAUGPT_MULTIMODAL_MODEL", "pilot:model")
    assert ep24_models.stage_ollama_model(7) == "pilot:model"
    assert ep24_models.stage_ollama_model(6) == "qwen3.8:27b"


def test_stage_override_is_local_and_reported(monkeypatch):
    clear(monkeypatch)
    monkeypatch.setenv("LACLAUGPT_MODEL_PROFILE", "small-candidate")
    monkeypatch.setenv("LACLAUGPT_STEP5_MODEL", "qwen3:8b-q4_K_M")
    assert ep24_models.stage_ollama_model(5) == "qwen3:8b-q4_K_M"
    assert ep24_models.stage_ollama_model_source(5) == "LACLAUGPT_STEP5_MODEL"
    assert ep24_models.stage_ollama_model(8) == ep24_models.STAGE_MODELS[8]
    assert ep24_models.stage_ollama_model(6) == "qwen3.8:27b"


def test_step6_rejects_accidental_small_override(monkeypatch):
    clear(monkeypatch)
    monkeypatch.setenv("LACLAUGPT_STEP6_MODEL", "qwen3:8b-q4_K_M")
    with pytest.raises(ValueError, match="pinned"):
        ep24_models.stage_ollama_model(6)


def test_unknown_profile_and_non_llm_stages_fail_closed(monkeypatch):
    clear(monkeypatch)
    monkeypatch.setenv("LACLAUGPT_MODEL_PROFILE", "typo")
    with pytest.raises(ValueError, match="Unknown model profile"):
        ep24_models.stage_ollama_model(7)
    for stage in (1, 3, 9):
        with pytest.raises(ValueError, match="Unsupported Ollama"):
            ep24_models.stage_ollama_model(stage)
