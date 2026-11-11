"""#378: every stage resolves its own model, and Step 6 cannot be dragged off 27B by a shared variable.

The failure this guards against is a specific, plausible accident: someone exports
``LACLAUGPT_MULTIMODAL_MODEL=qwen3:8b`` to speed up frame analysis, and the discourse analysis quietly
starts running on an 8B model. The pipeline would keep producing well-formed JSON, and the methodology
would have changed without a single error in the log.
"""

from __future__ import annotations

import json
# The module under test lives at the repository root, next to the stage scripts.
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ep24_models  # noqa: E402


def clear(monkeypatch):
    monkeypatch.delenv(ep24_models.SHARED_ENV, raising=False)
    for stage in ep24_models.STAGES:
        monkeypatch.delenv(ep24_models.STAGE_ENV.format(stage=stage), raising=False)


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
