"""Regression tests for #152: steps 7/8 must survive a minimal environment.

Steps 7 and 8 imported ``ollama`` and ``pydantic`` at module scope, so their
``--help`` (and any argument validation) died with ModuleNotFoundError unless
those packages were installed. The structured-output models are now built by a
factory that imports pydantic on demand, and the Ollama client is imported inside
``run_language``.

These tests execute the real scripts rather than grepping them, running them with
the two packages made unimportable. A source check alone would pass even if the
import crept back into a path that only runs on ``--help``.
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]

STEPS = {
    7: "step_7_roihu_discourse_network_analysis.py",
    8: "step_8_roihu_social_network_analysis.py",
}
HEAVY = ("ollama", "pydantic")

_BLOCKER = """
import sys
BLOCKED = {blocked!r}
class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in BLOCKED:
            raise ModuleNotFoundError(f"No module named {{name!r}}")
        return None
sys.meta_path.insert(0, _Blocker())
"""


def _run_hiding_heavy(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Run ``script`` as __main__ with ollama/pydantic forced to look absent."""
    target = ROOT / script
    wrapper = (
        _BLOCKER.format(blocked=HEAVY)
        + f"sys.argv = [{script!r}, *{list(args)!r}]\n"
        + f"exec(compile(open({str(target)!r}).read(), {script!r}, 'exec'),"
        + f" {{'__name__': '__main__', '__file__': {str(target)!r}}})\n"
    )
    return subprocess.run(
        [sys.executable, "-c", wrapper],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )


def _load(script: str, name: str) -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, ROOT / script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("step", sorted(STEPS))
def test_help_works_without_ollama_or_pydantic(step):
    result = _run_hiding_heavy(STEPS[step], "--help")
    combined = result.stdout + result.stderr
    assert "ModuleNotFoundError" not in combined, combined
    assert result.returncode == 0, combined
    assert "--country" in combined


@pytest.mark.parametrize("step", sorted(STEPS))
def test_invalid_country_still_rejected_without_heavy_deps(step):
    """Argument validation must run before any heavy import."""
    result = _run_hiding_heavy(STEPS[step], "--country", "atlantis")
    combined = result.stdout + result.stderr
    assert "ModuleNotFoundError" not in combined, combined
    assert result.returncode != 0
    assert "unknown country" in combined


def test_heavy_imports_are_not_at_module_scope():
    """A cheap textual backstop for the executed behaviour above."""
    for script in STEPS.values():
        module_lines = [
            line
            for line in (ROOT / script).read_text(encoding="utf-8").splitlines()
            if line.startswith(("import ", "from "))
        ]
        assert not any("ollama" in line for line in module_lines), script
        assert not any("pydantic" in line for line in module_lines), script


@pytest.mark.parametrize(
    ("step", "factory"),
    [(7, "_dna_models"), (8, "_sna_models")],
)
def test_schema_factories_require_pydantic_only_when_called(step, factory):
    """The models must still be real pydantic models once the factory runs.

    Skipped where pydantic is absent: the point of #152 is that the *CLI* must
    work without it, and keeping the test extra lean is option 1. Where pydantic
    is installed (Roihu, dev), this pins the factory's output shape.
    """
    pytest.importorskip("pydantic")
    module = _load(STEPS[step], f"_probe_{factory}")
    model = getattr(module, factory)()
    assert hasattr(model, "model_json_schema")
    assert model.model_json_schema()["type"] == "object"


def _fake_ollama(payload: dict[str, Any]):
    """A stand-in ``ollama`` module whose chat() returns a canned JSON reply."""
    import json
    import types

    module = types.ModuleType("ollama")

    def chat(**kwargs: Any) -> dict[str, Any]:
        return {"message": {"content": json.dumps(payload)}}

    module.chat = chat  # type: ignore[attr-defined]
    return module


def test_step8_run_language_reaches_inference_without_shadowing_bug(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Regression: a local ``source`` assignment shadowed the ``source()`` helper.

    ``run_language`` did ``p = source(lang)`` and later assigned a local named
    ``source`` inside the edge loop, which made ``source`` local for the whole
    function and raised UnboundLocalError on the very first line -- step 8 could
    never process a single row. ruff flagged it as F823; this executes the path.
    """
    pytest.importorskip("pydantic")
    monkeypatch.setenv("LACLAUGPT_MAX_ROWS", "1")

    csv_path = tmp_path / "ep24_fi.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["video_id", "allas_filename", "country", "summary_analysis"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "video_id": "v1",
                "allas_filename": "a.mp4",
                "country": "Finland",
                "summary_analysis": "evidence",
            }
        )
    monkeypatch.setenv("LACLAUGPT_INPUT_CSV", str(csv_path))

    module = _load(STEPS[8], "_probe_step8_runlang")
    payload = {
        "analysis_markdown": "md",
        "edges": [
            {
                "source_actor": "A",
                "target_actor": "B",
                "relation_type": "mention",
                "directed": True,
                "evidence_quote": "q",
                "confidence": 0.5,
            }
        ],
    }
    monkeypatch.setitem(sys.modules, "ollama", _fake_ollama(payload))

    module.run_language("")  # must not raise UnboundLocalError

    saved = csv.DictReader(csv_path.open(encoding="utf-8"))
    row = next(saved)
    assert row["sna_analysis_markdown"] == "md"
    assert '"relation_type": "mention"' in row["sna_edges_json"]
