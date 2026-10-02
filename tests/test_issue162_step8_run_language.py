"""Regression guard for #162: step 8 must actually process a row.

``run_language`` did ``p = source(lang)`` and later assigned a local named
``source`` inside the edge loop. That made ``source`` local for the whole
function, so the first line raised UnboundLocalError and **step 8 never processed
a single row** -- silently, because the per-row handler swallowed it.

The merged #152 work added ``--help``-only tests, which exit during CLI parsing
before ``run_language`` is reached, so this class of defect is invisible to them.
This test calls the function with a stubbed Ollama client, the way #128's lesson
requires: execute the code, do not grep it.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
STEP8 = ROOT / "step_8_roihu_social_network_analysis.py"


def _load_step8() -> Any:
    spec = importlib.util.spec_from_file_location("_probe_step8_162", STEP8)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fake_ollama(payload: dict[str, Any]) -> types.ModuleType:
    module = types.ModuleType("ollama")

    def chat(**kwargs: Any) -> dict[str, Any]:
        return {"message": {"content": json.dumps(payload)}}

    module.chat = chat  # type: ignore[attr-defined]
    return module


def test_run_language_processes_a_row_instead_of_raising_unbound_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    pytest.importorskip("pydantic")
    monkeypatch.setenv("LACLAUGPT_MAX_ROWS", "1")
    # configure_step_cli() in earlier tests may set this indirectly via
    # os.environ, which pytest's monkeypatch cannot automatically restore.
    # This regression exercises Step 8's historical in-place direct-call path.
    monkeypatch.delenv("LACLAUGPT_OUTPUT_CSV", raising=False)

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

    module = _load_step8()
    module.run_language("")  # must not raise UnboundLocalError

    with csv_path.open(encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["sna_analysis_markdown"] == "md"
    assert '"relation_type": "mention"' in row["sna_edges_json"]


def test_step8_has_no_f823_shadowing_of_the_source_helper():
    """``source`` (the module helper) must not also be a local in run_language."""
    source = STEP8.read_text(encoding="utf-8")
    body = source.split("def run_language", 1)[1]
    # A bare ``source =`` or ``source=`` assignment would shadow the helper.
    assert "\n        source =" not in body
    assert "\n                source=" not in body
