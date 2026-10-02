"""Regression tests for #157: Step 5 (roihu_postprocess) defects.

Covers the four concrete bugs found in review, without redesigning the task:

* A -- the Pydantic output schema did not match the prompt or the code.
* B -- Step 5 writes ``themes`` while the resolver reads ``topics``.
* C -- the prompt fed ``summary_analysis`` twice.
* D -- large raw upstream blobs were duplicated into the inference context.
* 3 -- the context window was a hard-coded 4096.

The prompt-rendering tests are the ones that execute real behaviour; the schema
test pins the exact JSON contract a model response must satisfy.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

# roihu_postprocess imports ollama/pydantic at module scope, which the CI `test`
# extra deliberately does not ship (#152 keeps it lean). Production reaches this
# module through step_5_roihu_postprocess.py, which defers the import, so the
# stage itself is fine; these tests simply cannot run without the runtime deps.
pytest.importorskip("ollama")
pytest.importorskip("pydantic")

from ep24_pipeline import metadata_context  # noqa: E402
from roihu_postprocess import (  # noqa: E402
    CONTEXT_EXCLUDED_FIELDS,
    DEFAULT_NUM_CTX,
    Sentiment,
    _num_ctx,
    _num_predict,
)


def test_sentiment_schema_matches_prompt_and_downstream_access():
    """Bug A: schema must expose exactly entities/themes/positive/neutral/negative."""
    props = Sentiment.model_json_schema()["properties"]
    assert set(props) == {"entities", "themes", "positive", "neutral", "negative"}
    assert "topics" not in props

    # A response that follows the documented system prompt must parse, and the
    # attributes the code reads must be present.
    parsed = Sentiment.model_validate_json(
        '{"entities": ["sanna marin"], "themes": ["ep elections"],'
        ' "positive": ["marin"], "neutral": [], "negative": ["perussuomalaiset"]}'
    )
    assert parsed.entities == ["sanna marin"]
    assert parsed.themes == ["ep elections"]
    assert parsed.negative == ["perussuomalaiset"]


def test_summary_analysis_is_not_duplicated_in_prompt_context():
    """Bug C: summary_analysis is added separately, so it must not be in context."""
    row = pd.Series(
        {
            "video_id": "v1",
            "allas_filename": "a.mp4",
            "summary_analysis": "THE SUMMARY TEXT",
            "frame_analysis_1": "a frame",
        }
    )
    context = metadata_context(row, exclude_fields=CONTEXT_EXCLUDED_FIELDS)
    assert "THE SUMMARY TEXT" not in context
    assert "summary_analysis" not in context
    # Non-excluded evidence is still present.
    assert "a frame" in context

    # Without the exclusion the summary *is* present -- proving the field is the
    # one being controlled, not silently absent for another reason.
    assert "THE SUMMARY TEXT" in metadata_context(row)


def test_large_raw_blobs_are_kept_out_of_context_but_stay_in_dataframe():
    """Bug D: reduction applies to the prompt only, never to stored data."""
    row = pd.Series(
        {
            "video_id": "v1",
            "allas_filename": "a.mp4",
            "vllm_video_raw_output": "X" * 5000,
            "vllm_video_structured_json": '{"big": "blob"}',
            "frame_analysis_1": "visible evidence",
        }
    )
    context = metadata_context(row, exclude_fields=CONTEXT_EXCLUDED_FIELDS)
    assert "X" * 100 not in context
    assert "visible evidence" in context
    # The dataframe value is untouched by prompt rendering.
    assert row["vllm_video_raw_output"] == "X" * 5000


def test_num_ctx_defaults_to_32768_and_is_configurable(monkeypatch):
    """Point 3: context must be configurable and default above the old 4096."""
    assert DEFAULT_NUM_CTX == 32768
    monkeypatch.delenv("LACLAUGPT_POSTPROCESS_NUM_CTX", raising=False)
    assert _num_ctx() == 32768
    monkeypatch.setenv("LACLAUGPT_POSTPROCESS_NUM_CTX", "16384")
    assert _num_ctx() == 16384
    monkeypatch.setenv("LACLAUGPT_POSTPROCESS_NUM_PREDICT", "512")
    assert _num_predict() == 512


def test_get_response_passes_the_configured_context(monkeypatch):
    """The configured num_ctx must reach the Ollama call options."""
    import roihu_postprocess as step5

    calls: dict = {}
    fake = types.ModuleType("ollama")

    def chat(**kwargs):
        calls.update(kwargs)
        return {"message": {"content": '{"entities": [], "themes": [],'
                                       ' "positive": [], "neutral": [], "negative": []}'}}

    fake.chat = chat  # type: ignore[attr-defined]
    monkeypatch.setattr(step5, "ollama", fake, raising=False)
    monkeypatch.setenv("LACLAUGPT_POSTPROCESS_NUM_CTX", "24576")

    result = step5.get_response("prompt", "system")
    assert result is not None
    assert calls["options"]["num_ctx"] == 24576
    assert calls["format"] == Sentiment.model_json_schema()


def test_stage_runs_end_to_end_writing_contracted_columns(tmp_path, monkeypatch):
    """The stage must emit entities/themes/sentiment from a cumulative row."""
    import roihu_postprocess as step5

    csv_path = tmp_path / "step5_in.csv"
    pd.DataFrame(
        [
            {
                "video_id": "v1",
                "allas_filename": "a.mp4",
                "country": "Finland",
                "summary_analysis": "summary",
                "entities": "researcher seed",
                "themes": "seed theme",
            }
        ]
    ).to_csv(csv_path, index=False)
    out_path = tmp_path / "step5_out.csv"

    fake = types.ModuleType("ollama")
    fake.chat = lambda **kw: {  # type: ignore[attr-defined]
        "message": {
            "content": '{"entities": ["sanna marin"], "themes": ["ep elections"],'
            ' "positive": ["marin"], "neutral": [], "negative": ["ps"]}'
        }
    }
    monkeypatch.setattr(step5, "ollama", fake, raising=False)
    monkeypatch.setenv("LACLAUGPT_INPUT_CSV", str(csv_path))
    monkeypatch.setenv("LACLAUGPT_OUTPUT_CSV", str(out_path))

    step5.analyze_responses(None)

    out = pd.read_csv(out_path, dtype=str, keep_default_na=False)
    row = out.iloc[0]
    # Upstream researcher seeds survive; model discoveries are appended.
    assert "researcher seed" in row["entities"]
    assert "sanna marin" in row["entities"]
    assert "seed theme" in row["themes"]
    assert "ep elections" in row["themes"]
    assert row["negative"] == "ps"
    assert row["postprocess_summary_md"]


def test_themes_column_is_the_one_normalization_resolves():
    """Bug B: enrich must read the same concept Step 5 writes (`themes`)."""
    enrich = (ROOT / "roihu_enrich.py").read_text(encoding="utf-8")
    assert '("entity", "entities", entity_ids)' in enrich
    assert '("topic", theme_column, topic_ids)' in enrich
    assert 'theme_column = "themes"' in enrich
    # The old `topic` loop resolved a column Step 5 never writes.
    assert '("topic", "topics", topic_ids)' not in enrich
