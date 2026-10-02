"""Tests for the Step 6 discourse analysis refactor (issue #170).

Synthetic fixtures only; no private data. The model call is stubbed at
``roihu_populism.call_model`` / ``roihu_populism._chat``.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import roihu_populism as step6  # noqa: E402
from roihu_populism import (  # noqa: E402
    AffectObservation,
    DiscourseParseError,
    FormulaOfPopulism,
    FrontierConstruct,
    UsConstruct,
    build_step6_context,
    compute_populist,
    get_step6_system_prompt,
    legacy_pairs,
    render_analysis_md,
)


def _row(**overrides) -> pd.Series:
    base = {
        "video_id": "v1",
        "country": "Finland",
        "author_username": "researcher",
        "allas_filename": "a.mp4",
        "entities": '["sanna marin"]',
        "themes": '["climate, sustainability and environmentalism"]',
        "asr_translated": "We the people deserve better.",
        "frame_analysis_1": "A podium with a flag.",
        "summary_analysis": "A campaign video.",
        "positive": "environment",
        "codebook_context_json": '[{"label": "sanna marin"}]',
        "rag_context_json": '[{"text": "prior analysis"}]',
    }
    base.update(overrides)
    return pd.Series(base)


# --- context builder ---------------------------------------------------------


def test_context_separates_the_four_provenance_classes():
    text, _sha, info = build_step6_context(_row())
    assert "CURRENT SOURCE EVIDENCE" in text
    assert "DERIVED PRIOR-STAGE ANALYSIS" in text
    assert "RESEARCHER/CODEBOOK MEMORY" in text
    assert "SOURCE METADATA" in text
    assert len(info["sections"]) == 4
    assert "We the people deserve better." in text


def test_context_is_deterministic_and_hashed():
    _t1, sha1, _ = build_step6_context(_row())
    _t2, sha2, _ = build_step6_context(_row())
    assert sha1 == sha2
    assert len(sha1) == 64


def test_context_marks_sentiment_as_auxiliary_not_affect():
    text, _sha, _ = build_step6_context(_row())
    assert "NOT affective investment" in text


def test_context_truncates_within_budget():
    row = _row(asr_translated="x" * 50000)
    text, _sha, info = build_step6_context(row, max_chars=2000)
    assert len(text) < 20000
    assert info["truncated"]


# --- schema ------------------------------------------------------------------


def test_empty_output_is_valid_and_yields_no_populist():
    result = FormulaOfPopulism()
    assert result.us_constructs == []
    assert compute_populist(result) is False


def test_confidence_bounds_are_enforced():
    with pytest.raises(Exception):
        UsConstruct(label="we", confidence=2.0)


def test_parse_error_retains_raw_response(monkeypatch):
    monkeypatch.setattr(step6, "_chat", lambda **_: {"message": {"content": "not json"}})
    with pytest.raises(DiscourseParseError) as excinfo:
        step6.call_model("sys", "user", model="m")
    assert "not json" in excinfo.value.raw_response
    assert excinfo.value.metadata["prompt_version"] == step6.PROMPT_VERSION


def test_valid_model_output_gets_provenance(monkeypatch):
    payload = FormulaOfPopulism(populism_analysis="ok").model_dump_json()
    monkeypatch.setattr(step6, "_chat", lambda **_: {"message": {"content": payload}})
    raw, parsed = step6.call_model("sys", "user", model="m")
    assert raw == payload
    assert parsed.model_metadata["model"] == "m"
    assert parsed.generated_at
    assert parsed.prompt_version == step6.PROMPT_VERSION


# --- theory / determinism ----------------------------------------------------


def test_populist_requires_us_and_antagonistic_frontier():
    us = UsConstruct(label="the people")
    assert compute_populist(FormulaOfPopulism(us_constructs=[us])) is False
    # A non-antagonistic boundary is not enough.
    assert compute_populist(
        FormulaOfPopulism(us_constructs=[us], frontier_constructs=[FrontierConstruct(them_side="elites", relation="blame")])
    ) is False
    assert compute_populist(
        FormulaOfPopulism(
            us_constructs=[us],
            frontier_constructs=[FrontierConstruct(them_side="elites", relation="antagonistic_boundary")],
        )
    ) is True


def test_legacy_pairs_only_emit_evidenced_affects():
    affects = [AffectObservation(affect="anger", target="the people")]
    assert legacy_pairs(["the people", "uninvested demand"], affects) == "the people^anger\n"


def test_legacy_pairs_format_matches_the_rdf_contract():
    affects = [AffectObservation(affect="hope", target="the people")]
    text = legacy_pairs(["the people"], affects)
    for line in text.splitlines():
        assert line.count("^") == 1
        element, affect = line.split("^")
        assert element.strip() and affect.strip()


def test_analysis_markdown_falls_back_deterministically():
    result = FormulaOfPopulism(us_constructs=[UsConstruct(label="we")])
    md = render_analysis_md(result)
    assert "populist (deterministic): false" in md
    assert render_analysis_md(result) == md  # deterministic


def test_system_prompt_encodes_the_theory_safeguards():
    prompt = get_step6_system_prompt()
    for phrase in (
        "Empty lists are valid",
        "not sentiment polarity",
        "Do NOT map Us to positive affect",
        "Ordinary disagreement",
        "never present them as evidence",
        "Do not output a populism score",
        "not a permanent label",
    ):
        assert phrase in prompt, phrase


def test_system_prompt_does_not_force_populism():
    prompt = get_step6_system_prompt().casefold()
    assert "you are not required to find populism" in prompt
    assert "positive affects" not in prompt


# --- pipeline contract -------------------------------------------------------


@pytest.fixture()
def stage_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LACLAUGPT_STEP6_SQLITE", str(tmp_path / "cache.db"))
    monkeypatch.delenv("LACLAUGPT_MONGO_ENABLED", raising=False)
    monkeypatch.delenv("LACLAUGPT_REDIS_URL", raising=False)
    monkeypatch.delenv("LACLAUGPT_MAX_ROWS", raising=False)
    return tmp_path


def _write_input(path: Path) -> None:
    columns = ["video_id", "country", "author_username", "allas_filename", "entities", "themes", "asr_translated", "summary_analysis"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for i in range(2):
            writer.writerow({
                "video_id": f"v{i}", "country": "Finland", "author_username": "r",
                "allas_filename": f"a{i}.mp4", "entities": '["sanna marin"]',
                "themes": '["ep elections"]', "asr_translated": "We deserve better.",
                "summary_analysis": "Campaign video.",
            })


def _stub_model(monkeypatch, result: FormulaOfPopulism):
    monkeypatch.setattr(step6, "call_model", lambda s, u, model=None: ("raw", result))


def test_stage_preserves_every_incoming_column(stage_env, monkeypatch):
    src = stage_env / "in.csv"
    out = stage_env / "out.csv"
    _write_input(src)
    _stub_model(
        monkeypatch,
        FormulaOfPopulism(
            populism_analysis="analysis",
            us_constructs=[UsConstruct(label="the people", confidence=0.8)],
            frontier_constructs=[FrontierConstruct(them_side="elites", relation="antagonistic_boundary", confidence=0.7)],
            affects=[AffectObservation(affect="anger", target="the people")],
        ),
    )
    monkeypatch.setenv("LACLAUGPT_INPUT_CSV", str(src))
    monkeypatch.setenv("LACLAUGPT_OUTPUT_CSV", str(out))
    step6.analyze_discourse(None)
    before = pd.read_csv(src, dtype=str, keep_default_na=False)
    after = pd.read_csv(out, dtype=str, keep_default_na=False)
    for column in before.columns:
        assert column in after.columns
        assert before[column].tolist() == after[column].tolist()
    assert after["entities"].tolist() == ['["sanna marin"]'] * 2
    assert after["themes"].tolist() == ['["ep elections"]'] * 2
    for column in step6.OUTPUT_COLUMNS:
        assert column in after.columns


def test_stage_writes_downstream_compatible_output(stage_env, monkeypatch):
    src = stage_env / "in.csv"
    out = stage_env / "out.csv"
    _write_input(src)
    _stub_model(
        monkeypatch,
        FormulaOfPopulism(
            us_constructs=[UsConstruct(label="the people")],
            frontier_constructs=[FrontierConstruct(them_side="elites", relation="antagonistic_boundary")],
            affects=[AffectObservation(affect="anger", target="the people")],
        ),
    )
    monkeypatch.setenv("LACLAUGPT_INPUT_CSV", str(src))
    monkeypatch.setenv("LACLAUGPT_OUTPUT_CSV", str(out))
    step6.analyze_discourse(None)
    after = pd.read_csv(out, dtype=str, keep_default_na=False)
    assert after.loc[0, "formula_of_populism_us"] == "the people^anger\n"
    assert after.loc[0, "formula_of_populism_populist"] == "true"
    assert after.loc[0, "laclau_summary_md"] != ""
    assert after.loc[0, "formula_of_populism_status"] == "ok"
    assert json.loads(after.loc[0, "formula_of_populism_json"])["prompt_version"] == step6.PROMPT_VERSION


def test_cache_hit_is_a_no_op_without_recalling_the_model(stage_env, monkeypatch):
    src = stage_env / "in.csv"
    out = stage_env / "out.csv"
    _write_input(src)
    _stub_model(monkeypatch, FormulaOfPopulism(populism_analysis="first"))
    monkeypatch.setenv("LACLAUGPT_INPUT_CSV", str(src))
    monkeypatch.setenv("LACLAUGPT_OUTPUT_CSV", str(out))
    step6.analyze_discourse(None)

    def _boom(*_a, **_k):
        raise AssertionError("model must not be called on a cache hit")

    monkeypatch.setattr(step6, "call_model", _boom)
    step6.analyze_discourse(None)
    after = pd.read_csv(out, dtype=str, keep_default_na=False)
    assert after.loc[0, "formula_of_populism_analysis"] == "first"
    assert after.loc[0, "laclau_summary_md"] == "first"
    assert after.loc[0, "formula_of_populism_status"] == "cache"


def test_parse_failure_does_not_kill_the_run(stage_env, monkeypatch):
    src = stage_env / "in.csv"
    out = stage_env / "out.csv"
    _write_input(src)

    def _raise(*_a, **_k):
        raise DiscourseParseError("bad", raw_response="{bad", metadata={})

    monkeypatch.setattr(step6, "call_model", _raise)
    monkeypatch.setenv("LACLAUGPT_INPUT_CSV", str(src))
    monkeypatch.setenv("LACLAUGPT_OUTPUT_CSV", str(out))
    step6.analyze_discourse(None)
    after = pd.read_csv(out, dtype=str, keep_default_na=False)
    assert len(after) == 2
    assert (after["formula_of_populism_status"] == "").all()


def test_stage_runs_without_mongo_or_redis(stage_env, monkeypatch):
    """Redis and Mongo are optional; the stage must run with both disabled."""
    src = stage_env / "in.csv"
    out = stage_env / "out.csv"
    _write_input(src)
    _stub_model(monkeypatch, FormulaOfPopulism(populism_analysis="ok"))
    monkeypatch.setenv("LACLAUGPT_INPUT_CSV", str(src))
    monkeypatch.setenv("LACLAUGPT_OUTPUT_CSV", str(out))
    step6.analyze_discourse("finland")
    assert Path(out).exists()
