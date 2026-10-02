"""Issue #170: theory, schema, persistence and compatibility tests for Step 6.

All fixtures are synthetic and contain no private EP24 research data.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

import roihu_populism as pop


def _result(*, with_affects: bool = True) -> pop.EP24DiscourseResult:
    affects = []
    if with_affects:
        affects = [
            pop.AffectObservation(
                affect="anger",
                target="citizens",
                text_span="we citizens are angry",
                confidence=0.9,
            ),
            pop.AffectObservation(
                affect="admiration",
                target="commission",
                text_span="I admire the commission",
                confidence=0.8,
            ),
        ]
    return pop.EP24DiscourseResult(
        analysis_markdown="Candidate document-level analysis.",
        us_constructs=[
            pop.UsConstruct(
                label="citizens",
                demands=["fair representation"],
                text_span="we citizens",
                confidence=0.9,
            )
        ],
        frontier_constructs=[
            pop.FrontierConstruct(
                us_side="citizens",
                them_side="commission",
                relation="boundary_construction",
                text_span="citizens versus the commission",
                confidence=0.8,
            )
        ],
        affects=affects,
        relations=[
            pop.RelationCandidate(
                relation="equivalence",
                left="fair representation",
                right="democratic control",
                text_span="fair representation and democratic control",
                confidence=0.7,
            )
        ],
        formula_minimum_conditions_met=True,
        counter_evidence=["The opposition is not consistently antagonistic."],
        uncertainty_notes=["Short social-media clip."],
        generated_at="2026-10-02T00:00:00+00:00",
    )


def test_system_prompt_encodes_current_theory_safeguards():
    prompt = pop.SYSTEM_PROMPT
    assert "Do not force an Us, Frontier, affect" in prompt
    assert "negative sentiment is not automatically an antagonistic Frontier" in prompt
    assert "Affect is affective investment, not detachable sentiment" in prompt
    assert "Never map Us -> positive or Frontier -> negative" in prompt
    assert "Co-occurrence is not a chain of equivalence" in prompt
    assert "Frequency/prominence is not hegemony" in prompt
    assert "Polysemy alone is not floating signification" in prompt
    assert "two-sided disagreement is not automatically political polarisation" in prompt
    assert "Empty lists" in prompt


def test_schema_allows_abstention_and_evidence_linked_empty_output():
    result = pop.EP24DiscourseResult(
        analysis_markdown="No supported Formula of Populism components.",
        formula_minimum_conditions_met=False,
        counter_evidence=["Political disagreement without a constitutive frontier."],
        uncertainty_notes=["No explicit collective subject."],
    )
    assert result.us_constructs == []
    assert result.frontier_constructs == []
    assert result.affects == []
    assert result.formula_minimum_conditions_met is False


def test_confidence_is_bounded():
    with pytest.raises(Exception):
        pop.UsConstruct(label="x", text_span="x", confidence=1.1)
    with pytest.raises(Exception):
        pop.AffectObservation(affect="hope", text_span="hope", confidence=-0.1)


def test_legacy_projection_never_fabricates_affect():
    no_affect = _result(with_affects=False)
    us, frontier = pop.legacy_formula_projection(no_affect)
    assert us == ""
    assert frontier == ""

    result = _result(with_affects=True)
    us, frontier = pop.legacy_formula_projection(result)
    assert us == "citizens^anger\n"
    # An opponent can carry admiration: compatibility must not force negative affect.
    assert frontier == "commission^admiration\n"


def test_context_keeps_four_provenance_classes_separate():
    row = pd.Series(
        {
            "video_id": "1",
            "allas_filename": "synthetic.mp4",
            "country": "Finland",
            "entities": '["Petteri Orpo"]',
            "themes": '["EU politics"]',
            "asr_transcript": "current spoken evidence",
            "ocr_1": "VISIBLE SOURCE TEXT",
            "summary_analysis": "derived upstream summary",
            "positive": "auxiliary sentiment",
        }
    )
    context, meta = pop.build_step6_context(
        row,
        memory_items=[
            {
                "kind": "entity",
                "label": "Petteri Orpo",
                "evidence_role": "normalization_context_not_source_evidence",
            }
        ],
        rag_items=[
            {
                "stage": "summary",
                "source_record_id": "other",
                "text": "prior model analysis",
            }
        ],
        codebook_block="Background context only, not evidence: synthetic codebook",
    )
    assert "CURRENT SOURCE EVIDENCE" in context
    assert "current spoken evidence" in context
    assert "DERIVED PRIOR-STAGE ANALYSIS" in context
    assert "derived upstream summary" in context
    assert "RESEARCHER/CODEBOOK MEMORY" in context
    assert "normalization_context_not_source_evidence" in context
    assert "RETRIEVED CORPUS CONTEXT" in context
    assert "prior model analysis" in context
    assert meta["truncated"] is False


def test_context_budget_is_explicit_and_hash_changes():
    row = pd.Series(
        {
            "video_id": "1",
            "allas_filename": "synthetic.mp4",
            "asr_transcript": "x" * 1000,
        }
    )
    context, meta = pop.build_step6_context(row, max_chars=200)
    assert len(context) == 200
    assert meta["truncated"] is True
    a = pop._context_hash(pop.SYSTEM_PROMPT, context, "model-a")
    b = pop._context_hash(pop.SYSTEM_PROMPT, context, "model-b")
    assert a != b


def test_invalid_llm_json_retains_raw_response(monkeypatch):
    class FakeOllama:
        @staticmethod
        def chat(**kwargs):
            return {"message": {"content": "not-json"}}

    monkeypatch.setitem(sys.modules, "ollama", FakeOllama)
    with pytest.raises(pop.DiscourseParseError) as exc:
        pop.analyze_context("synthetic evidence", model="test-model")
    assert exc.value.raw_response == "not-json"
    assert exc.value.metadata["model"] == "test-model"


def test_cache_is_stable_id_model_and_context_sensitive(tmp_path, monkeypatch):
    monkeypatch.setattr(pop, "CACHE_PATH", tmp_path / "step6.sqlite")
    conn = pop._open_cache()
    try:
        result = _result()
        pop._cache_store(
            conn,
            source_id="record-1",
            model="model-a",
            context_sha256="ctx-a",
            raw_response=result.model_dump_json(),
            result=result,
        )
        assert pop._cache_lookup(conn, "record-1", "model-a", "ctx-a") is not None
        assert pop._cache_lookup(conn, "record-1", "model-b", "ctx-a") is None
        assert pop._cache_lookup(conn, "record-1", "model-a", "ctx-b") is None
    finally:
        conn.close()


def test_mongo_patch_contains_full_cumulative_row():
    captured = {}

    class Storage:
        def patch_documents(self, purpose, documents):
            captured["purpose"] = purpose
            captured["documents"] = list(documents)
            return 1

    row = pd.Series(
        {
            "_storage_id": "record-1",
            "entities": '["Human Canonical Name"]',
            "themes": '["Human Theme"]',
            "summary_analysis": "upstream",
            "laclau_status": "ok",
        }
    )
    assert pop._persist_mongo_row(Storage(), row, "record-1") == 1
    doc = captured["documents"][0]
    assert captured["purpose"] == "dataframe"
    assert doc["entities"] == '["Human Canonical Name"]'
    assert doc["themes"] == '["Human Theme"]'
    assert doc["summary_analysis"] == "upstream"
    assert doc["step6_discourse_provenance"]["prompt_version"] == pop.PROMPT_VERSION


def test_step6_preserves_every_incoming_column_and_human_annotation(tmp_path, monkeypatch):
    source = tmp_path / "step5.csv"
    output = tmp_path / "step6.csv"
    incoming = pd.DataFrame(
        [
            {
                "country": "Finland",
                "author_username": "synthetic-author",
                "video_id": "video-1",
                "allas_filename": "synthetic/video-1.mp4",
                "entities": '["Petteri Orpo"]',
                "themes": '["EU politics"]',
                "asr_transcript": "we citizens oppose the commission",
                "summary_analysis": "synthetic summary",
                "postprocess_entities": '["Petteri Orpo"]',
                "postprocess_themes": '["EU politics"]',
                "arbitrary_future_field": "must survive",
            }
        ]
    )
    incoming.to_csv(source, index=False)

    monkeypatch.setenv("LACLAUGPT_INPUT_CSV", str(source))
    monkeypatch.setenv("LACLAUGPT_OUTPUT_CSV", str(output))
    monkeypatch.setenv("LACLAUGPT_MONGO_ENABLED", "0")
    monkeypatch.setenv("LACLAUGPT_COUNTRY", "finland")
    monkeypatch.setattr(pop, "CACHE_PATH", tmp_path / "cache.sqlite")
    monkeypatch.setattr(pop, "_codebook_context", lambda country, query: ("<none>", "", ""))
    monkeypatch.setattr(
        pop,
        "analyze_context",
        lambda context, model=None: (_result().model_dump_json(), _result()),
    )

    pop.run_step6(None)

    result = pd.read_csv(output, dtype=str, keep_default_na=False)
    for column in incoming.columns:
        assert result.loc[0, column] == str(incoming.loc[0, column]), column
    assert result.loc[0, "entities"] == '["Petteri Orpo"]'
    assert result.loc[0, "themes"] == '["EU politics"]'
    assert result.loc[0, "laclau_status"] == "ok"
    structured = json.loads(result.loc[0, "laclau_structured_json"])
    assert structured["formula_minimum_conditions_met"] is True
    assert result.loc[0, "formula_of_populism_us"] == "citizens^anger\n"
    assert result.loc[0, "formula_of_populism_frontier"] == "commission^admiration\n"


def test_limit_refuses_in_place_truncation(tmp_path, monkeypatch):
    source = tmp_path / "same.csv"
    pd.DataFrame(
        [
            {"video_id": "1", "allas_filename": "1.mp4"},
            {"video_id": "2", "allas_filename": "2.mp4"},
        ]
    ).to_csv(source, index=False)
    monkeypatch.setenv("LACLAUGPT_INPUT_CSV", str(source))
    monkeypatch.setenv("LACLAUGPT_OUTPUT_CSV", str(source))
    monkeypatch.setenv("LACLAUGPT_MAX_ROWS", "1")
    monkeypatch.setenv("LACLAUGPT_MONGO_ENABLED", "0")
    with pytest.raisesRegex(ValueError, "Refusing to truncate"):
        pop.run_step6(None)
