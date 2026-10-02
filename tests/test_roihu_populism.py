import json

import pandas as pd
import pytest
from pydantic import ValidationError

from roihu_populism import (
    AffectObservation,
    EP24DiscourseAnalysis,
    FrontierConstruct,
    SignifierCandidate,
    UsConstruct,
    build_prompt_context,
    compatibility_columns,
)


def _base_result(**overrides):
    data = dict(
        analysis_md="No formula is established.",
        us_constructs=[],
        frontier_constructs=[],
        affects=[],
        chains=[],
        signifier_candidates=[],
        rhetorical_performances=[],
        palonen_dynamic_evidence=[],
        formula_minimum_conditions_met=False,
        formula_abstention_reason="No collective Us and antagonistic frontier are both evidenced.",
        counter_evidence=[],
        uncertainty_notes=[],
        corpus_level_cautions=["Hegemony requires corpus-level evidence."],
    )
    data.update(overrides)
    return EP24DiscourseAnalysis(**data)


def test_empty_non_detection_is_valid():
    result = _base_result()
    assert result.us_constructs == []
    assert result.frontier_constructs == []
    assert result.affects == []
    assert result.formula_minimum_conditions_met is False


def test_confidence_bounds_are_enforced():
    with pytest.raises(ValidationError):
        UsConstruct(label="people", text_span="we", confidence=1.1)


def test_frontier_relation_distinguishes_criticism_from_antagonism():
    frontier = FrontierConstruct(
        label="government criticism",
        them_side="government",
        relation="criticism",
        text_span="the government made a bad decision",
        confidence=0.9,
    )
    assert frontier.relation == "criticism"
    assert frontier.relation != "antagonistic_frontier"


def test_affect_is_not_forced_by_side():
    result = _base_result(
        us_constructs=[
            UsConstruct(label="workers", text_span="we workers", confidence=0.9),
        ],
        affects=[
            AffectObservation(
                label="anger among workers",
                affect="anger",
                target="workers",
                text_span="we workers are furious",
                confidence=0.9,
            )
        ],
    )
    us, frontier = compatibility_columns(result)
    assert us == "workers^anger"
    assert frontier == ""


def test_legacy_compatibility_serialization_is_deterministic():
    result = _base_result(
        us_constructs=[
            UsConstruct(label="citizens", text_span="we citizens", confidence=0.8),
        ],
        frontier_constructs=[
            FrontierConstruct(
                label="boundary against elites",
                us_side="citizens",
                them_side="elites",
                relation="antagonistic_frontier",
                text_span="the elites block our future",
                confidence=0.8,
            )
        ],
        affects=[
            AffectObservation(
                label="hope",
                affect="hope",
                target="citizens",
                text_span="we still have hope",
                confidence=0.8,
            ),
            AffectObservation(
                label="anger",
                affect="anger",
                target="elites",
                text_span="anger at the elites",
                confidence=0.8,
            ),
        ],
        formula_minimum_conditions_met=True,
        formula_abstention_reason=None,
    )
    assert compatibility_columns(result) == ("citizens^hope", "elites^anger")


def test_signifier_candidate_requires_evidence_span_and_document_caution():
    candidate = SignifierCandidate(
        label="change",
        candidate_type="empty",
        text_span="change for everyone",
        confidence=0.4,
    )
    assert candidate.text_span
    assert candidate.document_level_only is True


def test_prompt_separates_evidence_from_memory_and_rag():
    row = pd.Series(
        {
            "_storage_id": "abc",
            "videoDescription": "Current source says reform.",
            "entities": json.dumps(["Researcher Canonical Name"]),
            "themes": json.dumps(["reform"]),
            "summary_analysis": "Prior model summary.",
            "codebook_context_json": json.dumps([{"label": "Party A"}]),
            "memory_context_json": json.dumps([{"label": "Canonical actor"}]),
            "rag_context_json": json.dumps([{"text": "Another document"}]),
        }
    )
    prompt, truncated = build_prompt_context(row, max_chars=10000)
    assert truncated is False
    assert "CURRENT SOURCE + CUMULATIVE RECORD" in prompt
    assert "RESEARCHER/CODEBOOK CONTEXT" in prompt
    assert "normalization_context_not_source_evidence" in prompt
    assert "prior_analysis_context_not_source_evidence" in prompt


def test_prompt_context_budget_is_enforced():
    row = pd.Series({"videoDescription": "x" * 500})
    prompt, truncated = build_prompt_context(row, max_chars=100)
    assert truncated is True
    assert len(prompt) == 100
