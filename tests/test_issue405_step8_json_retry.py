"""Issue #405: step 8 must not turn a truncated response into a permanent row error.

The real Finland export contains one row whose step-8 status is ``error`` with
``ValidationError: Invalid JSON: EOF`` — a response cut off mid-JSON under Ollama's default budget.
Step 7 had already been given an explicit budget, thinking disabled and bounded retries; step 8 had
none of the three. These tests pin the behaviour without a model or a corpus.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import step_8_roihu_social_network_analysis as step8  # noqa: E402


class FakeResult:
    @staticmethod
    def model_json_schema():
        return {"type": "object"}

    @staticmethod
    def model_validate_json(raw):
        import json

        try:
            payload = json.loads(raw)
        except ValueError as error:
            raise ValueError(f"Invalid JSON: {error}") from error
        if "edges" not in payload:
            raise ValueError("missing edges")
        return {"edges": [], "analysis_markdown": "synthetic"}


def _chat_sequence(payloads):
    calls = []

    def chat(**kwargs):
        calls.append(kwargs)
        content, reason = payloads[min(len(calls) - 1, len(payloads) - 1)]
        return {"message": {"content": content}, "done_reason": reason}

    return chat, calls


def test_a_truncated_response_is_retried_with_a_larger_budget():
    chat, calls = _chat_sequence([('{"edges": [', "length"), ('{"edges": []}', "stop")])
    parsed = step8.extract_with_retries(
        SimpleNamespace(chat=chat), model="synthetic", evidence="synthetic",
        base_budget=2048, max_attempts=3, validate=FakeResult.model_validate_json,
        system_prompt="synthetic system", schema={"type": "object"},
    )
    assert parsed["analysis_markdown"] == "synthetic"
    assert [call["options"]["num_predict"] for call in calls] == [2048, 4096]
    assert all(call["think"] is False for call in calls)


def test_the_failure_names_truncation_after_the_budget_is_exhausted():
    chat, calls = _chat_sequence([('{"edges": [', "length")])
    with pytest.raises(ValueError) as raised:
        step8.extract_with_retries(
            SimpleNamespace(chat=chat), model="synthetic", evidence="synthetic",
            base_budget=1024, max_attempts=2, validate=FakeResult.model_validate_json,
            system_prompt="synthetic system", schema={"type": "object"},
        )
    message = str(raised.value)
    assert "truncated before the JSON closed" in message
    assert "after 2 attempts" in message
    assert [call["options"]["num_predict"] for call in calls] == [1024, 2048]


def test_a_malformed_answer_is_not_reported_as_truncation():
    chat, _ = _chat_sequence([('{"wrong": 1}', "stop")])
    with pytest.raises(ValueError) as raised:
        step8.extract_with_retries(
            SimpleNamespace(chat=chat), model="synthetic", evidence="synthetic",
            base_budget=1024, max_attempts=1, validate=FakeResult.model_validate_json,
            system_prompt="synthetic system", schema={"type": "object"},
        )
    assert "not schema-valid" in str(raised.value)
    assert "truncated" not in str(raised.value)
