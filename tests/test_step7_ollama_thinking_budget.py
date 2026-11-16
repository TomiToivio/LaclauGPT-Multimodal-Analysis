"""Regression tests for the Monday500 Step 7 Qwen JSON budget failure.

An Ollama HTTP 200 can return empty content when Qwen uses its entire
num_predict budget for thinking (done_reason=length). The DNA output must
be generated with thinking disabled, while preserving bounded retries.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import step_7_roihu_discourse_network_analysis as step7


class FakeDNAResult:
    @staticmethod
    def model_json_schema():
        return {"type": "object"}

    @staticmethod
    def model_validate_json(raw):
        if raw != '{"statements":[]}':
            raise ValueError("invalid DNA JSON")
        return {"statements": []}


def test_step7_disables_thinking_for_structured_output(monkeypatch):
    calls = []

    def chat(**kwargs):
        calls.append(kwargs)
        return {"message": {"content": '{"statements":[]}'}, "done_reason": "stop"}

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=chat))
    monkeypatch.setattr(step7, "_models", lambda: FakeDNAResult)
    monkeypatch.setattr(step7, "_model_metadata", lambda: {
        "model": "qwen3.8:27b", "num_ctx": 16384, "num_predict": 4096
    })
    raw, parsed = step7.analyze_context("Synthetic political statement")
    assert parsed == {"statements": []}
    assert raw == '{"statements":[]}'
    assert calls[0]["think"] is False
    assert calls[0]["format"] == {"type": "object"}
    assert calls[0]["options"]["num_predict"] == 4096


def test_step7_retries_empty_json_without_enabling_thinking(monkeypatch):
    calls = []

    def chat(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return {"message": {"content": ""}, "done_reason": "length"}
        return {"message": {"content": '{"statements":[]}'}, "done_reason": "stop"}

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=chat))
    monkeypatch.setattr(step7, "_models", lambda: FakeDNAResult)
    monkeypatch.setattr(step7, "_model_metadata", lambda: {
        "model": "qwen3.8:27b", "num_ctx": 16384, "num_predict": 4096
    })
    monkeypatch.setenv("LACLAUGPT_STEP7_PARSE_ATTEMPTS", "3")
    _, parsed = step7.analyze_context("Synthetic transcript")
    assert parsed == {"statements": []}
    assert [c["options"]["num_predict"] for c in calls] == [4096, 8192]
    assert all(c["think"] is False for c in calls)


def test_step7_keeps_failed_json_retryable(monkeypatch):
    calls = []

    def chat(**kwargs):
        calls.append(kwargs)
        return {"message": {"content": ""}, "done_reason": "length"}

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=chat))
    monkeypatch.setattr(step7, "_models", lambda: FakeDNAResult)
    monkeypatch.setattr(step7, "_model_metadata", lambda: {
        "model": "qwen3.8:27b", "num_ctx": 16384, "num_predict": 4096
    })
    monkeypatch.setenv("LACLAUGPT_STEP7_PARSE_ATTEMPTS", "3")
    with pytest.raises(step7.Step7ParseError):
        step7.analyze_context("Synthetic transcript")
    assert len(calls) == 3
    assert all(c["think"] is False for c in calls)

def test_step7_length_retry_requests_compact_complete_json(monkeypatch):
    calls = []

    def chat(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return {"message": {"content": '{"analysis_markdown":"unfinished'}, "done_reason": "length"}
        return {"message": {"content": '{"statements":[]}'}, "done_reason": "stop"}

    monkeypatch.setitem(sys.modules, "ollama", SimpleNamespace(chat=chat))
    monkeypatch.setattr(step7, "_models", lambda: FakeDNAResult)
    monkeypatch.setattr(step7, "_model_metadata", lambda: {
        "model": "qwen3:8b", "num_ctx": 16384, "num_predict": 4096
    })
    monkeypatch.setenv("LACLAUGPT_STEP7_PARSE_ATTEMPTS", "3")
    _, parsed = step7.analyze_context("Source evidence that must remain available")
    assert parsed == {"statements": []}
    assert "RETRY:" not in calls[0]["messages"][0]["content"]
    assert "RETRY:" in calls[1]["messages"][0]["content"]
    assert "Preserve all distinct" in calls[1]["messages"][0]["content"]
    assert calls[0]["messages"][1] == calls[1]["messages"][1]
    assert calls[1]["think"] is False
