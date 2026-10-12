"""Regression tests for empty-but-HTTP-200 Ollama Step 4 responses (#236)."""
import sys
import types

import pytest

import roihu_summary


def _response(content):
    return {"message": {"content": content}, "done_reason": "stop"}


def test_empty_model_response_is_retried_without_changing_evidence(monkeypatch):
    requests = []
    replies = iter([_response(""), _response("   "), _response("Valid descriptive summary")])

    def chat(**kwargs):
        requests.append(kwargs)
        return next(replies)

    monkeypatch.setitem(sys.modules, "ollama", types.SimpleNamespace(chat=chat))
    monkeypatch.setenv("LACLAUGPT_SUMMARY_EMPTY_RETRIES", "2")
    assert roihu_summary.get_llama_summary_response("system", "evidence", model="test-model") == "Valid descriptive summary"
    assert len(requests) == 3
    assert all(req["messages"][1]["content"] == "evidence" for req in requests)


def test_permanent_empty_response_raises_without_returning_empty(monkeypatch):
    calls = []
    def chat(**kwargs):
        calls.append(1)
        return _response("")
    monkeypatch.setitem(sys.modules, "ollama", types.SimpleNamespace(chat=chat))
    monkeypatch.setenv("LACLAUGPT_SUMMARY_EMPTY_RETRIES", "1")
    with pytest.raises(ValueError, match="after 2 attempts"):
        roihu_summary.get_llama_summary_response("s", "u", model="test-model")
    assert len(calls) == 2


def test_nonempty_response_does_not_retry(monkeypatch):
    calls = []
    def chat(**kwargs):
        calls.append(1)
        return _response("  summary  ")
    monkeypatch.setitem(sys.modules, "ollama", types.SimpleNamespace(chat=chat))
    assert roihu_summary.get_llama_summary_response("s", "u", model="test-model") == "summary"
    assert len(calls) == 1
