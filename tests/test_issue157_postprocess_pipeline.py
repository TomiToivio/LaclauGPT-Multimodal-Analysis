from __future__ import annotations

import json
import re

import pandas as pd

import ep24_stage_contract as contract
import roihu_enrich as enrich
import roihu_postprocess as post
from roihu_codebooks import CodebookEntry


def _entry(entry_id, kind, label, *, aliases=(), country="FI"):
    return CodebookEntry(
        entry_id=entry_id,
        kind=kind,
        label=label,
        aliases=list(aliases),
        country=country,
        review_state="CANONICAL",
        locked=True,
    )


def _prompt_declared_keys() -> set[str]:
    """The keys the system prompt tells the model to emit.

    Read them from the authoritative "Use exactly these keys" line rather than by
    substring-searching the whole prompt: `video_status` also appears on the
    following "Value of ..." line, so a naive ``in`` check passes even when the key
    has been dropped from the contract. That near-miss is exactly how the schema and
    the prompt drifted apart before #204.
    """
    for line in post.get_system_prompt().splitlines():
        if "Use exactly these keys" in line:
            return set(re.findall(r"`([^`]+)`", line))
    raise AssertionError("system prompt has no 'Use exactly these keys' contract line")


def test_structured_schema_matches_prompt_contract_exactly():
    model = post._result_model()
    # Issue #204 made Step 5 the quality routing gate, so the system prompt now asks
    # the model for `video_status` alongside the extraction fields. The contract this
    # test protects is not a frozen field list: it is that the Pydantic schema and the
    # prompt agree exactly. Assert that agreement rather than pinning literals, or the
    # two drift apart silently the way they did before #204, where the prompt and
    # OUTPUT_COLUMNS carried `video_status` while the result model dropped it.
    declared = _prompt_declared_keys()
    assert declared == set(model.model_fields), (
        "structured-output schema and the prompt's declared keys disagree: "
        f"only in prompt {sorted(declared - set(model.model_fields))}, "
        f"only in schema {sorted(set(model.model_fields) - declared)}"
    )
    assert set(model.model_fields) == {
        "entities",
        "themes",
        "positive",
        "neutral",
        "negative",
        "video_status",
    }
    parsed = model.model_validate_json(
        json.dumps(
            {
                "entities": ["Person A"],
                "themes": ["climate policy"],
                "positive": ["Person A"],
                "neutral": [],
                "negative": ["climate policy"],
                "video_status": "OK",
            }
        )
    )
    assert parsed.entities == ["Person A"]
    assert parsed.themes == ["climate policy"]
    assert parsed.video_status == "OK"


def test_postprocess_context_defaults_to_32k_and_is_overridable(monkeypatch):
    monkeypatch.delenv("LACLAUGPT_POSTPROCESS_NUM_CTX", raising=False)
    assert post.postprocess_num_ctx() == 32768
    monkeypatch.setenv("LACLAUGPT_POSTPROCESS_NUM_CTX", "16384")
    assert post.postprocess_num_ctx() == 16384


def test_summary_is_present_once_and_giant_raw_fields_are_excluded():
    row = pd.Series(
        {
            "researcher_note": "human context",
            "summary_analysis": "UNIQUE SUMMARY TOKEN",
            "metadata": "previous rendered metadata",
            "vllm_video_raw_output": "HUGE RAW VLM BLOB",
            "vllm_video_prompt": "HUGE PREVIOUS PROMPT",
            "vllm_video_status": "ok",
            "entities": '["seed entity"]',
            "themes": '["seed theme"]',
        }
    )
    prompt = post.build_postprocess_prompt(row)
    assert prompt.count("UNIQUE SUMMARY TOKEN") == 1
    assert "human context" in prompt
    assert "vllm_video_status" in prompt
    assert "HUGE RAW VLM BLOB" not in prompt
    assert "HUGE PREVIOUS PROMPT" not in prompt


def test_theme_cleanup_uses_themes_column_not_topics_and_preserves_surface():
    entries = [
        _entry("T-EP", "theme", "EP elections", aliases=("European elections",)),
        _entry("E-ORPO", "entity", "Petteri Orpo", aliases=("Orpo",)),
    ]
    row = pd.Series(
        {
            "themes": '["European elections"]',
            "topics": "THIS MUST NOT DRIVE THEME CLEANUP",
            "positive": '["Orpo"]',
            "neutral": "[]",
            "negative": '["European elections"]',
        }
    )
    resolved = enrich.resolve_postprocess_fields(row, entries, country="FI")
    assert resolved["theme_canonical_names"] == ["EP elections"]
    assert resolved["theme_ids"] == ["T-EP"]
    assert all(
        item["observed"] != "THIS MUST NOT DRIVE THEME CLEANUP"
        for item in resolved["theme_results"]
    )
    assert row["themes"] == '["European elections"]'
    assert {
        (item["polarity"], item["canonical_label"])
        for item in resolved["sentiment_targets"]
        if item["decision"] == "EXISTING"
    } == {("positive", "Petteri Orpo"), ("negative", "EP elections")}


def test_ambiguous_or_fuzzy_cleanup_is_not_silently_accepted():
    entries = [
        _entry("E-A", "entity", "Anna Example"),
        _entry("E-B", "entity", "Anne Example"),
    ]
    row = pd.Series(
        {
            "themes": "[]",
            "positive": '["Ann Example"]',
            "neutral": "[]",
            "negative": "[]",
        }
    )
    resolved = enrich.resolve_postprocess_fields(row, entries, country="FI")
    target = resolved["sentiment_targets"][0]
    assert target["decision"] in {"CANDIDATE", "NEW", "AMBIGUOUS"}
    assert target.get("entry_id") is None


def test_stage5_contract_declares_raw_and_normalized_outputs():
    stage5 = contract.stage(5)
    required = {
        "postprocess_entities",
        "postprocess_themes",
        "positive",
        "neutral",
        "negative",
        "postprocess_summary_md",
        "ep24_entity_ids",
        "ep24_theme_resolution_json",
        "ep24_memory_theme_ids",
        "ep24_sentiment_targets_json",
    }
    assert required.issubset(set(stage5.appends))
    assert "entities" in contract.SOURCE_COLUMNS
    assert "themes" in contract.SOURCE_COLUMNS
    assert "entities" not in stage5.appends
    assert "themes" not in stage5.appends
    assert "entities" not in post.OUTPUT_COLUMNS
    assert "themes" not in post.OUTPUT_COLUMNS


def test_existing_values_are_preserved_when_model_adds_new_items():
    existing = post._existing_list('["seed entity", "another"]')
    combined = [*existing, "another", "new entity"]
    assert list(dict.fromkeys(combined)) == ["seed entity", "another", "new entity"]
