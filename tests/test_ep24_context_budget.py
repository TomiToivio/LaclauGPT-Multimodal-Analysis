"""Shared Steps 4-8 context budget and prompt contract (issue #241).

The four acceptance conditions are the four test classes: a normal packet, an empty one, an
oversized one, and the failure cases. Two further properties were missing from all three ad-hoc
implementations this module replaces, so they are pinned here: the reserve, and a cache key that
moves when the prompt, model or options move.
"""
from __future__ import annotations

import pytest

from ep24_context_budget import (
    NOTHING_USABLE, TRUNCATION_MARKER, BudgetError, Channel, build_packet, clean,
    estimate_tokens, prompt_contract_hash, resolve_budget, summarise,
)

EVIDENCE = "Speech [ASR original language; transcription may be inaccurate]"
CONTEXT = "Researcher normalization [NOT video evidence]"


class TestNormal:
    def test_a_fitting_packet_carries_every_channel_whole(self):
        packet = build_packet([
            Channel("asr", EVIDENCE, "a spoken line", priority=10),
            Channel("meta", "Source metadata [context; not proof of on-screen content]",
                    "country: Finland", priority=1),
        ])
        assert packet.truncated_fields == ()
        assert packet.omitted_fields == ()
        assert EVIDENCE in packet.text and "a spoken line" in packet.text
        assert packet.used <= packet.budget - packet.reserve_output - packet.reserve_schema

    
    def test_the_report_gives_chars_and_tokens_per_field(self):
        packet = build_packet([Channel("asr", EVIDENCE, "x" * 400, priority=1)])
        row = next(r for r in packet.report if r.channel == "asr")
        assert row.chars == 400
        assert row.est_tokens == 100
        assert "chars" in packet.render_report() and "tokens" in packet.render_report()

    def test_priority_decides_who_gets_the_budget(self):
        packet = build_packet(
            [
                Channel("low", "Context [not evidence]", "L" * 9000, priority=0),
                Channel("high", EVIDENCE, "H" * 100, priority=10),
            ],
            total=6000,
        )
        # the high-priority field is whole and comes first; the low-priority one fills what is
        # left and is explicitly truncated rather than dropped
        assert "H" * 100 in packet.text
        assert packet.text.index("H" * 100) < packet.text.index("L" * 50)
        assert packet.truncated_fields == ("low",)
        assert packet.omitted_fields == ()

    def test_provenance_labels_reach_the_model(self):
        packet = build_packet([Channel("memory", CONTEXT, "prior note", priority=1)])
        assert CONTEXT in packet.text


class TestEmpty:
    def test_no_channels_states_the_limitation_instead_of_inventing_content(self):
        packet = build_packet([])
        assert packet.text == NOTHING_USABLE
        assert packet.used > 0  # the limitation itself is the body

    def test_blank_and_pandas_missing_values_are_empty(self):
        assert clean("nan") == "" and clean("NaT") == "" and clean("<NA>") == ""
        assert clean(None) == "" and clean("  spaced  ") == "spaced"
        packet = build_packet([
            Channel("a", EVIDENCE, "", priority=2),
            Channel("b", EVIDENCE, "nan", priority=1),
        ])
        assert packet.text == NOTHING_USABLE
        assert set(packet.omitted_fields) == {"a", "b"}

    def test_an_empty_channel_is_reported_not_hidden(self):
        packet = build_packet([
            Channel("full", EVIDENCE, "present", priority=2),
            Channel("blank", EVIDENCE, "", priority=1),
        ])
        row = next(r for r in packet.report if r.channel == "blank")
        assert row.omitted and row.reason == "empty"
        assert summarise(packet)["omitted"] == ["blank"]


class TestOversized:
    def test_an_oversized_field_is_truncated_with_an_in_band_marker(self):
        packet = build_packet(
            [Channel("asr", EVIDENCE, "word " * 9000, priority=1, cap=500)], total=6000
        )
        row = next(r for r in packet.report if r.channel == "asr")
        assert row.truncated and TRUNCATION_MARKER in packet.text
        assert packet.truncated_fields == ("asr",)

    def test_truncation_never_exceeds_the_channel_budget(self):
        packet = build_packet(
            [Channel("asr", EVIDENCE, "word " * 9000, priority=1, cap=333)], total=6000
        )
        assert len(packet.text) <= 333 + len(EVIDENCE) + 8

    def test_the_total_budget_is_never_exceeded_across_channels(self):
        packet = build_packet(
            [Channel(f"c{i}", EVIDENCE, "z" * 4000, priority=i) for i in range(6)], total=9000
        )
        assert packet.used <= packet.budget - packet.reserve_output - packet.reserve_schema

    def test_the_marker_is_visibly_not_source_text(self):
        packet = build_packet(
            [Channel("asr", EVIDENCE, "a" * 5000, priority=1, cap=200)], total=6000
        )
        assert "TRUNCATED" in packet.text and "source is unchanged" in packet.text


class TestFailure:
    def test_a_budget_below_the_floor_is_refused(self):
        with pytest.raises(BudgetError):
            resolve_budget(500)

    def test_a_reserve_that_swallows_the_budget_is_refused_with_the_way_out(self):
        with pytest.raises(BudgetError) as exc:
            resolve_budget(4000, reserve_output=3000, reserve_schema=2000)
        assert "raise" in str(exc.value) or "lower the reserves" in str(exc.value)

    def test_a_negative_reserve_is_refused(self):
        with pytest.raises(BudgetError):
            resolve_budget(5000, reserve_output=-1)

    def test_a_non_integer_environment_value_is_refused_not_defaulted(self, monkeypatch):
        monkeypatch.setenv("LACLAUGPT_CONTEXT_MAX_CHARS", "twenty thousand")
        with pytest.raises(BudgetError):
            resolve_budget()

    def test_a_cap_of_zero_omits_the_field_rather_than_emitting_a_headless_block(self):
        packet = build_packet(
            [Channel("asr", EVIDENCE, "some evidence", priority=1, cap=0)], total=6000
        )
        assert packet.text == NOTHING_USABLE
        assert packet.omitted_fields == ("asr",)


class TestReserveAndCacheKey:
    def test_the_reserve_is_held_back_from_the_input(self):
        packet = build_packet(
            [Channel("asr", EVIDENCE, "w" * 50000, priority=1)], total=10000,
            reserve_output=1500, reserve_schema=500,
        )
        assert packet.used <= 10000 - 1500 - 500
        assert (packet.reserve_output, packet.reserve_schema) == (1500, 500)

    
    def test_the_cache_key_moves_when_the_prompt_model_or_options_move(self):
        base = prompt_contract_hash("prompt", "model-a", {"temperature": 0})
        assert prompt_contract_hash("other", "model-a", {"temperature": 0}) != base
        assert prompt_contract_hash("prompt", "model-b", {"temperature": 0}) != base
        assert prompt_contract_hash("prompt", "model-a", {"temperature": 1}) != base
        # unchanged inputs, and a reordered mapping, are the same contract
        assert prompt_contract_hash("prompt", "model-a", {"temperature": 0}) == base
        # extra (e.g. a schema version) is part of the contract too
        assert prompt_contract_hash("prompt", "model-a", {"temperature": 0}, extra=["v2"]) != base

    def test_options_are_order_insensitive(self):
        assert prompt_contract_hash("p", "m", {"a": 1, "b": 2}) == \
               prompt_contract_hash("p", "m", {"b": 2, "a": 1})

    def test_token_estimate_is_zero_for_empty_text(self):
        assert estimate_tokens("") == 0
        assert estimate_tokens("nan") == 0
        assert estimate_tokens("abcd") == 1
        assert estimate_tokens("a" * 5) == 2
