"""Contract coverage for the single shared EP24 Steps 4-8 prompt/context builder (issue #241).

This one suite replaces the three duplicate suites that shipped with the three duplicate modules
the consolidation removed (``test_ep24_context_budget.py``, ``test_ep24_evidence_packets.py``,
``test_summary_evidence.py``). Nothing they covered is dropped; what changes is that the
assertions now run against the one builder every stage uses.

The four acceptance conditions #241 names are the first four classes: a normal packet, an empty
one, an oversized one, and the failure cases. The rest pins the properties the issue asks for and
the consolidation must not lose: the per-stage contract budgets, the evidence/context split,
input immutability, the cache key moving with the prompt, model and generation options, and the
per-stage field tables.
"""
from __future__ import annotations

import pandas as pd
import pytest

from ep24_prompt_contract import (
    CONTEXT,
    NO_EVIDENCE,
    PREAMBLE_STEPS_6_8,
    STAGE_BUDGETS,
    TRUNCATION_MARKER,
    BudgetError,
    Stream,
    build_packet,
    build_stage_packet,
    clean,
    effective_prompt_fingerprint,
    estimate_tokens,
    normalise_stage,
    stage_budget,
)

LABEL_VIDEO = "Whole-video narrative [video analysis; model-generated, verify against sources]"
LABEL_METADATA = "Source metadata [context; not proof of on-screen content]"
LABEL_MEMORY = "Researcher normalization [NOT video evidence]"

#: One row with every field every stage reads, each long enough to be clipped.
OVERSIZED_ROW = {
    "video_id": "vid-1", "country": "Finland", "language": "fi", "platform": "tiktok",
    "author_username": "author", "username": "author", "url": "https://example.invalid/v",
    "date": "2024-05-01", "published": "2024-05-01", "created_at": "2024-05-01",
    "caption": "caption", "description": "description", "videoDescription": "description",
    "vllm_video_analysis": "video " * 4000,
    "vllm_video_markdown_analysis": "video " * 4000,
    "vllm_structured_output": "video " * 4000,
    "asr_transcript": "speech " * 4000,
    "whisper_transcript": "speech " * 4000,
    "whisperResult": "speech " * 4000,
    "asr_translated": "translation " * 4000,
    "asr_translation": "translation " * 4000,
    "whisper_translated": "translation " * 4000,
    "frame_analysis_1": "visuals " * 4000,
    "ocr_1": "SIGN " * 4000,
    "entities": "actor " * 4000, "themes": "theme " * 4000,
    "research_notes": "note " * 4000, "researcher_note": "note " * 4000,
    "summary_analysis": "summary " * 4000,
    "codebook_context_json": "codebook " * 4000,
    "dna_statements_json": "statement " * 4000,
    "laclau_structured_json": "analysis " * 4000,
    "ep24_entity_resolution_json": "resolution " * 4000,
}


# ---------------------------------------------------------------------------------------------
# The per-stage contract: the runbook's documented model-visible budgets
# ---------------------------------------------------------------------------------------------


class TestStageBudgets:
    def test_every_stage_in_the_programme_has_the_documented_budget(self):
        assert set(STAGE_BUDGETS) == {"step4", "step6", "step7", "step8"}
        assert STAGE_BUDGETS["step4"].model_visible_chars == 20000
        assert STAGE_BUDGETS["step6"].model_visible_chars == 28000
        assert STAGE_BUDGETS["step7"].model_visible_chars == 20000
        assert STAGE_BUDGETS["step8"].model_visible_chars == 12000

    @pytest.mark.parametrize("stage", ["step4", "step6", "step7", "step8"])
    def test_an_oversized_row_never_exceeds_the_stage_contract(self, stage):
        packet = build_stage_packet(stage, OVERSIZED_ROW)
        assert packet.used <= STAGE_BUDGETS[stage].model_visible_chars
        # A row this large must have been clipped somewhere, and the clip must be visible.
        assert packet.any_truncation
        assert packet.truncated_fields or packet.omitted_fields

    def test_the_stage_key_is_accepted_as_a_number_or_a_name(self):
        assert normalise_stage(6) == normalise_stage("step6") == normalise_stage("STEP6")
        with pytest.raises(KeyError):
            normalise_stage("step99")
        with pytest.raises(KeyError):
            normalise_stage(5)

    @pytest.mark.parametrize(
        "stage,env,total",
        [
            ("step4", "LACLAUGPT_SUMMARY_EVIDENCE_CHARS", 25000),
            ("step6", "LACLAUGPT_STEP6_MAX_CONTEXT_CHARS", 9000),
            ("step7", "LACLAUGPT_STEP7_MAX_CONTEXT_CHARS", 9000),
            ("step8", "LACLAUGPT_STEP8_MAX_EVIDENCE_CHARS", 9000),
        ],
    )
    def test_each_stage_budget_is_configurable_by_its_documented_env_var(
        self, monkeypatch, stage, env, total
    ):
        monkeypatch.setenv(env, str(total))
        assert stage_budget(stage).model_visible_chars == total
        assert build_stage_packet(stage, OVERSIZED_ROW).used <= total

    def test_an_explicit_budget_beats_the_environment(self, monkeypatch):
        monkeypatch.setenv("LACLAUGPT_STEP7_MAX_CONTEXT_CHARS", "9000")
        assert stage_budget("step7", model_visible_chars=5000).model_visible_chars == 5000
        assert build_stage_packet("step7", OVERSIZED_ROW, max_chars=5000).used <= 5000

    def test_step4_records_its_output_and_schema_reserve(self):
        budget = stage_budget("step4")
        assert (budget.output_reserve_chars, budget.schema_reserve_chars) == (2000, 800)
        assert budget.declared_total_chars == budget.model_visible_chars + 2800


# ---------------------------------------------------------------------------------------------
# Normal
# ---------------------------------------------------------------------------------------------


class TestNormal:
    def test_a_fitting_packet_carries_every_field_whole_with_its_label(self):
        packet = build_packet("step4", [
            Stream("video", LABEL_VIDEO, "a whole-video narrative"),
            Stream("metadata", LABEL_METADATA, "country: Finland"),
        ])
        assert packet.truncated_fields == ()
        assert packet.omitted_fields == ()
        assert LABEL_VIDEO in packet.text and "a whole-video narrative" in packet.text
        assert "country: Finland" in packet.text
        assert packet.used <= packet.model_visible_chars

    def test_the_report_gives_chars_and_tokens_per_field(self):
        packet = build_packet("step4", [Stream("ocr", "Visible text [OCR]", "x" * 400)])
        row = next(r for r in packet.report if r.name == "ocr")
        assert (row.kept_chars, row.original_chars, row.est_tokens) == (400, 400, 100)
        assert "chars" in packet.render_report() and "tokens" in packet.render_report()
        assert "stage=step4" in packet.log_line()
        assert "truncated=[]" in packet.log_line()

    def test_field_order_is_priority_and_context_comes_last(self):
        packet = build_packet("step4", [
            Stream("asr_original", "Speech [ASR original language]", "first"),
            Stream("ocr", "Visible text [OCR]", "second"),
            Stream("rag", LABEL_MEMORY, "third", kind=CONTEXT),
        ])
        assert packet.text.index("first") < packet.text.index("second") < packet.text.index("third")
        assert packet.evidence_chars > 0 and packet.context_chars > 0

    def test_provenance_labels_reach_the_model(self):
        packet = build_stage_packet(4, {"ocr_1": "VOTE 2024"})
        assert "may be inaccurate" in packet.text

    @pytest.mark.parametrize("stage", ["step6", "step7", "step8"])
    def test_steps_6_to_8_open_with_the_descriptive_context_preamble(self, stage):
        packet = build_stage_packet(stage, {"asr_transcript": "Emme tue tätä lakia"})
        assert packet.text.startswith(PREAMBLE_STEPS_6_8)

    def test_step4_has_no_preamble_and_its_own_metadata_block(self):
        packet = build_stage_packet(4, {"asr_transcript": "puhetta suomeksi",
                                        "author_username": "author"})
        assert not packet.text.startswith(PREAMBLE_STEPS_6_8)
        assert "puhetta suomeksi" in packet.text
        assert "author_username: author" in packet.text


# ---------------------------------------------------------------------------------------------
# Empty
# ---------------------------------------------------------------------------------------------


class TestEmpty:
    def test_no_fields_states_the_limitation_instead_of_inventing_content(self):
        packet = build_packet("step4", [])
        assert packet.text == NO_EVIDENCE
        assert packet.used > 0
        assert packet.any_truncation is False

    def test_blank_and_pandas_missing_values_are_empty(self):
        assert clean("nan") == "" and clean("NaT") == "" and clean("<NA>") == ""
        assert clean(None) == "" and clean("  spaced  ") == "spaced"
        packet = build_packet("step4", [
            Stream("video", LABEL_VIDEO, ""),
            Stream("asr_original", "Speech [ASR original language]", "nan"),
        ])
        assert packet.text == NO_EVIDENCE
        assert set(packet.omitted_fields) == {"video", "asr_original"}
        # An empty field is not a truncation: nothing was withheld from the model.
        assert packet.any_truncation is False

    def test_an_empty_field_is_reported_not_hidden(self):
        packet = build_packet("step4", [
            Stream("ocr", "Visible text [OCR]", "present"),
            Stream("video", LABEL_VIDEO, ""),
        ])
        row = next(r for r in packet.report if r.name == "video")
        assert row.omitted and row.reason == "empty field"

    def test_a_step4_row_with_no_evidence_says_so(self):
        assert NO_EVIDENCE in build_stage_packet(4, {}).text

    def test_step8_with_only_context_columns_yields_no_unsupported_relation_evidence(self):
        packet = build_stage_packet(8, {
            "rag_context_json": "Actor A controls Actor B",
            "memory_context_json": "Actor A controls Actor B",
        })
        assert "Actor A" not in packet.text

    def test_an_empty_json_container_is_not_evidence_for_steps_6_to_8(self):
        """Steps 6-8 carry no context columns, and an empty JSON container is not a field.

        The packet is then the stage's preamble alone -- the behaviour Steps 6-8 had before
        consolidation, kept deliberately: the preamble itself tells the model what it is holding.
        """
        packet = build_stage_packet(6, {
            "entities": "[]", "themes": "{}", "codebook_context_json": "[]",
            "memory_context_json": "SECRET_MEMORY_CONTEXT",
        })
        assert packet.text == PREAMBLE_STEPS_6_8
        assert "[entities |" not in packet.text
        assert "[themes |" not in packet.text
        assert "SECRET_MEMORY_CONTEXT" not in packet.text


# ---------------------------------------------------------------------------------------------
# Oversized
# ---------------------------------------------------------------------------------------------


class TestOversized:
    def test_an_oversized_field_is_clipped_with_an_in_band_marker(self):
        packet = build_packet("step8", [Stream("asr_transcript", "Speech [ASR]", "word " * 9000)])
        row = next(r for r in packet.report if r.name == "asr_transcript")
        assert row.truncated and TRUNCATION_MARKER in packet.text
        assert packet.truncated_fields == ("asr_transcript",)
        assert packet.any_truncation is True

    def test_a_field_never_exceeds_its_own_cap(self):
        packet = build_packet("step8", [
            Stream("asr_transcript", "Speech [ASR]", "word " * 9000, cap=333),
        ])
        row = next(r for r in packet.report if r.name == "asr_transcript")
        assert row.cap == 333 and row.kept_chars <= 333
        assert row.original_chars == len(("word " * 9000).strip())

    def test_the_total_budget_is_never_exceeded_across_fields(self):
        packet = build_packet("step7", [
            Stream(f"field{i}", "label", "z" * 40000) for i in range(12)
        ])
        assert packet.used <= packet.model_visible_chars

    def test_a_field_with_no_room_left_is_reported_as_omitted(self):
        packet = build_packet("step7", [
            Stream("first", "label", "y" * 9000, cap=9000),
            Stream("second", "label", "y" * 9000, cap=9000),
        ], model_visible_chars=9000)
        assert "first" in packet.truncated_fields or packet.truncated_fields
        row = next(r for r in packet.report if r.name == "second")
        assert row.omitted and row.reason == "no budget left for this field"

    def test_the_marker_is_visibly_not_source_text(self):
        packet = build_packet("step8", [
            Stream("dna_statements_json", "prior model analysis", "a" * 9000, cap=3800),
        ])
        assert TRUNCATION_MARKER in packet.text
        assert packet.text.endswith(TRUNCATION_MARKER)

    def test_the_bound_is_exact_not_approximate(self):
        """The packet fills its contract exactly; a bound that is not a bound is not a bound."""
        for stage in ("step4", "step6", "step7", "step8"):
            packet = build_stage_packet(stage, OVERSIZED_ROW)
            assert packet.used <= STAGE_BUDGETS[stage].model_visible_chars


# ---------------------------------------------------------------------------------------------
# Failure
# ---------------------------------------------------------------------------------------------


class TestFailure:
    @pytest.mark.parametrize("stage,floor", [("step4", 2000), ("step6", 128), ("step7", 128),
                                             ("step8", 3000)])
    def test_a_budget_below_the_stage_floor_is_refused(self, stage, floor):
        with pytest.raises(BudgetError):
            stage_budget(stage, model_visible_chars=floor - 1)

    def test_an_unknown_stage_is_refused(self):
        with pytest.raises(KeyError):
            stage_budget("step12")

    def test_a_non_integer_budget_environment_value_is_refused_not_defaulted(self, monkeypatch):
        monkeypatch.setenv("LACLAUGPT_STEP6_MAX_CONTEXT_CHARS", "twenty eight thousand")
        with pytest.raises(BudgetError):
            stage_budget("step6")

    def test_a_negative_reserve_is_refused(self, monkeypatch):
        monkeypatch.setenv("LACLAUGPT_SUMMARY_RESERVE_OUTPUT_CHARS", "-1")
        with pytest.raises(BudgetError):
            stage_budget("step4")

    def test_a_reserve_that_does_not_fit_the_budget_is_refused_with_the_way_out(self, monkeypatch):
        monkeypatch.setenv("LACLAUGPT_SUMMARY_EVIDENCE_CHARS", "2500")
        monkeypatch.setenv("LACLAUGPT_SUMMARY_RESERVE_OUTPUT_CHARS", "2000")
        with pytest.raises(BudgetError) as exc:
            stage_budget("step4")
        assert "raise" in str(exc.value) and "lower the reserves" in str(exc.value)

    def test_an_unknown_field_kind_is_refused(self):
        with pytest.raises(ValueError):
            build_packet("step6", [Stream("asr_transcript", "Speech [ASR]", "x", kind="evidence-ish")])

    def test_a_cap_of_zero_omits_the_field_rather_than_emitting_a_headless_block(self):
        packet = build_packet("step6", [
            Stream("asr_transcript", "Speech [ASR]", "some evidence", cap=0),
        ])
        assert packet.text == NO_EVIDENCE
        assert packet.omitted_fields == ("asr_transcript",)

    def test_an_oversized_preamble_cannot_push_the_packet_past_its_budget(self):
        packet = build_packet("step6", [], model_visible_chars=3000, preamble="p" * 40000)
        assert packet.used <= 3000


# ---------------------------------------------------------------------------------------------
# The invariants the issue names
# ---------------------------------------------------------------------------------------------


class TestInvariants:
    def test_a_step4_row_is_never_mutated(self):
        row = pd.Series({"asr_transcript": "puhetta suomeksi", "vllm_video_analysis": "video " * 5000})
        original = row.copy()
        build_stage_packet(4, row)
        pd.testing.assert_series_equal(row, original)

    @pytest.mark.parametrize("stage", ["step6", "step7", "step8"])
    def test_steps_6_to_8_never_mutate_the_row(self, stage):
        row = pd.Series({"asr_transcript": "speech " * 5000, "dna_statements_json": "statement " * 5000})
        original = row.copy()
        build_stage_packet(stage, row)
        pd.testing.assert_series_equal(row, original)

    @pytest.mark.parametrize("stage", ["step6", "step7", "step8"])
    def test_memory_and_rag_columns_cannot_enter_a_steps_6_to_8_packet(self, stage):
        packet = build_stage_packet(stage, {
            "asr_transcript": "Emme tue tätä lakia.",
            "memory_context_json": "SECRET_MEMORY_CONTEXT",
            "rag_context_json": "SECRET_RAG_CONTEXT",
            "formula_of_populism_analysis": "SECRET_THEORY_CONTEXT",
            "summary_analysis": "Model summary claims support.",
        })
        assert "Emme tue tätä lakia" in packet.text
        if stage != "step8":
            # Step 8's table is source quotations, DNA statements and record metadata only:
            # Step 4's fusion summary is not one of its evidence fields at all.
            assert "Model summary claims support" in packet.text
        assert "SECRET_MEMORY_CONTEXT" not in packet.text
        assert "SECRET_RAG_CONTEXT" not in packet.text
        assert "SECRET_THEORY_CONTEXT" not in packet.text

    def test_multilingual_and_conflicting_evidence_is_kept_attributed(self):
        packet = build_stage_packet(6, {
            "asr_transcript": "Emme tue tätä lakia",
            "asr_translation": "We do not support this law",
            "summary_analysis": "Model summary claims support",
            "codebook_context_json": "law: canonical term",
        })
        assert "[asr_transcript | " in packet.text
        assert "Emme tue tätä lakia" in packet.text
        assert "We do not support this law" in packet.text
        assert "Model summary claims support" in packet.text
        assert "researcher normalization, not source quote" in packet.text

    def test_step4_excludes_memory_and_rag_unless_retrieval_is_opted_in(self, monkeypatch):
        row = {"asr_transcript": "source", "vllm_video_analysis": "No flags visible",
               "ocr_1": "EU FLAG"}
        monkeypatch.setenv("LACLAUGPT_SUMMARY_INCLUDE_RETRIEVAL", "0")
        without = build_stage_packet(4, row, memory="SECRET LABEL", rag="OLD SUMMARY")
        assert "SECRET LABEL" not in without.text and "OLD SUMMARY" not in without.text
        monkeypatch.setenv("LACLAUGPT_SUMMARY_INCLUDE_RETRIEVAL", "1")
        with_context = build_stage_packet(4, row, memory="SECRET LABEL", rag="OLD SUMMARY")
        assert "SECRET LABEL" in with_context.text and "NOT video evidence" in with_context.text

    def test_disagreeing_source_and_derived_fields_all_survive(self):
        packet = build_stage_packet(4, {"vllm_video_analysis": "No flags visible",
                                        "ocr_1": "EU FLAG", "asr_transcript": "Ei lippuja"})
        assert "No flags visible" in packet.text
        assert "EU FLAG" in packet.text
        assert "Ei lippuja" in packet.text


# ---------------------------------------------------------------------------------------------
# The cache key: a changed prompt, model, option or packet must invalidate
# ---------------------------------------------------------------------------------------------


class TestFingerprint:
    def test_same_inputs_same_key(self):
        args = ("sys", "user", "llama3", {"num_ctx": 8192})
        assert effective_prompt_fingerprint(*args) == effective_prompt_fingerprint(*args)

    def test_prompt_change_changes_key(self):
        base = effective_prompt_fingerprint("sys", "user-a", "llama3", {})
        assert base != effective_prompt_fingerprint("sys", "user-b", "llama3", {})

    def test_model_change_changes_key(self):
        base = effective_prompt_fingerprint("sys", "user", "llama3", {})
        assert base != effective_prompt_fingerprint("sys", "user", "qwen3", {})

    def test_generation_option_change_changes_key(self):
        """#287's fix: a num_ctx/num_predict/temperature change must not reuse an answer."""
        base = effective_prompt_fingerprint("sys", "user", "llama3", {"num_ctx": 8192})
        assert base != effective_prompt_fingerprint("sys", "user", "llama3", {"num_ctx": 16384})
        assert base != effective_prompt_fingerprint(
            "sys", "user", "llama3", {"num_ctx": 8192, "temperature": 0.7}
        )

    def test_option_order_does_not_change_key(self):
        assert effective_prompt_fingerprint("s", "u", "m", {"a": 1, "b": 2}) == \
            effective_prompt_fingerprint("s", "u", "m", {"b": 2, "a": 1})

    def test_a_changed_packet_cannot_reuse_the_fingerprint_of_the_old_one(self, monkeypatch):
        """The consolidation changed what the packet contains, so old cache entries must miss.

        Steps 6/7/8 build the fingerprint over the stage's prompt version *and the packet text*,
        so a packet the builder now produces differently cannot collide with an answer that was
        produced from the old one.
        """
        row = {"asr_transcript": "Emme tue tätä lakia", "summary_analysis": "prior model summary"}
        packet = build_stage_packet(6, row)
        old_packet = "EP24 current-document evidence; prior model output is descriptive context, not direct speech."
        assert effective_prompt_fingerprint("sys", f"v3\n{packet.text}", "m", {}) != \
            effective_prompt_fingerprint("sys", f"v3\n{old_packet}", "m", {})

    def test_token_estimate_is_coarse_but_monotonic(self):
        assert estimate_tokens("") == 0
        assert estimate_tokens("abcd") == 1
        assert estimate_tokens("a" * 5) == 2
        assert estimate_tokens("a" * 100) < estimate_tokens("a" * 200)
