"""Contract coverage for the shared EP24 prompt/context packet (issue #241).

Normal, empty, oversized, multilingual and failure conditions, plus the two invariants the stage
issues depend on: continuity context can never displace source evidence, and the cache key covers
prompt, model and generation options.
"""
import unittest
from unittest.mock import patch

from ep24_prompt_contract import (
    CONTEXT,
    EVIDENCE,
    NO_EVIDENCE,
    STAGE_BUDGETS,
    Packet,
    StageBudget,
    Stream,
    build_packet,
    effective_prompt_fingerprint,
    estimate_tokens,
    stage_budget,
)


class BudgetTests(unittest.TestCase):
    def test_every_stage_in_the_programme_has_a_budget(self):
        self.assertEqual({"step4", "step6", "step7", "step8"}, set(STAGE_BUDGETS))

    def test_reserve_is_held_back_from_the_model_visible_budget(self):
        budget = stage_budget("step6")
        self.assertEqual(budget.total_chars, 32000)
        self.assertEqual(budget.output_reserve_chars, 4000)
        self.assertEqual(budget.evidence_chars, 28000)

    def test_unknown_stage_fails_loudly(self):
        with self.assertRaises(KeyError):
            stage_budget("step99")

    def test_env_override_is_honoured(self):
        with patch.dict("os.environ", {"LACLAUGPT_STEP7_EVIDENCE_CHARS": "9000"}):
            self.assertEqual(stage_budget("step7").total_chars, 9000)


class PacketTests(unittest.TestCase):
    def test_empty_evidence_says_so_instead_of_inventing(self):
        packet = build_packet("step4", [])
        self.assertEqual(packet.text, NO_EVIDENCE)
        self.assertEqual(packet.used_chars, len(NO_EVIDENCE))

    def test_blank_and_nan_values_are_not_evidence(self):
        packet = build_packet("step4", [
            Stream("speech", "nan", EVIDENCE),
            Stream("visuals", "  ", EVIDENCE),
        ])
        self.assertEqual(packet.text, NO_EVIDENCE)
        self.assertTrue(all(r.dropped for r in packet.streams))

    def test_normal_evidence_is_labelled_and_ordered_by_priority(self):
        packet = build_packet("step8", [
            Stream("quotes", "explicit quote", EVIDENCE, priority=2),
            Stream("source_context", "context first", EVIDENCE, priority=1),
        ])
        self.assertLess(packet.text.index("source_context"), packet.text.index("quotes"))
        self.assertIn("[evidence]", packet.text)

    def test_oversized_stream_is_truncated_and_reported(self):
        packet = build_packet("step8", [Stream("source_context", "x" * 40000, EVIDENCE)])
        report = packet.streams[0]
        self.assertTrue(report.truncated)
        self.assertLess(report.kept_chars, report.original_chars)
        self.assertIn("TRUNCATED", packet.text)
        self.assertLessEqual(packet.used_chars, packet.budget_chars)

    def test_budget_is_never_exceeded_across_many_streams(self):
        packet = build_packet("step7", [
            Stream(f"s{i}", "y" * 20000, EVIDENCE, priority=i) for i in range(9)
        ])
        self.assertLessEqual(packet.used_chars, packet.budget_chars)
        # Each stream was squeezed by its own cap; nothing had to be dropped.
        self.assertTrue(packet.truncated_streams)

    def test_packet_stays_within_budget_when_streams_outnumber_the_room(self):
        packet = build_packet("step7", [
            Stream(f"s{i}", "y" * 5000, EVIDENCE, priority=i) for i in range(30)
        ])
        self.assertLessEqual(packet.used_chars, packet.budget_chars)
        self.assertTrue(packet.dropped_streams)

    def test_context_never_displaces_evidence(self):
        """#239: no primary quotation may be displaced by retrieval context.

        The budget is tightened so the evidence stream can fill it exactly; the context stream
        then has nowhere to go and must be dropped rather than the quotation being trimmed.
        """
        quote = "the primary political quote"
        tight = StageBudget("step7", 18000, 6000, {"utterances": 12000})
        packet = build_packet("step7", [
            Stream("retrieval", "r" * 30000, CONTEXT, priority=0),
            Stream("utterances", quote + "z" * 17000, EVIDENCE, priority=99),
        ], budget=tight)
        self.assertIn(quote, packet.text)
        self.assertEqual(packet.context_chars, 0)
        self.assertIn("retrieval", packet.dropped_streams)
        self.assertLessEqual(packet.used_chars, packet.budget_chars)

    def test_context_is_labelled_as_such_and_can_be_excluded(self):
        packet = build_packet("step4", [
            Stream("speech", "puhetta suomeksi", EVIDENCE),
            Stream("memory", "aiempi analyysi", CONTEXT),
        ])
        self.assertIn("[context]", packet.text)
        without = build_packet("step4", [
            Stream("speech", "puhetta suomeksi", EVIDENCE),
            Stream("memory", "aiempi analyysi", CONTEXT),
        ], include_context=False)
        self.assertNotIn("aiempi analyysi", without.text)

    def test_multilingual_evidence_survives_intact(self):
        text = "Tekoäly ja populismi. Sztuczna inteligencja. Inteligência artificial."
        packet = build_packet("step6", [Stream("speech", text, EVIDENCE)])
        self.assertIn(text, packet.text)

    def test_log_line_reports_truncation_by_field(self):
        packet = build_packet("step4", [
            Stream("video", "v" * 30000, EVIDENCE),
            Stream("ocr", "VOTE", EVIDENCE),
        ])
        line = packet.log_line()
        self.assertIn("stage=step4", line)
        self.assertIn("truncated=['video']", line)
        self.assertIn("tokens~", line)

    def test_inputs_are_never_mutated(self):
        row = {"speech": "puhetta"}
        stream = Stream("speech", row["speech"], EVIDENCE)
        build_packet("step4", [stream])
        self.assertEqual(row, {"speech": "puhetta"})
        self.assertEqual(stream.text, "puhetta")

    def test_token_estimate_is_coarse_but_monotonic(self):
        self.assertEqual(estimate_tokens(""), 0)
        self.assertLess(estimate_tokens("a" * 100), estimate_tokens("a" * 200))

    def test_packet_reports_evidence_and_context_separately(self):
        packet = build_packet("step4", [
            Stream("speech", "evidence text", EVIDENCE),
            Stream("rag", "context text", CONTEXT),
        ])
        self.assertGreater(packet.evidence_chars, 0)
        self.assertGreater(packet.context_chars, 0)
        self.assertIsInstance(packet, Packet)


class FingerprintTests(unittest.TestCase):
    def test_same_inputs_same_key(self):
        args = ("sys", "user", "llama3", {"num_ctx": 8192})
        self.assertEqual(effective_prompt_fingerprint(*args), effective_prompt_fingerprint(*args))

    def test_prompt_change_changes_key(self):
        base = effective_prompt_fingerprint("sys", "user-a", "llama3", {})
        self.assertNotEqual(base, effective_prompt_fingerprint("sys", "user-b", "llama3", {}))

    def test_model_change_changes_key(self):
        base = effective_prompt_fingerprint("sys", "user", "llama3", {})
        self.assertNotEqual(base, effective_prompt_fingerprint("sys", "user", "qwen3", {}))

    def test_option_change_changes_key(self):
        """#244's fix, generalised: a num_ctx/num_predict change must invalidate the cache."""
        base = effective_prompt_fingerprint("sys", "user", "llama3", {"num_ctx": 8192})
        self.assertNotEqual(
            base, effective_prompt_fingerprint("sys", "user", "llama3", {"num_ctx": 16384})
        )

    def test_option_order_does_not_change_key(self):
        self.assertEqual(
            effective_prompt_fingerprint("s", "u", "m", {"a": 1, "b": 2}),
            effective_prompt_fingerprint("s", "u", "m", {"b": 2, "a": 1}),
        )


if __name__ == "__main__":
    unittest.main()
