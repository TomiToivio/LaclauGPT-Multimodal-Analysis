"""Regression coverage for bounded Step 4 evidence packets."""
import os
import unittest
from unittest.mock import patch
from ep24_summary_evidence import build_packet


class SummaryEvidenceTests(unittest.TestCase):
    def test_empty_source(self):
        self.assertIn("No usable source evidence", build_packet({}))

    def test_priority_and_provenance(self):
        row = {"vllm_video_analysis": "video sequence",
               "asr_transcript": "puhetta suomeksi",
               "asr_translated": "speech in English",
               "frame_analysis_1": "red banner",
               "ocr_1": "VOTE",
               "author_username": "author",
               "summary_analysis": "DO NOT INCLUDE",
               "raw_prompt": "DO NOT INCLUDE"}
        packet = build_packet(row)
        for value in ("video sequence", "puhetta suomeksi", "speech in English",
                      "red banner", "VOTE", "author"):
            self.assertIn(value, packet)
        self.assertNotIn("DO NOT INCLUDE", packet)
        self.assertLess(packet.index("video sequence"), packet.index("puhetta suomeksi"))
        self.assertIn("not proof", packet)
        self.assertEqual(row["summary_analysis"], "DO NOT INCLUDE")

    def test_oversized_stays_bounded_and_reports_truncation(self):
        row = {"vllm_video_analysis": "x" * 100000, "asr_transcript": "y" * 100000}
        packet = build_packet(row, total=12000)
        self.assertLessEqual(len(packet), 12000)
        self.assertIn("TRUNCATED", packet)

    def test_disagreement_kept(self):
        packet = build_packet({"vllm_video_analysis": "No flags visible",
                               "ocr_1": "EU FLAG", "asr_transcript": "Ei lippuja"})
        self.assertIn("No flags visible", packet)
        self.assertIn("EU FLAG", packet)
        self.assertIn("Ei lippuja", packet)

    def test_default_excludes_external_retrieval(self):
        with patch.dict(os.environ, {"LACLAUGPT_SUMMARY_INCLUDE_RETRIEVAL": "0"}):
            packet = build_packet({"asr_transcript": "source"}, memory="SECRET LABEL", rag="OLD SUMMARY")
        self.assertNotIn("SECRET LABEL", packet)
        self.assertNotIn("OLD SUMMARY", packet)

    def test_optional_retrieval_is_labeled(self):
        with patch.dict(os.environ, {"LACLAUGPT_SUMMARY_INCLUDE_RETRIEVAL": "1"}):
            packet = build_packet({"asr_transcript": "source"}, memory="label")
        self.assertIn("NOT video evidence", packet)

    def test_invalid_budget(self):
        with self.assertRaises(ValueError):
            build_packet({}, total=1999)

    def test_missing_and_nan(self):
        packet = build_packet({"asr_transcript": float("nan"), "ocr_1": "valid"})
        self.assertNotIn("nan", packet.lower())
        self.assertIn("valid", packet)


if __name__ == "__main__":
    unittest.main()
