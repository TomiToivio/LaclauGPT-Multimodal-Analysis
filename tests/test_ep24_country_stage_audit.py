"""Synthetic regression tests for the read-only country status auditor."""
import unittest
from scripts.ep24.audit_country_stages import dataframe_collections, tally


class CountryAuditTests(unittest.TestCase):
    def test_excludes_archived_collections(self):
        self.assertEqual(dataframe_collections([
            "laclauGPT_ep2024_reprocess_finland_dataframe",
            "laclauGPT_ep2024_reprocess_finland_dataframe_reset_20261011",
            "laclaugpt_ep2024_reprocess_poland_analysis",
        ]), {"finland": "laclauGPT_ep2024_reprocess_finland_dataframe"})

    def test_missing_not_confused_with_complete(self):
        rows = [
            {"video_id": "video-1", "_pipeline": {
                "step_01": {"status": "complete"},
                "step_02": {"status": "error"}}},
            {"video_id": "", "_pipeline": {
                "step_01": {"status": "complete"}}},
        ]
        output = tally(rows)
        self.assertEqual(output["records_read"], 2)
        self.assertEqual(output["identity_missing"], 1)
        self.assertEqual(output["stage_counts"][1]["complete"], 2)
        self.assertEqual(output["stage_counts"][2]["error"], 1)
        self.assertEqual(output["stage_counts"][2]["missing"], 1)
        self.assertEqual(output["all_steps_status_complete_with_valid_identity"], 0)
        self.assertFalse(output["output_payloads_verified"])

    def test_upstream_gaps_and_source_media_identity(self):
        result = tally([
            {"_storage_id": "storage-only", "_pipeline": {
                "step_01": {"status": "error"},
                "step_02": {"status": "complete", "run_id": "run-2"}}},
            {"video_id": "real-media", "_pipeline": {
                "step_01": {"status": "complete"},
                "step_02": {"status": "pending"}}},
        ])
        self.assertEqual(result["source_media_identity_missing"], 1)
        self.assertEqual(result["identity_valid"], 2)
        self.assertEqual(result["stage_counts"][2]["complete_with_upstream_gap"], 1)
        self.assertEqual(result["stage_counts"][2]["eligible_by_previous_stage"], 1)
        self.assertIn("run-2", result["run_ids"])

    def test_complete_chain_is_status_only(self):
        output = tally([{"_storage_id": "synthetic",
                         "_pipeline": {f"step_{n:02d}": {"status": "complete"}
                                       for n in range(1, 10)}}])
        self.assertEqual(output["all_steps_status_complete_with_valid_identity"], 1)
        self.assertFalse(output["run_coherence_verified"])


if __name__ == "__main__":
    unittest.main()
