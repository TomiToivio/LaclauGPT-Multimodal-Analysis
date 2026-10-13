"""Regression for orchestrated Step 5 batch records and Step 6 quality gate."""
import unittest

import pandas as pd

from ep24_stage_orchestrator import eligible_query
from roihu_postprocess import select_step5_checkpoint


class Step5RecordPreservationTests(unittest.TestCase):
    def setUp(self):
        self.before = pd.DataFrame({
            "_storage_id": [f"id-{i}" for i in range(25)],
            "summary_analysis": ["source"] * 25,
        })
        self.after = self.before.copy()
        self.after["processing_status"] = ["OK"] * 15 + ["REPROCESS"] * 8 + ["DELETE"] * 2
        self.after["postprocess_entities"] = ["[]"] * 25

    def test_orchestrated_step5_returns_all_25_claimed_records(self):
        original, output = select_step5_checkpoint(self.before, self.after, orchestrated=True)
        self.assertEqual(len(output), 25)
        self.assertEqual(set(output["_storage_id"]), set(original["_storage_id"]))
        self.assertEqual(output["processing_status"].value_counts().to_dict(),
                         {"OK": 15, "REPROCESS": 8, "DELETE": 2})

    def test_standalone_legacy_output_stays_ok_only(self):
        original, output = select_step5_checkpoint(self.before, self.after, orchestrated=False)
        self.assertEqual(len(output), 15)
        self.assertEqual(original["_storage_id"].tolist(), output["_storage_id"].tolist())
        self.assertTrue((output["processing_status"] == "OK").all())

    def test_step6_requires_approved_quality(self):
        query = eligible_query(6, retry_errors=True, force=False)
        self.assertEqual(query["processing_status"], {"$in": ["OK", "ok"]})
        self.assertEqual(query["_pipeline.step_05.status"], "complete")
        self.assertIn("error", query["$or"][1]["_pipeline.step_06.status"]["$in"])

    def test_steps_below_quality_gate_unchanged(self):
        self.assertNotIn("processing_status", eligible_query(5, retry_errors=False, force=False))


if __name__ == "__main__":
    unittest.main()
