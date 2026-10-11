"""Coverage for the Step 4 rerun-gate worklist (#236).

The manual scoring in the runbook needs the runtime, but the selection and the coverage arithmetic
do not -- and they are what a reviewer has to trust before spending model time. Offline, read-only.
"""
import csv
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "ep24"))

from step4_evidence_coverage import (  # noqa: E402
    country_code,
    main,
    report_record,
    sample_fingerprint,
    select_sample,
)

Huge = "narrative " * 3000


def row(source_id: str, country: str, **extra) -> dict:
    base = {
        "video_id": source_id,
        "country": country,
        "vllm_video_analysis": Huge,
        "asr_transcript": "puhetta suomeksi",
        "ocr_1": "VOTE",
        "author_username": "author",
    }
    base.update(extra)
    return base


class SelectionTests(unittest.TestCase):
    def test_selection_is_deterministic_and_capped_per_country(self):
        rows = [row(f"{c}{i}", c) for c in ("finland", "poland", "portugal") for i in range(15)]
        first = select_sample(rows, per_country=10)
        second = select_sample(list(reversed(rows)), per_country=10)
        self.assertEqual([r["video_id"] for r in first], [r["video_id"] for r in second])
        self.assertEqual(len(first), 30)

    def test_country_field_accepts_names_and_codes(self):
        rows = [row("a", "Finland"), row("b", "PL"), row("c", "portugal")]
        codes = sorted(country_code(r) for r in select_sample(rows, per_country=5))
        self.assertEqual(codes, ["FI", "PL", "PT"])

    def test_small_sample_returns_what_exists(self):
        self.assertEqual(len(select_sample([row("a", "finland")], per_country=10)), 1)

    def test_fingerprint_is_stable_and_sample_dependent(self):
        rows = [row("a", "finland"), row("b", "poland")]
        reports = [report_record(r) for r in rows]
        self.assertEqual(sample_fingerprint(reports), sample_fingerprint(list(reports)))
        self.assertNotEqual(sample_fingerprint(reports), sample_fingerprint(reports[:1]))


class CoverageTests(unittest.TestCase):
    def test_oversized_source_evidence_is_reported_as_low_coverage(self):
        report = report_record(row("v1", "finland"))
        self.assertLess(report.coverage, 1.0)
        self.assertGreater(report.packet_chars, 0)
        self.assertEqual(report.packet_tokens, report.packet_chars // 4 + (report.packet_chars % 4 > 0))

    def test_small_record_is_fully_covered(self):
        report = report_record({
            "video_id": "v2", "country": "poland",
            "asr_transcript": "krótka wypowiedź polityczna", "ocr_1": "VOTE",
        })
        self.assertEqual(report.omitted_fields, ())
        self.assertEqual(report.coverage, 1.0)

    def test_machinery_fields_are_not_scored_as_omissions(self):
        report = report_record(row("v3", "portugal", raw_prompt="x" * 500,
                                   summary_analysis="y" * 500, _storage_id="z" * 500))
        self.assertNotIn("raw_prompt", report.omitted_fields)
        self.assertNotIn("summary_analysis", report.omitted_fields)
        self.assertNotIn("_storage_id", report.omitted_fields)

    def test_a_short_field_is_not_counted_as_evidence(self):
        report = report_record({"video_id": "v4", "country": "finland", "ocr_1": "AB"})
        self.assertEqual(report.row_chars, 0)


class CliTests(unittest.TestCase):
    def test_writes_a_worklist_and_optional_json(self):
        rows = [row(f"f{i}", "finland") for i in range(3)] + [row("p1", "poland")]
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "in.csv"
            with open(csv_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            out = io.StringIO()
            with redirect_stdout(out):
                code = main([str(csv_path), "--per-country", "2", "--json", str(Path(tmp) / "out.json")])
            self.assertEqual(code, 0)
            self.assertIn("per country:", out.getvalue())
            self.assertIn("sample fingerprint:", out.getvalue())
            payload = json.loads((Path(tmp) / "out.json").read_text(encoding="utf-8"))
            self.assertEqual(len(payload), 3)
            self.assertEqual(sorted(r["country"] for r in payload), ["FI", "FI", "PL"])
            self.assertIn("coverage", payload[0])

    def test_empty_input_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "empty.csv"
            csv_path.write_text("video_id,country\n", encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main([str(csv_path)]), 2)


if __name__ == "__main__":
    unittest.main()
