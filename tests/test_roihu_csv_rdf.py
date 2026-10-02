"""Synthetic compatibility, provenance and restart tests; no private EP24 input."""

import argparse
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roihu_csv_rdf import NS, export, project_row  # noqa: E402


class RDFExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.input = self.root / "ep24_fi.csv"
        self.rows = [{"videoId": "00012345678901234567890", "authorUniqueId": "synthetic",
                      "summary_analysis": 'Line one\nŁódź "quoted" \\ Ω\t\x01',
                      "entities": "Example, Party, ambiguous name",
                      "formula_of_populism_us": "justice^hope\n",
                      "formula_of_populism_frontier": "elite^anger\nmalformed\n",
                      "human_locked": "true", "researcher_note": "retain exactly", "empty": ""}]
        self.write(self.rows)

    def write(self, rows):
        with self.input.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def args(self, name="run", **changes):
        values = dict(input=self.input, output_dir=self.root / name,
                      checkpoint=self.root / "state.sqlite", limit=None, dry_run=False,
                      base_uri="https://data.example/laclaugpt", project="ep24", dataset_id="fi")
        values.update(changes)
        return argparse.Namespace(**values)

    def test_original_values_and_source_bytes_retained(self):
        before = self.input.read_bytes()
        result = export(self.args())
        with (self.root / "run/final.csv").open(newline="", encoding="utf-8") as stream:
            actual = next(csv.DictReader(stream))
        self.assertEqual(self.rows[0], {key: actual[key] for key in self.rows[0]})
        self.assertEqual(self.input.read_bytes(), before)
        self.assertEqual((self.root / "run/source.csv").read_bytes(), before)
        self.assertEqual(result["warnings"], 1)
        self.assertEqual(result["rows"], 1)

    def test_cache_hit_then_human_edit_invalidates(self):
        export(self.args("a"))
        self.assertEqual(export(self.args("b"))["cache_hits"], 1)
        self.rows[0]["researcher_note"] = "manually revised"
        self.write(self.rows)
        self.assertEqual(export(self.args("c"))["cache_hits"], 0)

    def test_identity_and_config_scoping(self):
        row = self.rows[0]
        a = project_row(row, base="https://example.org", project="ep24", dataset="fi", row_number=1)
        b = project_row(row, base="https://example.org", project="ep24", dataset="fi", row_number=2)
        c = project_row(row, base="https://example.org", project="ep24", dataset="pl", row_number=1)
        self.assertEqual(a[0], b[0])
        self.assertNotEqual(a[1], b[1])
        self.assertNotEqual(a[0], c[0])
        export(self.args("a"))
        self.assertEqual(export(self.args("b", project="other"))["cache_hits"], 0)

    def test_refuses_to_overwrite_outputs(self):
        export(self.args())
        path = self.root / "run/final.csv"
        path.write_text("researcher edited this", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "already exists"):
            export(self.args())
        self.assertEqual(path.read_text(), "researcher edited this")

    def test_dry_run_does_not_create_files(self):
        self.assertEqual(export(self.args(dry_run=True))["status"], "dry-run")
        self.assertFalse((self.root / "run").exists())
        self.assertFalse((self.root / "state.sqlite").exists())

    def test_bad_headers_and_reserved_names_are_rejected(self):
        for data in ("a,a\n1,2\n", "a,\n1,2\n", "rdf_status\nok\n", ""):
            self.input.write_text(data)
            with self.assertRaises(ValueError):
                export(self.args(dry_run=True))

    def test_structural_failure_retains_cache_without_success_manifest(self):
        with self.input.open("a") as stream:
            stream.write("too,few,fields\n")
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            export(self.args())
        self.assertFalse((self.root / "run/manifest.json").exists())
        self.assertFalse((self.root / "run/final.csv").exists())
        self.write(self.rows)
        self.assertEqual(export(self.args("retry"))["cache_hits"], 1)

    def test_missing_id_is_visible_and_not_conflated(self):
        a = project_row({"text": "one"}, base="https://example.org", project="p", dataset="d", row_number=1)
        b = project_row({"text": "one"}, base="https://example.org", project="p", dataset="d", row_number=2)
        self.assertNotEqual(a[0], b[0])
        self.assertTrue(a[3])

    def test_no_invented_actor_or_social_edges(self):
        _, _, graph, warnings = project_row(self.rows[0], base="https://example.org",
                                            project="p", dataset="d", row_number=1)
        self.assertIn("coded-origin-unspecified", graph)
        self.assertNotIn("supports", graph)
        self.assertNotIn(NS + "Actor", graph)
        self.assertEqual(len(warnings), 1)

    def test_bare_populism_elements_preserve_coding_without_fabricated_affect(self):
        row = {
            "videoId": "bare-1",
            "formula_of_populism_us": "the people\n",
            "formula_of_populism_frontier": "elites\n",
        }
        _, _, graph, warnings = project_row(
            row, base="https://example.org", project="p", dataset="d", row_number=1
        )
        self.assertEqual(warnings, [])
        self.assertIn(' <' + NS + 'element> "the people" .\n', graph)
        self.assertIn(' <' + NS + 'element> "elites" .\n', graph)
        self.assertNotIn(' <' + NS + 'affect> ', graph)


    def test_limit_and_empty_input(self):
        self.write(self.rows * 3)
        self.assertEqual(export(self.args("sample", limit=1))["rows"], 1)
        self.input.write_text("videoId,text\n")
        self.assertEqual(export(self.args("empty"))["rows"], 0)

    def test_malformed_base_uri_rejected(self):
        for base in ("relative", "https://example.org/a>b", "https://example.org/#fragment"):
            with self.assertRaises(ValueError):
                export(self.args(base_uri=base))

    def test_rdf_parser_roundtrip(self):
        try:
            from rdflib import Graph, Namespace
        except ImportError:
            self.skipTest("Install rdflib for independent N-Triples syntax validation")
        export(self.args())
        graph = Graph().parse(self.root / "run/graph.nt", format="nt")
        ns = Namespace(NS)
        values = {str(value) for _, _, value in graph.triples((None, ns.value, None))}
        self.assertIn(self.rows[0]["summary_analysis"], values)
        self.assertIn("", values)
        self.assertEqual(len(list(graph.triples((None, ns.column, None)))), len(self.rows[0]))
        manifest = json.loads((self.root / "run/manifest.json").read_text())
        self.assertEqual(manifest["status"], "complete-with-warnings")


if __name__ == "__main__":
    unittest.main()
