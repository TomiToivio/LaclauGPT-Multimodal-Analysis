"""Synthetic regressions for exported coverage diagnostics."""
import csv
import tempfile
import unittest
from pathlib import Path

from scripts.ep24.report_stage_coverage import report_country, render_markdown


class ExportCoverageTests(unittest.TestCase):
    def report(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'synthetic.csv'
            with path.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    'record_id', 'laclau_status', 'laclau_structured_json'])
                writer.writeheader()
                writer.writerows(rows)
            return report_country(path)

    def test_empty_country_renders_unknown_share(self):
        report = self.report([])
        self.assertIsNone(report['steps']['6']['share'])
        self.assertIn('—', render_markdown({'synthetic': report}))

    def test_serialized_empty_results_are_not_output(self):
        values = ['[]', '{}', 'null', '""', '  []  ', '{"claim":"synthetic"}']
        report = self.report([
            {'record_id': str(i), 'laclau_status': 'complete',
             'laclau_structured_json': value}
            for i, value in enumerate(values)])
        self.assertEqual(report['steps']['6']['text_present'], 1)
        self.assertEqual(report['steps']['6']['status_present'], 6)

    def test_status_is_not_labelled_as_execution(self):
        report = self.report([{'record_id': 'synthetic', 'laclau_status': 'skipped'}])
        markdown = render_markdown({'synthetic': report})
        self.assertIn('status present', markdown)
        self.assertNotIn('ran (status)', markdown)
        self.assertIn('does not establish', markdown)


if __name__ == '__main__':
    unittest.main()
