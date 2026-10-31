"""Regression guards for the preview-only selective retry entrypoint (#342)."""
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "roihu"))
import reprocess_selected  # noqa: E402


class PreviewOnlyTest(unittest.TestCase):
    def test_execute_is_rejected_before_touching_missing_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            export = Path(tmp) / "does-not-exist"
            output = Path(tmp) / "manifest.json"
            with contextlib.redirect_stderr(io.StringIO()) as errors:
                rc = reprocess_selected.main(
                    ["--export", str(export), "--steps", "6", "--execute",
                     "--apply", "--manifest", str(output)]
                )
            self.assertEqual(rc, 2)
            self.assertIn("not implemented", errors.getvalue())
            self.assertFalse(output.exists())

    def test_placeholder_commands_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stderr(io.StringIO()) as errors:
                rc = reprocess_selected.main(
                    ["--export", tmp, "--steps", "6", "--commands"]
                )
            self.assertEqual(rc, 2)
            self.assertIn("not implemented", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
