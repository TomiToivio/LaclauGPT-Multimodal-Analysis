"""Run-scoped collection prefixes (#369).

The exporter used to hardcode ``laclaugpt_ep2024_reprocess_``. A second reprocessing run
(Monday500) wrote ``laclaugpt_ep24_monday500_20261031_*`` beside it, so a reader that
cannot name a prefix silently exports the previous run's records. These tests pin the
prefix contract: explicit, validated, and never silently defaulted on a typo.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ep24_mongo_export as export  # noqa: E402


def _script(name: str):
    path = ROOT / "scripts" / "ep24" / name
    spec = importlib.util.spec_from_file_location(f"_script_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestPrefixValidation:
    def test_default_prefix_is_the_historical_contract(self):
        assert export.DEFAULT_COLLECTION_PREFIX == "laclaugpt_ep2024_reprocess_"
        assert export.collection_name("finland", "_dataframe") == (
            "laclaugpt_ep2024_reprocess_finland_dataframe"
        )

    def test_run_scoped_prefix_names_the_run(self):
        prefix = "laclaugpt_ep24_monday500_20261031_"
        assert export.collection_name("poland", "_dataframe", prefix=prefix) == (
            "laclaugpt_ep24_monday500_20261031_poland_dataframe"
        )
        assert export.child_collection("portugal", "entities", prefix=prefix) == (
            "laclaugpt_ep24_monday500_20261031_portugal_entities"
        )

    def test_a_prefix_must_end_with_an_underscore(self):
        with pytest.raises(export.ExportError, match="unsafe collection prefix"):
            export.collection_name("finland", "_dataframe", prefix="laclaugpt_ep24_monday500")

    def test_a_missing_prefix_fails_closed(self):
        with pytest.raises(export.ExportError):
            export.collection_name("finland", "_dataframe", prefix="")

    @pytest.mark.parametrize("prefix", [
        "laclaugpt ep24_",
        "laclaugpt-ep24_",
        "laclaugpt.ep24_",
        "$where_",
        "../laclaugpt_",
    ])
    def test_prefixes_that_cannot_be_collection_names_are_rejected(self, prefix):
        with pytest.raises(export.ExportError):
            export.validate_collection_prefix(prefix)


class TestCliWiring:
    def test_export_cli_accepts_a_prefix(self):
        module = _script("export_mongo_results.py")
        args = module.parse_args(["--output", "/tmp/out", "--collection-prefix",
                                  "laclaugpt_ep24_monday500_20261031_"])
        assert args.collection_prefix == "laclaugpt_ep24_monday500_20261031_"

    def test_export_cli_default_is_the_historical_prefix(self):
        module = _script("export_mongo_results.py")
        args = module.parse_args(["--output", "/tmp/out"])
        assert args.collection_prefix == export.DEFAULT_COLLECTION_PREFIX

    def test_audit_cli_accepts_a_prefix(self):
        module = _script("audit_stage_coverage.py")
        args = module.parse_args(["--output", "/tmp/coverage.json", "--collection-prefix",
                                  "laclaugpt_ep24_monday500_20261031_"])
        assert args.collection_prefix == "laclaugpt_ep24_monday500_20261031_"
