"""Safety checks for the archive-first EP24 reset command."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "reset_ep24", ROOT / "scripts/roihu/reset_ep24_and_restart.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_targets_only_dataset_country_and_excludes_curated_and_prior_archives():
    cfg = SimpleNamespace(prefix="laclaugpt_ep2024_reprocess_finland")
    class DB:
        def list_collection_names(self):
            return [
                "laclaugpt_ep2024_reprocess_finland_dataframe",
                "laclaugpt_ep2024_reprocess_finland_rag",
                "laclaugpt_ep2024_reprocess_finland_memory",
                "laclaugpt_ep2024_reprocess_finland_entities",
                "laclaugpt_ep2024_reprocess_finland_dataframe_reset_archive_20261010",
                "laclaugpt_ep2024_reprocess_poland_dataframe",
                "other_collection",
            ]
    assert module.targets(DB(), cfg) == [
        "laclaugpt_ep2024_reprocess_finland_dataframe",
        "laclaugpt_ep2024_reprocess_finland_rag",
    ]


def test_archive_name_is_reversible():
    assert module.archive_name("prefix_dataframe", "20261011T120000Z") == (
        "prefix_dataframe_reset_archive_20261011T120000Z")


def test_importer_and_sbatch_resolve_from_repository_root():
    assert module.REPO_ROOT == ROOT
    assert module.IMPORTER == ROOT / "step_0_roihu_import.py"
    assert module.STAGE1 == ROOT / "scripts/roihu/step_1_roihu_preprocess.sbatch"
    assert module.IMPORTER.is_file()
    assert module.STAGE1.is_file()
