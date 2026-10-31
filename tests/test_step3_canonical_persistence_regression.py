"""Regression tests for Step 3 canonical identity and orchestrated persistence."""
import ast
from pathlib import Path
import pandas as pd
from experiments.vllm_video_test import write_local_checkpoint, OUTPUT_COLUMNS, persist_mongo_patch

SOURCE = Path("experiments/vllm_video_test.py").read_text()


def test_orchestrator_is_the_only_mongo_writer():
    tree = ast.parse(SOURCE)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    assert 'LACLAUGPT_ORCHESTRATED_BATCH' in ast.get_source_segment(SOURCE, main)
    assert '"orchestrator_pending"' in ast.get_source_segment(SOURCE, main)


def test_direct_mongo_patch_never_upserts_or_recomputes_id():
    fn = next(n for n in ast.parse(SOURCE).body if isinstance(n, ast.FunctionDef) and n.name == "persist_mongo_patch")
    source = ast.get_source_segment(SOURCE, fn)
    assert 'upsert=False' in source
    assert 'row.get("_storage_id"' in source
    assert 'stable_source_id(row)' not in source
    assert 'matched_count' in source


def test_retry_replaces_canonical_video_result(tmp_path):
    before = pd.DataFrame([{
        "_storage_id": "canonical-001", "video_id": "v1",
        "frame_analysis_1": "original frame",
        "vllm_video_analysis": "stale analysis",
        "vllm_video_status": "error",
    }])
    record = {field: "" for field in OUTPUT_COLUMNS}
    record["vllm_video_analysis"] = "fresh analysis"
    record["vllm_video_status"] = "ok"
    result = write_local_checkpoint(
        before, [0], [record], tmp_path / "checkpoint.csv",
        tmp_path / "checkpoint.sqlite3",
        __import__("logging").getLogger(__name__),
    )
    assert result.loc[0, "vllm_video_analysis"] == "fresh analysis"
    assert result.loc[0, "frame_analysis_1"] == "original frame"
    assert "step3_vllm_video_analysis" not in result.columns
    assert result.loc[0, "_storage_id"] == "canonical-001"
