"""Contract: reprocessing may update Step 1 results, never source metadata."""
import pandas as pd
import pytest
from ep24_pipeline import assert_source_metadata_preserved
import ast
from pathlib import Path


def _step1_columns():
    # Parse a constant without importing GPU/OpenCV/NeMo production dependencies.
    path = Path(__file__).resolve().parents[1] / 'roihu_preprocess.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == 'PREPROCESS_COLUMNS'
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError('PREPROCESS_COLUMNS not found')


PREPROCESS_COLUMNS = _step1_columns()


def test_existing_step1_results_can_change_without_source_mutation():
    before = pd.DataFrame([{
        "video_id": "source-id", "entities": "researcher-coded",
        "themes": "researcher-theme", "asr_backend": "",
        "asr_transcript": "old", "preprocess_status": "error",
    }])
    after = before.copy()
    after["asr_backend"] = "canary"
    after["asr_transcript"] = "new transcription"
    after["preprocess_status"] = "ok"
    assert_source_metadata_preserved(before, after, mutable_columns=PREPROCESS_COLUMNS)


def test_source_and_researcher_annotations_remain_protected():
    before = pd.DataFrame([{
        "video_id": "source-id", "entities": "researcher-coded",
        "themes": "researcher-theme", "asr_backend": "",
    }])
    after = before.copy()
    after["asr_backend"] = "canary"
    after["entities"] = "incorrect overwrite"
    with pytest.raises(AssertionError, match="stage mutated incoming column: entities"):
        assert_source_metadata_preserved(before, after, mutable_columns=PREPROCESS_COLUMNS)
