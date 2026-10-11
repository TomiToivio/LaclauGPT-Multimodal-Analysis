"""Contract: reprocessing may update Step 1 results, never source metadata."""
import pandas as pd
import pytest
from ep24_pipeline import assert_source_metadata_preserved
from roihu_preprocess import PREPROCESS_COLUMNS


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
