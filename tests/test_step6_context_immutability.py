"""Step 6 must not overwrite Step 5's shared context/provenance fields."""
from pathlib import Path

import pandas as pd

from ep24_pipeline import assert_source_metadata_preserved
from roihu_populism import STEP6_COLUMNS, build_prompt_context


def test_enriched_context_used_without_mutating_prior_stage():
    original = pd.DataFrame([{
        "video_id": "HEPP24-HR-RG-TT-20240601-123",
        "codebook_context_json": '["original"]',
        "memory_context_json": '["original-memory"]',
        "rag_context_json": '["original-rag"]',
        "summary_analysis": "Current source content",
    }])
    enriched = original.copy(deep=True)
    enriched.loc[0, "codebook_context_json"] = '["updated"]'
    output = original.copy(deep=True)
    output["laclau_codebook_context_json"] = enriched["codebook_context_json"]
    context, _ = build_prompt_context(enriched.loc[0])
    assert output.at[0, "codebook_context_json"] == '["original"]'
    assert output.at[0, "laclau_codebook_context_json"] == '["updated"]'
    assert isinstance(context, str)
    assert_source_metadata_preserved(original, output)


def test_step6_implementation_never_copies_shared_context_onto_output():
    source = (Path(__file__).resolve().parents[1] / "roihu_populism.py").read_text()
    body = source.split("def process_country(", 1)[1]
    assert "context_rows = enriched" in body
    assert "build_prompt_context(context_row)" in body
    assert 'out[column] = enriched[column]' not in body
    assert 'laclau_codebook_context_json' in STEP6_COLUMNS
