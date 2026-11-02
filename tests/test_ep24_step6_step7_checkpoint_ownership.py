"""Regression: EP24 stages 6/7 may update their own outputs, not source context."""
from pathlib import Path

import pandas as pd
import pytest

from ep24_pipeline import write_cumulative_csv
from roihu_populism import STEP6_COLUMNS
from step_7_roihu_discourse_network_analysis import DNA_COLUMNS


@pytest.mark.parametrize(
    "owned_column,owned_fields",
    [
        ("formula_of_populism_analysis", STEP6_COLUMNS),
        ("dna_actor_unresolved_count", DNA_COLUMNS),
        ("dna_persistence_status", DNA_COLUMNS),
    ],
)
def test_resumed_stage_can_update_own_existing_field(tmp_path, owned_column, owned_fields):
    before = pd.DataFrame([{
        "_storage_id": "stable-id",
        "codebook_context_json": '{"source": true}',
        owned_column: "old",
    }])
    after = before.copy(deep=True)
    after.loc[0, owned_column] = "new"
    target = tmp_path / "checkpoint.csv"

    write_cumulative_csv(before, after, target, mutable_columns=owned_fields)

    reread = pd.read_csv(target, dtype=str, keep_default_na=False)
    assert reread.loc[0, owned_column] == "new"
    assert reread.loc[0, "codebook_context_json"] == '{"source": true}'


@pytest.mark.parametrize("owned_fields", [STEP6_COLUMNS, DNA_COLUMNS])
def test_stage_cannot_mutate_existing_shared_context(tmp_path, owned_fields):
    before = pd.DataFrame([{
        "_storage_id": "stable-id",
        "codebook_context_json": '{"source": true}',
    }])
    after = before.copy(deep=True)
    after.loc[0, "codebook_context_json"] = '{"source": false}'
    with pytest.raises(AssertionError, match="codebook_context_json"):
        write_cumulative_csv(
            before, after, tmp_path / "bad.csv", mutable_columns=owned_fields
        )


def test_stage_call_sites_use_owned_fields_and_step7_isolates_enrichment():
    root = Path(__file__).resolve().parents[1]
    step6 = (root / "roihu_populism.py").read_text()
    step7 = (root / "step_7_roihu_discourse_network_analysis.py").read_text()
    assert step6.count("mutable_columns=STEP6_COLUMNS") == 2
    assert step7.count("mutable_columns=DNA_COLUMNS") == 2
    assert "_prepare_context(out.copy(deep=True), normalized_country)" in step7
