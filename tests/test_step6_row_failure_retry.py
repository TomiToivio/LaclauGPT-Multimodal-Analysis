"""Step 6's Laclau row result must remain retryable, not false-complete."""
from unittest.mock import MagicMock

import pandas as pd
import pytest

from ep24_stage_orchestrator import persist_batch


@pytest.mark.parametrize("status,expected", [
    ("ok", "complete"),
    ("error", "error"),
    ("", "error"),
])
def test_step6_row_status_in_mongo(status, expected):
    storage = MagicMock()
    storage.collection_name.return_value = "test_finland_dataframe"
    output = pd.DataFrame([{"_storage_id": "test-1", "laclau_status": status}])
    if expected == "error":
        with pytest.raises(RuntimeError, match="1/1 rows failed"):
            persist_batch(storage, 6, "test-run", output, source_ids={"test-1"})
    else:
        persist_batch(storage, 6, "test-run", output, source_ids={"test-1"})
    args, kwargs = storage.db["test_finland_dataframe"].update_one.call_args
    assert args[0] == {"_storage_id": "test-1"}
    assert args[1]["$set"]["_pipeline.step_06.status"] == expected
    assert kwargs["upsert"] is False
