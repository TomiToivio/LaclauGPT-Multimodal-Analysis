"""Synthetic verification for read-only live sample export (#373)."""
from scripts.ep24.export_live_sample import presentation_frame, snapshot_summary


def _doc(key, statuses):
    return {
        "_storage_id": key, "video_id": key, "country": "finland",
        "source_field": "retain me",
        "_pipeline": {f"step_{n:02d}": {"status": state}
                      for n, state in statuses.items()},
        "_provenance": {"run_id": "fixture"},
    }


def test_partial_claimed_error_rows_retained_and_flattened():
    docs = [
        _doc("c", {1: "complete", 2: "claimed"}),
        _doc("a", {1: "complete", 2: "complete", 7: "complete"}),
        _doc("b", {1: "error"}),
    ]
    original = [dict(doc) for doc in docs]
    frame = presentation_frame(docs, "finland")
    assert frame["_storage_id"].tolist() == ["a", "b", "c"]
    assert frame["source_field"].tolist() == ["retain me"] * 3
    assert frame["step_07_status"].tolist() == ["complete", "pending", "pending"]
    assert frame["step_01_status"].tolist() == ["complete", "error", "complete"]
    assert frame["step_02_status"].tolist() == ["complete", "pending", "claimed"]
    assert "_provenance" in frame.columns
    assert docs == original   # never mutates Mongo-derived documents
    summary = snapshot_summary(docs)
    assert summary["observed_records"] == 3
    assert summary["step_07_complete"] == 1
    assert not summary["all_step_07_complete"]


def test_duplicate_and_missing_storage_ids_rejected():
    import pytest
    with pytest.raises(ValueError, match="Duplicate"):
        presentation_frame([_doc("a", {}), _doc("a", {})], "finland")
    with pytest.raises(ValueError, match="missing"):
        presentation_frame([{"video_id": "no-id"}], "finland")


def test_mixed_countries_rejected():
    import pytest
    with pytest.raises(ValueError, match="mixed"):
        presentation_frame([_doc("one", {}), {**_doc("two", {}), "country": "poland"}], "finland")
