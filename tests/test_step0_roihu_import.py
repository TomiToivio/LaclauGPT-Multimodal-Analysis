"""Step 0 regression tests use synthetic CSVs and an in-memory Mongo stand-in."""
from pathlib import Path

import pandas as pd
import pytest

from step_0_roihu_import import import_country, prepare_dataframe_with_offset


def _sample(n=3):
    return pd.DataFrame({
        "video_id": [f"vid{i}" for i in range(n)],
        "allas_filename": [f"file{i}.mp4" for i in range(n)],
        "entities": ["reviewed entity"] * n,
        "themes": ["reviewed theme"] * n,
        "source_recording": ["rec"] * n,
        "sequence_number": list(map(str, range(n))),
    })


def test_step0_ids_match_existing_bootstrap_across_chunks():
    from ep24_bootstrap import prepare_dataframe
    frame = _sample(5)
    whole = prepare_dataframe(frame, country="finland")
    pieces = pd.concat([
        prepare_dataframe_with_offset(frame.iloc[0:2], country="finland", offset=0),
        prepare_dataframe_with_offset(frame.iloc[2:4], country="finland", offset=2),
        prepare_dataframe_with_offset(frame.iloc[4:5], country="finland", offset=4),
    ])
    assert pieces["_storage_id"].tolist() == whole["_storage_id"].tolist()


def test_step0_dry_run_checks_source_without_mongo(tmp_path, monkeypatch):
    from roihu_storage import StorageConfig
    frame = _sample()
    path = tmp_path / "ep24_finland.csv"
    frame.to_csv(path, index=False)
    monkeypatch.setenv("LACLAUGPT_MONGO_ENABLED", "0")
    monkeypatch.setenv("LACLAUGPT_DATASET", "ep2024_reprocess")
    assert import_country(path, limit=2, batch_size=1, dry_run=True) == (0, 2)


def test_step0_rejects_unmigrated_csv(tmp_path, monkeypatch):
    path = tmp_path / "ep24_poland.csv"
    pd.DataFrame([{"video_id": "a"}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing"):
        import_country(path, limit=0, batch_size=2, dry_run=True)


def test_step0_rejects_lfs_pointer(tmp_path):
    path = tmp_path / "ep24_finland.csv"
    path.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 100\n")
    with pytest.raises(RuntimeError, match="Git LFS pointer"):
        import_country(path, limit=0, batch_size=10, dry_run=True)
