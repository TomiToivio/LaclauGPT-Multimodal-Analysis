"""Regression tests for missing Allas media metadata in claimed Step 1 batches."""
from pathlib import Path

import pandas as pd

from ep24_pipeline import local_media_path, recover_allas_filename
from ep24_allas import stage_media


def test_recover_structured_hepp24_id(monkeypatch, tmp_path):
    monkeypatch.setenv("LACLAUGPT_ALLAS_LOCAL_ROOT", str(tmp_path))
    row = pd.Series({
        "country": "portugal",
        "video_id": "HEPP24-PT-RG-TT-20240606-1060000",
        "allas_filename": "",
    })
    assert recover_allas_filename(row) == (
        "https://a3s.fi/swift/v1/HEPP24/PT/RG/TT/20240606/"
        "HEPP24-PT-RG-TT-20240606-1060000.mp4"
    )
    path = local_media_path(row)
    assert path.is_relative_to(tmp_path / "_http")
    assert path.name == "HEPP24-PT-RG-TT-20240606-1060000.mp4"


def test_existing_media_key_is_authoritative():
    row = pd.Series({"video_id": "HEPP24-PT-RG-TT-20240606-1060000",
                     "allas_filename": "special/custom.mp4"})
    assert recover_allas_filename(row) == "special/custom.mp4"


def test_unknown_id_is_not_guessed():
    assert recover_allas_filename(pd.Series({
        "video_id": "not-a-structured-id", "allas_filename": ""
    })) == ""


def test_country_mismatch_is_not_guessed():
    assert recover_allas_filename(pd.Series({
        "country": "portugal", "video_id": "HEPP24-FI-RG-TT-20240606-1060000"
    })) == ""


def test_staging_recovers_filename_and_downloads_to_correct_path(tmp_path, monkeypatch):
    monkeypatch.setenv("LACLAUGPT_ALLAS_LOCAL_ROOT", str(tmp_path))
    frame = pd.DataFrame([{
        "country": "portugal",
        "video_id": "HEPP24-PT-RG-TT-20240606-1060000",
        "allas_filename": "",
    }])
    calls = []

    def fake_download(url, dest):
        calls.append(url)
        Path(dest).write_bytes(b"test mp4 bytes")

    monkeypatch.setattr("ep24_allas.urllib.request.urlretrieve", fake_download)
    assert stage_media(frame) == 1
    assert frame.at[0, "allas_filename"] == calls[0]
    assert local_media_path(frame.iloc[0]).read_bytes() == b"test mp4 bytes"


def test_one_unrecoverable_row_does_not_abort_batch(tmp_path, monkeypatch):
    monkeypatch.setenv("LACLAUGPT_ALLAS_LOCAL_ROOT", str(tmp_path))
    frame = pd.DataFrame([
        {"country": "portugal", "video_id": "unknown", "allas_filename": ""},
        {"country": "portugal", "video_id": "HEPP24-PT-RG-TT-20240606-1060000", "allas_filename": ""},
    ])
    def fake_download(url, dest):
        Path(dest).write_bytes(b"media")

    monkeypatch.setattr("ep24_allas.urllib.request.urlretrieve", fake_download)
    assert stage_media(frame) == 1
    assert frame.at[0, "allas_filename"] == ""
    assert frame.at[1, "allas_filename"].startswith("https://")
