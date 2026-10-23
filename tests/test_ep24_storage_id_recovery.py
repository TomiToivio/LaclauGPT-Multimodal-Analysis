"""Recover missing Step 1 media identities without guessing source records."""
import pandas as pd

from ep24_pipeline import recover_media_identity, recover_allas_filename


VIDEO = "HEPP24-PT-RG-TT-20240606-1060000"
URL = f"https://a3s.fi/swift/v1/HEPP24/PT/RG/TT/20240606/{VIDEO}.mp4"
STORAGE_ID = f"{VIDEO}|Portugal|{URL}"


def test_both_fields_can_be_recovered_from_composite_storage_id():
    row = pd.Series({"_storage_id": STORAGE_ID, "country": "portugal",
                     "video_id": "", "allas_filename": ""})
    assert recover_media_identity(row) == (VIDEO, URL)
    assert recover_allas_filename(row) == URL


def test_hash_only_id_is_not_guessed():
    row = pd.Series({"_storage_id": "f" * 64, "country": "portugal",
                     "video_id": "", "allas_filename": ""})
    assert recover_media_identity(row) == ("", "")
    assert recover_allas_filename(row) == ""


def test_cross_country_identity_is_not_used():
    row = pd.Series({"_storage_id": STORAGE_ID, "country": "poland",
                     "video_id": "", "allas_filename": ""})
    assert recover_media_identity(row) == ("", "")


def test_conflicting_existing_video_id_remains_authoritative():
    row = pd.Series({"_storage_id": STORAGE_ID, "country": "portugal",
                     "video_id": "another-video", "allas_filename": ""})
    assert recover_media_identity(row) == ("another-video", "")


def test_invalid_url_from_composite_is_rejected():
    row = pd.Series({"_storage_id": f"{VIDEO}|Portugal|https://evil.example/{VIDEO}.mp4",
                     "country": "portugal", "video_id": "", "allas_filename": ""})
    assert recover_media_identity(row) == ("", "")
