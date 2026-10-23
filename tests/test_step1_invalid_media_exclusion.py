"""Invalid media rows must leave the active Mongo dataframe, not poison Step 1."""
from types import SimpleNamespace

from ep24_stage_orchestrator import (
    exclude_unidentifiable_media, valid_media_identity,
)


class FakeCollection:
    def __init__(self):
        self.saved = []
        self.deleted = []

    def replace_one(self, selector, document, upsert=False):
        self.saved.append((selector, document, upsert))

    def delete_one(self, selector):
        self.deleted.append(selector)
        return SimpleNamespace(deleted_count=1)


class FakeStorage:
    def __init__(self):
        self.active = FakeCollection()
        self.audit = FakeCollection()
        self.db = {"dataset_dataframe": self.active,
                   "dataset_dataframe_excluded": self.audit}

    def collection_name(self, purpose):
        return "dataset_" + purpose


def test_invalid_rows_are_removed_from_processing_and_audited():
    video = "HEPP24-PT-RG-TT-20240606-1060000"
    url = f"https://a3s.fi/swift/v1/HEPP24/PT/RG/TT/20240606/{video}.mp4"
    valid = {"_id": 1, "_storage_id": video + "|Portugal|" + url,
             "country": "portugal", "video_id": video, "allas_filename": url}
    invalid = {"_id": 2, "_storage_id": "hashonly", "country": "portugal",
               "video_id": "", "allas_filename": ""}
    storage = FakeStorage()
    retained = exclude_unidentifiable_media(storage, [valid, invalid],
                                            run_id="job-0", country="portugal")
    assert retained == [valid]
    assert len(storage.active.deleted) == 1
    assert storage.active.deleted[0]["_id"] == 2
    assert len(storage.audit.saved) == 1
    assert storage.audit.saved[0][1]["_exclusion"]["reason"] == (
        "missing_video_id_or_allas_filename"
    )
    assert "_id" not in storage.audit.saved[0][1]


def test_recoverable_composite_id_is_not_dropped():
    video = "HEPP24-PT-RG-TT-20240606-1060000"
    url = f"https://a3s.fi/swift/v1/HEPP24/PT/RG/TT/20240606/{video}.mp4"
    row = {"_storage_id": f"{video}|Portugal|{url}",
           "video_id": "", "allas_filename": "", "country": "portugal"}
    assert valid_media_identity(row)


def test_broken_or_partial_identity_is_excluded():
    assert not valid_media_identity({"_storage_id": "hashonly", "video_id": "",
                                     "allas_filename": "", "country": "portugal"})
    assert not valid_media_identity({"video_id": "", "allas_filename": "valid.mp4"})
    assert not valid_media_identity({"video_id": "random-video", "allas_filename": ""})
