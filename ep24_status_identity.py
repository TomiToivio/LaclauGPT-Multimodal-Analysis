"""Identity arithmetic for the EP24 status tool, kept out of the script's import side effects (#379).

``ep24_status.py`` loads the private environment and constructs storage at import time, which is right
for a CLI and wrong for a test: importing it sets ``LACLAUGPT_MONGO_*`` for the whole pytest process and
breaks unrelated tests. The arithmetic lives here instead, over the two reads a collection must answer.

This is *reporting*, not the gate: the resume hard stop is ``scripts/ep24/preflight_pilot.py``, which also
checks dataset isolation and stale claims. One gate, one home.
"""
from __future__ import annotations


def identity_summary(collection) -> dict:
    """Total vs distinct stable ids, without assuming the collection is clean.

    ``collection`` is anything that answers ``count_documents`` and ``distinct`` — a real pymongo
    collection or a test double.
    """
    total = collection.count_documents({})
    ids = [value for value in collection.distinct("_storage_id") if value not in (None, "")]
    missing = collection.count_documents({"_storage_id": {"$in": [None, ""]}})
    unique = len(ids)
    return {
        "total": total,
        "unique": unique,
        # A row that has no id is not also a duplicate; subtracting it keeps the two findings separate.
        "duplicates": max(total - unique - missing, 0),
        "missing_id": missing,
    }
