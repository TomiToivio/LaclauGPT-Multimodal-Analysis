"""Small EP24-facing database helpers over the existing roihu_storage layer."""
from __future__ import annotations

import os
from contextlib import contextmanager

from roihu_storage import MongoStorage, StorageConfig


@contextmanager
def country_storage(country: str):
    old = os.environ.get("LACLAUGPT_COUNTRY")
    os.environ["LACLAUGPT_COUNTRY"] = country
    storage = MongoStorage(StorageConfig.from_env())
    try:
        yield storage
    finally:
        storage.close()
        if old is None:
            os.environ.pop("LACLAUGPT_COUNTRY", None)
        else:
            os.environ["LACLAUGPT_COUNTRY"] = old
