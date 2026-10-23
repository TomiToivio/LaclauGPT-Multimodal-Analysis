"""Stage EP24 media from CSC Allas before legacy Step 1 runs.

If allas_filename is an HTTP(S) URL it is downloaded directly. For object keys,
configure LACLAUGPT_ALLAS_FETCH_COMMAND in the private .env. The template receives
{object} and {destination}. Example site-specific commands stay private.
"""
from __future__ import annotations

import logging
import os
import shlex
import subprocess
import urllib.request
from pathlib import Path

import pandas as pd

from ep24_pipeline import recover_allas_filename, recover_media_identity, local_media_path

LOG = logging.getLogger("ep24_allas")


def stage_media(df: pd.DataFrame) -> int:
    root = Path(os.getenv("LACLAUGPT_ALLAS_LOCAL_ROOT", "./Allas"))
    root.mkdir(parents=True, exist_ok=True)
    template = os.getenv("LACLAUGPT_ALLAS_FETCH_COMMAND", "").strip()
    staged = 0
    if "allas_filename" not in df.columns:
        df["allas_filename"] = ""
    if "video_id" not in df.columns:
        df["video_id"] = ""
    for index, row in df.iterrows():
        recovered_id, recovered_url = recover_media_identity(row)
        if not str(row.get("video_id", "") or "").strip() and recovered_id:
            df.at[index, "video_id"] = recovered_id
            LOG.info("Allas recovered video_id from _storage_id=%s", recovered_id)
        if not str(row.get("allas_filename", "") or "").strip() and recovered_url:
            df.at[index, "allas_filename"] = recovered_url
        key = recover_allas_filename(df.loc[index])
        if not key:
            LOG.error("Allas unresolved media identity video_id=%s", row.get("video_id", ""))
            continue
        if not str(row.get("allas_filename", "") or "").strip():
            df.at[index, "allas_filename"] = key
            LOG.info("Allas recovered filename video_id=%s", row.get("video_id", ""))
        try:
            destination = local_media_path(df.loc[index], root=root)
            if destination.is_file() and destination.stat().st_size > 0:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            LOG.debug("Allas stage object=%s destination=%s", key, destination)
            if key.startswith(("https://", "http://")):
                tmp = destination.with_name(destination.name + ".part")
                try:
                    urllib.request.urlretrieve(key, tmp)
                    if not tmp.is_file() or tmp.stat().st_size == 0:
                        raise RuntimeError("Downloaded media was empty")
                    os.replace(tmp, destination)
                finally:
                    tmp.unlink(missing_ok=True)
            elif template:
                command = template.format(object=key, destination=str(destination))
                subprocess.run(shlex.split(command), check=True)
            else:
                LOG.error("Allas fetch command missing for object=%s", key)
                continue
            staged += 1
        except Exception:
            LOG.exception(
                "Allas staging failed video_id=%s object=%s; row remains retryable",
                row.get("video_id", ""), key,
            )
    return staged
