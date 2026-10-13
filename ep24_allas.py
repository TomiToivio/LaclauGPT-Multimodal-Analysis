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

LOG = logging.getLogger("ep24_allas")


def stage_media(df: pd.DataFrame) -> int:
    root = Path(os.getenv("LACLAUGPT_ALLAS_LOCAL_ROOT", "./Allas"))
    root.mkdir(parents=True, exist_ok=True)
    template = os.getenv("LACLAUGPT_ALLAS_FETCH_COMMAND", "").strip()
    staged = 0
    for _, row in df.iterrows():
        key = str(row.get("allas_filename", "")).strip()
        if not key:
            continue
        direct = Path(key)
        if direct.exists():
            continue
        destination = root / key.lstrip("/")
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        LOG.debug("Allas stage object=%s destination=%s", key, destination)
        if key.startswith(("https://", "http://")):
            tmp = destination.with_suffix(destination.suffix + ".part")
            urllib.request.urlretrieve(key, tmp)
            os.replace(tmp, destination)
        elif template:
            command = template.format(object=key, destination=str(destination))
            subprocess.run(shlex.split(command), check=True)
        else:
            LOG.warning(
                "media not local and no LACLAUGPT_ALLAS_FETCH_COMMAND configured: %s", key
            )
            continue
        staged += 1
    return staged
