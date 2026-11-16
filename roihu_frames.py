"""Small pure helpers shared by the Roihu stage implementations (#413).

`_clean_document` and `_bounded` were byte-identical copies in `roihu_discourse_network.py` and
`roihu_populism.py`. They are pure — no module globals, no I/O — so keeping two copies only created the
risk of the two drifting apart. This module is their one home.

`_prompt_hash` deliberately stays in each stage module: it reads that module's own `SYSTEM_PROMPT`,
`PROMPT_VERSION` and model metadata, so moving it here would either change behaviour or require threading
three globals through every call. Two prompt hashes over two different prompts are not a duplication.

Behaviour is unchanged: these are the same functions, moved.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def clean_document(row: pd.Series) -> dict[str, Any]:
    """Convert a dataframe row into a JSON-safe document.

    Missing values become ``None``; every key is coerced to ``str``. Values that cannot be tested for
    nullness (containers, arrays) are passed through untouched rather than dropped.
    """
    doc: dict[str, Any] = {}
    for key, value in row.to_dict().items():
        try:
            if pd.isna(value):
                value = None
        except (TypeError, ValueError):
            pass
        doc[str(key)] = value
    return doc


def bounded_text(value: str, limit: int) -> tuple[str, bool]:
    """Truncate ``value`` to ``limit`` characters, reporting whether it was cut.

    Returns ``(text, truncated)``. No ellipsis is appended: callers that want one add it, and callers that
    hash the result must not have unseen characters change it.
    """
    if len(value) <= limit:
        return value, False
    return value[:limit], True
