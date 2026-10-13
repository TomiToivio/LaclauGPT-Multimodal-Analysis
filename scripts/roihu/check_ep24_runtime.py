#!/usr/bin/env python3
"""Fail fast when the restartable EP24 orchestration dependencies are missing."""
from __future__ import annotations

import importlib
import sys

REQUIRED = ("pandas", "pymongo", "redis")

missing = []
for name in REQUIRED:
    try:
        importlib.import_module(name)
    except ImportError:
        missing.append(name)

if missing:
    print(
        "Missing EP24 runtime dependencies: "
        + ", ".join(missing)
        + ". Install requirements.txt in the selected Roihu venv.",
        file=sys.stderr,
    )
    raise SystemExit(2)
