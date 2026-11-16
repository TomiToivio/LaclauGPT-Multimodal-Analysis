#!/usr/bin/env python3
"""Step 8 CLI compatibility entry point (Social Network Analysis).

The evidence-backed SNA implementation and its readable research prompt
live in roihu_sna.py. Preserve this executable name for Slurm and users.
Importing this filename aliases the implementation itself, maintaining
existing Python imports, monkeypatching, and source-code introspection.
"""
from __future__ import annotations

if __name__ == "__main__":
    import runpy
    runpy.run_module("roihu_sna", run_name="__main__")
else:
    import sys
    import roihu_sna as _implementation
    sys.modules[__name__] = _implementation
