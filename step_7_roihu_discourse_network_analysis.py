#!/usr/bin/env python3
"""Step 7 CLI compatibility entry point (Discourse Network Analysis).

The research implementation, including its readable Leifeld prompt and
source-grounding checks, lives in roihu_discourse_network.py.
Keep this file name: Slurm scripts and researchers invoke it directly.
When imported, expose the implementation module itself so test monkeypatches
and existing Python callers continue to modify the actual function globals.
"""
from __future__ import annotations

if __name__ == "__main__":
    import runpy
    runpy.run_module("roihu_discourse_network", run_name="__main__")
else:
    import sys
    import roihu_discourse_network as _implementation
    sys.modules[__name__] = _implementation
