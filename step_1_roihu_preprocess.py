#!/usr/bin/env python3
"""Step 1: legacy-compatible EP24 preprocessing on CSC Roihu.

Canonical numbered batch entry point. The historical roihu_preprocess.py remains
available as a compatibility implementation.
"""
import runpy
runpy.run_path("roihu_preprocess.py", run_name="__main__")
