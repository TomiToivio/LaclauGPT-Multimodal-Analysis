#!/usr/bin/env python3
"""Step 5: structured post-processing of the summary output on CSC Roihu."""
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).with_name("roihu_postprocess.py")), run_name="__main__")
