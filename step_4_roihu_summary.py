#!/usr/bin/env python3
"""Step 4: multimodal social-semiotic summary/fusion on CSC Roihu."""
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).with_name("roihu_summary.py")), run_name="__main__")
