#!/usr/bin/env python3
"""Step 2: sampled-frame multimodal analysis on CSC Roihu."""
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).with_name("roihu_frame.py")), run_name="__main__")
