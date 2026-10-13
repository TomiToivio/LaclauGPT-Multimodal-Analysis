#!/usr/bin/env python3
"""Step 2: deep single-keyframe analysis on CSC Roihu.

Consumes the complete Step 1 dataframe and analyzes exactly one keyframe at
original source t=1.0s. Every incoming column is preserved and frame-analysis
fields are appended.
"""
import runpy
import sys
from pathlib import Path
from ep24_cli import configure_step_cli

selection = configure_step_cli(2, sys.argv[1:])
sys.argv = [sys.argv[0], *selection.remaining_argv]
runpy.run_path(str(Path(__file__).with_name("roihu_frame.py")), run_name="__main__")
