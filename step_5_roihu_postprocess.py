#!/usr/bin/env python3
"""Step 5: structured post-processing of the summary output on CSC Roihu."""
import runpy
import sys
from pathlib import Path
from ep24_cli import configure_step_cli

selection = configure_step_cli(5, sys.argv[1:])
sys.argv = [sys.argv[0], *selection.remaining_argv]
runpy.run_path(str(Path(__file__).with_name("roihu_postprocess.py")), run_name="__main__")
