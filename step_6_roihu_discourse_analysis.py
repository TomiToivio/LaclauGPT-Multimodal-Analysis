#!/usr/bin/env python3
"""Step 6: Laclau/Palonen discourse analysis.

This is the canonical numbered name for the historical roihu_populism.py stage.
"""
import runpy
import sys
from pathlib import Path
from ep24_cli import configure_step_cli

selection = configure_step_cli(6, sys.argv[1:])
sys.argv = [sys.argv[0], *selection.remaining_argv]
runpy.run_path(str(Path(__file__).with_name("roihu_populism.py")), run_name="__main__")
