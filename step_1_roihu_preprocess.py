#!/usr/bin/env python3
"""Step 1: EP24 preprocess on CSC Roihu.

Production semantics:
- preserve every incoming CSV field;
- extract exactly one original-video keyframe at t=1.0s;
- run OCR exactly once on that keyframe;
- run backend-neutral ASR/translation on the full staged video;
- record ASR/OCR backend and model provenance;
- append outputs and pass the cumulative dataframe to Step 2.
"""
import runpy
import sys
from pathlib import Path
from ep24_cli import configure_step_cli

selection = configure_step_cli(1, sys.argv[1:])
sys.argv = [sys.argv[0], *selection.remaining_argv]
runpy.run_path(str(Path(__file__).with_name("roihu_preprocess.py")), run_name="__main__")
