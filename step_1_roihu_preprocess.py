#!/usr/bin/env python3
"""Step 1: EP24 preprocess on CSC Roihu.

Production semantics:
- preserve every incoming CSV field;
- extract exactly one original-video keyframe at t=1.0s;
- run OCR exactly once on that keyframe;
- run backend-neutral ASR/translation on the full staged video;
- record ASR/OCR backend and model provenance;
- append outputs and pass the cumulative dataframe to Step 2.

Set LACLAUGPT_INPUT_CSV and LACLAUGPT_OUTPUT_CSV for explicit stage chaining.
"""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name("roihu_preprocess.py")), run_name="__main__")
