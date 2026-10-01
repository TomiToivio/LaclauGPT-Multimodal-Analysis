#!/usr/bin/env python3
"""Step 1: EP24 preprocess on CSC Roihu.

Production semantics:
- preserve every incoming CSV field;
- start media analysis at original t=1.0s after the known feed-scroll artifact;
- extract exactly one keyframe at t=1.0s;
- run OCR only on that keyframe;
- run Whisper/translation on the non-destructive clip from t=1.0s onward;
- append outputs and pass the cumulative dataframe to Step 2.

Set LACLAUGPT_INPUT_CSV and LACLAUGPT_OUTPUT_CSV for explicit stage chaining.
"""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name("roihu_preprocess.py")), run_name="__main__")
