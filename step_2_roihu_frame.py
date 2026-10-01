#!/usr/bin/env python3
"""Step 2: deep single-keyframe analysis on CSC Roihu.

Consumes the complete Step 1 dataframe, including source metadata, Whisper,
translation and OCR. Analyzes only the t=1.0s keyframe, emphasizing visible
platform/video metadata, text, scene details and social-semiotic description.

Every incoming column is preserved and frame-analysis fields are appended.
Set LACLAUGPT_INPUT_CSV to Step 1 output and LACLAUGPT_OUTPUT_CSV to Step 2 output.
"""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name("roihu_frame.py")), run_name="__main__")
