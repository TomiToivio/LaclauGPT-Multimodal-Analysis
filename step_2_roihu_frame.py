#!/usr/bin/env python3
"""Step 2: deep single-keyframe analysis on CSC Roihu.

Consumes the complete Step 1 dataframe, including source metadata, Whisper,
translation and OCR. Analyzes exactly one image: the Step 1 keyframe extracted
at original source t=1.0s. It does not sample later frames. Temporal narrative
belongs to Step 3 native-video analysis; audio/language evidence comes from the
Whisper transcript produced by Step 1.

Every incoming column is preserved and frame-analysis fields are appended.
Set LACLAUGPT_INPUT_CSV to Step 1 output and LACLAUGPT_OUTPUT_CSV to Step 2 output.
"""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name("roihu_frame.py")), run_name="__main__")
