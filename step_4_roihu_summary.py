#!/usr/bin/env python3
"""Step 4: fuse multimodal evidence into a readable research summary.

Research purpose:
    Bring together the Step 1 transcript/OCR, Step 2 frame interpretation,
    and optional Step 3 temporal video interpretation without erasing the
    individual evidence streams.

Inputs and outputs:
    Reads the cumulative EP24 record selected by the shared CLI/environment
    contract. The historical implementation appends summary fields while
    retaining prior source metadata. See ep24_stage_contract.py for the exact
    field contract and docs/EP24_PIPELINE_CONTRACT.md for its meaning.

Implementation:
    This numbered script deliberately runs roihu_summary.py. The separate
    historical implementation is retained for research compatibility; the
    wrapper is an entry point, not a duplicate analysis implementation.
"""
import runpy
import sys
from pathlib import Path
from ep24_cli import configure_step_cli

selection = configure_step_cli(4, sys.argv[1:])
sys.argv = [sys.argv[0], *selection.remaining_argv]
runpy.run_path(str(Path(__file__).with_name("roihu_summary.py")), run_name="__main__")
