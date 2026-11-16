#!/usr/bin/env python3
"""Step 5: structure and normalize the Step 4 multimodal summary.

Research purpose:
    Extract legacy-compatible structured post-processing information such as
    entities, topics and sentiment targets from the accumulated analysis.

Inputs and outputs:
    Consumes the cumulative record from the previous step and appends fields
    without discarding earlier evidence. The precise schema is specified by
    ep24_stage_contract.py and docs/EP24_PIPELINE_CONTRACT.md.

Implementation:
    This stable numbered CLI entry point delegates to roihu_postprocess.py.
    Keeping the historical implementation separate is intentional: it
    protects compatibility with the published human-coded research workflow.
"""
import runpy
import sys
from pathlib import Path
from ep24_cli import configure_step_cli

selection = configure_step_cli(5, sys.argv[1:])
sys.argv = [sys.argv[0], *selection.remaining_argv]
runpy.run_path(str(Path(__file__).with_name("roihu_postprocess.py")), run_name="__main__")
