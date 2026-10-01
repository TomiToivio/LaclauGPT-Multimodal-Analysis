#!/usr/bin/env python3
"""Step 6: Laclau/Palonen discourse analysis.

This is the canonical numbered name for the historical roihu_populism.py stage.
The old filename remains as a compatibility implementation.
"""
import runpy
from pathlib import Path
runpy.run_path(str(Path(__file__).with_name("roihu_populism.py")), run_name="__main__")
