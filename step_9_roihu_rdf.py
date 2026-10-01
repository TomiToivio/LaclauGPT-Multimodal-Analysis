#!/usr/bin/env python3
"""Step 9: deterministic RDF export after discourse/network analysis."""
import os
import sys
from roihu_csv_rdf import main

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--limit" not in args and os.getenv("LACLAUGPT_MAX_ROWS"):
        args += ["--limit", os.environ["LACLAUGPT_MAX_ROWS"]]
    raise SystemExit(main(args))
