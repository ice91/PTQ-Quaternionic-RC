#!/usr/bin/env python
"""
Convert raw SPARC per-galaxy files to the tidy CSV format used here.

Usage:
  python scripts/prepare_sparc_subset.py --raw-dir /path/to/SPARC/raw --out dataset/sparc_miniset.csv --list GalaxyA GalaxyB ...

This is a template; adapt to your raw data file naming.
"""

import argparse
import pandas as pd
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=str, required=True)
    ap.add_argument("--out", type=str, default="dataset/sparc_miniset.csv")
    ap.add_argument("--list", nargs="+", required=True, help="Galaxy names to include")
    args = ap.parse_args()

    rows = []
    raw = Path(args.raw_dir)
    for name in args.list:
        # TODO: adapt parsing here to your SPARC raw format.
        # As a placeholder we raise to remind you to wire this up.
        raise NotImplementedError("Adapt parser to SPARC raw files for: " + name)

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

if __name__ == "__main__":
    main()
