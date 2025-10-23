# scripts/assemble_h_catalog.py
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from ptquat.geometry import assemble_h_catalog

def main():
    ap = argparse.ArgumentParser(description="Assemble geometry catalog (h) from multiple sources into kpc.")
    ap.add_argument("--sparc", default="dataset/sparc_tidy.csv",
                    help="SPARC tidy CSV (for distances to convert arcsec→kpc)")
    ap.add_argument("--sources", nargs="+", default=None,
                    help="List of CSV files or directories containing source CSVs.")
    ap.add_argument("--out", default="dataset/geometry/h_catalog.csv",
                    help="Output CSV path")
    ap.add_argument("--prefer-thin", action="store_true", help="Prefer thin-disk entries when tie.")
    ap.add_argument("--default-rel-err", type=float, default=0.25,
                    help="Relative error used when a source lacks h uncertainty.")
    args = ap.parse_args()

    sources: list[str] = []
    if args.sources:
        for s in args.sources:
            p = Path(s)
            if p.is_dir():
                sources.extend([str(q) for q in p.glob("*.csv")])
            elif p.suffix.lower() == ".csv":
                sources.append(str(p))
    else:
        # 預設尋找 dataset/geometry/sources/*.csv
        d = Path("dataset/geometry/sources")
        if d.exists():
            sources = [str(p) for p in d.glob("*.csv")]

    if not sources:
        raise SystemExit("找不到任何來源 CSV。請用 --sources 指定，或放到 dataset/geometry/sources/*.csv")

    df = assemble_h_catalog(args.sparc, sources, out_csv=args.out,
                            prefer_thin=args.prefer_thin, default_rel_err=args.default_rel_err)
    print(f"[assemble] {len(df)} rows → {args.out}")

if __name__ == "__main__":
    main()
