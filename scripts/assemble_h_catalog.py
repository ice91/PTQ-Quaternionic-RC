# scripts/assemble_h_catalog.py
from __future__ import annotations
import argparse
from ptquat.geometry import build_s4g_h_catalog

def main():
    ap = argparse.ArgumentParser(description="Assemble S4G edge-on h_catalog.csv")
    ap.add_argument("--sparc", default="dataset/sparc_tidy.csv")
    ap.add_argument("--out", default="dataset/geometry/h_catalog.csv")
    ap.add_argument("--prefer", choices=["thin","thick"], default="thin")
    ap.add_argument("--default-rel-err", type=float, default=0.25)
    args = ap.parse_args()
    df = build_s4g_h_catalog(args.sparc, args.out, prefer=args.prefer, default_rel_err=args.default_rel_err)
    print(f"Saved {args.out} (N={len(df)})")

if __name__ == "__main__":
    main()
