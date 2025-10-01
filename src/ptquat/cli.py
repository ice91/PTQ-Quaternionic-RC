from __future__ import annotations
import argparse
from pathlib import Path
from .fetch_vizier import fetch_sparc_to_csv
from .preprocess import build_tidy_csv
from .fit_global import run as run_fit

def main(argv=None):
    ap = argparse.ArgumentParser(prog="ptquat", description="PT-Quaternionic SPARC workflow")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p0 = sub.add_parser("fetch", help="Download SPARC (VizieR) CSVs")
    p0.add_argument("--out", default="dataset/raw", help="Output directory")
    
    p1 = sub.add_parser("preprocess", help="Merge table1/2 to tidy CSV with quality cuts")
    p1.add_argument("--raw", default="dataset/raw", help="Directory containing vizier_table1/2.csv")
    p1.add_argument("--out", default="dataset/sparc_tidy.csv", help="Output tidy CSV")
    p1.add_argument("--i-min", type=float, default=30.0)
    p1.add_argument("--reldmax", type=float, default=0.2)
    p1.add_argument("--qual-max", type=int, default=2)

    p2 = sub.add_parser("fit", help="Run global fits (PTQ / Baryon / MOND / NFW-1p)")
    p2.add_argument("--data", default="dataset/sparc_tidy.csv")
    p2.add_argument("--outdir", default="results/mini")
    p2.add_argument("--model", choices=["ptq","ptq-split","baryon","mond","nfw1p"], default="ptq")
    p2.add_argument("--prior", choices=["galaxies-only","planck-anchored"], default="galaxies-only")
    p2.add_argument("--sigma-sys", type=float, default=4.0)
    p2.add_argument("--H0-kms-mpc", type=float, default=None)
    p2.add_argument("--nwalkers", default="4x")
    p2.add_argument("--steps", type=int, default=12000)
    p2.add_argument("--seed", type=int, default=42)
    # MOND
    p2.add_argument("--a0-si", type=float, default=None)
    p2.add_argument("--a0-range", type=str, default="5e-11,2e-10")
    # NFW
    p2.add_argument("--c0", type=float, default=10.0)
    p2.add_argument("--c-slope", type=float, default=-0.1)
    p2.add_argument("--logM200-range", type=str, default="9,13")
    # backend
    p2.add_argument("--backend-hdf5", type=str, default=None)
    p2.add_argument("--thin-by", type=int, default=10)
    p2.add_argument("--resume", action="store_true")

    args = ap.parse_args(argv)

    if args.cmd == "fetch":
        paths = fetch_sparc_to_csv(args.out)
        print(f"Saved: {paths}")
    elif args.cmd == "preprocess":
        raw = Path(args.raw)
        t1 = raw / "vizier_table1.csv"
        t2 = raw / "vizier_table2.csv"
        out = build_tidy_csv(
            t1, t2, args.out, i_min_deg=args.i_min,
            relD_max=args.reldmax, qual_max=args.qual_max
        )
        print(f"Tidy CSV saved to {out}")
    elif args.cmd == "fit":
        # 轉發參數（保持 CLI 介面簡潔）
        run_fit([
            f"--data-path={args.data}",
            f"--outdir={args.outdir}",
            f"--model={args.model}",
            f"--prior={args.prior}",
            f"--sigma-sys={args.sigma_sys}",
            f"--steps={args.steps}",
            f"--nwalkers={args.nwalkers}",
            f"--seed={args.seed}",
            f"--a0-range={args.a0_range}",
            f"--c0={args.c0}",
            f"--c-slope={args.c_slope}",
            f"--logM200-range={args.logM200_range}",
        ] +
        ([] if args.H0_kms_mpc is None else [f"--H0-kms-mpc={args.H0_kms_mpc}"]) +
        ([] if args.a0_si is None else [f"--a0-si={args.a0_si}"]) +
        ([] if args.backend_hdf5 is None else [f"--backend-hdf5={args.backend_hdf5}"]) +
        ([f"--thin-by={args.thin_by}"] if args.thin_by else []) +
        (["--resume"] if args.resume else []))
