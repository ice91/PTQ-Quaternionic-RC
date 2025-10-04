# src/ptquat/cli.py
from __future__ import annotations
import argparse
from pathlib import Path
from .fetch_vizier import fetch_sparc_to_csv
from .preprocess import build_tidy_csv
from .fit_global import run as run_fit

def main(argv=None):
    ap = argparse.ArgumentParser(prog="ptquat", description="PT-Quaternionic SPARC workflow")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # fetch
    p0 = sub.add_parser("fetch", help="Download SPARC (VizieR) CSVs")
    p0.add_argument("--out", default="dataset/raw", help="Output directory")

    # preprocess
    p1 = sub.add_parser("preprocess", help="Merge table1/2 to tidy CSV with quality cuts")
    p1.add_argument("--raw", default="dataset/raw", help="Directory containing vizier_table1/2.csv")
    p1.add_argument("--out", default="dataset/sparc_tidy.csv", help="Output tidy CSV")
    p1.add_argument("--i-min", type=float, default=30.0)
    p1.add_argument("--reldmax", type=float, default=0.2)
    p1.add_argument("--qual-max", type=int, default=2)

    # fit
    p2 = sub.add_parser("fit", help="Global fits (PTQ / PTQ-sat / PTQ-nu / Baryon / MOND / NFW-1p) with HDF5 backend")
    p2.add_argument("--data", default="dataset/sparc_tidy.csv")
    p2.add_argument("--outdir", default="results/mini")
    p2.add_argument("--prior", choices=["galaxies-only","planck-anchored"], default="galaxies-only")
    p2.add_argument("--sigma-sys", type=float, default=4.0)           # 初始值 + 弱先驗尺度
    p2.add_argument("--H0-kms-mpc", type=float, default=None)
    p2.add_argument("--nwalkers", default="4x")
    p2.add_argument("--steps", type=int, default=12000)
    p2.add_argument("--seed", type=int, default=42)

    # 模型選擇與其引數（轉傳到 fit_global）
    p2.add_argument("--model", choices=["ptq","ptq-split","ptq-sat","ptq-nu","baryon","mond","nfw1p"], default="ptq")
    p2.add_argument("--a0-si", type=float, default=None, help="Fix MOND a0 [m/s^2]; if omitted, sample it.")
    p2.add_argument("--a0-range", type=str, default="5e-11,2e-10", help="Uniform prior range for a0 when sampling.")
    p2.add_argument("--logM200-range", type=str, default="9,13", help="Uniform prior for log10 M200 [Msun] in nfw1p.")
    p2.add_argument("--c0", type=float, default=10.0, help="c(M) normalization at 1e12 Msun in nfw1p.")
    p2.add_argument("--c-slope", type=float, default=-0.1, help="c(M) slope beta in nfw1p.")
    p2.add_argument("--r0-range", type=str, default="0.1,30.0", help="Uniform prior on r0 [kpc] in log10-space (ptq-sat).")
    p2.add_argument("--kappa", type=float, default=(1.0/(2.0*3.141592653589793)),
                    help="Scale factor in a_* = kappa * eps * c * H0 (for ptq-nu).")

    # HDF5 backend / thinning / resume
    p2.add_argument("--backend-hdf5", type=str, default=None, help="emcee HDF5 backend path")
    p2.add_argument("--thin-by", type=int, default=10, help="Thin factor when loading samples")
    p2.add_argument("--resume", action="store_true", help="Resume if HDF5 backend has iterations")

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
        run_fit([
            f"--data-path={args.data}",
            f"--outdir={args.outdir}",
            f"--prior={args.prior}",
            f"--sigma-sys={args.sigma_sys}",
            f"--steps={args.steps}",
            f"--nwalkers={args.nwalkers}",
            f"--seed={args.seed}",
            f"--model={args.model}",
            f"--a0-range={args.a0_range}",
            f"--logM200-range={args.logM200_range}",
            f"--c0={args.c0}",
            f"--c-slope={args.c_slope}",
            f"--r0-range={args.r0_range}",
            f"--kappa={args.kappa}",
        ]
        + ([] if args.H0_kms_mpc is None else [f"--H0-kms-mpc={args.H0_kms_mpc}"])
        + ([] if args.a0_si is None else [f"--a0-si={args.a0_si}"])
        + ([] if args.backend_hdf5 is None else [f"--backend-hdf5={args.backend_hdf5}"])
        + ([f"--thin-by={args.thin_by}"] if args.thin_by else [])
        + (["--resume"] if args.resume else []))
