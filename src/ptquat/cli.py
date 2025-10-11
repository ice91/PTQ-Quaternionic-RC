# src/ptquat/cli.py
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .fetch_vizier import fetch_sparc_to_csv
from .preprocess import build_tidy_csv
from .fit_global import run as run_fit
from . import experiments as EXP


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
    p2 = sub.add_parser("fit", help="Global fits (PTQ / PTQ-ν / PTQ-screen / Baryon / MOND / NFW-1p) with HDF5 backend")
    p2.add_argument("--data", default="dataset/sparc_tidy.csv")
    p2.add_argument("--outdir", default="results/mini")
    p2.add_argument("--prior", choices=["galaxies-only","planck-anchored"], default="galaxies-only")
    p2.add_argument("--sigma-sys", type=float, default=4.0)
    p2.add_argument("--H0-kms-mpc", type=float, default=None)
    p2.add_argument("--nwalkers", default="4x")
    p2.add_argument("--steps", type=int, default=12000)
    p2.add_argument("--seed", type=int, default=42)
    p2.add_argument("--model", choices=["ptq","ptq-nu","ptq-screen","ptq-split","baryon","mond","nfw1p"], default="ptq")
    p2.add_argument("--a0-si", type=float, default=None)
    p2.add_argument("--a0-range", type=str, default="5e-11,2e-10")
    p2.add_argument("--logM200-range", type=str, default="9,13")
    p2.add_argument("--c0", type=float, default=10.0)
    p2.add_argument("--c-slope", type=float, default=-0.1)
    p2.add_argument("--backend-hdf5", type=str, default=None)
    p2.add_argument("--thin-by", type=int, default=10)
    p2.add_argument("--resume", action="store_true")
    p2.add_argument("--likelihood", choices=["gauss","t"], default="gauss")
    p2.add_argument("--t-dof", type=float, default=8.0)

    # experiments group
    px = sub.add_parser("exp", help="Supplementary experiments & robustness")
    sx = px.add_subparsers(dest="exp_cmd", required=True)

    # exp ppc
    ppc = sx.add_parser("ppc", help="Posterior(-like) predictive check on an existing results dir")
    ppc.add_argument("--results", required=True, help="Path to a finished results dir")
    ppc.add_argument("--data",   default="dataset/sparc_tidy.csv")
    ppc.add_argument("--prefix", default="ppc")

    # exp stress
    pst = sx.add_parser("stress", help="Inflate i_err/D_err and re-fit")
    pst.add_argument("--data", default="dataset/sparc_tidy.csv")
    pst.add_argument("--outroot", default="results/stress")
    pst.add_argument("--model", choices=["ptq","ptq-nu","ptq-screen","ptq-split","baryon","mond","nfw1p"], default="ptq-screen")
    pst.add_argument("--scale-i", type=float, default=2.0)
    pst.add_argument("--scale-D", type=float, default=2.0)
    # passthrough fit args
    pst.add_argument("--prior", choices=["galaxies-only","planck-anchored"], default="galaxies-only")
    pst.add_argument("--sigma-sys", type=float, default=4.0)
    pst.add_argument("--H0-kms-mpc", type=float, default=None)
    pst.add_argument("--nwalkers", default="4x")
    pst.add_argument("--steps", type=int, default=12000)
    pst.add_argument("--seed", type=int, default=42)
    pst.add_argument("--likelihood", choices=["gauss","t"], default="gauss")
    pst.add_argument("--t-dof", type=float, default=8.0)

    # exp mask
    pmk = sx.add_parser("mask", help="Mask inner radii r<rmin_kpc and re-fit")
    pmk.add_argument("--data", default="dataset/sparc_tidy.csv")
    pmk.add_argument("--outroot", default="results/mask")
    pmk.add_argument("--model", choices=["ptq","ptq-nu","ptq-screen","ptq-split","baryon","mond","nfw1p"], default="ptq-screen")
    pmk.add_argument("--rmin-kpc", type=float, default=2.0)
    # passthrough
    pmk.add_argument("--prior", choices=["galaxies-only","planck-anchored"], default="galaxies-only")
    pmk.add_argument("--sigma-sys", type=float, default=4.0)
    pmk.add_argument("--H0-kms-mpc", type=float, default=None)
    pmk.add_argument("--nwalkers", default="4x")
    pmk.add_argument("--steps", type=int, default=12000)
    pmk.add_argument("--seed", type=int, default=42)
    pmk.add_argument("--likelihood", choices=["gauss","t"], default="gauss")
    pmk.add_argument("--t-dof", type=float, default=8.0)

    # exp H0
    ph0 = sx.add_parser("H0", help="Scan H0 sensitivity")
    ph0.add_argument("--data", default="dataset/sparc_tidy.csv")
    ph0.add_argument("--outroot", default="results/H0_scan")
    ph0.add_argument("--model", choices=["ptq","ptq-nu","ptq-screen"], default="ptq-screen")
    ph0.add_argument("--H0-list", type=float, nargs="+", default=[60.0, 67.4, 70.0, 73.0, 76.0])
    # passthrough
    ph0.add_argument("--prior", choices=["galaxies-only","planck-anchored"], default="galaxies-only")
    ph0.add_argument("--sigma-sys", type=float, default=4.0)
    ph0.add_argument("--nwalkers", default="4x")
    ph0.add_argument("--steps", type=int, default=12000)
    ph0.add_argument("--seed", type=int, default=42)
    ph0.add_argument("--likelihood", choices=["gauss","t"], default="gauss")
    ph0.add_argument("--t-dof", type=float, default=8.0)

    # exp plateau
    ppl = sx.add_parser("plateau", help="Make stacked residual-acceleration plateau plot from a results dir")
    ppl.add_argument("--results", required=True)
    ppl.add_argument("--data", default="dataset/sparc_tidy.csv")
    ppl.add_argument("--nbins", type=int, default=24)
    ppl.add_argument("--prefix", default="plateau")

    # exp closure
    pcl = sx.add_parser("closure", help="Cross-scale closure test: epsilon_cos vs epsilon_RC")
    pcl.add_argument("--results", required=True)
    group = pcl.add_mutually_exclusive_group(required=True)
    group.add_argument("--epsilon-cos", type=float, default=None)
    group.add_argument("--omega-lambda", type=float, default=None)

    # exp kappa-gal (per-galaxy kappa check)
    kgal = sx.add_parser("kappa-gal", help="Per-galaxy kappa check: eps_eff(r*)/eps_cos vs kappa_pred=eta*Rd/r*")
    kgal.add_argument("--results", required=True)
    kgal.add_argument("--data", default="dataset/sparc_tidy.csv")
    kgal.add_argument("--eta", type=float, default=0.15)
    kgal.add_argument("--frac-vmax", type=float, default=0.9)
    kgal.add_argument("--nsamp", type=int, default=300)
    kgal.add_argument("--prefix", default="kappa_gal")
    # NEW: source of y
    kgal.add_argument("--y-source", choices=["model","obs","obs-debias"], default="model")
    grp_k = kgal.add_mutually_exclusive_group(required=False)
    grp_k.add_argument("--epsilon-cos", type=float, default=None)
    grp_k.add_argument("--omega-lambda", type=float, default=None)

    # exp kappa-prof (radius-resolved stacked profile)
    kpro = sx.add_parser("kappa-prof", help="Radius-resolved stacked kappa profile")
    kpro.add_argument("--results", required=True)
    kpro.add_argument("--data", default="dataset/sparc_tidy.csv")
    kpro.add_argument("--eta", type=float, default=0.15)
    kpro.add_argument("--nbins", type=int, default=24)
    kpro.add_argument("--min-per-bin", type=int, default=20)
    kpro.add_argument("--x-kind", choices=["r_over_Rd","r_kpc"], default="r_over_Rd")
    kpro.add_argument("--prefix", default="kappa_profile")
    grp_k2 = kpro.add_mutually_exclusive_group(required=False)
    grp_k2.add_argument("--epsilon-cos", type=float, default=None)
    grp_k2.add_argument("--omega-lambda", type=float, default=None)

    args = ap.parse_args(argv)

    if args.cmd == "fetch":
        paths = fetch_sparc_to_csv(args.out)
        print(f"Saved: {paths}")

    elif args.cmd == "preprocess":
        raw = Path(args.raw)
        t1 = raw / "vizier_table1.csv"
        t2 = raw / "vizier_table2.csv"
        out = build_tidy_csv(t1, t2, args.out, i_min_deg=args.i_min, relD_max=args.reldmax, qual_max=args.qual_max)
        print(f"Tidy CSV saved to {out}")

    elif args.cmd == "fit":
        run_fit([
            f"--data-path={args.data}",
            f"--outdir={args.outdir}",
            f"--prior={args.prior}",
            f"--sigma-sys={args.sigma_s
ys}",
            f"--steps={args.steps}",
            f"--nwalkers={args.nwalkers}",
            f"--seed={args.seed}",
            f"--model={args.model}",
            f"--a0-range={args.a0_range}",
            f"--logM200-range={args.logM200_range}",
            f"--c0={args.c0}",
            f"--c-slope={args.c_slope}",
            f"--likelihood={args.likelihood}",
            f"--t-dof={args.t_dof}",
        ]
        + ([] if args.H0_kms_mpc is None else [f"--H0-kms-mpc={args.H0_kms_mpc}"])
        + ([] if args.a0_si is None else [f"--a0-si={args.a0_si}"])
        + ([] if args.backend_hdf5 is None else [f"--backend-hdf5={args.backend_hdf5}"])
        + ([f"--thin-by={args.thin_by}"] if args.thin_by else [])
        + (["--resume"] if args.resume else []))

    elif args.cmd == "exp":
        if args.exp_cmd == "ppc":
            out = EXP.ppc_check(args.results, args.data, out_prefix=args.prefix)
            print(json.dumps(out, indent=2))

        elif args.exp_cmd == "stress":
            out = EXP.stress_errors(
                data_path=args.data, out_root=args.outroot, model=args.model,
                scale_i=args.scale_i, scale_D=args.scale_D,
                prior=args.prior, sigma_sys=args.sigma_sys,
                H0_kms_mpc=args.H0_kms_mpc, nwalkers=args.nwalkers,
                steps=args.steps, seed=args.seed,
                likelihood=args.likelihood, t_dof=args.t_dof
            )
            print(json.dumps(out, indent=2))

        elif args.exp_cmd == "mask":
            out = EXP.mask_inner(
                data_path=args.data, out_root=args.outroot, model=args.model,
                rmin_kpc=args.rmin_kpc,
                prior=args.prior, sigma_sys=args.sigma_sys,
                H0_kms_mpc=args.H0_kms_mpc, nwalkers=args.nwalkers,
                steps=args.steps, seed=args.seed,
                likelihood=args.likelihood, t_dof=args.t_dof
            )
            print(json.dumps(out, indent=2))

        elif args.exp_cmd == "H0":
            df = EXP.scan_H0(
                data_path=args.data, out_root=args.outroot, model=args.model,
                H0_list=args.H0_list, prior=args.prior, sigma_sys=args.sigma_sys,
                nwalkers=args.nwalkers, steps=args.steps, seed=args.seed,
                likelihood=args.likelihood, t_dof=args.t_dof
            )
            print(df.to_string(index=False))

        elif args.exp_cmd == "plateau":
            df, png = EXP.residual_plateau(args.results, args.data, nbins=args.nbins, out_prefix=args.prefix)
            print(f"Saved: {png}")

        elif args.exp_cmd == "closure":
            out = EXP.closure_test(args.results, epsilon_cos=args.epsilon_cos, omega_lambda=args.omega_lambda)
            print(json.dumps(out, indent=2))

        elif args.exp_cmd == "kappa-gal":
            out = EXP.kappa_per_galaxy(
                results_dir=args.results, data_path=args.data,
                eta=args.eta, frac_vmax=args.frac_vmax, nsamp=args.nsamp,
                epsilon_cos=args.epsilon_cos, omega_lambda=args.omega_lambda,
                y_source=args.y_source,        # NEW: plumb through
                out_prefix=args.prefix
            )
            print(json.dumps(out, indent=2))

        elif args.exp_cmd == "kappa-prof":
            df, png = EXP.kappa_radius_resolved(
                results_dir=args.results, data_path=args.data,
                eta=args.eta, epsilon_cos=args.epsilon_cos, omega_lambda=args.omega_lambda,
                nbins=args.nbins, min_per_bin=args.min_per_bin,
                x_kind=args.x_kind, out_prefix=args.prefix
            )
            print(f"Saved: {png}")
