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

    # geometry tools
    geom = sub.add_parser("geom", help="Geometry catalogs (e.g., vertical thickness h)")
    gsub = geom.add_subparsers(dest="geom_cmd", required=True)

    s4g = gsub.add_parser("s4g-hcat", help="Assemble dataset/geometry/h_catalog.csv from S4G edge-on catalogs")
    s4g.add_argument("--sparc", default="dataset/sparc_tidy.csv", help="Path to tidy SPARC CSV (for distances & name map)")
    s4g.add_argument("--out", default="dataset/geometry/h_catalog.csv", help="Output h-catalog CSV path")
    s4g.add_argument("--prefer", choices=["thin","thick"], default="thin", help="Prefer thin or thick disk thickness")
    s4g.add_argument("--default-rel-err", type=float, default=0.25, help="Fallback relative error when no error column")
    # 明確 IDs、鏡像、逾時、重試、verbose
    s4g.add_argument("--ids", nargs="*", default=None, help="Explicit VizieR catalog IDs (e.g. J/A+A/548/A126)")
    s4g.add_argument("--mirror", default=None, help="Preferred VizieR mirror host (e.g. vizier.cfa.harvard.edu)")
    s4g.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout (seconds) per request")
    s4g.add_argument("--retries", type=int, default=2, help="Retries per catalog per mirror")
    s4g.add_argument("--verbose", action="store_true")

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
    exp = sub.add_parser("exp", help="Supplementary experiments & robustness")
    sx = exp.add_subparsers(dest="exp_cmd", required=True)

    # ppc
    ppc = sx.add_parser("ppc", help="Posterior(-like) predictive check on an existing results dir")
    ppc.add_argument("--results", required=True, help="Path to a finished results dir")
    ppc.add_argument("--data",   default="dataset/sparc_tidy.csv")
    ppc.add_argument("--prefix", default="ppc")

    # stress
    pst = sx.add_parser("stress", help="Inflate i_err/D_err and re-fit")
    pst.add_argument("--data", default="dataset/sparc_tidy.csv")
    pst.add_argument("--outroot", default="results/stress")
    pst.add_argument("--model", choices=["ptq","ptq-nu","ptq-screen","ptq-split","baryon","mond","nfw1p"], default="ptq-screen")
    pst.add_argument("--scale-i", type=float, default=2.0)
    pst.add_argument("--scale-D", type=float, default=2.0)
    pst.add_argument("--prior", choices=["galaxies-only","planck-anchored"], default="galaxies-only")
    pst.add_argument("--sigma-sys", type=float, default=4.0)
    pst.add_argument("--H0-kms-mpc", type=float, default=None)
    pst.add_argument("--nwalkers", default="4x")
    pst.add_argument("--steps", type=int, default=12000)
    pst.add_argument("--seed", type=int, default=42)
    pst.add_argument("--likelihood", choices=["gauss","t"], default="gauss")
    pst.add_argument("--t-dof", type=float, default=8.0)

    # mask
    pmk = sx.add_parser("mask", help="Mask inner radii r<rmin_kpc and re-fit")
    pmk.add_argument("--data", default="dataset/sparc_tidy.csv")
    pmk.add_argument("--outroot", default="results/mask")
    pmk.add_argument("--model", choices=["ptq","ptq-nu","ptq-screen","ptq-split","baryon","mond","nfw1p"], default="ptq-screen")
    pmk.add_argument("--rmin-kpc", type=float, default=2.0)
    pmk.add_argument("--prior", choices=["galaxies-only","planck-anchored"], default="galaxies-only")
    pmk.add_argument("--sigma-sys", type=float, default=4.0)
    pmk.add_argument("--H0-kms-mpc", type=float, default=None)
    pmk.add_argument("--nwalkers", default="4x")
    pmk.add_argument("--steps", type=int, default=12000)
    pmk.add_argument("--seed", type=int, default=42)
    pmk.add_argument("--likelihood", choices=["gauss","t"], default="gauss")
    pmk.add_argument("--t-dof", type=float, default=8.0)

    # H0
    ph0 = sx.add_parser("H0", help="Scan H0 sensitivity")
    ph0.add_argument("--data", default="dataset/sparc_tidy.csv")
    ph0.add_argument("--outroot", default="results/H0_scan")
    ph0.add_argument("--model", choices=["ptq","ptq-nu","ptq-screen"], default="ptq-screen")
    ph0.add_argument("--H0-list", type=float, nargs="+", default=[60.0, 67.4, 70.0, 73.0, 76.0])
    ph0.add_argument("--prior", choices=["galaxies-only","planck-anchored"], default="galaxies-only")
    ph0.add_argument("--sigma-sys", type=float, default=4.0)
    ph0.add_argument("--nwalkers", default="4x")
    ph0.add_argument("--steps", type=int, default=12000)
    ph0.add_argument("--seed", type=int, default=42)
    ph0.add_argument("--likelihood", choices=["gauss","t"], default="gauss")
    ph0.add_argument("--t-dof", type=float, default=8.0)

    # plateau
    ppl = sx.add_parser("plateau", help="Make stacked residual-acceleration plateau plot from a results dir")
    ppl.add_argument("--results", required=True)
    ppl.add_argument("--data", default="dataset/sparc_tidy.csv")
    ppl.add_argument("--nbins", type=int, default=24)
    ppl.add_argument("--prefix", default="plateau")

    # closure
    pcl = sx.add_parser("closure", help="Cross-scale closure test: epsilon_cos vs epsilon_RC")
    pcl.add_argument("--results", required=True)
    group = pcl.add_mutually_exclusive_group(required=True)
    group.add_argument("--epsilon-cos", type=float, default=None)
    group.add_argument("--omega-lambda", type=float, default=None)

    # kappa-gal
    kgal = sx.add_parser("kappa-gal", help="Per-galaxy kappa check")
    kgal.add_argument("--results", required=True)
    kgal.add_argument("--data", default="dataset/sparc_tidy.csv")
    kgal.add_argument("--eta", type=float, default=0.15)
    kgal.add_argument("--frac-vmax", type=float, default=0.9)
    kgal.add_argument("--nsamp", type=int, default=300)
    kgal.add_argument("--prefix", default="kappa_gal")
    kgal.add_argument("--y-source", choices=["model","obs","obs-debias"], default="model")
    kgal.add_argument("--eps-norm", choices=["fit","cos"], default="fit")
    kgal.add_argument("--rstar-from", choices=["model","obs"], default="model")
    # regression + lambda + r* interpolation
    kgal.add_argument("--regression", choices=["ols","deming"], default="deming")
    kgal.add_argument("--deming-lambda", type=float, default=1.0)
    kgal.add_argument("--no-interp-rstar", action="store_false", dest="interp_rstar",
                      help="Disable linear interpolation for r* (default: enabled)")
    kgal.set_defaults(interp_rstar=True)
    grp_k = kgal.add_mutually_exclusive_group(required=False)
    grp_k.add_argument("--epsilon-cos", type=float, default=None)
    grp_k.add_argument("--omega-lambda", type=float, default=None)

    # kappa-prof
    kpro = sx.add_parser("kappa-prof", help="Radius-resolved stacked kappa profile")
    kpro.add_argument("--results", required=True)
    kpro.add_argument("--data", default="dataset/sparc_tidy.csv")
    kpro.add_argument("--eta", type=float, default=0.15)
    kpro.add_argument("--nbins", type=int, default=24)
    kpro.add_argument("--min-per-bin", type=int, default=20)
    kpro.add_argument("--x-kind", choices=["r_over_Rd","r_kpc"], default="r_over_Rd")
    kpro.add_argument("--eps-norm", choices=["fit","cos"], default="fit")
    kpro.add_argument("--prefix", default="kappa_profile")
    grp_k2 = kpro.add_mutually_exclusive_group(required=False)
    grp_k2.add_argument("--epsilon-cos", type=float, default=None)
    grp_k2.add_argument("--omega-lambda", type=float, default=None)

    # kappa-fit
    kfit = sx.add_parser("kappa-fit", help="Fit y=A/x+B from kappa_profile outputs")
    kfit.add_argument("--results", required=True)
    kfit.add_argument("--prefix", default="kappa_profile")
    kfit.add_argument("--eps-norm", choices=["fit","cos"], default="cos")
    grp_k3 = kfit.add_mutually_exclusive_group(required=False)
    grp_k3.add_argument("--epsilon-cos", type=float, default=None)
    grp_k3.add_argument("--omega-lambda", type=float, default=None)
    kfit.add_argument("--bootstrap", type=int, default=0, help="N bootstrap draws; 0 to skip")
    kfit.add_argument("--min-per-bin", type=int, default=20)
    kfit.add_argument("--seed", type=int, default=1234)

    # zprof
    zp = sx.add_parser("zprof", help="Radius-resolved zero-parameter collapse in z=g_N/a0")
    zp.add_argument("--results", required=True)
    zp.add_argument("--data", default="dataset/sparc_tidy.csv")
    zp.add_argument("--nbins", type=int, default=24)
    zp.add_argument("--min-per-bin", type=int, default=20)
    zp.add_argument("--eps-norm", choices=["fit","cos"], default="cos")
    zp.add_argument("--prefix", default="z_profile")
    zp.add_argument("--z-qlo", type=float, default=0.01)
    zp.add_argument("--z-qhi", type=float, default=0.99)
    zp.add_argument("--no-theory", action="store_true")
    grp_z1 = zp.add_mutually_exclusive_group(required=False)
    grp_z1.add_argument("--epsilon-cos", type=float, default=None)
    grp_z1.add_argument("--omega-lambda", type=float, default=None)

    # zgal
    zg = sx.add_parser("zgal", help="Per-galaxy single-point at r*: z=g_N/a0 vs y=eps_eff/eps_den")
    zg.add_argument("--results", required=True)
    zg.add_argument("--data", default="dataset/sparc_tidy.csv")
    zg.add_argument("--frac-vmax", type=float, default=0.9)
    zg.add_argument("--y-source", choices=["model","obs","obs-debias"], default="obs-debias")
    zg.add_argument("--rstar-from", choices=["model","obs"], default="obs")
    zg.add_argument("--eps-norm", choices=["fit","cos"], default="cos")
    zg.add_argument("--nsamp", type=int, default=300)
    zg.add_argument("--prefix", default="z_gal")
    zg.add_argument("--no-interp-rstar", action="store_false", dest="interp_rstar",
                    help="Disable linear interpolation for r* (default: enabled)")
    zg.set_defaults(interp_rstar=True)
    grp_z2 = zg.add_mutually_exclusive_group(required=False)
    grp_z2.add_argument("--epsilon-cos", type=float, default=None)
    grp_z2.add_argument("--omega-lambda", type=float, default=None)

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
            f"--sigma-sys={args.sigma_sys}",
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

    elif args.cmd == "geom":
        if args.geom_cmd == "s4g-hcat":
            from .geometry import build_s4g_h_catalog
            df = build_s4g_h_catalog(
                sparc_tidy_csv=args.sparc,
                out_csv=args.out,
                prefer=args.prefer,
                default_rel_err=args.default_rel_err,
                ids=args.ids,
                mirror=args.mirror,
                timeout_sec=args.timeout,
                retries=args.retries,
                verbose=args.verbose,
            )
            print(f"Saved {args.out}  (N={len(df)})")
            print(df.head(10).to_string(index=False))

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
                y_source=args.y_source, eps_norm=args.eps_norm,
                rstar_from=args.rstar_from,
                regression=args.regression, deming_lambda=args.deming_lambda,
                interpolate_rstar=args.interp_rstar,
                out_prefix=args.prefix
            )
            print(json.dumps(out, indent=2))

        elif args.exp_cmd == "kappa-prof":
            df, png = EXP.kappa_radius_resolved(
                results_dir=args.results, data_path=args.data,
                eta=args.eta, epsilon_cos=args.epsilon_cos, omega_lambda=args.omega_lambda,
                nbins=args.nbins, min_per_bin=args.min_per_bin,
                x_kind=args.x_kind, eps_norm=args.eps_norm, out_prefix=args.prefix
            )
            print(f"Saved: {png}")

        elif args.exp_cmd == "kappa-fit":
            out = EXP.kappa_two_param_fit(
                results_dir=args.results, prefix=args.prefix,
                eps_norm=args.eps_norm,
                epsilon_cos=args.epsilon_cos, omega_lambda=args.omega_lambda
            )
            print(json.dumps(out, indent=2))
            if args.bootstrap and args.bootstrap > 0:
                boot = EXP.kappa_two_param_bootstrap(
                    results_dir=args.results, prefix=args.prefix,
                    eps_norm=args.eps_norm,
                    epsilon_cos=args.epsilon_cos, omega_lambda=args.omega_lambda,
                    n_boot=args.bootstrap, min_per_bin=args.min_per_bin, seed=args.seed
                )
                print(json.dumps(boot, indent=2))

        elif args.exp_cmd == "zprof":
            df, png = EXP.z_profile(
                results_dir=args.results, data_path=args.data,
                nbins=args.nbins, min_per_bin=args.min_per_bin,
                eps_norm=args.eps_norm,
                epsilon_cos=args.epsilon_cos, omega_lambda=args.omega_lambda,
                out_prefix=args.prefix,
                z_quantile_clip=(args.z_qlo, args.z_qhi),
                do_theory=(not args.no_theory)
            )
            print(f"Saved: {png}")

        elif args.exp_cmd == "zgal":
            out = EXP.z_per_galaxy(
                results_dir=args.results, data_path=args.data,
                frac_vmax=args.frac_vmax, y_source=args.y_source,
                rstar_from=args.rstar_from, eps_norm=args.eps_norm,
                epsilon_cos=args.epsilon_cos, omega_lambda=args.omega_lambda,
                nsamp=args.nsamp, interpolate_rstar=args.interp_rstar,
                out_prefix=args.prefix
            )
            print(json.dumps(out, indent=2))
