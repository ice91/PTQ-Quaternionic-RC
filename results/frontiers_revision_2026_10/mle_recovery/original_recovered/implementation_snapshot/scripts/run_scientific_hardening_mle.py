#!/usr/bin/env python
"""R1B scientific-hardening true-MLE runner.

Writes ONLY under results/scientific_hardening_r1/01_mle/
Never touches historical result trees.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

# Ensure src authority when executed as a script
IMPL_ROOT = Path(__file__).resolve().parents[1]
SRC = IMPL_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ptquat.constants import C_LIGHT, H0_SI  # noqa: E402
from ptquat.data import load_tidy_sparc  # noqa: E402
from ptquat.fit_global import _layout  # noqa: E402
from ptquat.mle import (  # noqa: E402
    a0_to_epsilon,
    build_multistart,
    epsilon_to_a0,
    evaluate_loglike,
    load_historical_median_theta,
    map_theta_mond_to_ptqnu,
    map_theta_mondscreen_to_ptqscreen,
    map_theta_ptqnu_to_mond,
    map_theta_ptqscreen_to_mondscreen,
    matched_epsilon_bounds,
    model_dirname,
    optimize_mle,
    parameter_domain_manifest,
    write_model_outputs,
)

MODELS = ["baryon", "mond", "ptq", "ptq-nu", "ptq-screen", "mond-screen", "nfw1p"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="R1B true MLE reconstruction")
    ap.add_argument("--data", default=str(IMPL_ROOT / "dataset" / "sparc_tidy.csv"))
    ap.add_argument(
        "--outdir",
        default=str(IMPL_ROOT / "results" / "scientific_hardening_r1" / "01_mle"),
    )
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--maxiter", type=int, default=20, help="Block-coordinate outer iterations")
    ap.add_argument("--n-perturb", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    gdict = load_tidy_sparc(args.data)
    galaxies = [gdict[k] for k in sorted(gdict.keys())]
    G = len(galaxies)
    N_points = int(sum(len(g.r_kpc) for g in galaxies))
    print(f"[R1B] N_gal={G} N_points={N_points}")

    manifest = parameter_domain_manifest(H0_SI)
    (outdir / "parameter_domain_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    compare_rows = []
    hist_rows = []
    results = {}

    for model in args.models:
        print(f"[R1B] optimizing {model} ...", flush=True)
        _, hist_meta = load_historical_median_theta(IMPL_ROOT, model, galaxies, H0_SI)
        starts = build_multistart(
            model, galaxies, IMPL_ROOT, H0_SI, n_perturb=args.n_perturb, seed=args.seed
        )
        # NFW: allow more outer iterations
        maxiter = args.maxiter if model != "nfw1p" else max(args.maxiter, 12)
        res = optimize_mle(
            model, galaxies, starts, H0_si=H0_SI, domain="matched", maxiter=maxiter
        )
        mdir = outdir / model_dirname(model)
        write_model_outputs(mdir, res, hist_meta)
        results[model] = res

        compare_rows.append(
            {
                "model": model,
                "max_loglike": res["max_loglike"],
                "AIC": res["AIC"],
                "BIC_NPOINTS": res["BIC_NPOINTS"],
                "k": res["k"],
                "N_data": res["N_data"],
                "N_gal": res["N_gal"],
                "optimizer": res["optimizer"],
                "converged": res["converged"],
                "termination_status": res["termination_status"],
                "best_start": res["best_start"],
                "cross_start_delta_logL": res["cross_start_delta_logL"],
                "n_near_boundary": res["boundary"]["n_near_boundary"],
                "eps": res["parameter_vector"].get("eps"),
                "a0": res["parameter_vector"].get("a0"),
                "q": res["parameter_vector"].get("q"),
                "sigma_sys": res["parameter_vector"].get("sigma_sys"),
            }
        )
        hist_rows.append(
            {
                "model": model,
                "historical_posterior_median_logL": hist_meta["historical_logL_total_full"],
                "historical_posterior_median_AIC": hist_meta["historical_AIC_full"],
                "true_MLE_logL": res["max_loglike"],
                "true_MLE_AIC": res["AIC"],
                "delta_logL": res["max_loglike"] - hist_meta["historical_logL_total_full"],
                "delta_AIC": res["AIC"] - hist_meta["historical_AIC_full"],
                "historical_source": hist_meta["source_dir"],
            }
        )
        print(
            f"[R1B] {model}: max_logL={res['max_loglike']:.6f} AIC={res['AIC']:.3f} "
            f"best_start={res['best_start']} crossΔ={res['cross_start_delta_logL']:.3e}",
            flush=True,
        )

    pd.DataFrame(compare_rows).to_csv(outdir / "model_compare_mle.csv", index=False)
    pd.DataFrame(hist_rows).to_csv(outdir / "historical_vs_true_mle.csv", index=False)

    # Equivalence pairs
    eq_rows = []
    # PTQ-nu / MOND
    if "ptq-nu" in results and "mond" in results:
        dlog = results["ptq-nu"]["max_loglike"] - results["mond"]["max_loglike"]
        # also evaluate mapped thetas
        th_m = results["mond"]["theta"]
        th_p_from_m = map_theta_mond_to_ptqnu(th_m, G, H0_SI)
        ll_map = evaluate_loglike(th_p_from_m, galaxies, "ptq-nu", H0_SI)
        th_p = results["ptq-nu"]["theta"]
        th_m_from_p = map_theta_ptqnu_to_mond(th_p, G, H0_SI)
        ll_map2 = evaluate_loglike(th_m_from_p, galaxies, "mond", H0_SI)
        eq_rows.append(
            {
                "pair": "ptq-nu_vs_mond",
                "deterministic_identity": True,
                "delta_max_logL": dlog,
                "delta_AIC": results["ptq-nu"]["AIC"] - results["mond"]["AIC"],
                "delta_BIC_NPOINTS": results["ptq-nu"]["BIC_NPOINTS"] - results["mond"]["BIC_NPOINTS"],
                "mapped_mond_theta_on_ptqnu_logL": ll_map,
                "mapped_ptqnu_theta_on_mond_logL": ll_map2,
                "abs_delta_logL": abs(dlog),
                "pass_1e-5": abs(dlog) < 1e-5,
                "eps_lo_hi": matched_epsilon_bounds(H0_SI),
                "a0_mond": results["mond"]["parameter_vector"]["a0"],
                "eps_ptqnu": results["ptq-nu"]["parameter_vector"]["eps"],
                "a0_from_eps": epsilon_to_a0(results["ptq-nu"]["parameter_vector"]["eps"], H0_SI),
            }
        )

    if "ptq-screen" in results and "mond-screen" in results:
        dlog = results["ptq-screen"]["max_loglike"] - results["mond-screen"]["max_loglike"]
        th_m = results["mond-screen"]["theta"]
        th_p_from_m = map_theta_mondscreen_to_ptqscreen(th_m, G, H0_SI)
        ll_map = evaluate_loglike(th_p_from_m, galaxies, "ptq-screen", H0_SI)
        th_p = results["ptq-screen"]["theta"]
        th_m_from_p = map_theta_ptqscreen_to_mondscreen(th_p, G, H0_SI)
        ll_map2 = evaluate_loglike(th_m_from_p, galaxies, "mond-screen", H0_SI)
        eq_rows.append(
            {
                "pair": "ptq-screen_vs_mond-screen",
                "deterministic_identity": True,
                "delta_max_logL": dlog,
                "delta_AIC": results["ptq-screen"]["AIC"] - results["mond-screen"]["AIC"],
                "delta_BIC_NPOINTS": results["ptq-screen"]["BIC_NPOINTS"] - results["mond-screen"]["BIC_NPOINTS"],
                "mapped_mondscreen_theta_on_ptqscreen_logL": ll_map,
                "mapped_ptqscreen_theta_on_mondscreen_logL": ll_map2,
                "abs_delta_logL": abs(dlog),
                "pass_1e-5": abs(dlog) < 1e-5,
                "a0_mondscreen": results["mond-screen"]["parameter_vector"]["a0"],
                "eps_ptqscreen": results["ptq-screen"]["parameter_vector"]["eps"],
                "q_ptqscreen": results["ptq-screen"]["parameter_vector"]["q"],
                "q_mondscreen": results["mond-screen"]["parameter_vector"]["q"],
            }
        )

    pd.DataFrame(eq_rows).to_csv(outdir / "equivalence_pairs.csv", index=False)

    # Manifest
    lines = [
        "# MLE_MANIFEST",
        "",
        f"- IMPL_ROOT: `{IMPL_ROOT}`",
        f"- N_gal: {G}",
        f"- N_points: {N_points}",
        f"- H0_SI: {H0_SI}",
        f"- c*H0: {C_LIGHT * H0_SI}",
        f"- matched a0 bounds: {manifest['matched_a0_bounds_si']}",
        f"- matched epsilon bounds: {manifest['matched_epsilon_bounds']}",
        "",
        "## Models",
    ]
    for row in compare_rows:
        lines.append(
            f"- `{row['model']}`: max_logL={row['max_loglike']:.6f}, "
            f"AIC={row['AIC']:.3f}, BIC_NPOINTS={row['BIC_NPOINTS']:.3f}, k={row['k']}"
        )
    lines.append("")
    lines.append("## Equivalence")
    for row in eq_rows:
        lines.append(
            f"- `{row['pair']}`: abs(delta_logL)={row['abs_delta_logL']:.3e}, "
            f"pass_1e-5={row['pass_1e-5']}"
        )
    (outdir / "MLE_MANIFEST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[R1B] done", outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
