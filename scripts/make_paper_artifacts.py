#!/usr/bin/env python3
# scripts/make_paper_artifacts.py
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import json
import yaml
import pandas as pd

from ptquat.fit_global import run as run_fit_direct
from ptquat import experiments as EXP


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copyfile(src, dst)

def _run_fit(model: str, tidy_csv: Path, outdir: Path,
             likelihood: str, fast: bool, prior: str = "galaxies-only") -> Path:
    _ensure_dir(outdir)
    steps = "6000" if not fast else "200"
    nwalk = "4x"   if not fast else "2x"
    argv = [
        f"--data-path={tidy_csv}",
        f"--outdir={outdir}",
        f"--model={model}",
        f"--likelihood={likelihood}",
        f"--steps={steps}",
        f"--nwalkers={nwalk}",
        f"--seed=42",
        f"--prior={prior}",
        f"--sigma-sys=4.0",
    ]
    print("+ ptquat.fit_global.run", " ".join(argv), flush=True)
    run_fit_direct(argv)
    gpath = outdir / "global_summary.yaml"
    if not gpath.exists():
        raise RuntimeError(f"[make] {model} fit finished but {gpath} not found.")
    return gpath

def build_compare_csv(run_root: Path, out_csv: Path, models: list[str], like: str) -> pd.DataFrame:
    rows = []
    for m in models:
        rdir = run_root / f"{m}_{like}"
        gpath = rdir / "global_summary.yaml"
        if not gpath.exists():
            print(f"[make][warn] skip {m}: {gpath} not found", flush=True); continue
        y = yaml.safe_load(gpath.read_text())
        rows.append(dict(
            model=y.get("model", m),
            likelihood=y.get("likelihood", like),
            AIC_full=y.get("AIC_full"),
            BIC_full=y.get("BIC_full"),
            chi2_total=y.get("chi2_total"),
            k_parameters=y.get("k_parameters"),
            N_total=y.get("N_total"),
            outdir=str(rdir)
        ))
    df = pd.DataFrame(rows).sort_values("AIC_full")
    _ensure_dir(out_csv.parent)
    df.to_csv(out_csv, index=False)
    return df

def run_diagnostics_for(results_dir: Path,
                        tidy_csv: Path,
                        figdir: Path,
                        omega_lambda: float,
                        eta: float,
                        nbins: int) -> None:
    """在 results_dir 上跑 plateau/ppc/closure + kappa (fit & cos) 圖與擬合，並拷貝主圖到 figdir。"""
    # 1) plateau
    _, plateau_png = EXP.residual_plateau(str(results_dir), str(tidy_csv), nbins=nbins, out_prefix="plateau")
    # 2) PPC coverage
    ppc_json = EXP.ppc_check(str(results_dir), str(tidy_csv), out_prefix="ppc")
    with open(results_dir/"ppc_coverage.json","w") as f: json.dump(ppc_json, f, indent=2)
    # 3) closure
    cl = EXP.closure_test(str(results_dir), omega_lambda=omega_lambda)
    with open(results_dir/"closure_test.json","w") as f: json.dump(cl, f, indent=2)

    # 4) kappa-profile（fit 與 cos）
    for eps_norm, prefix in [("fit", "kappa_profile_A_fit"), ("cos","kappa_profile_A_cos")]:
        _, png = EXP.kappa_radius_resolved(
            results_dir=str(results_dir), data_path=str(tidy_csv),
            eta=eta, omega_lambda=(omega_lambda if eps_norm=="cos" else None),
            nbins=nbins, min_per_bin=20,
            x_kind="r_over_Rd", eps_norm=eps_norm, out_prefix=prefix
        )
        _copy_if_exists(png, figdir / f"{prefix}_{results_dir.name}.png")
        # y=A/x+B 擬合 + bootstrap
        fit_sum = EXP.kappa_profile_fit(
            results_dir=str(results_dir), prefix=prefix, eps_norm=eps_norm,
            omega_lambda=(omega_lambda if eps_norm=="cos" else None), bootstrap=2000, seed=1234
        )
        with open(results_dir/f"{prefix}_fit_summary.json","w") as f: json.dump(fit_sum, f, indent=2)

    # 5) per-galaxy：obs-debias + cos（主要論文本體），另附 model + fit（補充）
    _ = EXP.kappa_per_galaxy(
        results_dir=str(results_dir), data_path=str(tidy_csv),
        eta=eta, frac_vmax=0.9, nsamp=300, omega_lambda=omega_lambda,
        y_source="obs-debias", eps_norm="cos", rstar_from="obs",
        out_prefix="kappa_gal_obsdebiased_cos"
    )
    _copy_if_exists(results_dir/"kappa_gal_obsdebiased_cos.png", figdir / f"kappa_gal_obsdebiased_cos_{results_dir.name}.png")

    _ = EXP.kappa_per_galaxy(
        results_dir=str(results_dir), data_path=str(tidy_csv),
        eta=eta, frac_vmax=0.9, nsamp=300,
        y_source="model", eps_norm="fit", rstar_from="model",
        out_prefix="kappa_gal_model_fit"
    )
    _copy_if_exists(results_dir/"kappa_gal_model_fit.png", figdir / f"kappa_gal_model_fit_{results_dir.name}.png")

    # 6) 拷貝主圖
    _copy_if_exists(plateau_png, figdir / f"plateau_{results_dir.name}.png")

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build EJPC artifacts: fits, diagnostics, κ-profiles (fit & cos), and model-compare.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--figdir", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--likelihood", choices=["gauss","t"], default="gauss")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--omega-lambda", type=float, default=0.685)
    ap.add_argument("--eta", type=float, default=0.15)
    ap.add_argument("--nbins", type=int, default=24)
    return ap.parse_args()

def main():
    args = parse_args()
    tidy_csv = Path(args.data)
    run_root = Path(args.out)
    figdir   = Path(args.figdir)
    like     = args.likelihood
    _ensure_dir(run_root); _ensure_dir(figdir)

    # 1) 擬合各模型
    results_dirs: dict[str, Path] = {}
    for m in args.models:
        outdir = run_root / f"{m}_{like}"
        _run_fit(model=m, tidy_csv=tidy_csv, outdir=outdir, likelihood=like, fast=args.fast, prior="galaxies-only")
        results_dirs[m] = outdir

    # 2) 診斷（優先 ptq-screen）
    diag_model = "ptq-screen" if "ptq-screen" in results_dirs else list(results_dirs.keys())[-1]
    run_diagnostics_for(results_dirs[diag_model], tidy_csv, figdir,
                        omega_lambda=args.omega_lambda, eta=args.eta, nbins=args.nbins)

    # 3) 模型比較表
    cmp_csv = run_root / "ejpc_model_compare.csv"
    df = build_compare_csv(run_root, cmp_csv, list(results_dirs.keys()), like)
    print(df.to_string(index=False))
    print(f"\n[make] Compare CSV: {cmp_csv}")
    print(f"[make] Figures collected in: {figdir}")

if __name__ == "__main__":
    main()
