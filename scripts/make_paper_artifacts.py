#!/usr/bin/env python3
# scripts/make_paper_artifacts.py
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import sys
import json
import yaml
import pandas as pd

# 直接呼叫內部模組，避免子程序「靜默成功但未寫檔」
from ptquat.fit_global import run as run_fit_direct
from ptquat import experiments as EXP


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _run_fit(model: str, tidy_csv: Path, outdir: Path,
             likelihood: str, fast: bool, prior: str = "galaxies-only") -> Path:
    """直接呼叫 ptquat.fit_global.run 並檢查 global_summary.yaml 存在"""
    _ensure_dir(outdir)

    steps = "200" if fast else "6000"
    nwalk = "2x" if fast else "4x"

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
        raise RuntimeError(
            f"[make] {model} fit finished but {gpath} not found. "
            "Please check stdout above for any warnings/exceptions."
        )
    return gpath


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copyfile(src, dst)


def build_compare_csv(run_root: Path, out_csv: Path, models: list[str], like: str) -> pd.DataFrame:
    rows = []
    for m in models:
        rdir = run_root / f"{m}_{like}"
        gpath = rdir / "global_summary.yaml"
        if not gpath.exists():
            print(f"[make][warn] skip {m}: {gpath} not found", flush=True)
            continue
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
    """
    在 results_dir 上跑 plateau / ppc / kappa-gal / kappa-prof / closure，
    並將圖複製到 figdir。
    """
    # plateau
    _, plateau_png = EXP.residual_plateau(str(results_dir), str(tidy_csv), nbins=nbins, out_prefix="plateau")
    # ppc
    _ = EXP.ppc_check(str(results_dir), str(tidy_csv), out_prefix="ppc")
    # kappa-gal
    summ_gal = EXP.kappa_per_galaxy(
        results_dir=str(results_dir), data_path=str(tidy_csv),
        eta=eta, frac_vmax=0.9, nsamp=300, omega_lambda=omega_lambda,
        out_prefix="kappa_gal"
    )
    # kappa-prof
    _, kprof_png = EXP.kappa_radius_resolved(
        results_dir=str(results_dir), data_path=str(tidy_csv),
        eta=eta, omega_lambda=omega_lambda, nbins=nbins, min_per_bin=10,
        x_kind="r_over_Rd", out_prefix="kappa_profile"
    )
    # closure
    _ = EXP.closure_test(str(results_dir), omega_lambda=omega_lambda)

    # 將主圖複製到 figdir
    _ensure_dir(figdir)
    _copy_if_exists(plateau_png, figdir / f"plateau_{results_dir.name}.png")
    _copy_if_exists(Path(results_dir / "kappa_gal.png"), figdir / f"kappa_gal_{results_dir.name}.png")
    _copy_if_exists(kprof_png, figdir / f"kappa_profile_{results_dir.name}.png")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build EJPC artifacts: quick fits, diagnostics, and model-compare CSV."
    )
    ap.add_argument("--data", required=True, help="Tidy SPARC CSV (after preprocess).")
    ap.add_argument("--out", required=True, help="Root output directory (will contain per-model results).")
    ap.add_argument("--figdir", required=True, help="Directory to collect main figures.")
    ap.add_argument("--models", nargs="+", required=True,
                    help="Models to run, e.g. baryon ptq-screen mond nfw1p ...")
    ap.add_argument("--likelihood", choices=["gauss", "t"], default="gauss")
    ap.add_argument("--fast", action="store_true", help="Use small steps/walkers for smoke run.")
    ap.add_argument("--skip-fetch", action="store_true", help="Kept for compatibility; ignored here.")
    ap.add_argument("--omega-lambda", type=float, default=0.685, help="Closure test anchor (ΩΛ).")
    ap.add_argument("--eta", type=float, default=0.15, help="η used in kappa checks.")
    ap.add_argument("--nbins", type=int, default=24, help="Binning for plateau / kappa-profile.")
    return ap.parse_args()


def main():
    args = parse_args()
    tidy_csv = Path(args.data)
    run_root = Path(args.out)
    figdir   = Path(args.figdir)
    like     = args.likelihood

    _ensure_dir(run_root)
    _ensure_dir(figdir)

    # 1) 逐模型擬合
    results_dirs: dict[str, Path] = {}
    for m in args.models:
        outdir = run_root / f"{m}_{like}"
        _run_fit(model=m, tidy_csv=tidy_csv, outdir=outdir,
                 likelihood=like, fast=args.fast, prior="galaxies-only")
        results_dirs[m] = outdir

    # 2) 跑診斷：優先對 ptq-screen（若在列表中），否則用最後一個模型
    diag_model = "ptq-screen" if "ptq-screen" in results_dirs else list(results_dirs.keys())[-1]
    try:
        run_diagnostics_for(results_dirs[diag_model], tidy_csv, figdir,
                            omega_lambda=args.omega_lambda, eta=args.eta, nbins=args.nbins)
    except Exception as e:
        print(f"[make][warn] diagnostics for {diag_model} failed: {e}", flush=True)

    # 3) 匯出比較表
    cmp_csv = run_root / "ejpc_model_compare.csv"
    df = build_compare_csv(run_root, cmp_csv, list(results_dirs.keys()), like)
    print(df.to_string(index=False))

    print(f"\n[make] Done. Compare CSV: {cmp_csv}")
    print(f"[make] Figures collected in: {figdir}")


if __name__ == "__main__":
    main()
