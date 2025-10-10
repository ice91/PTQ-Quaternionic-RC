#!/usr/bin/env python3
# scripts/make_paper_artifacts.py
from __future__ import annotations
import argparse, subprocess, json, shutil, sys
from pathlib import Path
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

def sh(*cmd, cwd: Path | None = None):
    print("+", " ".join(map(str, cmd))); sys.stdout.flush()
    subprocess.check_call(list(map(str, cmd)), cwd=cwd or ROOT)

def ensure_data(tidy_path: Path, raw_dir: Path, skip_fetch: bool):
    if tidy_path.exists():
        return
    if skip_fetch:
        raise SystemExit(f"[make] --skip-fetch 開啟，但找不到 {tidy_path}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    sh("python", "-m", "ptquat.cli", "fetch", "--out", str(raw_dir))
    sh("python", "-m", "ptquat.cli", "preprocess", "--raw", str(raw_dir), "--out", str(tidy_path))

def run_model(model: str, tidy: Path, outdir: Path, like: str, fast: bool, prior: str = "galaxies-only"):
    steps = "6000"; nwalk = "4x"
    if fast:
        steps = "200"; nwalk = "2x"
    args = [
        "python","-m","ptquat.cli","fit",
        "--data", str(tidy),
        "--outdir", str(outdir),
        "--model", model,
        "--likelihood", like,
        "--steps", steps,
        "--nwalkers", nwalk,
        "--seed","42",
        "--prior", prior,
        "--sigma-sys","4.0",
    ]
    sh(*args)

def copy_all(patterns, src_dir: Path, dst_dir: Path):
    for pat in patterns:
        for p in src_dir.glob(pat):
            if p.is_file():
                shutil.copy(p, dst_dir / p.name)

def main():
    ap = argparse.ArgumentParser(description="Reproduce paper artifacts (figures & tables) with one command.")
    ap.add_argument("--data", default=str(ROOT/"dataset/sparc_tidy.csv"), help="Path to tidy SPARC CSV.")
    ap.add_argument("--raw", default=str(ROOT/"dataset/raw"), help="Directory for VizieR raw tables.")
    ap.add_argument("--out", default=str(ROOT/"results/ejpc_run"), help="Root output directory for runs.")
    ap.add_argument("--figdir", default=str(ROOT/"paper_figs"), help="Directory to collect figures.")
    ap.add_argument("--models", nargs="+", default=["baryon","mond","nfw1p","ptq","ptq-nu","ptq-screen"],
                    help="Models to run.")
    ap.add_argument("--like", choices=["gauss","t"], default="gauss")
    ap.add_argument("--fast", action="store_true", help="Fast mode (few steps, fewer walkers).")
    ap.add_argument("--skip-fetch", action="store_true", help="Do not fetch if dataset missing.")
    ap.add_argument("--omega-lambda", type=float, default=0.685, help="For closure test if used.")
    args = ap.parse_args()

    TIDY = Path(args.data); RAW = Path(args.raw); OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
    FIGS = Path(args.figdir); FIGS.mkdir(parents=True, exist_ok=True)

    # 0) data
    ensure_data(TIDY, RAW, skip_fetch=args.skip_fetch)

    # 1) run models
    run_dirs = {}
    for m in args.models:
        outdir = OUT / f"{m}_{args.like}"
        outdir.mkdir(parents=True, exist_ok=True)
        run_model(m, TIDY, outdir, like=args.like, fast=args.fast)
        run_dirs[m] = outdir

    # 2) diagnostics on ptq-screen if present; else用 MOND
    diag_key = "ptq-screen" if "ptq-screen" in run_dirs else ("mond" if "mond" in run_dirs else list(run_dirs)[0])
    rdir = run_dirs[diag_key]
    # plateau
    sh("python","-m","ptquat.cli","exp","plateau","--results",str(rdir),"--data",str(TIDY),"--nbins","24","--prefix","plateau")
    # ppc
    sh("python","-m","ptquat.cli","exp","ppc","--results",str(rdir),"--data",str(TIDY),"--prefix","ppc")
    # kappa (if PTQ-family; 若非 PTQ 仍可跑，會以當前模型的殘差定義 eps_eff)
    sh("python","-m","ptquat.cli","exp","kappa-gal","--results",str(rdir),"--data",str(TIDY),"--eta","0.15","--prefix","kappa_gal")
    sh("python","-m","ptquat.cli","exp","kappa-prof","--results",str(rdir),"--data",str(TIDY),"--eta","0.15","--nbins","24","--prefix","kappa_profile")
    # closure（僅 PTQ 家族含 epsilon；若沒有 epsilon，會失敗，故保護性包起來）
    try:
        sh("python","-m","ptquat.cli","exp","closure","--results",str(rdir),"--omega-lambda",str(args.omega_lambda))
    except subprocess.CalledProcessError:
        print(f"[make] closure skipped for model={diag_key} (non-PTQ family).")

    # 3) model comparison table
    rows = []
    for m, d in run_dirs.items():
        gpath = d / "global_summary.yaml"
        y = yaml.safe_load(gpath.read_text())
        rows.append({
            "model": y["model"],
            "k": y["k_parameters"],
            "N": y["N_total"],
            "chi2_total": y["chi2_total"],
            "logL_full": y["logL_total_full"],
            "AIC_full": y["AIC_full"],
            "BIC_full": y["BIC_full"],
            "AIC_quad": y["AIC_quad"],
            "BIC_quad": y["BIC_quad"],
            "results_dir": str(d),
        })
    df = pd.DataFrame(rows).sort_values("BIC_full")
    df.to_csv(OUT/"ejpc_model_compare.csv", index=False)

    # 4) closure table (if any)
    cpath = rdir / "closure_test.yaml"
    if cpath.exists():
        c = yaml.safe_load(cpath.read_text())
        pd.DataFrame([c]).to_csv(OUT/"closure_table.csv", index=False)

    # 5) collect figures
    copy_all(["plateau*.png", "kappa_*.png", "ppc_coverage.json"], rdir, FIGS)
    # copy a few per-galaxy plots
    count = 0
    for p in sorted(rdir.glob("plot_*.png")):
        shutil.copy(p, FIGS / p.name)
        count += 1
        if count >= 6: break

    print("\n[make] Done. Artifacts:")
    print(" -", OUT/"ejpc_model_compare.csv")
    if (OUT/"closure_table.csv").exists():
        print(" -", OUT/"closure_table.csv")
    print(" - figures in", FIGS)

if __name__ == "__main__":
    main()
