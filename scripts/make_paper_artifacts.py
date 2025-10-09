#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate paper-ready artifacts (tables + figures) from existing results.

Outputs (default into paper_outputs/):
  - model_compare.csv / .tex
  - fig_delta_AIC.png, fig_delta_BIC.png
  - fig_ppc_coverage.png (if ppc_coverage.json exists)
  - fig_h0_AIC.png, fig_h0_epsilon.png (if H0 scan CSV exists)
  - fig_kappa_gal.png (if kappa_gal CSV/JSON exist)
  - fig_kappa_profile.png (if kappa_profile_binned.csv exists)
  - stress_mask_summary.csv / .tex (auto-scan subdirs)
  - closure_summary.tex (if closure_test.yaml exists)

Usage examples
--------------
Default paths assumed by README (just run):
    python scripts/make_paper_artifacts.py

Custom model mapping:
    python scripts/make_paper_artifacts.py \
      --model PTQ-screen=results/ptq_screen \
      --model MOND=results/ejpc_mond \
      --model PTQ-v=results/ptq_nu \
      --model NFW-1p=results/ejpc_nfw1p \
      --model PTQ=results/ejpc_main \
      --model Baryon=results/ejpc_baryon

Note
----
This script only *reads* the outputs produced by `ptquat fit` / `ptquat exp ...`.
It does not run fits itself.
"""

from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt


# --------------------------
# Utilities
# --------------------------

def _ensure_outdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _load_yaml(p: Path) -> dict:
    with open(p, "r") as f:
        return yaml.safe_load(f)

def _safe(path: Path) -> bool:
    try:
        return path.exists()
    except Exception:
        return False

def _nice_model_order(names: List[str]) -> List[str]:
    # Preferred order for the paper
    key = {k:i for i, k in enumerate([
        "PTQ-screen","MOND","PTQ-v","NFW-1p","PTQ","Baryon"
    ])}
    return sorted(names, key=lambda x: key.get(x, 9999))

def _akaike_weights(delta_aic: np.ndarray) -> np.ndarray:
    # w_i = exp(-0.5 ΔAIC_i) / Σ_j exp(-0.5 ΔAIC_j)
    z = np.exp(-0.5 * np.asarray(delta_aic, float))
    s = np.sum(z)
    return z / s if s > 0 else np.full_like(z, np.nan, dtype=float)

def _fmt(x, nd=2):
    if x is None or (isinstance(x, float) and (not np.isfinite(x))):
        return ""
    return f"{x:.{nd}f}"

def _try_read_json(p: Path) -> Optional[dict]:
    if _safe(p):
        try:
            return json.load(open(p, "r"))
        except Exception:
            return None
    return None


# --------------------------
# Model comparison table + ΔIC plots
# --------------------------

def make_model_comparison(models_map: Dict[str, Path], outdir: Path) -> pd.DataFrame:
    rows = []
    for name, rdir in models_map.items():
        summ = _load_yaml(rdir / "global_summary.yaml")
        rows.append(dict(
            model=name,
            k=summ["k_parameters"],
            N=summ["N_total"],
            chi2=summ["chi2_total"],
            AIC_full=summ["AIC_full"],
            BIC_full=summ["BIC_full"],
            AIC_quad=summ.get("AIC_quad"),
            BIC_quad=summ.get("BIC_quad"),
            sigma_sys_med=summ.get("sigma_sys_median"),
            epsilon_med=summ.get("epsilon_median"),
            q_med=summ.get("q_median"),
            a0_med=summ.get("a0_median"),
            likelihood=summ.get("likelihood","gauss"),
        ))
    df = pd.DataFrame(rows).set_index("model")

    # Δ relative to best (minimum) for AIC_full/BIC_full
    for ic in ["AIC_full","BIC_full"]:
        m = float(df[ic].min())
        df[f"d{ic}"] = df[ic] - m

    # Akaike weights (from AIC_full)
    df["akaike_w"] = _akaike_weights(df["dAIC_full"].values)

    # nicer order & save
    df = df.loc[_nice_model_order(list(df.index))]
    _ensure_outdir(outdir)
    csv_path = outdir / "model_compare.csv"
    tex_path = outdir / "model_compare.tex"
    df_round = df.copy()
    for c in ["chi2","AIC_full","BIC_full","AIC_quad","BIC_quad","dAIC_full","dBIC_full","sigma_sys_med","epsilon_med","q_med","a0_med","akaike_w"]:
        if c in df_round.columns:
            df_round[c] = pd.to_numeric(df_round[c], errors="coerce").round(3)
    df_round.to_csv(csv_path)
    with open(tex_path, "w") as f:
        f.write(df_round.reset_index()[[
            "model","k","N","chi2","AIC_full","BIC_full","dAIC_full","dBIC_full","akaike_w","epsilon_med","q_med","a0_med","sigma_sys_med","likelihood"
        ]].to_latex(index=False, escape=True))
    print(f"[OK] model comparison → {csv_path}, {tex_path}")

    # ΔAIC / ΔBIC bar plots
    def _bar_delta(series: pd.Series, title: str, outpng: Path):
        plt.figure(figsize=(6.0, 4.0), dpi=160)
        x = np.arange(len(series))
        plt.bar(x, series.values)
        plt.xticks(x, series.index, rotation=30, ha="right")
        plt.ylabel("Δ " + title.split()[-1])
        plt.title(title)
        plt.tight_layout()
        plt.savefig(outpng)
        plt.close()
        print(f"[OK] {outpng}")

    _bar_delta(df["dAIC_full"], "Δ AIC (full-likelihood)", outdir/"fig_delta_AIC.png")
    _bar_delta(df["dBIC_full"], "Δ BIC (full-likelihood)",  outdir/"fig_delta_BIC.png")

    return df


# --------------------------
# PPC coverage plot
# --------------------------

def make_ppc_plot(ppc_json: Path, outdir: Path) -> None:
    dat = _try_read_json(ppc_json)
    if not dat:
        print(f"[WARN] skip PPC: {ppc_json} not found.")
        return
    cov68 = float(dat.get("coverage68", np.nan))
    cov95 = float(dat.get("coverage95", np.nan))

    plt.figure(figsize=(4.6,3.8), dpi=160)
    x = np.arange(2)
    y = np.array([cov68, cov95])
    plt.bar(x, y, width=0.5)
    plt.xticks(x, ["68%", "95%"])
    plt.axhline(0.68, linestyle="--", linewidth=1.2)
    plt.axhline(0.95, linestyle="--", linewidth=1.2)
    plt.ylim(0, 1.05)
    plt.ylabel("Predictive coverage")
    plt.title("Posterior(-like) predictive coverage")
    plt.tight_layout()
    p = outdir / "fig_ppc_coverage.png"
    plt.savefig(p)
    plt.close()
    print(f"[OK] {p}")


# --------------------------
# H0 scan plots
# --------------------------

def make_h0_scan_plots(h0_csv: Path, outdir: Path) -> None:
    if not _safe(h0_csv):
        print(f"[WARN] skip H0 scan: {h0_csv} not found.")
        return
    df = pd.read_csv(h0_csv)
    if "H0_kms_mpc" not in df.columns:
        print(f"[WARN] H0 scan CSV missing H0_kms_mpc: {h0_csv}")
        return

    # AIC_full vs H0
    plt.figure(figsize=(5.0,3.8), dpi=160)
    plt.plot(df["H0_kms_mpc"], df["AIC_full"], marker="o", linewidth=1.5)
    plt.xlabel(r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]")
    plt.ylabel("AIC (full)")
    plt.title("H0 sensitivity — AIC")
    plt.tight_layout()
    p1 = outdir / "fig_h0_AIC.png"
    plt.savefig(p1); plt.close(); print(f"[OK] {p1}")

    # epsilon median vs H0 (if present)
    if "epsilon_median" in df.columns and df["epsilon_median"].notna().any():
        plt.figure(figsize=(5.0,3.8), dpi=160)
        plt.plot(df["H0_kms_mpc"], df["epsilon_median"], marker="o", linewidth=1.5)
        plt.xlabel(r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]")
        plt.ylabel(r"$\epsilon_{\rm median}$")
        plt.title("H0 sensitivity — epsilon")
        plt.tight_layout()
        p2 = outdir / "fig_h0_epsilon.png"
        plt.savefig(p2); plt.close(); print(f"[OK] {p2}")
    else:
        print("[INFO] H0 scan has no epsilon_median column; skip epsilon plot.")


# --------------------------
# Kappa checks plots
# --------------------------

def make_kappa_gal_plot(csv_path: Path, summary_json: Path, outdir: Path) -> None:
    if not _safe(csv_path):
        print(f"[WARN] skip kappa-gal: {csv_path} not found.")
        return
    df = pd.read_csv(csv_path)
    summ = _try_read_json(summary_json) or {}
    a = float(summ.get("slope", np.nan))
    b = float(summ.get("intercept", np.nan))
    r2 = float(summ.get("R2", np.nan))

    plt.figure(figsize=(5.3,4.1), dpi=160)
    plt.scatter(df["kappa_pred"], df["eps_eff_over_epscos"], s=14, alpha=0.75)
    x = np.linspace(0.0, max(0.22, df["kappa_pred"].max()*1.05), 200)
    plt.plot(x, x, linestyle="--", linewidth=1.2, label="y = x")
    if np.isfinite(a) and np.isfinite(b):
        plt.plot(x, a*x + b, linewidth=1.4, label=f"fit: y={a:.2f}x+{b:.2f}, R²={r2:.2f}")
    plt.xlabel(r"$\kappa_{\rm pred}=\eta\,R_d/r_\ast$")
    plt.ylabel(r"$\varepsilon_{\rm eff}(r_\ast)/\varepsilon_{\rm cos}$")
    plt.title("Per-galaxy κ check")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    p = outdir / "fig_kappa_gal.png"
    plt.savefig(p); plt.close(); print(f"[OK] {p}")

def make_kappa_profile_plot(binned_csv: Path, outdir: Path,
                            eta: Optional[float]=0.15,
                            kappa_from_gal_summary: Optional[Path]=None) -> None:
    if not _safe(binned_csv):
        print(f"[WARN] skip kappa-profile: {binned_csv} not found.")
        return
    dfb = pd.read_csv(binned_csv)
    plt.figure(figsize=(5.6,4.0), dpi=160)
    plt.fill_between(dfb["x_mid"], dfb["q16"], dfb["q84"], alpha=0.25, label="stacked 16–84%")
    plt.plot(dfb["x_mid"], dfb["q50"], linewidth=1.6, label="stacked median")

    # Optional overlay: kappa*eta/x using kappa from kappa-gal summary
    label_pred = None
    if eta is not None and kappa_from_gal_summary and _safe(kappa_from_gal_summary):
        summ = _try_read_json(kappa_from_gal_summary) or {}
        kappa = float(summ.get("slope", np.nan))
        if np.isfinite(kappa):
            x = np.linspace(max(dfb["x_mid"].min(), 1e-3), max(dfb["x_mid"].max(), 2.0), 400)
            y = (kappa * eta) / np.maximum(x, 1e-6)
            plt.plot(x, y, linestyle="--", linewidth=1.2)
            label_pred = r"$\kappa\,\eta/x$"
    if label_pred:
        plt.legend()

    plt.xlabel("x = r/Rd")
    plt.ylabel(r"$\varepsilon_{\rm eff}(r)/\varepsilon_{\rm cos}$")
    plt.title("Radius-resolved κ profile")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    p = outdir / "fig_kappa_profile.png"
    plt.savefig(p); plt.close(); print(f"[OK] {p}")


# --------------------------
# Stress / Mask summaries
# --------------------------

def _collect_summaries_glob(root: Path) -> List[Tuple[str, dict]]:
    """Collect all direct subdirs containing 'global_summary.yaml'."""
    out = []
    if not _safe(root):
        return out
    for sub in sorted(root.glob("*")):
        if sub.is_dir() and _safe(sub / "global_summary.yaml"):
            out.append((sub.name, _load_yaml(sub / "global_summary.yaml")))
    return out

def make_stress_mask_summaries(stress_root: Path, mask_root: Path, outdir: Path) -> None:
    rows = []
    for tag, root in [("stress", stress_root), ("mask", mask_root)]:
        for name, summ in _collect_summaries_glob(root):
            rows.append(dict(
                kind=tag, run=name, model=summ.get("model"),
                sigma_sys_med=summ.get("sigma_sys_median"),
                epsilon_med=summ.get("epsilon_median"),
                q_med=summ.get("q_median"),
                a0_med=summ.get("a0_median"),
                chi2=summ.get("chi2_total"),
                AIC_full=summ.get("AIC_full"), BIC_full=summ.get("BIC_full"),
                k=summ.get("k_parameters"), N=summ.get("N_total"),
                likelihood=summ.get("likelihood","gauss")
            ))
    if not rows:
        print("[WARN] no stress/mask summaries found; skip.")
        return
    df = pd.DataFrame(rows).sort_values(["kind","run"])
    _ensure_outdir(outdir)
    csvp = outdir / "stress_mask_summary.csv"
    texp = outdir / "stress_mask_summary.tex"
    dfr = df.copy()
    for c in ["sigma_sys_med","epsilon_med","q_med","a0_med","chi2","AIC_full","BIC_full"]:
        if c in dfr.columns:
            dfr[c] = pd.to_numeric(dfr[c], errors="coerce").round(3)
    dfr.to_csv(csvp, index=False)
    with open(texp, "w") as f:
        f.write(dfr.to_latex(index=False, escape=True))
    print(f"[OK] stress/mask → {csvp}, {texp}")


# --------------------------
# Closure summary
# --------------------------

def make_closure_summary(closure_yaml: Path, outdir: Path) -> None:
    if not _safe(closure_yaml):
        print(f"[WARN] skip closure: {closure_yaml} not found.")
        return
    cl = _load_yaml(closure_yaml)
    # Build a 1-row tex table
    df = pd.DataFrame([dict(
        epsilon_RC=cl.get("epsilon_RC"),
        epsilon_cos=cl.get("epsilon_cos"),
        sigma_RC=cl.get("sigma_RC"),
        diff=cl.get("diff"),
        pass_within_3sigma=cl.get("pass_within_3sigma"),
    )])
    for c in ["epsilon_RC","epsilon_cos","sigma_RC","diff"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(4)
    with open(outdir / "closure_summary.tex", "w") as f:
        f.write(df.to_latex(index=False, escape=True))
    print(f"[OK] closure → {outdir/'closure_summary.tex'}")


# --------------------------
# Main
# --------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Build paper tables/figures from PTQ results.")
    ap.add_argument("--outdir", default="paper_outputs", type=str)

    # model mapping
    ap.add_argument("--model", action="append", default=None,
                    help="Mapping Name=results_dir (repeatable). "
                         "If omitted, use defaults from README.")

    # optional inputs (defaults assume README layout)
    ap.add_argument("--ppc-json", default="results/ptq_screen/ppc_coverage.json")
    ap.add_argument("--h0-scan-csv", default="results/H0_scan/ptq-screen_H0_scan.csv")

    ap.add_argument("--kappa-gal-csv", default="results/ptq_screen/kappa_gal_per_galaxy.csv")
    ap.add_argument("--kappa-gal-summary", default="results/ptq_screen/kappa_gal_summary.json")
    ap.add_argument("--kappa-prof-binned", default="results/ptq_screen/kappa_profile_binned.csv")

    ap.add_argument("--plateau-binned", default="results/ptq_screen/plateau_binned.csv")  # optional

    ap.add_argument("--stress-root", default="results/stress")
    ap.add_argument("--mask-root", default="results/mask")

    ap.add_argument("--closure-yaml", default="results/ptq_screen/closure_test.yaml")

    return ap.parse_args()

def _default_models_map() -> Dict[str, Path]:
    return {
        "PTQ-screen": Path("results/ptq_screen"),
        "MOND":       Path("results/ejpc_mond"),
        "PTQ-v":      Path("results/ptq_nu"),
        "NFW-1p":     Path("results/ejpc_nfw1p"),
        "PTQ":        Path("results/ejpc_main"),
        "Baryon":     Path("results/ejpc_baryon"),
    }

def _parse_models_map(items: Optional[List[str]]) -> Dict[str, Path]:
    if not items:
        return _default_models_map()
    out = {}
    for s in items:
        if "=" not in s:
            raise ValueError(f"--model expects Name=path, got: {s}")
        name, p = s.split("=", 1)
        out[name.strip()] = Path(p.strip())
    return out

def main():
    args = parse_args()
    outdir = Path(args.outdir); _ensure_outdir(outdir)
    models_map = _parse_models_map(args.model)

    # 1) Model comparison + ΔIC plots
    try:
        make_model_comparison(models_map, outdir)
    except FileNotFoundError as e:
        print(f"[ERROR] model comparison failed: {e}")

    # 2) PPC plot
    make_ppc_plot(Path(args.ppc_json), outdir)

    # 3) H0 scan plots
    make_h0_scan_plots(Path(args.h0_scan_csv), outdir)

    # 4) Kappa-gal + Kappa-profile
    make_kappa_gal_plot(Path(args.kappa_gal_csv), Path(args.kappa_gal_summary), outdir)
    make_kappa_profile_plot(Path(args.kappa_prof_binned), outdir,
                            eta=0.15, kappa_from_gal_summary=Path(args.kappa_gal_summary))

    # 5) Stress/Mask summaries
    make_stress_mask_summaries(Path(args.stress_root), Path(args.mask_root), outdir)

    # 6) Closure summary
    make_closure_summary(Path(args.closure_yaml), outdir)

    # 7) (Optional) plateau overlay: replot binned for consistent style (if present)
    pb = Path(args.plateau_binned)
    if _safe(pb):
        dfb = pd.read_csv(pb)
        plt.figure(figsize=(6.2,4.0), dpi=160)
        x, y, y1, y2 = dfb["r_mid_kpc"], dfb["q50"], dfb["q16"], dfb["q84"]
        plt.fill_between(x, y1, y2, alpha=0.25)
        plt.plot(x, y, linewidth=1.6)
        plt.xlabel("r [kpc]")
        plt.ylabel(r"$\Delta a$  [m s$^{-2}$]")
        plt.title("Stacked residual-acceleration plateau")
        plt.tight_layout()
        p = outdir / "fig_plateau.png"
        plt.savefig(p); plt.close(); print(f"[OK] {p}")
    else:
        print(f"[INFO] skip plateau (no {pb}).")

    print(f"\nAll done. Artifacts in: {outdir.resolve()}")

if __name__ == "__main__":
    main()
