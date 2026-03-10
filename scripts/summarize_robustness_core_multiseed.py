#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Layer B: three-model multi-seed robustness aggregation.

This script reads, for each seed in {0, 7, 13}:
- results/revision_robustness_core_multiseed12000/seed_<seed>/<model>_gauss/global_summary.yaml
- results/revision_robustness_core_multiseed12000/seed_<seed>/paper_extra/model_compare_core/compare_table.csv
- results/revision_robustness_core_multiseed12000/seed_<seed>/paper_extra/loo_compare_core/loo_table.csv

and writes robustness-level summaries to:
- results/revision_robustness_core_multiseed12000/aggregate/

Roles:
- Robustness / ranking-stability audit for the three key models:
  ptq-screen, mond, mond-screen.
- Complements (but does NOT replace) the six-model baseline landscape.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import yaml


RUNROOT = Path("results/revision_robustness_core_multiseed12000")
AGG_DIR = RUNROOT / "aggregate"

SEEDS = [0, 7, 13]
MODELS = ["ptq-screen", "mond", "mond-screen"]


def _read_yaml(p: Path) -> Dict[str, Any] | None:
    if not p.exists():
        return None
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return None


def build_fit_long() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        seed_root = RUNROOT / f"seed_{seed}"
        for m in MODELS:
            yml = seed_root / f"{m}_gauss" / "global_summary.yaml"
            s = _read_yaml(yml)
            if s is None:
                continue
            rows.append(
                {
                    "seed": seed,
                    "model": m,
                    "chi2_total": s.get("chi2_total"),
                    "AIC_full": s.get("AIC_full"),
                    "BIC_full": s.get("BIC_full"),
                    "k_parameters": s.get("k_parameters"),
                    "epsilon_median": s.get("epsilon_median"),
                    "q_median": s.get("q_median"),
                    "a0_median": s.get("a0_median"),
                    "steps": s.get("steps"),
                    "nwalkers": s.get("nwalkers"),
                    "burn_in": s.get("burn_in"),
                    "resumed": s.get("resumed"),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["seed", "model"])
    df = pd.DataFrame(rows)
    df = df.sort_values(["seed", "model"]).reset_index(drop=True)
    return df


def build_compare_long() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for seed in SEEDS:
        seed_root = RUNROOT / f"seed_{seed}"
        cmp_csv = seed_root / "paper_extra" / "model_compare_core" / "compare_table.csv"
        loo_csv = seed_root / "paper_extra" / "loo_compare_core" / "loo_table.csv"

        df_cmp = pd.read_csv(cmp_csv) if cmp_csv.exists() else pd.DataFrame()
        df_loo = pd.read_csv(loo_csv) if loo_csv.exists() else pd.DataFrame()

        if not df_cmp.empty:
            if "delta_waic" not in df_cmp.columns and "waic" in df_cmp.columns:
                w_min = df_cmp["waic"].min()
                df_cmp["delta_waic"] = df_cmp["waic"] - w_min
            if "rank" not in df_cmp.columns and "waic" in df_cmp.columns:
                df_cmp["rank"] = df_cmp["waic"].rank().astype(int)
            df_cmp.rename(columns={"rank": "waic_rank"}, inplace=True)

        if not df_loo.empty:
            if "delta_looic" not in df_loo.columns and "looic" in df_loo.columns:
                l_min = df_loo["looic"].min()
                df_loo["delta_looic"] = df_loo["looic"] - l_min
            if "rank" not in df_loo.columns and "looic" in df_loo.columns:
                df_loo["rank"] = df_loo["looic"].rank().astype(int)
            df_loo.rename(columns={"rank": "loo_rank"}, inplace=True)

        base = pd.DataFrame({"model": MODELS})
        merged = base.merge(df_cmp, on="model", how="left", suffixes=("", "_waic"))
        merged = merged.merge(df_loo, on="model", how="left", suffixes=("", "_loo"))

        for _, r in merged.iterrows():
            rows.append(
                {
                    "seed": seed,
                    "model": r["model"],
                    "waic": r.get("waic"),
                    "delta_waic": r.get("delta_waic"),
                    "p_waic": r.get("p_waic"),
                    "looic": r.get("looic"),
                    "delta_looic": r.get("delta_looic"),
                    "elpd_loo": r.get("elpd_loo"),
                    "p_loo": r.get("p_loo"),
                    "waic_rank": r.get("waic_rank"),
                    "loo_rank": r.get("loo_rank"),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["seed", "model"])
    df = pd.DataFrame(rows)
    df = df.sort_values(["seed", "model"]).reset_index(drop=True)
    return df


def write_fit_long(df: pd.DataFrame) -> None:
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    out = AGG_DIR / "multiseed_fit_metrics_long.csv"
    df.to_csv(out, index=False)


def write_compare_long(df: pd.DataFrame) -> None:
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    out = AGG_DIR / "multiseed_compare_metrics_long.csv"
    df.to_csv(out, index=False)


def build_model_summary(df_fit: pd.DataFrame, df_cmp: pd.DataFrame) -> pd.DataFrame:
    """Aggregate across seeds for each model."""
    rows: List[Dict[str, Any]] = []

    # Helper: collect metrics safely
    def _safe_series(df: pd.DataFrame, col: str) -> List[float]:
        if col not in df.columns:
            return []
        return [float(x) for x in df[col].dropna().tolist()]

    for m in MODELS:
        df_f = df_fit[df_fit["model"] == m]
        df_c = df_cmp[df_cmp["model"] == m]

        metrics = {}
        for col in ("AIC_full", "BIC_full"):
            vals = _safe_series(df_f, col)
            if vals:
                metrics[f"{col}_mean"] = statistics.mean(vals)
                metrics[f"{col}_median"] = statistics.median(vals)
                metrics[f"{col}_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
                metrics[f"{col}_min"] = min(vals)
                metrics[f"{col}_max"] = max(vals)
        for col in ("waic", "looic"):
            vals = _safe_series(df_c, col)
            if vals:
                metrics[f"{col}_mean"] = statistics.mean(vals)
                metrics[f"{col}_median"] = statistics.median(vals)
                metrics[f"{col}_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
                metrics[f"{col}_min"] = min(vals)
                metrics[f"{col}_max"] = max(vals)

        # Best-counts per metric (smaller is better)
        best_counts = {
            "best_count_by_AIC": 0,
            "best_count_by_BIC": 0,
            "best_count_by_WAIC": 0,
            "best_count_by_LOOIC": 0,
        }
        for seed in SEEDS:
            df_seed_f = df_fit[df_fit["seed"] == seed]
            df_seed_c = df_cmp[df_cmp["seed"] == seed]
            if not df_seed_f.empty:
                for col, key in (("AIC_full", "best_count_by_AIC"), ("BIC_full", "best_count_by_BIC")):
                    if col not in df_seed_f.columns:
                        continue
                    svals = df_seed_f[col].dropna()
                    if svals.empty:
                        continue
                    best_val = svals.min()
                    row = df_seed_f[(df_seed_f["model"] == m) & (df_seed_f[col] == best_val)]
                    if not row.empty:
                        best_counts[key] += 1
            if not df_seed_c.empty:
                for col, key in (("waic", "best_count_by_WAIC"), ("looic", "best_count_by_LOOIC")):
                    if col not in df_seed_c.columns:
                        continue
                    svals = df_seed_c[col].dropna()
                    if svals.empty:
                        continue
                    best_val = svals.min()
                    row = df_seed_c[(df_seed_c["model"] == m) & (df_seed_c[col] == best_val)]
                    if not row.empty:
                        best_counts[key] += 1

        row = {"model": m, "n_seeds": len(set(df_cmp[df_cmp["model"] == m]["seed"]))}
        row.update(metrics)
        row.update(best_counts)
        rows.append(row)

    return pd.DataFrame(rows)


def build_rank_stability(df_cmp: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for m in MODELS:
        df_m = df_cmp[df_cmp["model"] == m]
        if df_m.empty:
            rows.append(
                {
                    "model": m,
                    "waic_rank_mean": None,
                    "waic_rank_std": None,
                    "loo_rank_mean": None,
                    "loo_rank_std": None,
                    "n_seeds": 0,
                }
            )
            continue
        wr = df_m["waic_rank"].dropna().tolist() if "waic_rank" in df_m.columns else []
        lr = df_m["loo_rank"].dropna().tolist() if "loo_rank" in df_m.columns else []
        def _m_and_s(vals: List[float]) -> tuple[Any, Any]:
            if not vals:
                return None, None
            if len(vals) == 1:
                return float(vals[0]), 0.0
            return float(statistics.mean(vals)), float(statistics.pstdev(vals))
        wr_m, wr_s = _m_and_s(wr)
        lr_m, lr_s = _m_and_s(lr)
        rows.append(
            {
                "model": m,
                "waic_rank_mean": wr_m,
                "waic_rank_std": wr_s,
                "loo_rank_mean": lr_m,
                "loo_rank_std": lr_s,
                "n_seeds": len(set(df_m["seed"])),
            }
        )
    return pd.DataFrame(rows)


def write_model_summary(df_model: pd.DataFrame) -> None:
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    out = AGG_DIR / "multiseed_model_summary.csv"
    df_model.to_csv(out, index=False)


def write_rank_stability(df_rank: pd.DataFrame) -> None:
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    out = AGG_DIR / "multiseed_rank_stability.csv"
    df_rank.to_csv(out, index=False)


def write_multiseed_summary_md(
    df_fit: pd.DataFrame, df_cmp: pd.DataFrame, df_model: pd.DataFrame, df_rank: pd.DataFrame
) -> None:
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []

    lines.append("# Multi-seed robustness summary (Layer B)")
    lines.append("")
    lines.append(
        "- This robustness layer uses three models (`ptq-screen`, `mond`, `mond-screen`) "
        "with seeds {0, 7, 13}, each run with steps = 12000 and nwalkers = 192."
    )
    lines.append(
        "- Its role is to audit seed sensitivity and ranking stability, not to replace the "
        "six-model baseline landscape (Layer A)."
    )
    lines.append("")

    # Rank stability overview
    if not df_rank.empty:
        lines.append("## Rank stability (WAIC / LOO)")
        lines.append("")
        for _, r in df_rank.iterrows():
            lines.append(
                f"- `{r['model']}`: "
                f"WAIC rank mean ≈ {r['waic_rank_mean']}, std ≈ {r['waic_rank_std']}; "
                f"LOO rank mean ≈ {r['loo_rank_mean']}, std ≈ {r['loo_rank_std']} "
                f"(n_seeds = {int(r['n_seeds'])})."
            )
        lines.append("")
    else:
        lines.append("## Rank stability (WAIC / LOO)")
        lines.append("")
        lines.append("- (no compare/LOO metrics found across seeds)")
        lines.append("")

    # ptq-screen vs others: seed sensitivity
    lines.append("## Seed sensitivity for key models")
    lines.append("")
    if not df_cmp.empty:
        for m in MODELS:
            df_m = df_cmp[df_cmp["model"] == m]
            if df_m.empty:
                lines.append(f"- `{m}`: missing across all seeds.")
                continue
            waics = df_m["waic"].dropna().tolist() if "waic" in df_m.columns else []
            looics = df_m["looic"].dropna().tolist() if "looic" in df_m.columns else []
            if waics:
                lines.append(
                    f"- `{m}` WAIC across seeds: "
                    f"min ≈ {min(waics):.1f}, max ≈ {max(waics):.1f}, "
                    f"spread ≈ {max(waics) - min(waics):.1f}."
                )
            if looics:
                lines.append(
                    f"  LOOIC across seeds: "
                    f"min ≈ {min(looics):.1f}, max ≈ {max(looics):.1f}, "
                    f"spread ≈ {max(looics) - min(looics):.1f}."
                )
    else:
        lines.append("- Compare/LOO metrics missing; seed sensitivity cannot be assessed.")
    lines.append("")

    # Relationship to baseline
    lines.append("## Relationship to baseline (Layer A)")
    lines.append("")
    lines.append(
        "- The six-model baseline (Layer A) provides a single long-chain comparison at seed 0; "
        "this multi-seed audit checks whether that snapshot is representative across seeds."
    )
    lines.append(
        "- Results here should be interpreted as evidence for ranking stability or variability "
        "under different seeds, not as new single 'best' runs."
    )
    lines.append("")

    # Missing data notice
    missing_seeds = [s for s in SEEDS if f"seed_{s}" not in {p.name for p in RUNROOT.iterdir() if p.is_dir()}]
    if missing_seeds:
        lines.append("## Missing seeds / runs")
        lines.append("")
        lines.append(f"- The following seeds have no directory under RUNROOT: {missing_seeds}")
        lines.append("")

    out_md = AGG_DIR / "multiseed_summary.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")


def write_multiseed_latex(df_model: pd.DataFrame) -> None:
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    out_tex = AGG_DIR / "multiseed_robustness_table.tex"

    lines: List[str] = []
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"Model & AIC$_{\mathrm{full}}$ median$\pm$std & "
        r"BIC$_{\mathrm{full}}$ median$\pm$std & "
        r"WAIC median$\pm$std & LOOIC median$\pm$std \\"
    )
    lines.append(r"\midrule")
    for _, r in df_model.iterrows():
        def _fmt(med_key: str, std_key: str) -> str:
            med = r.get(med_key)
            sd = r.get(std_key)
            if med is None or pd.isna(med):
                return r"$\cdots$"
            if sd is None or pd.isna(sd):
                return f"{med:.1f}"
            return f"{med:.1f}$\\pm${sd:.1f}"

        lines.append(
            f"{r['model']} & "
            f"{_fmt('AIC_full_median', 'AIC_full_std')} & "
            f"{_fmt('BIC_full_median', 'BIC_full_std')} & "
            f"{_fmt('waic_median', 'waic_std')} & "
            f"{_fmt('looic_median', 'looic_std')} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    out_tex.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    AGG_DIR.mkdir(parents=True, exist_ok=True)

    df_fit = build_fit_long()
    df_cmp = build_compare_long()

    write_fit_long(df_fit)
    write_compare_long(df_cmp)

    df_model = build_model_summary(df_fit, df_cmp)
    df_rank = build_rank_stability(df_cmp)

    write_model_summary(df_model)
    write_rank_stability(df_rank)
    write_multiseed_summary_md(df_fit, df_cmp, df_model, df_rank)
    write_multiseed_latex(df_model)

    print(f"[summarize_robustness_core_multiseed] wrote aggregate artifacts under {AGG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

