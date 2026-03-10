#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Layer A: six-model single-seed baseline aggregation.

This script reads:
- results/revision_baseline_sixmodels_12000/<model>_gauss/global_summary.yaml
- results/revision_baseline_sixmodels_12000/paper_extra/model_compare_baseline/compare_table.csv
- results/revision_baseline_sixmodels_12000/paper_extra/loo_compare_baseline/loo_table.csv

and writes baseline-level summary artifacts to:
- results/revision_baseline_sixmodels_12000/aggregate/

Roles:
- Baseline landscape for the paper (Table-style main results).
- NOT a robustness audit; Layer B handles multi-seed robustness separately.
"""

from __future__ import annotations

import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List


RUNROOT = Path("results/revision_baseline_sixmodels_12000")
AGG_DIR = RUNROOT / "aggregate"

MODELS = ["ptq-screen", "mond", "ptq-nu", "nfw1p", "ptq", "baryon"]


def _read_yaml(p: Path) -> Dict[str, Any] | None:
    if not p.exists():
        return None
    try:
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return None


def _resolve_compare_csv() -> Path | None:
    """Prefer baseline compare dir, fall back to any legacy if needed."""
    c1 = RUNROOT / "paper_extra" / "model_compare_baseline" / "compare_table.csv"
    if c1.exists():
        return c1
    c2 = RUNROOT / "paper_extra" / "model_compare_core" / "compare_table.csv"
    if c2.exists():
        return c2
    c3 = RUNROOT / "paper_extra" / "model_compare" / "compare_table.csv"
    if c3.exists():
        return c3
    return None


def _resolve_loo_csv() -> Path | None:
    """Prefer baseline loo dir, fall back to any legacy if needed."""
    c1 = RUNROOT / "paper_extra" / "loo_compare_baseline" / "loo_table.csv"
    if c1.exists():
        return c1
    c2 = RUNROOT / "paper_extra" / "loo_compare_core" / "loo_table.csv"
    if c2.exists():
        return c2
    c3 = RUNROOT / "paper_extra" / "loo_compare" / "loo_table.csv"
    if c3.exists():
        return c3
    return None


def build_fit_metrics() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for m in MODELS:
        yml = RUNROOT / f"{m}_gauss" / "global_summary.yaml"
        s = _read_yaml(yml)
        if s is None:
            continue
        rows.append(
            {
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
        return pd.DataFrame(columns=["model"])
    df = pd.DataFrame(rows)
    df = df.sort_values("BIC_full", na_position="last").reset_index(drop=True)
    return df


def build_compare_metrics(
    df_fit: pd.DataFrame,
) -> pd.DataFrame:
    cmp_csv = _resolve_compare_csv()
    loo_csv = _resolve_loo_csv()

    df_cmp = pd.read_csv(cmp_csv) if cmp_csv is not None and cmp_csv.exists() else pd.DataFrame()
    df_loo = pd.read_csv(loo_csv) if loo_csv is not None and loo_csv.exists() else pd.DataFrame()

    if not df_cmp.empty:
        if "delta_waic" not in df_cmp.columns and "waic" in df_cmp.columns:
            w_min = df_cmp["waic"].min()
            df_cmp["delta_waic"] = df_cmp["waic"] - w_min
        if "rank" not in df_cmp.columns and "waic" in df_cmp.columns:
            df_cmp["rank"] = df_cmp["waic"].rank().astype(int)
        df_cmp.rename(columns={"rank": "waic_rank"}, inplace=True)
    else:
        df_cmp = pd.DataFrame(columns=["model"])

    if not df_loo.empty:
        if "delta_looic" not in df_loo.columns and "looic" in df_loo.columns:
            l_min = df_loo["looic"].min()
            df_loo["delta_looic"] = df_loo["looic"] - l_min
        if "rank" not in df_loo.columns and "looic" in df_loo.columns:
            df_loo["rank"] = df_loo["looic"].rank().astype(int)
        df_loo.rename(columns={"rank": "loo_rank"}, inplace=True)
    else:
        df_loo = pd.DataFrame(columns=["model"])

    # Merge on model, but restrict to the six baseline models
    base = pd.DataFrame({"model": MODELS})
    out = base.merge(df_cmp, on="model", how="left", suffixes=("", "_waic"))
    out = out.merge(df_loo, on="model", how="left", suffixes=("", "_loo"))

    # Harmonize column names for output
    cols = {
        "waic": "waic",
        "delta_waic": "delta_waic",
        "p_waic": "p_waic",
        "elpd_loo": "elpd_loo",
        "p_loo": "p_loo",
        "looic": "looic",
        "delta_looic": "delta_looic",
        "waic_rank": "waic_rank",
        "loo_rank": "loo_rank",
        "n_data": "n_data",
    }
    for old, new in list(cols.items()):
        if old not in out.columns and f"{old}_loo" in out.columns:
            cols[old] = f"{old}_loo"

    keep_cols = ["model"]
    for old, _ in cols.items():
        if cols[old] in out.columns:
            keep_cols.append(cols[old])
    keep_cols = list(dict.fromkeys(keep_cols))  # remove duplicates, preserve order

    sub = out.loc[:, keep_cols].copy()
    sub = sub.rename(
        columns={cols.get("waic", "waic"): "waic", cols.get("delta_waic", "delta_waic"): "delta_waic",
                 cols.get("p_waic", "p_waic"): "p_waic",
                 cols.get("elpd_loo", "elpd_loo"): "elpd_loo",
                 cols.get("p_loo", "p_loo"): "p_loo",
                 cols.get("looic", "looic"): "looic",
                 cols.get("delta_looic", "delta_looic"): "delta_looic",
                 cols.get("waic_rank", "waic_rank"): "waic_rank",
                 cols.get("loo_rank", "loo_rank"): "loo_rank",
                 cols.get("n_data", "n_data"): "n_data"}
    )
    return sub


def write_fit_metrics_csv(df_fit: pd.DataFrame) -> None:
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    out = AGG_DIR / "baseline_sixmodels_fit_metrics.csv"
    df_fit.to_csv(out, index=False)


def write_compare_metrics_csv(df_cmp: pd.DataFrame) -> None:
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    out = AGG_DIR / "baseline_sixmodels_compare_metrics.csv"
    df_cmp.to_csv(out, index=False)


def write_summary_markdown(df_fit: pd.DataFrame, df_cmp: pd.DataFrame) -> None:
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []

    lines.append("# Baseline six-model summary (Layer A)")
    lines.append("")
    lines.append(
        "- This baseline uses a single long chain per model "
        "(steps = 12000, nwalkers = 192, seed = 0) on the same SPARC dataset."
    )
    lines.append(
        "- It provides the main six-model landscape for the paper; "
        "multi-seed robustness is handled separately in Layer B."
    )
    lines.append("")

    # AIC/BIC rankings
    if not df_fit.empty:
        lines.append("## AIC/BIC ordering")
        lines.append("")
        df_bic = df_fit.sort_values("BIC_full", na_position="last")
        for _, r in df_bic.iterrows():
            lines.append(
                f"- `{r['model']}`: AIC_full ≈ {r['AIC_full']:.1f}, "
                f"BIC_full ≈ {r['BIC_full']:.1f}, k ≈ {r['k_parameters']}"
            )
        lines.append("")
    else:
        lines.append("## AIC/BIC ordering")
        lines.append("")
        lines.append("- (missing fit metrics: no global_summary.yaml files found)")
        lines.append("")

    # WAIC/LOO rankings
    if not df_cmp.empty and "waic" in df_cmp.columns:
        lines.append("## WAIC / LOO ordering")
        lines.append("")
        df_waic = df_cmp.sort_values("waic", na_position="last")
        for _, r in df_waic.iterrows():
            waic = r.get("waic")
            dwaic = r.get("delta_waic")
            wr = r.get("waic_rank")
            looic = r.get("looic")
            dl = r.get("delta_looic")
            lr = r.get("loo_rank")
            lines.append(
                f"- `{r['model']}`: WAIC ≈ {waic:.1f} (ΔWAIC ≈ {dwaic:.1f}, rank={wr}), "
                f"LOOIC ≈ {looic:.1f} (ΔLOOIC ≈ {dl:.1f}, rank={lr})"
            )
        lines.append("")
    else:
        lines.append("## WAIC / LOO ordering")
        lines.append("")
        lines.append("- (missing compare / loo metrics)")
        lines.append("")

    # Negative controls and key models
    lines.append("## Interpretation: key models and negative controls")
    lines.append("")
    if not df_fit.empty:
        present = set(df_fit["model"])
    else:
        present = set(df_cmp["model"]) if not df_cmp.empty else set()

    def _is_present(name: str) -> bool:
        return name in present

    # Negative controls
    nc = []
    if _is_present("baryon"):
        nc.append("`baryon` (baryon-only)")
    if _is_present("ptq"):
        nc.append("`ptq` (linear PTQ)")
    if nc:
        lines.append("- Negative controls present in this baseline: " + ", ".join(nc) + ".")
    else:
        lines.append("- Negative controls: missing or not included in this run.")

    # PTQ-nu / NFW / MOND / PTQ-screen relative position (based on WAIC if available)
    if not df_cmp.empty and "waic" in df_cmp.columns:
        focus = ["ptq-screen", "mond", "ptq-nu", "nfw1p"]
        df_waic = df_cmp.set_index("model")
        ordered: List[str] = []
        for m in focus:
            if m in df_waic.index:
                r = df_waic.loc[m]
                ordered.append(
                    f"{m} (WAIC ≈ {r['waic']:.1f}, ΔWAIC ≈ {r.get('delta_waic', float('nan')):.1f}, "
                    f"rank={int(r.get('waic_rank', 0) or 0)})"
                )
        if ordered:
            lines.append(
                "- Among key models (ptq-screen, mond, ptq-nu, nfw1p), "
                "the WAIC ordering is: " + "; ".join(ordered) + "."
            )
    else:
        lines.append(
            "- WAIC-based ordering of ptq-screen / mond / ptq-nu / nfw1p "
            "is unavailable (compare_table.csv missing)."
        )

    lines.append("")

    out_md = AGG_DIR / "baseline_sixmodels_summary.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")


def write_latex_table(df_fit: pd.DataFrame, df_cmp: pd.DataFrame) -> None:
    AGG_DIR.mkdir(parents=True, exist_ok=True)
    out_tex = AGG_DIR / "baseline_sixmodels_table.tex"

    # Merge a concise view for the table
    df = df_fit.merge(df_cmp, on="model", how="left", suffixes=("", "_cmp"))
    df = df.sort_values("BIC_full", na_position="last")

    lines: List[str] = []
    lines.append(r"\begin{tabular}{lrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Model & AIC$_{\mathrm{full}}$ & BIC$_{\mathrm{full}}$ & WAIC & LOOIC & Rank (WAIC/LOO) \\")
    lines.append(r"\midrule")
    for _, r in df.iterrows():
        waic = r.get("waic")
        looic = r.get("looic")
        wr = r.get("waic_rank")
        lr = r.get("loo_rank")
        rank_str = ""
        if pd.notnull(wr) or pd.notnull(lr):
            rank_str = f"{int(wr) if pd.notnull(wr) else '-'} / {int(lr) if pd.notnull(lr) else '-'}"
        lines.append(
            f"{r['model']} & "
            f"{(r['AIC_full'] if pd.notnull(r['AIC_full']) else float('nan')):.1f} & "
            f"{(r['BIC_full'] if pd.notnull(r['BIC_full']) else float('nan')):.1f} & "
            f"{(waic if pd.notnull(waic) else float('nan')):.1f} & "
            f"{(looic if pd.notnull(looic) else float('nan')):.1f} & "
            f"{rank_str} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    out_tex.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    AGG_DIR.mkdir(parents=True, exist_ok=True)

    df_fit = build_fit_metrics()
    df_cmp = build_compare_metrics(df_fit)

    write_fit_metrics_csv(df_fit)
    write_compare_metrics_csv(df_cmp)
    write_summary_markdown(df_fit, df_cmp)
    write_latex_table(df_fit, df_cmp)

    print(f"[summarize_baseline_sixmodels] wrote aggregate artifacts under {AGG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

