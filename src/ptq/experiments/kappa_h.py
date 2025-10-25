# -*- coding: utf-8 -*-
"""
ptq.experiments.kappa_h
-----------------------

回歸實驗：以 (log10 h) 對 (log10 κ, log10 Σ_tot) 做線性回歸，並提供：
- κ-only / Σ-only / κ+Σ 的 WLS/OLS 估計、R^2、AICc、ΔAICc
- LOO（逐一留一交叉驗證）與 bootstrap 參數區間
- 「距離不變」檢驗（DIST-INV）：log(h/D) 對 log(κD), log Σ 的回歸
- 可選擇忽略 h 的 outliers 與自訂排除清單
- 輸出 CSV／PNG 圖／JSON 報告（報告欄位對應 summarize_kappa_sigma_reports.py）

注意：
- 本程式假設輸入 merged CSV 內已有欄位：kappa、h_kpc。
- Σ_tot 若不存在，會嘗試由 *星＋氣體* 表面密度欄位推算（見 _build_sigma_tot）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt


TINY = 1e-12


# ---------------------------
# 小工具
# ---------------------------
def _canon_name(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).upper().strip()
    for ch in [" ", "-", "_", "/"]:
        s = s.replace(ch, "")
    # 常見前綴已都是字母，不額外處理
    return s


def _finite_mask(*arrs) -> np.ndarray:
    m = np.ones_like(np.asarray(arrs[0], dtype=float), dtype=bool)
    for a in arrs:
        m &= np.isfinite(a)
    return m


def _ols_wls(X: np.ndarray, y: np.ndarray, w: np.ndarray | None):
    X = sm.add_constant(X, has_constant="add")
    if w is None:
        model = sm.OLS(y, X).fit()
    else:
        model = sm.WLS(y, X, weights=w).fit()
    return model


def _rss(model) -> float:
    res = model.resid
    return float(np.dot(res, res))


def _aicc(n: int, k: int, rss: float) -> float:
    if n <= k + 1:
        return np.nan
    aic = n * np.log(rss / n) + 2 * k
    return float(aic + (2 * k * (k + 1)) / (n - k - 1))


def _loo_coeffs(X: np.ndarray, y: np.ndarray, w: np.ndarray | None) -> Tuple[np.ndarray, np.ndarray]:
    n = len(y)
    ps: List[np.ndarray] = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        m = _ols_wls(X[mask], y[mask], None if w is None else w[mask])
        ps.append(m.params)
    P = np.stack(ps, axis=0)
    return P.mean(axis=0), P.std(axis=0)


def _bootstrap_coeffs(
    X: np.ndarray, y: np.ndarray, w: np.ndarray | None, nboot: int = 2000, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = len(y)
    B: List[np.ndarray] = []
    for _ in range(nboot):
        idx = rng.integers(0, n, size=n)
        m = _ols_wls(X[idx], y[idx], None if w is None else w[idx])
        B.append(m.params)
    B = np.stack(B, axis=0)
    med = np.median(B, axis=0)
    lo = np.percentile(B, 16, axis=0)
    hi = np.percentile(B, 84, axis=0)
    return med, lo, hi


def _build_sigma_tot(df: pd.DataFrame, ml36: float | None, rgas_mult: float | None, gas_helium: float | None) -> pd.Series:
    """
    嘗試建構 Σ_tot（質量表面密度，單位假設相容），優先順序：
    1) 若已有 'Sigma_tot' 欄，直接使用。
    2) 若有星與氣體：Sigma_star, Sigma_gas -> Sigma_star + Sigma_gas
    3) 若有 3.6um 星面密度：Sigma_star_36 -> 乘上 ml36（預設使用者提供）
    4) 氣體可套用 rgas_mult 與 helium 係數（若提供）。
    找欄位時同時嘗試大小寫變種。
    """
    # 直接存在
    for name in ["Sigma_tot", "sigma_tot", "SIGMA_TOT"]:
        if name in df.columns:
            return df[name].astype(float)

    # 蒐集候選欄位
    def pick(*cands) -> pd.Series | None:
        for c in cands:
            if c in df.columns:
                return df[c].astype(float)
        return None

    s_star = pick("Sigma_star", "sigma_star", "SIGMA_STAR", "Sigma_*,star", "SigmaStar", "st_sigma")
    s_star36 = pick("Sigma_star_36", "sigma_star_36", "Sigma36", "ml36_sigma")
    s_gas = pick("Sigma_gas", "sigma_gas", "SIGMA_GAS", "Sigma_HI+H2", "SigmaGas", "gas_sigma")
    s_hi = pick("Sigma_HI", "sigma_HI")
    s_h2 = pick("Sigma_H2", "sigma_H2")

    # 若只有 star_36，需 ml36 轉質量面密度
    if s_star is None and s_star36 is not None and ml36 is not None:
        s_star = s_star36 * float(ml36)

    # 若沒有總氣體，嘗試 HI/H2 合成
    if s_gas is None and (s_hi is not None or s_h2 is not None):
        s_hi = s_hi if s_hi is not None else pd.Series(0.0, index=df.index)
        s_h2 = s_h2 if s_h2 is not None else pd.Series(0.0, index=df.index)
        s_gas = s_hi + s_h2

    # 氣體 multiplicative 調整
    if s_gas is not None:
        if rgas_mult is not None:
            s_gas = s_gas * float(rgas_mult)
        if gas_helium is not None:
            s_gas = s_gas * float(gas_helium)

    # 組合
    if s_star is not None and s_gas is not None:
        return s_star + s_gas
    if s_star is not None:
        return s_star
    if s_gas is not None:
        return s_gas

    # 無法建構，回傳 NaN series
    return pd.Series(np.nan, index=df.index, dtype=float)


def _load_exclude_keys(path: str | None) -> set[str]:
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        return set()
    df = pd.read_csv(p)
    cols = list(df.columns)
    # 嘗試找 gkey/galaxy
    for cand in ["gkey", "galaxy", "Galaxy", "name", cols[0]]:
        if cand in df.columns:
            return set(_canon_name(x) for x in df[cand].astype(str))
    # fallback: 所有第一欄
    return set(_canon_name(x) for x in df[cols[0]].astype(str))


def _load_h_outlier_keys(path: str | None) -> set[str]:
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        return set()
    df = pd.read_csv(p)
    # 允許 gkey 或 galaxy_h 皆可
    for cand in ["gkey", "galaxy_h"]:
        if cand in df.columns:
            return set(_canon_name(x) for x in df[cand].astype(str))
    # fallback: 第一欄
    return set(_canon_name(x) for x in df[df.columns[0]].astype(str))


def _ensure_gkey(df: pd.DataFrame) -> pd.DataFrame:
    if "gkey" in df.columns:
        return df
    # 嘗試以常見名稱欄位建立
    for cand in ["galaxy", "Galaxy", "name", "Name", "galaxy_s"]:
        if cand in df.columns:
            df = df.copy()
            df["gkey"] = df[cand].astype(str).map(_canon_name)
            return df
    # 若找不到，使用第一欄
    df = df.copy()
    df["gkey"] = df[df.columns[0]].astype(str).map(_canon_name)
    return df


# ---------------------------
# 主流程
# ---------------------------
def run(args: argparse.Namespace) -> Dict:
    df = pd.read_csv(args.sparc_with_h)
    df = _ensure_gkey(df)

    # 排除清單
    ex_keys = _load_exclude_keys(args.exclude_csv)
    if ex_keys:
        df = df[~df["gkey"].isin(ex_keys)]

    # 移除 h outliers
    if args.drop_h_outliers:
        ol_keys = _load_h_outlier_keys(args.h_outliers_csv)
        if ol_keys:
            df = df[~df["gkey"].isin(ol_keys)]

    # 建 Σ_tot（若缺）
    sigma_tot = _build_sigma_tot(df, args.ml36, args.rgas_mult, args.gas_helium)
    if "Sigma_tot" not in df.columns:
        df["Sigma_tot"] = sigma_tot

    # 必要欄位
    if "kappa" not in df.columns or "h_kpc" not in df.columns:
        raise RuntimeError("輸入檔缺少必要欄位（kappa / h_kpc）。請先完成特徵計算與合併。")

    # log 變數
    logk = np.log10(np.maximum(df["kappa"].astype(float).to_numpy(), TINY))
    logs = np.log10(np.maximum(df["Sigma_tot"].astype(float).to_numpy(), TINY))
    logh = np.log10(np.maximum(df["h_kpc"].astype(float).to_numpy(), TINY))

    # 權重（WLS）
    w = None
    if args.wls:
        # 優先使用 sigma_h；否則以 h_kpc_err_hi/lo 的平均
        if "sigma_h" in df.columns and np.isfinite(df["sigma_h"]).any():
            sig_lin = df["sigma_h"].astype(float).to_numpy()
        else:
            lo = df.get("h_kpc_err_lo", pd.Series(np.nan, index=df.index)).astype(float).to_numpy()
            hi = df.get("h_kpc_err_hi", pd.Series(np.nan, index=df.index)).astype(float).to_numpy()
            sig_lin = np.nanmean(np.vstack([lo, hi]), axis=0)
        # 轉為 log 空間近似誤差
        h_lin = np.maximum(df["h_kpc"].astype(float).to_numpy(), TINY)
        rel = np.where(np.isfinite(sig_lin) & (sig_lin > 0), sig_lin / h_lin, np.nan)
        sig_log = rel / np.log(10.0)
        w = 1.0 / np.square(sig_log)
        w[~np.isfinite(w)] = np.nan

    # 過濾缺值
    m = _finite_mask(logk, logs, logh)
    if w is not None:
        m &= np.isfinite(w)
    logk, logs, logh = logk[m], logs[m], logh[m]
    if w is not None:
        w = w[m]

    n_used = len(logh)
    if n_used == 0:
        raise RuntimeError("沒有可用樣本（kappa / Sigma_tot / h_kpc 有缺或權重無效）。")

    # 統計
    r, _ = pearsonr(logk, logh)
    rho, _ = spearmanr(logk, logh)

    print(f"Saved -> {args.out_csv} (rows={n_used})") if args.out_csv else None
    print(f"Sample size (used in fit): {n_used}")
    print(f"Pearson r (log–log): {r:.3f}")
    print(f"Spearman ρ (log–log): {rho:.3f}")

    # κ-only
    Xk = logk.reshape(-1, 1)
    mk = _ols_wls(Xk, logh, w)
    print("Weighted OLS (log10 h = a + b log10 κ):")
    print(f"  a = {mk.params[0]:.3f} ± {mk.bse[0]:.3f}")
    print(f"  b = {mk.params[1]:.3f} ± {mk.bse[1]:.3f}")
    print(f"  R^2 = {mk.rsquared:.3f}")

    # Σ-only
    Xs = logs.reshape(-1, 1)
    ms = _ols_wls(Xs, logh, w)

    # κ + Σ
    X2 = np.stack([logk, logs], axis=1)
    m2 = _ols_wls(X2, logh, w)

    aicc_k = _aicc(n_used, k=2, rss=_rss(mk))  # const+1
    aicc_s = _aicc(n_used, k=2, rss=_rss(ms))
    aicc_ks = _aicc(n_used, k=3, rss=_rss(m2))  # const+2

    print(f"[log h ~ log κ + log Sigma_tot] N={n_used}, p=3")
    print(
        f"  beta: {m2.params[0]:.3f}±{m2.bse[0]:.3f}, "
        f"{m2.params[1]:.3f}±{m2.bse[1]:.3f}, "
        f"{m2.params[2]:.3f}±{m2.bse[2]:.3f}"
    )
    print(f"  R^2 = {m2.rsquared:.3f}, AICc = {aicc_ks:.2f}")
    print(f"  AICc (κ-only/Σ-only/κ+Σ) = {aicc_k:.2f} / {aicc_s:.2f} / {aicc_ks:.2f}")
    print(
        f"  ΔAICc = {{'kappa_only': {aicc_k - aicc_ks}, 'sigma_only': {aicc_s - aicc_ks}, 'kappa_sigma': 0.0}}"
    )

    # LOO
    if args.loo:
        mu, sd = _loo_coeffs(X2, logh, w)
        print(f"  LOO mean: {list(mu)}")
        print(f"  LOO std : {list(sd)}")

    # Bootstrap
    if args.bootstrap and args.bootstrap > 0:
        med, lo, hi = _bootstrap_coeffs(X2, logh, w, nboot=args.bootstrap, seed=42)
        print(f"  Bootstrap medians: {list(med)}")
        print(f"  16–84 percentiles: low={list(lo)}  high={list(hi)}")

    # --------------------------------
    # 輸出 CSV（用到的樣本與 log 值）
    # --------------------------------
    if args.out_csv:
        df_used = pd.DataFrame(
            {
                "log10_h": logh,
                "log10_kappa": logk,
                "log10_Sigma_tot": logs,
            }
        )
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        df_used.to_csv(args.out_csv, index=False)
        print(f"Saved -> {args.out_csv} (rows={len(df_used)})")

    # ---------------
    # 畫圖（可選）
    # ---------------
    if args.out_plot:
        Path(args.out_plot).parent.mkdir(parents=True, exist_ok=True)
        # 圖1：log κ vs log h（含 κ-only 擬合線）
        plt.figure(figsize=(5, 4))
        plt.scatter(logk, logh, s=28, alpha=0.85)
        # 擬合線
        xgrid = np.linspace(logk.min() - 0.1, logk.max() + 0.1, 200)
        ygrid = mk.params[0] + mk.params[1] * xgrid
        plt.plot(xgrid, ygrid, linewidth=2)
        plt.xlabel(r"$\log_{10}\,\kappa$")
        plt.ylabel(r"$\log_{10}\,h\ \mathrm{(kpc)}$")
        plt.tight_layout()
        plt.savefig(args.out_plot, dpi=180)
        print(f"Saved plot -> {args.out_plot}")

        # 圖2：同一張散點，但用 log Σ 顏色呈現（對應 _sigma）
        root = Path(args.out_plot)
        out_sigma = root.with_name(root.stem.replace(".png", "") + "_sigma.png") if root.suffix == "" else root.with_name(root.stem + "_sigma" + root.suffix)
        plt.figure(figsize=(5, 4))
        sc = plt.scatter(logk, logh, s=28, c=logs, alpha=0.9)
        plt.xlabel(r"$\log_{10}\,\kappa$")
        plt.ylabel(r"$\log_{10}\,h\ \mathrm{(kpc)}$")
        cb = plt.colorbar(sc)
        cb.set_label(r"$\log_{10}\,\Sigma_{\rm tot}$")
        plt.tight_layout()
        plt.savefig(out_sigma, dpi=180)
        print(f"Saved plot -> {out_sigma}")

    # -----------------------
    # 距離不變（DIST-INV）
    # -----------------------
    dist_col = args.dist_col or "D_Mpc_h"
    use_dist = dist_col if dist_col in df.columns else ("D_Mpc_gal" if "D_Mpc_gal" in df.columns else None)
    if use_dist is None:
        print("[DIST-INV] skipped: no distance column found")
        dist_report = {"dist_col_used": None, "beta": [np.nan, np.nan, np.nan], "bse": [np.nan, np.nan, np.nan], "R2": np.nan}
        dist_n = 0
        md = None
    else:
        D = np.maximum(df[use_dist].astype(float).to_numpy(), TINY)[m]
        y_d = np.log10(np.maximum(df["h_kpc"].astype(float).to_numpy()[m] / D, TINY))  # log(h/D)
        x1_d = np.log10(np.maximum(df["kappa"].astype(float).to_numpy()[m] * D, TINY))  # log(κD)
        x2_d = logs  # log Σ_tot（與前一致）
        Xd = np.stack([x1_d, x2_d], axis=1)
        md = _ols_wls(Xd, y_d, w)
        dist_n = len(y_d)
        print(
            f"[DIST-INV] log(h/D) = {md.params[0]:.3f} + {md.params[1]:.3f} log(κD) + {md.params[2]:.3f} log Σ ; R^2={md.rsquared:.3f}"
        )
        dist_report = {
            "dist_col_used": use_dist,
            "beta": md.params.tolist(),
            "bse": md.bse.tolist(),
            "R2": float(md.rsquared),
        }

    # -----------------------
    # JSON 報告（與 summarize 腳本相容）
    # -----------------------
    report = {
        # 主要（κ+Σ）
        "a": float(m2.params[0]),
        "a_se": float(m2.bse[0]),
        "b_kappa": float(m2.params[1]),
        "b_se": float(m2.bse[1]),
        "c_sigma": float(m2.params[2]),
        "c_se": float(m2.bse[2]),
        "R2": float(m2.rsquared),
        "AICc": float(aicc_ks),
        "dAIC_kappa_only": float(aicc_k - aicc_ks),
        "dAIC_sigma_only": float(aicc_s - aicc_ks),
        "dAIC_kappa_sigma": 0.0,
        # 距離不變
        "dist_n": int(dist_n),
        "dist_b_kappa": float(md.params[1]) if md is not None else np.nan,
        "dist_c_sigma": float(md.params[2]) if md is not None else np.nan,
        "dist_R2": float(md.rsquared) if md is not None else np.nan,
        # 附帶原始區塊（保留向下相容）
        "aicc": {"kappa_only": float(aicc_k), "sigma_only": float(aicc_s), "kappa_sigma": float(aicc_ks)},
        "params": {
            "kappa_only": {"beta": mk.params.tolist(), "bse": mk.bse.tolist(), "R2": float(mk.rsquared)},
            "kappa_sigma": {"beta": m2.params.tolist(), "bse": m2.bse.tolist(), "R2": float(m2.rsquared)},
            "dist_inv": dist_report,
        },
        "n_used": int(n_used),
    }

    if args.report_json:
        Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Saved report -> {args.report_json}")

    return report


# ---------------------------
# CLI
# ---------------------------
def main():
    ap = argparse.ArgumentParser(prog="ptq.experiments.kappa_h")

    # I/O
    ap.add_argument("--sparc-with-h", type=str, required=True, help="merged CSV（含 kappa / h_kpc；若可，最好含 Sigma_tot 與 D_Mpc_h）")
    ap.add_argument("--out-csv", type=str, default=None, help="輸出用樣本（log 值）")
    ap.add_argument("--out-plot", type=str, default=None, help="散點圖輸出檔（會再產生 _sigma 版）")
    ap.add_argument("--report-json", type=str, default=None, help="回歸摘要報告 JSON")

    # 權重與健壯度分析
    ap.add_argument("--wls", action="store_true", help="使用 WLS（以 h 的誤差推估 log-space 權重）")
    ap.add_argument("--loo", action="store_true", help="進行留一交叉驗證（係數平均／標準差）")
    ap.add_argument("--bootstrap", type=int, default=0, help="bootstrap 次數（0 表示不執行）")

    # Σ_tot 組合（若輸入檔沒有 Sigma_tot 時才會用到這些）
    ap.add_argument("--use-total-sigma", action="store_true", help="保留原旗標供腳本相容（實際上自動偵測/建構 Sigma_tot）")
    ap.add_argument("--use-exp", action="store_true", help="相容旗標（不影響此檔計算）")
    ap.add_argument("--ml36", type=float, default=None, help="M/L (3.6μm) 乘數（若僅提供星光面密度 Sigma_star_36）")
    ap.add_argument("--rgas-mult", type=float, default=None, help="氣體倍率（如需修正）")
    ap.add_argument("--gas-helium", type=float, default=None, help="氦修正倍率（如 1.33）")

    # 資料過濾
    ap.add_argument("--drop-h-outliers", action="store_true", help="移除 h 的 outliers（需要 --h-outliers-csv）")
    ap.add_argument("--h-outliers-csv", type=str, default=None, help="由 build-h 產出的 outliers CSV")
    ap.add_argument("--exclude-csv", type=str, default=None, help="自訂排除清單（含 gkey 或 galaxy 欄位）")

    # DIST-INV 距離欄位
    ap.add_argument("--dist-col", type=str, default="D_Mpc_h", help="距離欄位（預設使用 S4G 的 D_Mpc_h；若無則回退 D_Mpc_gal）")

    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
