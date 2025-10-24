# src/ptq/experiments/kappa_h.py
import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# matplotlib 是可選；沒有就跳過畫圖
try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


# -----------------------------
# 小工具
# -----------------------------
def _finite(arr):
    return np.isfinite(arr)


def _smooth_series(x, y, window=5):
    """簡單 rolling mean 平滑（置中、缺值跳過），確保 window 為奇數且不超過資料長度。"""
    n = len(y)
    if n < 3:
        return y.copy()
    w = max(3, min(window, n if n % 2 == 1 else n - 1))
    s = pd.Series(y).rolling(window=w, center=True, min_periods=1).mean().to_numpy()
    return s


def _kappa_profile(r_kpc, v_kms, v_err=None):
    """
    由 (r, V) 計算 κ(R)；單位≈(km/s)/kpc。回傳 (r, κ, κ_err)。
    κ^2 = 2 (V/R)^2 (1 + d ln V / d ln R)
    """
    r = np.asarray(r_kpc, dtype=float)
    v = np.asarray(v_kms, dtype=float)

    ok = _finite(r) & _finite(v)
    r, v = r[ok], v[ok]
    if v_err is not None:
        v_e = np.asarray(v_err, dtype=float)[ok]
    else:
        v_e = None

    if len(r) < 3:
        return r, np.full_like(r, np.nan), (np.full_like(r, np.nan) if v_e is not None else None)

    order = np.argsort(r)
    r = r[order]
    v = v[order]
    if v_e is not None:
        v_e = v_e[order]

    v_s = _smooth_series(r, v, window=5)
    dvdr = np.gradient(v_s, r)
    with np.errstate(divide="ignore", invalid="ignore"):
        dlnv_dlnr = (r / v_s) * dvdr
        term = 1.0 + dlnv_dlnr
        term[term < 0] = np.nan
        kappa = np.sqrt(2.0) * (v_s / r) * np.sqrt(term)

    # 由 V 誤差粗略傳播 κ 的不確定度：以 V±σ 重新計算 κ，取差的一半
    if v_e is not None:
        v_hi = v_s + v_e
        v_lo = np.clip(v_s - v_e, 1e-6, None)
        dvdr_hi = np.gradient(v_hi, r)
        dvdr_lo = np.gradient(v_lo, r)
        with np.errstate(divide="ignore", invalid="ignore"):
            dln_hi = (r / v_hi) * dvdr_hi
            dln_lo = (r / v_lo) * dvdr_lo
            term_hi = 1.0 + dln_hi
            term_lo = 1.0 + dln_lo
            term_hi[term_hi < 0] = np.nan
            term_lo[term_lo < 0] = np.nan
            k_hi = np.sqrt(2.0) * (v_hi / r) * np.sqrt(term_hi)
            k_lo = np.sqrt(2.0) * (v_lo / r) * np.sqrt(term_lo)
        k_err = 0.5 * np.abs(k_hi - k_lo)
    else:
        k_err = None

    return r, kappa, k_err


def _pick_characteristic_kappa(df_one: pd.DataFrame) -> Tuple[float, float, float, str, float]:
    """
    挑 κ 的代表值：
      1) 找 v_disk 最大的半徑；若全為 NaN，找 v_obs 最大的半徑
      2) 在該半徑插值 κ（就近取樣）
    回傳: (kappa_star, r_star, v_at_r, how, kappa_err)
    """
    r = df_one["r_kpc"].to_numpy()
    v = df_one["v_obs_kms"].to_numpy()
    v_err = df_one["v_err_kms"].to_numpy() if "v_err_kms" in df_one else None

    r_prof, k_prof, k_err = _kappa_profile(r, v, v_err=v_err)

    if df_one["v_disk_kms"].notna().any():
        idx = int(df_one["v_disk_kms"].to_numpy().argmax())
        how = "R_at_max_vdisk"
    else:
        idx = int(df_one["v_obs_kms"].to_numpy().argmax())
        how = "R_at_max_vobs"

    r_star = float(df_one["r_kpc"].iloc[idx])
    if len(r_prof) == 0 or not np.isfinite(r_star):
        return (np.nan, np.nan, np.nan, how, np.nan)

    j = int(np.nanargmin(np.abs(r_prof - r_star)))
    k_star = float(k_prof[j]) if np.isfinite(k_prof[j]) else np.nan
    v_star = float(df_one["v_obs_kms"].iloc[idx])
    k_err_star = float(k_err[j]) if (k_err is not None and np.isfinite(k_err[j])) else np.nan
    return (k_star, r_star, v_star, how, k_err_star)


def _first_non_nan(s: pd.Series) -> float:
    return s.dropna().iloc[0] if s.notna().any() else np.nan


def _weighted_ols(X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    加權/不加權最小平方法。
    回傳 (beta, se_beta, R2)，其中 beta=[a,b,...]、se_beta 同長度。
    """
    n, p = X.shape
    if w is None:
        W = np.eye(n)
        ybar = y.mean()
    else:
        W = np.diag(w)
        ybar = np.average(y, weights=w)

    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ y
    beta = np.linalg.solve(XtWX, XtWy)
    yhat = X @ beta
    res = y - yhat

    # R^2（加權 / 不加權）
    if w is None:
        ss_res = float(np.sum(res**2))
        ss_tot = float(np.sum((y - ybar)**2))
    else:
        ss_res = float(np.sum(w * res**2))
        ss_tot = float(np.sum(w * (y - ybar)**2))
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    dof = n - p
    s2 = ss_res / dof if dof > 0 else np.nan
    cov = s2 * np.linalg.inv(XtWX)
    se = np.sqrt(np.diag(cov))
    return beta, se, R2


def _aicc(n: int, rss: float, k: int) -> float:
    # k 含截距；AICc = n ln(rss/n) + 2k + 2k(k+1)/(n-k-1)
    return n * math.log(rss / n) + 2 * k + (2 * k * (k + 1)) / (n - k - 1)


def _loo_ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Leave-One-Out：回傳每次擬合的 beta 堆疊成 (n, p)"""
    n = len(y)
    out = []
    for i in range(n):
        m = np.ones(n, dtype=bool)
        m[i] = False
        b, _, _ = _weighted_ols(X[m], y[m], w=None)
        out.append(b)
    return np.vstack(out)


def _bootstrap_WLS(rng: np.random.Generator, X: np.ndarray, y: np.ndarray, w: np.ndarray, B: int = 2000) -> np.ndarray:
    n = len(y)
    out = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        b, _, _ = _weighted_ols(X[idx], y[idx], w=w[idx])
        out.append(b)
    return np.vstack(out)


def _cook_vif(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Cook's D 與兩變數的 VIF（若只有一個解釋變數就回傳 nan）。"""
    beta, _, _ = _weighted_ols(X, y, None)
    yhat = X @ beta
    res = y - yhat
    H = X @ np.linalg.inv(X.T @ X) @ X.T
    h = np.diag(H)
    p = X.shape[1]
    s2 = (res**2).sum() / (len(y) - p)
    D = (res**2 / (p * s2)) * (h / (1 - h) ** 2)  # Cook's D
    if p >= 3:
        x1 = X[:, 1]
        x2 = X[:, 2]
        r12 = np.corrcoef(x1, x2)[0, 1]
        VIF = 1 / (1 - r12**2)
    else:
        VIF = float("nan")
    return D, float(np.nanmax(D)), float(VIF)


# -----------------------------
# 主流程
# -----------------------------
def run(
    sparc_with_h: str,
    out_csv: str,
    out_plot: Optional[str] = None,
    min_points: int = 6,
    use_stellar_sigma: bool = False,
    use_total_sigma: bool = False,
    use_exp: bool = False,
    ml36: float = 0.5,
    rgas_mult: float = 1.7,
    gas_helium: float = 1.33,
    do_wls: bool = False,
    do_loo: bool = False,
    bootstrap: int = 0,
    report_json: Optional[str] = None,
    seed: int = 42,
) -> int:

    df = pd.read_csv(sparc_with_h)
    # 只留有 h 的星系（保留全部徑向點）
    df = df[df["h_kpc"].notna()].copy()

    # galaxy-level 快取（避免每迴圈 groupby）
    g_agg = (
        df.groupby("gkey", as_index=False)
          .agg(
              D_Mpc_gal=("D_Mpc", "median"),
              L36_disk_first=("L36_disk", _first_non_nan),
              L36_tot=("L36_tot", "max"),
              L36_bulge=("L36_bulge", "max"),
              M_HI=("M_HI", "max"),
          )
    )
    g_agg["L36_bulge"] = g_agg["L36_bulge"].fillna(0.0)
    g_agg["L36_disk_gal"] = g_agg["L36_disk_first"].fillna(g_agg["L36_tot"] - g_agg["L36_bulge"])
    g_agg = g_agg.drop(columns=["L36_disk_first"])

    # 逐星系建立特徵
    rows: List[Dict] = []
    for gkey, sub in df.groupby("gkey", as_index=False):
        sub = sub.sort_values("r_kpc")
        if sub.shape[0] < min_points:
            continue

        k_star, r_star, v_star, how, k_err = _pick_characteristic_kappa(sub)
        if not np.isfinite(k_star) or not np.isfinite(r_star):
            continue

        # 取 galaxy-level 參數
        gg = g_agg[g_agg["gkey"] == gkey]
        L36_disk = float(gg["L36_disk_gal"].iloc[0]) if len(gg) else np.nan
        M_HI = float(gg["M_HI"].iloc[0]) if len(gg) else np.nan
        D_Mpc_gal = float(gg["D_Mpc_gal"].iloc[0]) if len(gg) else np.nan

        # 由 r_char 推 Rd
        Rd_kpc = r_star / 2.2

        # 星光面密度（單位以 1e9 Msun/kpc^2 作相對量，在回歸只影響截距）
        Sigma_star0 = ml36 * L36_disk / (2 * math.pi * Rd_kpc**2) if (np.isfinite(L36_disk) and Rd_kpc > 0) else np.nan
        Sigma_star = Sigma_star0 * math.exp(-r_star / Rd_kpc) if (use_exp and np.isfinite(Sigma_star0)) else Sigma_star0

        # 氣體面密度（指數盤、R_gas ≈ rgas_mult * Rd；乘 1.33 補 He）
        Rgas = rgas_mult * Rd_kpc
        Sigma_gas0 = (gas_helium * M_HI) / (2 * math.pi * Rgas**2) if (np.isfinite(M_HI) and Rgas > 0) else np.nan
        Sigma_gas = Sigma_gas0 * math.exp(-r_star / Rgas) if (use_exp and np.isfinite(Sigma_gas0)) else Sigma_gas0

        if use_total_sigma:
            Sigma_tot = np.nansum([Sigma_star, Sigma_gas])
        else:
            Sigma_tot = Sigma_star  # 若用 stellar 或沒選任何，就等同 stellar

        # h 的對稱誤差（若只提供正/負方向，取平均）
        h = float(sub["h_kpc"].iloc[0])
        hp = sub["h_err_plus_kpc"].iloc[0] if "h_err_plus_kpc" in sub else np.nan
        hm = sub["h_err_minus_kpc"].iloc[0] if "h_err_minus_kpc" in sub else np.nan
        if np.isfinite(hp) and np.isfinite(hm):
            h_sigma = 0.5 * (float(hp) + float(hm))
        elif np.isfinite(hp):
            h_sigma = float(hp)
        elif np.isfinite(hm):
            h_sigma = float(hm)
        else:
            h_sigma = np.nan

        rows.append(dict(
            gkey=gkey,
            galaxy=sub["galaxy"].iloc[0] if "galaxy" in sub else gkey,
            kappa_char_kms_per_kpc=k_star,
            kappa_char_err=k_err,
            r_char_kpc=r_star,
            vobs_at_rchar_kms=v_star,
            choose_rule=how,
            h_kpc=h,
            h_sigma=h_sigma,
            Rd_kpc=Rd_kpc,
            Sigma_star=Sigma_star,
            Sigma_gas=Sigma_gas,
            Sigma_tot=Sigma_tot,
            D_Mpc=D_Mpc_gal,
        ))

    out = pd.DataFrame(rows)
    # 只保留 κ 與 h 皆有限的樣本
    out = out[np.isfinite(out["kappa_char_kms_per_kpc"]) & np.isfinite(out["h_kpc"])].copy()
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"Saved -> {out_csv} (rows={len(out)})")

    # ------- (A) 舊版 κ–h：保持相容輸出 -------
    x_k = np.log10(out["kappa_char_kms_per_kpc"].to_numpy())
    y_h = np.log10(out["h_kpc"].to_numpy())
    m = np.isfinite(x_k) & np.isfinite(y_h)
    pearson = np.corrcoef(x_k[m], y_h[m])[0, 1] if m.sum() >= 2 else np.nan
    rx = pd.Series(x_k[m]).rank(method="average").to_numpy()
    ry = pd.Series(y_h[m]).rank(method="average").to_numpy()
    spearman = np.corrcoef(rx, ry)[0, 1] if m.sum() >= 2 else np.nan

    y_sigma_log = None
    if out["h_sigma"].notna().any():
        y_sigma_log = (out["h_sigma"] / out["h_kpc"]) / math.log(10.0)
    if y_sigma_log is not None and np.all(np.isfinite(y_sigma_log)):
        w1 = 1 / np.square(y_sigma_log)
        X1 = np.vstack([np.ones(len(x_k[m])), x_k[m]]).T
        b1, se1, R2_1 = _weighted_ols(X1, y_h[m], w=w1[m])
    else:
        X1 = np.vstack([np.ones(len(x_k[m])), x_k[m]]).T
        b1, se1, R2_1 = _weighted_ols(X1, y_h[m], w=None)

    print(f"Sample size (used in fit): {int(m.sum())}")
    print(f"Pearson r (log–log): {pearson:.3f}")
    print(f"Spearman ρ (log–log): {spearman:.3f}")
    print("Weighted OLS (log10 h = a + b log10 κ):")
    print(f"  a = {b1[0]:.3f} ± {se1[0]:.3f}")
    print(f"  b = {b1[1]:.3f} ± {se1[1]:.3f}")
    print(f"  R^2 = {R2_1:.3f}")

    # 預設畫圖：沿用舊圖（h vs κ）
    if out_plot is not None and HAVE_MPL and int(m.sum()) >= 2:
        fig, ax = plt.subplots(figsize=(5, 4))
        if out["h_sigma"].notna().any():
            ax.errorbar(out["kappa_char_kms_per_kpc"], out["h_kpc"],
                        yerr=out["h_sigma"], fmt="o", alpha=0.8, label="galaxies")
        else:
            ax.plot(out["kappa_char_kms_per_kpc"], out["h_kpc"], "o", label="galaxies")
        xx = np.linspace(np.nanmin(out["kappa_char_kms_per_kpc"]), np.nanmax(out["kappa_char_kms_per_kpc"]), 200)
        yy = (10 ** b1[0]) * (xx ** b1[1])
        ax.plot(xx, yy, "-", lw=2, label=f"fit: h = 10^{b1[0]:.2f} κ^{b1[1]:.2f}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(r"$\kappa_\star\;[{\rm km\,s^{-1}\,kpc^{-1}}]$")
        ax.set_ylabel(r"$h\;[{\rm kpc}]$")
        ax.grid(True, which="both", alpha=0.3); ax.legend()
        fig.tight_layout()
        Path(out_plot).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_plot, dpi=200)
        print(f"Saved plot -> {out_plot}")
    elif out_plot is not None and not HAVE_MPL:
        print("matplotlib not available; skip plot.")

    # ------- (B) 多變量：加入 Σ -------
    do_multivar = (use_stellar_sigma or use_total_sigma)
    report: Dict = {}
    if do_multivar:
        # 選擇變數
        sigma_col = "Sigma_tot" if use_total_sigma else "Sigma_star"
        # 清洗
        jj = out.dropna(subset=[sigma_col]).copy()
        jj = jj[(jj["h_kpc"] > 0) & (jj["kappa_char_kms_per_kpc"] > 0) & (jj[sigma_col] > 0)]
        if len(jj) < 3:
            print(f"[WARN] multivariate needs >=3 galaxies; got {len(jj)} after cleaning.")
            # 仍輸出單變量結果
        else:
            X = np.vstack([
                np.ones(len(jj)),
                np.log10(jj["kappa_char_kms_per_kpc"].to_numpy()),
                np.log10(jj[sigma_col].to_numpy())
            ]).T
            y = np.log10(jj["h_kpc"].to_numpy())

            w = None
            if do_wls and jj["h_sigma"].notna().any():
                sig_logh = (jj["h_sigma"] / jj["h_kpc"]) / math.log(10.0)
                m_w = np.isfinite(sig_logh) & (sig_logh > 0)
                # 若有極少數權重無法算，統一退回 OLS 以簡化流程
                if m_w.sum() == len(jj):
                    w = 1 / np.square(sig_logh.to_numpy())

            beta, se, R2 = _weighted_ols(X, y, w=w)
            yhat = X @ beta
            rss = float(((y - yhat) ** 2 if w is None else (w * (y - yhat) ** 2)).sum())
            k_eff = X.shape[1]  # 含截距
            aicc = _aicc(len(y), rss, k_eff)

            print(f"[log h ~ log κ + log {sigma_col}] N={len(jj)}, p=3")
            print(f"  beta: {beta[0]:.3f}±{se[0]:.3f}, {beta[1]:.3f}±{se[1]:.3f}, {beta[2]:.3f}±{se[2]:.3f}")
            print(f"  R^2 = {R2:.3f}, AICc = {aicc:.2f}")

            # 診斷
            Di, Dmax, VIF = _cook_vif(X, y)
            # LOO / bootstrap
            loo_stats = {}
            boot_stats = {}
            if do_loo:
                B_loo = _loo_ols(X, y)
                loo_stats = {
                    "mean": B_loo.mean(axis=0).tolist(),
                    "std":  B_loo.std(axis=0, ddof=1).tolist(),
                }
                print(f"  LOO mean: {loo_stats['mean']}")
                print(f"  LOO std : {loo_stats['std']}")
            if bootstrap and bootstrap > 0:
                rng = np.random.default_rng(seed)
                B_boot = _bootstrap_WLS(rng, X, y, w=(w if w is not None else np.ones(len(y))), B=bootstrap)
                pct = np.percentile(B_boot, [16, 50, 84], axis=0)
                boot_stats = {
                    "p16": pct[0].tolist(),
                    "p50": pct[1].tolist(),
                    "p84": pct[2].tolist(),
                }
                print(f"  Bootstrap medians: {boot_stats['p50']}")
                print(f"  16–84 percentiles: low={boot_stats['p16']}  high={boot_stats['p84']}")

            # 輸出第二張圖：h 對 X=Σ/κ^2
            if out_plot is not None and HAVE_MPL and len(jj) >= 3:
                fig2, ax2 = plt.subplots(figsize=(5, 4))
                Xaxis = jj[sigma_col].to_numpy() / np.square(jj["kappa_char_kms_per_kpc"].to_numpy())
                ax2.plot(Xaxis, jj["h_kpc"], "o", label="galaxies")
                # 以回歸係數畫指數 1,-2 的等效線：h ∝ Σ κ^-2
                xx = np.logspace(np.log10(np.nanmin(Xaxis)), np.log10(np.nanmax(Xaxis)), 200)
                # 用回歸係數重建：log h = a + b log κ + c log Σ
                # 換成 X=Σ/κ^2 時，斜率 ~ c 對 log Σ、~(-2b) 對 κ^2；這裡用直接代值畫線較簡單
                # 取代表 κ 與 Σ 比例，以中位數還原
                k_med = np.median(jj["kappa_char_kms_per_kpc"].to_numpy())
                s_med = np.median(jj[sigma_col].to_numpy())
                # 解出 a' 使線通過 (s_med/k_med^2, h_med)
                h_med = np.median(jj["h_kpc"].to_numpy())
                # 直接用模型在 (k_med, s_med) 的預測做為錨點
                h_med_fit = 10 ** (beta[0] + beta[1] * math.log10(k_med) + beta[2] * math.log10(s_med))
                # 假設 h ∝ X^{beta2}，用 beta2 作斜率（在 log–log）
                yy = h_med_fit * (xx / (s_med / (k_med**2))) ** beta[2]
                ax2.plot(xx, yy, "-", lw=2, label=f"fit ~ X^{beta[2]:.2f}")
                ax2.set_xscale("log"); ax2.set_yscale("log")
                ax2.set_xlabel(r"$X \equiv \Sigma / \kappa^2$  (arb. units)")
                ax2.set_ylabel(r"$h\;[{\rm kpc}]$")
                ax2.grid(True, which="both", alpha=0.3); ax2.legend()
                fig2.tight_layout()
                p2 = Path(out_plot).with_name(Path(out_plot).stem + "_sigma.png")
                fig2.savefig(p2, dpi=200)
                print(f"Saved plot -> {p2}")

            # 整理報表
            report = {
                "n": int(len(jj)),
                "model": f"log10(h) = a + b log10(kappa) + c log10({sigma_col})",
                "beta": beta.tolist(),
                "se_beta": se.tolist(),
                "R2": R2,
                "AICc": aicc,
                "sigma_col": sigma_col,
                "diagnostics": {
                    "cooksD_max": float(Dmax),
                    "VIF_kappa_sigma": float(VIF),
                    "top5_cooksD_indices": np.argsort(Di)[-5:][::-1].tolist(),
                },
                "loo": loo_stats,
                "bootstrap": boot_stats,
                "units": {
                    "kappa": "km/s/kpc",
                    "h": "kpc",
                    "Sigma": "1e9 Msun/kpc^2 (relative units; intercept shifts with unit choices)",
                }
            }

    # 報表輸出（含單變量結果）
    if report_json:
        core = {
            "kappa_only": {
                "n": int(m.sum()),
                "beta": b1.tolist(),
                "se_beta": se1.tolist(),
                "R2": R2_1,
                "pearson": float(pearson),
                "spearman": float(spearman),
                "model": "log10(h) = a + b log10(kappa)",
            },
            "multivar": report if do_multivar else {},
            "paths": {
                "in": sparc_with_h,
                "out_csv": out_csv,
                "out_plot": out_plot,
            }
        }
        Path(report_json).parent.mkdir(parents=True, exist_ok=True)
        with open(report_json, "w", encoding="utf-8") as f:
            json.dump(core, f, ensure_ascii=False, indent=2)
        print(f"Saved report -> {report_json}")

    return 0


def main():
    p = argparse.ArgumentParser(description="κ–h / (κ,Σ)–h experiment on SPARC×S4G")
    p.add_argument("--sparc-with-h", type=str, required=True, help="dataset/geometry/sparc_with_h.csv")
    p.add_argument("--out-csv", type=str, default="dataset/geometry/kappa_h_sample.csv")
    p.add_argument("--out-plot", type=str, default="dataset/geometry/kappa_h_scatter.png")
    p.add_argument("--min-points", type=int, default=6, help="min radial points per galaxy")

    # 新增：Σ 與模型選項
    g = p.add_mutually_exclusive_group()
    g.add_argument("--use-stellar-sigma", action="store_true", help="use Σ_* only")
    g.add_argument("--use-total-sigma", action="store_true", help="use Σ_tot = Σ_* + Σ_gas")
    p.add_argument("--use-exp", action="store_true", help="evaluate Σ at r_char: multiply exp(-r/Rd) (and gas exp(-r/Rgas))")

    p.add_argument("--ml36", type=float, default=0.5, help="stellar M/L at 3.6μm (default 0.5)")
    p.add_argument("--rgas-mult", type=float, default=1.7, help="R_gas = rgas_mult * Rd (default 1.7)")
    p.add_argument("--gas-helium", type=float, default=1.33, help="Helium correction factor for HI (default 1.33)")

    p.add_argument("--wls", action="store_true", help="weighted least squares using h uncertainties")
    p.add_argument("--loo", action="store_true", help="leave-one-out coefficients")
    p.add_argument("--bootstrap", type=int, default=0, help="bootstrap iterations for WLS (0 to disable)")
    p.add_argument("--report-json", type=str, default=None, help="save full metrics to JSON")
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    return sys.exit(run(
        sparc_with_h=args.sparc_with_h,
        out_csv=args.out_csv,
        out_plot=args.out_plot,
        min_points=args.min_points,
        use_stellar_sigma=args.use_stellar_sigma,
        use_total_sigma=args.use_total_sigma,
        use_exp=args.use_exp,
        ml36=args.ml36,
        rgas_mult=args.rgas_mult,
        gas_helium=args.gas_helium,
        do_wls=args.wls,
        do_loo=args.loo,
        bootstrap=args.bootstrap,
        report_json=args.report_json,
        seed=args.seed,
    ))


if __name__ == "__main__":
    main()
