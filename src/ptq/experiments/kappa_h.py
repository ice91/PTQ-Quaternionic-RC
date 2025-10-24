# src/ptq/experiments/kappa_h.py
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# matplotlib 是可選；沒有就跳過畫圖
try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


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
    """由 (r, V) 計算 κ(R)；單位≈(km/s)/kpc。回傳 κ 及一個由 V 誤差近似的 κ 誤差。"""
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

    # 平滑 + 數值微分
    v_s = _smooth_series(r, v, window=5)
    dvdr = np.gradient(v_s, r)
    with np.errstate(divide="ignore", invalid="ignore"):
        dlnv_dlnr = (r / v_s) * dvdr
        term = 1.0 + dlnv_dlnr
        term[term < 0] = np.nan  # 噪音造成的負值捨棄
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


def _pick_characteristic_kappa(df_one):
    """
    挑 κ 的代表值：
      1) 找 v_disk 最大的半徑；若全為 NaN，找 v_obs 最大的半徑
      2) 在該半徑插值 κ（就近取樣）
    回傳: (kappa_star, r_star, v_at_r, how)
    """
    r = df_one["r_kpc"].to_numpy()
    v = df_one["v_obs_kms"].to_numpy()
    v_err = df_one["v_err_kms"].to_numpy() if "v_err_kms" in df_one else None

    r_prof, k_prof, k_err = _kappa_profile(r, v, v_err=v_err)

    # 代表半徑：先看 v_disk，否則用 v_obs
    if df_one["v_disk_kms"].notna().any():
        idx = df_one["v_disk_kms"].to_numpy().argmax()
        how = "R_at_max_vdisk"
    else:
        idx = df_one["v_obs_kms"].to_numpy().argmax()
        how = "R_at_max_vobs"

    r_star = float(df_one["r_kpc"].iloc[int(idx)])
    # 在 κ(R) 上找就近點
    if len(r_prof) == 0 or not np.isfinite(r_star):
        return (np.nan, np.nan, np.nan, how, np.nan)

    j = np.nanargmin(np.abs(r_prof - r_star))
    k_star = float(k_prof[j]) if np.isfinite(k_prof[j]) else np.nan
    v_star = float(df_one["v_obs_kms"].iloc[int(idx)])
    k_err_star = float(k_err[j]) if (k_err is not None and np.isfinite(k_err[j])) else np.nan
    return (k_star, r_star, v_star, how, k_err_star)


def _weighted_ols(x, y, y_sigma=None):
    """在 log–log 空間做加權 OLS： y = a + b x。回傳 (a, b, a_err, b_err, R2, n)"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    m = np.isfinite(x) & np.isfinite(y)
    if y_sigma is not None:
        s = np.asarray(y_sigma, float)
        m = m & np.isfinite(s) & (s > 0)
        w = 1.0 / (s[m] ** 2)
    else:
        w = np.ones(m.sum())

    x, y = x[m], y[m]
    if len(x) < 3:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, len(x))

    X = np.vstack([np.ones_like(x), x]).T
    W = np.diag(w)
    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ y
    beta = np.linalg.solve(XtWX, XtWy)
    a, b = beta

    # 殘差與 R2
    yhat = X @ beta
    ss_res = float(((y - yhat) ** 2 * w).sum())
    ss_tot = float(((y - np.average(y, weights=w)) ** 2 * w).sum())
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # 共變異矩陣（加權殘差方差）
    dof = len(x) - 2
    s2 = ss_res / dof if dof > 0 else np.nan
    cov = s2 * np.linalg.inv(XtWX)
    a_err = float(np.sqrt(cov[0, 0])) if np.isfinite(cov[0, 0]) else np.nan
    b_err = float(np.sqrt(cov[1, 1])) if np.isfinite(cov[1, 1]) else np.nan
    return (float(a), float(b), a_err, b_err, float(R2), int(len(x)))


def run(sparc_with_h, out_csv, out_plot=None, min_points=6):
    df = pd.read_csv(sparc_with_h)
    # 只留有 h 的星系
    df = df[df["h_kpc"].notna()].copy()

    results = []
    for gkey, sub in df.groupby("gkey", as_index=False):
        # 資料品質：至少 min_points 個徑向點
        sub = sub.sort_values("r_kpc")
        if sub.shape[0] < min_points:
            continue

        k_star, r_star, v_star, how, k_err = _pick_characteristic_kappa(sub)

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

        results.append({
            "gkey": gkey,
            "galaxy": sub["galaxy"].iloc[0] if "galaxy" in sub else gkey,
            "kappa_char_kms_per_kpc": k_star,
            "kappa_char_err": k_err,
            "r_char_kpc": r_star,
            "vobs_at_rchar_kms": v_star,
            "choose_rule": how,
            "h_kpc": h,
            "h_sigma": h_sigma
        })

    out = pd.DataFrame(results)
    out = out[np.isfinite(out["kappa_char_kms_per_kpc"]) & np.isfinite(out["h_kpc"])].copy()
    out.to_csv(out_csv, index=False)
    print(f"Saved -> {out_csv} (rows={len(out)})")

    # 相關性與回歸（log–log）
    x = np.log10(out["kappa_char_kms_per_kpc"].to_numpy())
    y = np.log10(out["h_kpc"].to_numpy())
    y_sigma = None
    if "h_sigma" in out.columns and out["h_sigma"].notna().any():
        # σ_log10(h) ≈ σ_h / (h * ln 10)
        y_sigma = (out["h_sigma"] / out["h_kpc"]) / math.log(10.0)

    # 皮爾森
    m = np.isfinite(x) & np.isfinite(y)
    pearson = np.corrcoef(x[m], y[m])[0, 1] if m.sum() >= 2 else np.nan
    # 史匹爾曼（用 rank 近似）
    rx = pd.Series(x[m]).rank(method="average").to_numpy()
    ry = pd.Series(y[m]).rank(method="average").to_numpy()
    spearman = np.corrcoef(rx, ry)[0, 1] if m.sum() >= 2 else np.nan

    a, b, a_err, b_err, R2, n = _weighted_ols(x, y, y_sigma=y_sigma if y_sigma is not None and np.all(np.isfinite(y_sigma[m])) else None)

    print(f"Sample size (used in fit): {n}")
    print(f"Pearson r (log–log): {pearson:.3f}")
    print(f"Spearman ρ (log–log): {spearman:.3f}")
    print(f"Weighted OLS (log10 h = a + b log10 κ):")
    print(f"  a = {a:.3f} ± {a_err:.3f}")
    print(f"  b = {b:.3f} ± {b_err:.3f}")
    print(f"  R^2 = {R2:.3f}")

    # 畫圖（可選）
    if out_plot is not None and HAVE_MPL and n >= 2:
        fig, ax = plt.subplots(figsize=(5, 4))
        # 誤差棒（只做 h 的誤差；κ 的誤差圖略）
        if out["h_sigma"].notna().any():
            ax.errorbar(out["kappa_char_kms_per_kpc"], out["h_kpc"],
                        yerr=out["h_sigma"], fmt="o", alpha=0.8, label="galaxies")
        else:
            ax.plot(out["kappa_char_kms_per_kpc"], out["h_kpc"], "o", label="galaxies")

        # 擬合線
        xx = np.linspace(np.nanmin(out["kappa_char_kms_per_kpc"]), np.nanmax(out["kappa_char_kms_per_kpc"]), 200)
        yy = (10 ** a) * (xx ** b)
        ax.plot(xx, yy, "-", lw=2, label=f"fit: h = 10^{a:.2f} κ^{b:.2f}")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\kappa_\star\;[{\rm km\,s^{-1}\,kpc^{-1}}]$")
        ax.set_ylabel(r"$h\;[{\rm kpc}]$")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        Path(out_plot).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_plot, dpi=200)
        print(f"Saved plot -> {out_plot}")
    elif out_plot is not None and not HAVE_MPL:
        print("matplotlib not available; skip plot.")

    return 0


def main():
    p = argparse.ArgumentParser(description="κ–h experiment on SPARC×S4G intersection")
    p.add_argument("--sparc-with-h", type=str, required=True, help="dataset/geometry/sparc_with_h.csv")
    p.add_argument("--out-csv", type=str, default="dataset/geometry/kappa_h_sample.csv")
    p.add_argument("--out-plot", type=str, default="dataset/geometry/kappa_h_scatter.png")
    p.add_argument("--min-points", type=int, default=6, help="min radial points per galaxy")
    args = p.parse_args()
    return sys.exit(run(args.sparc_with_h, args.out_csv, args.out_plot, min_points=args.min_points))


if __name__ == "__main__":
    main()
