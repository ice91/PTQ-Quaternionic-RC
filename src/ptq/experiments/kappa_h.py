# -*- coding: utf-8 -*-
"""
ptq.experiments.kappa_h
=======================

目的
----
以星系盤厚度尺度 h 與 κ（環振頻率）/ Σ_tot（總質量表面密度）的關係做線性回歸分析，
輸出 κ-only、Σ-only、κ+Σ 的 OLS/WLS 結果、交叉驗證（LOO）、bootstrap 區間、
以及距離不變（DIST-INV）檢驗。程式同時支援在「輸入 CSV 缺少 kappa 欄位」時，
直接由旋轉曲線 (r_kpc, v_obs_kms) 計算 κ(R)。

κ(R) 定義
---------
    κ(R) = sqrt(2) * V(R)/R * sqrt(1 + d ln V / d ln R)

這裡以「局部對數空間線性回歸」估計 d ln V / d ln R（可調視窗），
對不等距半徑也可靠；如提供速度誤差 v_err_kms，會做加權回歸。

假設與單位
----------
- 半徑 r_kpc：kpc
- 速度 v 以 km/s，κ 輸出單位 km s^-1 kpc^-1（與 r,v 單位相容）。
- 若指定 --deproject-velocity，則以 v_obs_kms / sin(i_deg) 作為圓周速度；
  命令列保護避免 i_deg 接近 0 時的數值不穩定（可調門檻）。
- Σ_tot 若不存在，會嘗試由星/氣體面密度欄位組合（見 _build_sigma_tot）。

輸入欄位（至少其一）
--------------------
必需：h_kpc（由前處理 merge 後加入）
必要其一：
  1) kappa（若已預先計算）
  2) r_kpc + v_obs_kms（本程式將自動計算 kappa）

可選：
  - v_err_kms：推斜率時的加權（建議）
  - i_deg（配合 --deproject-velocity）
  - Sigma_tot 或相關星/氣體面密度欄位（若無則自動嘗試組合）

相依套件
--------
numpy, pandas, statsmodels, scipy（僅用到 scipy.stats）

作者註
------
- 邊界處理：為避免 sqrt 負值，會將 d ln V / d ln R 下限截在 -0.99。
- 回歸：WLS 權重來自 h 的不確定度（若存在）；kappa 的誤差不導入權重，
  以免耦合導致偏置（這亦常見於盤厚度的對數回歸處理）。
- 加回「沒有 Σ 就退化成 κ-only」與「WLS 權重無效即退回 OLS」兩層保險。
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
# 名稱與篩選工具
# ---------------------------
def _canon_name(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).upper().strip()
    for ch in [" ", "-", "_", "/", "."]:
        s = s.replace(ch, "")
    return s


def _finite_mask(*arrs) -> np.ndarray:
    m = np.ones_like(np.asarray(arrs[0], dtype=float), dtype=bool)
    for a in arrs:
        m &= np.isfinite(a)
    return m


# ---------------------------
# 回歸輔助
# ---------------------------
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


# ---------------------------
# Σ_tot 建構（若缺）
# ---------------------------
def _build_sigma_tot(df: pd.DataFrame, ml36: float | None, rgas_mult: float | None, gas_helium: float | None) -> pd.Series:
    """
    嘗試建構 Σ_tot（質量表面密度；單位由資料自行維持一致性），優先順序：
      1) 直接使用 'Sigma_tot'
      2) Sigma_star + Sigma_gas
      3) Sigma_star_36 * ml36 （若僅有 3.6um 星光面密度）
      4) 氣體允許以 HI/H2 組合，並套用 rgas_mult / gas_helium（若提供）
    """
    # 直接存在
    for name in ["Sigma_tot", "sigma_tot", "SIGMA_TOT"]:
        if name in df.columns:
            return df[name].astype(float)

    def pick(*cands) -> pd.Series | None:
        for c in cands:
            if c in df.columns:
                return df[c].astype(float)
        return None

    s_star = pick("Sigma_star", "sigma_star", "SIGMA_STAR", "SigmaStar", "st_sigma")
    s_star36 = pick("Sigma_star_36", "sigma_star_36", "Sigma36", "ml36_sigma")
    s_gas = pick("Sigma_gas", "sigma_gas", "SIGMA_GAS", "SigmaGas", "gas_sigma")
    s_hi = pick("Sigma_HI", "sigma_HI")
    s_h2 = pick("Sigma_H2", "sigma_H2")

    if s_star is None and s_star36 is not None and ml36 is not None:
        s_star = s_star36 * float(ml36)

    if s_gas is None and (s_hi is not None or s_h2 is not None):
        s_hi = s_hi if s_hi is not None else pd.Series(0.0, index=df.index)
        s_h2 = s_h2 if s_h2 is not None else pd.Series(0.0, index=df.index)
        s_gas = s_hi + s_h2

    if s_gas is not None:
        if rgas_mult is not None:
            s_gas = s_gas * float(rgas_mult)
        if gas_helium is not None:
            s_gas = s_gas * float(gas_helium)

    if s_star is not None and s_gas is not None:
        return s_star + s_gas
    if s_star is not None:
        return s_star
    if s_gas is not None:
        return s_gas

    return pd.Series(np.nan, index=df.index, dtype=float)


# ---------------------------
# κ 計算：局部線性回歸估斜率
# ---------------------------
def _safe_sin_deg(deg: np.ndarray, min_deg: float = 5.0) -> np.ndarray:
    """避免 i→0 的發散問題；角度缺值或過小時以 min_deg 代替。"""
    x = np.asarray(deg, dtype=float)
    x = np.where(np.isfinite(x), x, min_deg)
    x = np.clip(x, min_deg, 89.9)
    return np.sin(np.deg2rad(x))


def _local_log_slope(lnR: np.ndarray, lnV: np.ndarray, w: np.ndarray | None, half_window: int, min_points: int) -> np.ndarray:
    """
    在 lnR 軸上（不等距）做局部線性回歸，估各點的 d ln V / d ln R。
    - half_window：左右各取多少點（總視窗約 2*half_window+1）
    - min_points：回歸至少需要的點數
    權重 w 指的是對 lnV 的權重（建議 w=1/sigma_lnV^2）；若為 None 則 OLS。
    """
    n = len(lnR)
    slope = np.full(n, np.nan, dtype=float)

    for j in range(n):
        i0 = max(0, j - half_window)
        i1 = min(n, j + half_window + 1)
        x = lnR[i0:i1]
        y = lnV[i0:i1]
        if len(x) < min_points:
            continue

        # 權重
        if w is not None:
            ww = w[i0:i1].copy()
            ww = np.where(np.isfinite(ww) & (ww > 0), ww, np.nan)
            good = np.isfinite(x) & np.isfinite(y) & np.isfinite(ww)
        else:
            good = np.isfinite(x) & np.isfinite(y)

        if good.sum() < min_points:
            continue

        xg, yg = x[good], y[good]
        if w is not None:
            wg = ww[good]
            # 加權最小平方法：解 (X' W X) beta = X' W y
            X = np.column_stack([np.ones_like(xg), xg])
            W = np.diag(wg)
        else:
            X = np.column_stack([np.ones_like(xg), xg])
            W = None

        try:
            if W is None:
                beta, *_ = np.linalg.lstsq(X, yg, rcond=None)
            else:
                XtW = X.T @ W
                beta = np.linalg.solve(XtW @ X, XtW @ yg)
            slope[j] = float(beta[1])
        except np.linalg.LinAlgError:
            continue

    # 對無法估計的點，退而用數值梯度補
    bad = ~np.isfinite(slope)
    if bad.any():
        grad = np.gradient(lnV, lnR, edge_order=1)
        slope[bad] = grad[bad]

    # 物理下限：避免 1 + slope < 0
    slope = np.maximum(slope, -0.99)
    return slope


def _compute_kappa_for_group(
    r_kpc: np.ndarray,
    v_kms: np.ndarray,
    v_err: np.ndarray | None,
    i_deg: np.ndarray | None,
    deproject: bool,
    half_window: int,
    min_points: int,
    min_incl_deg: float,
) -> np.ndarray:
    """對單一星系（已依半徑排序）計算 κ(R)。回傳與輸入同長度的陣列，缺值以 NaN 表示。"""
    r = np.asarray(r_kpc, dtype=float)
    v = np.asarray(v_kms, dtype=float)

    m = np.isfinite(r) & np.isfinite(v) & (r > 0) & (v > 0)
    if m.sum() < min_points:
        return np.full_like(r, np.nan, dtype=float)

    r = r[m]
    v = v[m]

    # 反投影（可選）
    if deproject:
        if i_deg is None:
            raise RuntimeError("要求 --deproject-velocity 但缺少 i_deg 欄位。")
        sini = _safe_sin_deg(np.asarray(i_deg, dtype=float)[m], min_deg=min_incl_deg)
        v = v / np.maximum(sini, 1e-3)

    # 權重：若有 v_err，則 sigma_lnV ≈ v_err / v，w = 1/sigma_lnV^2
    w = None
    if v_err is not None:
        verr = np.asarray(v_err, dtype=float)[m]
        sigma_lnV = np.where(np.isfinite(verr) & (verr > 0), verr / v, np.nan)
        w = 1.0 / np.square(sigma_lnV)
        w[~np.isfinite(w)] = np.nan

    lnR = np.log(np.maximum(r, TINY))
    lnV = np.log(np.maximum(v, TINY))

    slope = _local_log_slope(lnR, lnV, w, half_window=half_window, min_points=min_points)
    kappa = np.sqrt(2.0) * (v / r) * np.sqrt(np.maximum(1.0 + slope, 1e-6))

    # 回填到原索引長度
    out = np.full(r_kpc.shape, np.nan, dtype=float)
    out[np.where(m)[0]] = kappa
    return out


def _ensure_kappa(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """
    若 'kappa' 欄位不存在或全為 NaN，則根據 r_kpc 與 v_obs_kms 自動計算 κ。
    可選參數：
      --velocity-column：優先使用的速度欄（預設自動，先找 v_circ_kms 再 v_obs_kms）
      --deproject-velocity：以 i_deg 反投影
      --deriv-window：局部回歸視窗大小（總點數約 2*half+1）
      --deriv-minpoints：每次回歸的最少點數
      --min-inclination-deg：i 的最小值保護
    """
    need = ("kappa" not in df.columns) or (~np.isfinite(df["kappa"]).any())
    if not need:
        return df

    # 決定速度欄
    vcol = args.velocity_column
    if vcol is None or vcol.lower() == "auto":
        if "v_circ_kms" in df.columns:
            vcol = "v_circ_kms"
        elif "v_obs_kms" in df.columns:
            vcol = "v_obs_kms"
        else:
            raise RuntimeError("找不到速度欄位（v_circ_kms 或 v_obs_kms）。")
    if vcol not in df.columns:
        raise RuntimeError(f"指定的速度欄位不存在：{vcol}")

    verr_col = "v_err_kms" if "v_err_kms" in df.columns else None
    ideg_col = "i_deg" if "i_deg" in df.columns else None

    kappas = []
    groups = df.groupby("galaxy", sort=False, dropna=False)
    for g, gdf in groups:
        # 依半徑排序計算；保留原索引順序以回填
        order = np.argsort(gdf["r_kpc"].to_numpy())
        idx_sorted = gdf.index.to_numpy()[order]

        r = gdf.loc[idx_sorted, "r_kpc"].to_numpy()
        v = gdf.loc[idx_sorted, vcol].to_numpy()
        ve = gdf.loc[idx_sorted, verr_col].to_numpy() if verr_col else None
        idg = gdf.loc[idx_sorted, ideg_col].to_numpy() if ideg_col else None

        k = _compute_kappa_for_group(
            r_kpc=r,
            v_kms=v,
            v_err=ve,
            i_deg=idg,
            deproject=args.deproject_velocity,
            half_window=max(1, args.deriv_window // 2),
            min_points=max(3, args.deriv_minpoints),
            min_incl_deg=float(args.min_inclination_deg),
        )

        # 回填到原順序
        k_back = np.full_like(gdf["r_kpc"].to_numpy(), np.nan, dtype=float)
        k_back[order] = k
        kappas.append(pd.Series(k_back, index=gdf.index))

    df = df.copy()
    df["kappa"] = pd.concat(kappas).sort_index()
    n_valid = int(np.isfinite(df["kappa"]).sum())
    print(f"[kappa] computed from rotation curves: valid points = {n_valid}")
    return df


# ---------------------------
# 排除/異常 樣本清單
# ---------------------------
def _load_exclude_keys(path: str | None) -> set[str]:
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        return set()
    df = pd.read_csv(p)
    cols = list(df.columns)
    for cand in ["gkey", "galaxy", "Galaxy", "name", cols[0]]:
        if cand in df.columns:
            return set(_canon_name(x) for x in df[cand].astype(str))
    return set(_canon_name(x) for x in df[cols[0]].astype(str))


def _load_h_outlier_keys(path: str | None) -> set[str]:
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        return set()
    df = pd.read_csv(p)
    for cand in ["gkey", "galaxy_h"]:
        if cand in df.columns:
            return set(_canon_name(x) for x in df[cand].astype(str))
    return set(_canon_name(x) for x in df[df.columns[0]].astype(str))


def _ensure_gkey(df: pd.DataFrame) -> pd.DataFrame:
    if "gkey" in df.columns:
        return df
    for cand in ["galaxy", "Galaxy", "name", "Name", "galaxy_s"]:
        if cand in df.columns:
            df = df.copy()
            df["gkey"] = df[cand].astype(str).map(_canon_name)
            return df
    df = df.copy()
    df["gkey"] = df[df.columns[0]].astype(str).map(_canon_name)
    return df


# ---------------------------
# 主流程
# ---------------------------
def run(args: argparse.Namespace) -> Dict:
    df = pd.read_csv(args.sparc_with_h)
    df = _ensure_gkey(df)

    # 需要的核心欄位檢查
    for c in ["r_kpc", "galaxy"]:
        if c not in df.columns:
            raise RuntimeError(f"輸入缺少必要欄位：{c}")

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

    # 若缺少 kappa，就計算
    df = _ensure_kappa(df, args)

    # 必要欄位：kappa 與 h_kpc
    if "kappa" not in df.columns or "h_kpc" not in df.columns:
        raise RuntimeError("輸入檔缺少必要欄位（kappa / h_kpc）。請先完成特徵計算與合併。")

    # 是否有可用的 Σ 剖面？
    has_sigma = np.isfinite(df["Sigma_tot"].astype(float)).any()

    # 組回歸資料（log 空間）
    logk = np.log10(np.maximum(df["kappa"].astype(float).to_numpy(), TINY))
    logh = np.log10(np.maximum(df["h_kpc"].astype(float).to_numpy(), TINY))
    logs = None
    if has_sigma:
        logs = np.log10(np.maximum(df["Sigma_tot"].astype(float).to_numpy(), TINY))
    else:
        print("[Σ] not found or entirely NaN; falling back to κ-only model.")

    # 權重（WLS）；若權重不可用則退回 OLS
    w = None
    used_wls = False
    if args.wls:
        # 優先使用 sigma_h；否則以 h_kpc_err_hi/lo 的「逐列」平均（兩者其一存在就用其值，兩者皆 NaN 才 NaN）
        if "sigma_h" in df.columns and np.isfinite(df["sigma_h"]).any():
            sig_lin = df["sigma_h"].astype(float).to_numpy()
        else:
            lo = df.get("h_kpc_err_lo", pd.Series(np.nan, index=df.index)).astype(float).to_numpy()
            hi = df.get("h_kpc_err_hi", pd.Series(np.nan, index=df.index)).astype(float).to_numpy()
            sig_lin = np.where(
                np.isfinite(lo) & np.isfinite(hi), 0.5 * (lo + hi),
                np.where(np.isfinite(lo), lo, np.where(np.isfinite(hi), hi, np.nan))
            )

        h_lin = np.maximum(df["h_kpc"].astype(float).to_numpy(), TINY)
        rel = np.where(np.isfinite(sig_lin) & (sig_lin > 0), sig_lin / h_lin, np.nan)
        sig_log = rel / np.log(10.0)
        w = 1.0 / np.square(sig_log)
        w[~np.isfinite(w)] = np.nan

        if np.isfinite(w).any():
            used_wls = True
        else:
            print("[WLS] no usable h uncertainties; falling back to OLS.")
            w = None

    # 過濾缺值/無效權重
    if has_sigma:
        m = _finite_mask(logk, logs, logh)
    else:
        m = _finite_mask(logk, logh)
    if w is not None:
        m &= np.isfinite(w)

    logk = logk[m]
    logh = logh[m]
    if has_sigma:
        logs = logs[m]
    if w is not None:
        w = w[m]

    n_used = len(logh)
    if n_used == 0:
        need = "kappa / h_kpc" + (" / Sigma_tot" if has_sigma else "")
        raise RuntimeError(f"沒有可用樣本（{need} 有缺或權重無效）。")

    # 統計量（κ vs h）
    r, _ = pearsonr(logk, logh)
    rho, _ = spearmanr(logk, logh)
    print(f"Sample size (used in fit): {n_used}")
    print(f"Pearson r (log–log): {r:.3f}")
    print(f"Spearman ρ (log–log): {rho:.3f}")

    # --- κ-only ---
    Xk = logk.reshape(-1, 1)
    mk = _ols_wls(Xk, logh, w)
    print(("Weighted " if used_wls else "") + "OLS (log10 h = a + b log10 κ):")
    print(f"  a = {mk.params[0]:.3f} ± {mk.bse[0]:.3f}")
    print(f"  b = {mk.params[1]:.3f} ± {mk.bse[1]:.3f}")
    print(f"  R^2 = {mk.rsquared:.3f}")

    # --- Σ-only / κ+Σ（若有 Σ）---
    ms = None
    m2 = None
    aicc_k = _aicc(n_used, k=2, rss=_rss(mk))  # const+1
    aicc_s = np.nan
    aicc_ks = np.nan

    if has_sigma:
        Xs = logs.reshape(-1, 1)
        ms = _ols_wls(Xs, logh, w)
        X2 = np.stack([logk, logs], axis=1)
        m2 = _ols_wls(X2, logh, w)

        aicc_s = _aicc(n_used, k=2, rss=_rss(ms))
        aicc_ks = _aicc(n_used, k=3, rss=_rss(m2))  # const+2

        print(f"[log h ~ log κ + log Σ_tot] N={n_used}, p=3")
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
    else:
        print(f"[log h ~ log κ] N={n_used}, p=2")
        print(f"  AICc (κ-only) = {aicc_k:.2f}")

    # LOO
    if args.loo:
        if has_sigma:
            mu, sd = _loo_coeffs(np.stack([logk, logs], axis=1), logh, w)
        else:
            mu, sd = _loo_coeffs(logk.reshape(-1, 1), logh, w)
        print(f"  LOO mean: {list(mu)}")
        print(f"  LOO std : {list(sd)}")

    # Bootstrap
    if args.bootstrap and args.bootstrap > 0:
        if has_sigma:
            med, lo, hi = _bootstrap_coeffs(np.stack([logk, logs], axis=1), logh, w, nboot=args.bootstrap, seed=42)
        else:
            med, lo, hi = _bootstrap_coeffs(logk.reshape(-1, 1), logh, w, nboot=args.bootstrap, seed=42)
        print(f"  Bootstrap medians: {list(med)}")
        print(f"  16–84 percentiles: low={list(lo)}  high={list(hi)}")

    # 輸出 CSV（用到的樣本與 log 值）
    if args.out_csv:
        if has_sigma:
            df_used = pd.DataFrame({"log10_h": logh, "log10_kappa": logk, "log10_Sigma_tot": logs})
        else:
            df_used = pd.DataFrame({"log10_h": logh, "log10_kappa": logk})
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        df_used.to_csv(args.out_csv, index=False)
        print(f"Saved -> {args.out_csv} (rows={len(df_used)})")

    # 畫圖（可選）
    if args.out_plot:
        Path(args.out_plot).parent.mkdir(parents=True, exist_ok=True)
        # 圖1：log κ vs log h（含 κ-only 擬合線）
        plt.figure(figsize=(5, 4))
        plt.scatter(logk, logh, s=28, alpha=0.85)
        xgrid = np.linspace(logk.min() - 0.1, logk.max() + 0.1, 200)
        ygrid = mk.params[0] + mk.params[1] * xgrid
        plt.plot(xgrid, ygrid, linewidth=2)
        plt.xlabel(r"$\log_{10}\,\kappa$")
        plt.ylabel(r"$\log_{10}\,h\ \mathrm{(kpc)}$")
        plt.tight_layout()
        plt.savefig(args.out_plot, dpi=180)
        print(f"Saved plot -> {args.out_plot}")

        # 圖2：用 log Σ 著色（只有在有 Σ 時才畫）
        if has_sigma:
            root = Path(args.out_plot)
            out_sigma = (
                root.with_name(root.stem + "_sigma" + root.suffix)
                if root.suffix
                else root.with_name(root.stem.replace(".png", "") + "_sigma.png")
            )
            plt.figure(figsize=(5, 4))
            sc = plt.scatter(logk, logh, s=28, c=logs, alpha=0.9)
            plt.xlabel(r"$\log_{10}\,\kappa$")
            plt.ylabel(r"$\log_{10}\,h\ \mathrm{(kpc)}$")
            cb = plt.colorbar(sc)
            cb.set_label(r"$\log_{10}\,\Sigma_{\rm tot}$")
            plt.tight_layout()
            plt.savefig(out_sigma, dpi=180)
            print(f"Saved plot -> {out_sigma}")
        else:
            print("[plot] skipped colored-by-Σ panel because Σ is unavailable.")

    # 距離不變（DIST-INV）
    dist_col = args.dist_col or "D_Mpc_h"
    use_dist = dist_col if dist_col in df.columns else ("D_Mpc_gal" if "D_Mpc_gal" in df.columns else None)
    if use_dist is None:
        print("[DIST-INV] skipped: no distance column found")
        dist_report = {"dist_col_used": None, "beta": [np.nan, np.nan, np.nan], "bse": [np.nan, np.nan, np.nan], "R2": np.nan}
        dist_n = 0
        md = None
    else:
        D_all = np.maximum(df[use_dist].astype(float).to_numpy(), TINY)
        D = D_all[m]
        y_d = np.log10(np.maximum(df["h_kpc"].astype(float).to_numpy()[m] / D, TINY))  # log(h/D)
        x1_d = np.log10(np.maximum(df["kappa"].astype(float).to_numpy()[m] * D, TINY))  # log(κD)

        if has_sigma:
            x2_d = logs  # 已經 masked 過
            Xd = np.stack([x1_d, x2_d], axis=1)
            md = _ols_wls(Xd, y_d, w)
            beta = md.params.tolist()
            bse = md.bse.tolist()
            # pad to length 3: [const, b_kappa, c_sigma]
            if len(beta) == 2:
                beta = [beta[0], beta[1], np.nan]
                bse = [bse[0], bse[1], np.nan]
        else:
            Xd = x1_d.reshape(-1, 1)
            md = _ols_wls(Xd, y_d, w)
            # pad to 3 (const, b_kappa, c_sigma=nan)
            beta = md.params.tolist() + [np.nan] if len(md.params) == 2 else [np.nan, np.nan, np.nan]
            bse = md.bse.tolist() + [np.nan] if len(md.bse) == 2 else [np.nan, np.nan, np.nan]

        dist_n = len(y_d)
        if has_sigma:
            print(f"[DIST-INV] log(h/D) = {md.params[0]:.3f} + {md.params[1]:.3f} log(κD) + {md.params[2]:.3f} log Σ ; R^2={md.rsquared:.3f}")
        else:
            print(f"[DIST-INV] log(h/D) = {md.params[0]:.3f} + {md.params[1]:.3f} log(κD) ; R^2={md.rsquared:.3f}")
        dist_report = {
            "dist_col_used": use_dist,
            "beta": beta,
            "bse": bse,
            "R2": float(md.rsquared),
        }

    # JSON 報告（在沒有 Σ 時，相關欄位以 NaN/None 補齊）
    report = {
        # 主要迴歸（若有 Σ 則是 κ+Σ，否則是 κ-only）
        "a": float(m2.params[0]) if has_sigma else float(mk.params[0]),
        "a_se": float(m2.bse[0]) if has_sigma else float(mk.bse[0]),
        "b_kappa": float(m2.params[1]) if has_sigma else float(mk.params[1]),
        "b_se": float(m2.bse[1]) if has_sigma else float(mk.bse[1]),
        "c_sigma": float(m2.params[2]) if has_sigma else np.nan,
        "c_se": float(m2.bse[2]) if has_sigma else np.nan,
        "R2": float(m2.rsquared) if has_sigma else float(mk.rsquared),
        "AICc": float(aicc_ks) if has_sigma else float(aicc_k),
        "dAIC_kappa_only": float(aicc_k - aicc_ks) if has_sigma else 0.0,
        "dAIC_sigma_only": float(aicc_s - aicc_ks) if has_sigma else np.nan,
        "dAIC_kappa_sigma": 0.0 if has_sigma else np.nan,
        # 距離不變
        "dist_n": int(dist_n),
        "dist_b_kappa": float(md.params[1]) if md is not None and len(md.params) >= 2 else np.nan,
        "dist_c_sigma": float(md.params[2]) if md is not None and len(md.params) >= 3 else np.nan,
        "dist_R2": float(md.rsquared) if md is not None else np.nan,
        # 附帶原始區塊（向下相容）
        "aicc": {
            "kappa_only": float(aicc_k),
            "sigma_only": float(aicc_s) if has_sigma else np.nan,
            "kappa_sigma": float(aicc_ks) if has_sigma else np.nan,
        },
        "params": {
            "kappa_only": {"beta": mk.params.tolist(), "bse": mk.bse.tolist(), "R2": float(mk.rsquared)},
            "kappa_sigma": ({"beta": m2.params.tolist(), "bse": m2.bse.tolist(), "R2": float(m2.rsquared)} if has_sigma else None),
            "dist_inv": dist_report,
        },
        "n_used": int(n_used),
        "used_wls": bool(used_wls),
        "has_sigma": bool(has_sigma),
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
    ap.add_argument("--sparc-with-h", type=str, required=True, help="merged CSV（含 h_kpc；若無 kappa 則由本程式計算）")
    ap.add_argument("--out-csv", type=str, default=None, help="輸出：回歸使用的對數樣本")
    ap.add_argument("--out-plot", type=str, default=None, help="輸出：散點圖（另附 _sigma 版本；無 Σ 則略過）")
    ap.add_argument("--report-json", type=str, default=None, help="輸出：回歸摘要 JSON")

    # 權重與健壯度分析
    ap.add_argument("--wls", action="store_true", help="使用 WLS（以 h 的線性不確定度推估 log-space 權重；若無效則退回 OLS）")
    ap.add_argument("--loo", action="store_true", help="逐一留一交叉驗證（係數平均／標準差）")
    ap.add_argument("--bootstrap", type=int, default=0, help="bootstrap 次數（0 表示不執行）")

    # Σ_tot 組合（若輸入檔沒有 Sigma_tot 時才會用到這些）
    ap.add_argument("--use-total-sigma", action="store_true", help="相容旗標（實際自動偵測/建構 Sigma_tot）")
    ap.add_argument("--use-exp", action="store_true", help="相容旗標（不影響計算）")
    ap.add_argument("--ml36", type=float, default=None, help="M/L[3.6μm] 乘數（若僅提供星光面密度 Sigma_star_36）")
    ap.add_argument("--rgas-mult", type=float, default=None, help="氣體倍率（如需修正）")
    ap.add_argument("--gas-helium", type=float, default=None, help="氦修正倍率（如 1.33）")

    # κ 計算選項（當缺少 kappa 欄位時生效）
    ap.add_argument("--velocity-column", type=str, default="auto", help="用於 κ 計算的速度欄位（auto|v_obs_kms|v_circ_kms）")
    ap.add_argument("--deproject-velocity", action="store_true", help="以 i_deg 反投影速度（v/sin i）")
    ap.add_argument("--deriv-window", type=int, default=5, help="局部回歸視窗大小（總點數約 2*half+1；建議奇數≥5）")
    ap.add_argument("--deriv-minpoints", type=int, default=3, help="每次回歸最少點數（≥3）")
    ap.add_argument("--min-inclination-deg", type=float, default=5.0, help="i 的最小度數保護（避免 sin(i) 太小）")

    # 資料過濾
    ap.add_argument("--drop-h-outliers", action="store_true", help="移除 h 的 outliers（需要 --h-outliers-csv）")
    ap.add_argument("--h-outliers-csv", type=str, default=None, help="由 build-h 產出的 outliers CSV")
    ap.add_argument("--exclude-csv", type=str, default=None, help="自訂排除清單（含 gkey 或 galaxy 欄位）")

    # DIST-INV 距離欄位
    ap.add_argument("--dist-col", type=str, default="D_Mpc_h", help="距離欄位（預設用 S4G 的 D_Mpc_h；若無則回退 D_Mpc_gal）")

    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()

