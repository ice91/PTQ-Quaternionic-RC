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

若輸入中 **沒有每半徑的 Σ 欄位**，本版會自動以 galaxy-level 方式在代表半徑 r★
（v_disk 最大；若無則 v_obs 最大）合成 Σ_*（以及可選擇 Σ_gas、Σ_tot），
回復你舊版工作流程的 κ+Σ 模型（參數：--use-total-sigma/--use-exp/--ml36/--rgas-mult/--gas-helium）。

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
- Σ_tot 若不存在，先嘗試由每半徑星/氣體面密度欄位組合；若仍無，啟用 fallback：
  以 L36_disk、M_HI 等**星系整體量**在 r★ 合成 Σ。

相依套件
--------
numpy, pandas, statsmodels, scipy（僅用到 scipy.stats）, matplotlib
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
# Σ_tot 建構（若缺：先試每半徑，再試 galaxy-level fallback）
# ---------------------------
def _build_sigma_tot(df: pd.DataFrame, ml36: float | None, rgas_mult: float | None, gas_helium: float | None) -> pd.Series:
    """
    嘗試建構 **每半徑** Σ_tot（若輸入已有 per-radius 面密度欄位）。
      1) 直接使用 'Sigma_tot'
      2) Sigma_star + Sigma_gas
      3) Sigma_star_36 * ml36
      4) HI/H2 組合後套用 rgas_mult / gas_helium
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


def _first_valid(x: pd.Series) -> float:
    s = pd.Series(x)
    return s.dropna().iloc[0] if s.notna().any() else np.nan


def _build_fallback_sigma_samples(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """
    Galaxy-level Σ fallback：
    - 代表半徑 r★：v_disk_kms 最大；若無此欄或全 NaN，改用 v_obs_kms 最大。
    - 在 r★ 取 κ★（近鄰非缺值）；用 L36_disk / M_HI 與參數合成 Σ_*、Σ_gas（可選 exp 衰減）。
    - 回傳每星系一列的樣本表（gkey, galaxy, kappa_char, Sigma_tot_fallback, h_kpc, h_sigma, D_Mpc）。
    """
    rows: List[Dict] = []
    need_ml = args.ml36 is not None  # 沒有 ml36 也可僅用 gas，但一般會用 stellar 為主
    groups = df.groupby("gkey", sort=False, dropna=False)

    for gkey, sub in groups:
        if "h_kpc" not in sub.columns or not np.isfinite(sub["h_kpc"]).any():
            continue
        sub = sub.sort_values("r_kpc")

        # 代表半徑 r★
        vdisk_col = "v_disk_kms" if "v_disk_kms" in sub.columns else None
        vobs_col = "v_obs_kms" if "v_obs_kms" in sub.columns else None
        if vdisk_col and np.isfinite(sub[vdisk_col]).any():
            idx = sub[vdisk_col].astype(float).idxmax()
            how = "R_at_max_vdisk"
        elif vobs_col and np.isfinite(sub[vobs_col]).any():
            idx = sub[vobs_col].astype(float).idxmax()
            how = "R_at_max_vobs"
        else:
            continue

        r_star = float(sub.loc[idx, "r_kpc"])
        if not np.isfinite(r_star) or r_star <= 0:
            continue

        # κ★：取 r★ 鄰近的非 NaN 值
        sub2 = sub.copy()
        sub2["absdr"] = np.abs(sub2["r_kpc"] - r_star)
        sub2 = sub2.sort_values("absdr")
        k_ok = sub2["kappa"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        if k_ok.empty:
            continue
        kappa_char = float(k_ok.iloc[0])

        # h 與不確定度（取 galaxy-level 第一筆）
        h = float(_first_valid(sub["h_kpc"]))
        hi = _first_valid(sub.get("h_kpc_err_hi", pd.Series([np.nan])))
        lo = _first_valid(sub.get("h_kpc_err_lo", pd.Series([np.nan])))
        if np.isfinite(hi) and np.isfinite(lo):
            h_sigma = 0.5 * (hi + lo)
        elif np.isfinite(hi):
            h_sigma = hi
        elif np.isfinite(lo):
            h_sigma = lo
        else:
            h_sigma = np.nan

        # galaxy-level 量：L36_disk, L36_tot, L36_bulge, M_HI, D
        L_tot = _first_valid(sub.get("L36_tot", pd.Series([np.nan])))
        L_bulge = _first_valid(sub.get("L36_bulge", pd.Series([np.nan])))
        L_disk = _first_valid(sub.get("L36_disk", pd.Series([np.nan])))
        if not np.isfinite(L_disk):
            if np.isfinite(L_tot) and np.isfinite(L_bulge):
                L_disk = L_tot - L_bulge
        M_HI = _first_valid(sub.get("M_HI", pd.Series([np.nan])))

        # Rd, Rgas
        Rd = r_star / 2.2 if r_star > 0 else np.nan
        Rgas = (args.rgas_mult * Rd) if (args.rgas_mult is not None and np.isfinite(Rd) and Rd > 0) else np.nan

        # Σ_* (中心) 與 r★ 衰減
        if need_ml and np.isfinite(L_disk) and np.isfinite(Rd) and Rd > 0:
            Sigma_star0 = float(args.ml36) * L_disk / (2 * np.pi * Rd**2)
            Sigma_star = Sigma_star0 * np.exp(-r_star / Rd) if args.use_exp else Sigma_star0
        else:
            Sigma_star = np.nan

        # Σ_gas (中心) 與 r★ 衰減（只用 HI 做近似；乘以氦修正；Rgas 由 rgas_mult 決定）
        if (args.gas_helium is not None) and np.isfinite(M_HI) and np.isfinite(Rgas) and Rgas > 0:
            Sigma_gas0 = float(args.gas_helium) * M_HI / (2 * np.pi * Rgas**2)
            Sigma_gas = Sigma_gas0 * np.exp(-r_star / Rgas) if args.use_exp else Sigma_gas0
        else:
            Sigma_gas = np.nan

        if args.use_total_sigma and np.isfinite(Sigma_star) and np.isfinite(Sigma_gas):
            Sigma_tot = Sigma_star + Sigma_gas
        else:
            # 預設至少回 Σ_*（與舊版一致）
            Sigma_tot = Sigma_star

        rows.append(
            dict(
                gkey=gkey,
                galaxy=_first_valid(sub.get("galaxy", pd.Series([gkey]))),
                kappa_char=kappa_char,
                r_char=r_star,
                Sigma_tot=Sigma_tot,
                Sigma_star=Sigma_star,
                Sigma_gas=Sigma_gas,
                h_kpc=h,
                h_sigma=h_sigma,
                D_Mpc=_first_valid(sub.get("D_Mpc_h", pd.Series([np.nan]))),
                choose_rule=how,
            )
        )

    out = pd.DataFrame(rows)
    # 清理：至少要 κ★ 與 h 可用；Σ_tot 若全 NaN 就讓上層判斷失敗
    out = out[np.isfinite(out["kappa_char"]) & np.isfinite(out["h_kpc"])]
    return out


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
    for _, gdf in groups:
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

    # 必要欄位檢查
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

    # 建 Σ_tot（若有 per-radius 欄位）
    sigma_tot = _build_sigma_tot(df, args.ml36, args.rgas_mult, args.gas_helium)
    if "Sigma_tot" not in df.columns:
        df["Sigma_tot"] = sigma_tot

    # 若缺少 κ，就計算
    df = _ensure_kappa(df, args)

    # 必要：kappa 與 h_kpc
    if "kappa" not in df.columns or "h_kpc" not in df.columns:
        raise RuntimeError("輸入檔缺少必要欄位（kappa / h_kpc）。請先完成特徵計算與合併。")

    # 是否有 per-radius Σ？
    per_radius_sigma_ok = np.isfinite(df["Sigma_tot"].astype(float)).any()

    # 權重（WLS；若不可用則退回 OLS）
    def _build_w_from_h(series_h: pd.Series, sig_hi: pd.Series, sig_lo: pd.Series) -> tuple[np.ndarray | None, bool]:
        h_lin = np.maximum(series_h.astype(float).to_numpy(), TINY)
        lo = sig_lo.astype(float).to_numpy() if sig_lo is not None else np.full_like(h_lin, np.nan)
        hi = sig_hi.astype(float).to_numpy() if sig_hi is not None else np.full_like(h_lin, np.nan)
        sig_lin = np.where(np.isfinite(lo) & np.isfinite(hi), 0.5 * (lo + hi),
                           np.where(np.isfinite(lo), lo, np.where(np.isfinite(hi), hi, np.nan)))
        rel = np.where(np.isfinite(sig_lin) & (sig_lin > 0), sig_lin / h_lin, np.nan)
        sig_log = rel / np.log(10.0)
        w = 1.0 / np.square(sig_log)
        w[~np.isfinite(w)] = np.nan
        return (w if np.isfinite(w).any() else None, bool(np.isfinite(w).any()))

    # ---- 路徑 1：per-radius Σ 存在（維持新版行為） ----
    if per_radius_sigma_ok:
        logk = np.log10(np.maximum(df["kappa"].astype(float).to_numpy(), TINY))
        logh = np.log10(np.maximum(df["h_kpc"].astype(float).to_numpy(), TINY))
        logs = np.log10(np.maximum(df["Sigma_tot"].astype(float).to_numpy(), TINY))

        w = None
        used_wls = False
        if args.wls:
            w, used_wls = _build_w_from_h(df["h_kpc"], df.get("h_kpc_err_hi"), df.get("h_kpc_err_lo"))

        m = _finite_mask(logk, logs, logh)
        if w is not None:
            m &= np.isfinite(w)
            w = w[m]
        logk, logs, logh = logk[m], logs[m], logh[m]

        n_used = len(logh)
        if n_used == 0:
            raise RuntimeError("沒有可用樣本（kappa / Sigma_tot / h_kpc 有缺或權重無效）。")

        # 統計量（κ vs h）
        r, _ = pearsonr(logk, logh)
        rho, _ = spearmanr(logk, logh)
        print(f"Sample size (used in fit): {n_used}")
        print(f"Pearson r (log–log): {r:.3f}")
        print(f"Spearman ρ (log–log): {rho:.3f}")

        # κ-only / Σ-only / κ+Σ
        mk = _ols_wls(logk.reshape(-1, 1), logh, w)
        ms = _ols_wls(logs.reshape(-1, 1), logh, w)
        m2 = _ols_wls(np.stack([logk, logs], axis=1), logh, w)

        aicc_k = _aicc(n_used, k=2, rss=_rss(mk))
        aicc_s = _aicc(n_used, k=2, rss=_rss(ms))
        aicc_ks = _aicc(n_used, k=3, rss=_rss(m2))

        print(("Weighted " if used_wls else "") + "OLS (log10 h = a + b log10 κ):")
        print(f"  a = {mk.params[0]:.3f} ± {mk.bse[0]:.3f}")
        print(f"  b = {mk.params[1]:.3f} ± {mk.bse[1]:.3f}")
        print(f"  R^2 = {mk.rsquared:.3f}")
        print(f"[log h ~ log κ + log Σ_tot] N={n_used}, p=3")
        print(f"  beta: {m2.params[0]:.3f}±{m2.bse[0]:.3f}, {m2.params[1]:.3f}±{m2.bse[1]:.3f}, {m2.params[2]:.3f}±{m2.bse[2]:.3f}")
        print(f"  R^2 = {m2.rsquared:.3f}, AICc = {aicc_ks:.2f}")
        print(f"  AICc (κ-only/Σ-only/κ+Σ) = {aicc_k:.2f} / {aicc_s:.2f} / {aicc_ks:.2f}")
        print(f"  ΔAICc = {{'kappa_only': {aicc_k - aicc_ks}, 'sigma_only': {aicc_s - aicc_ks}, 'kappa_sigma': 0.0}}")

        # LOO / Bootstrap
        if args.loo:
            mu, sd = _loo_coeffs(np.stack([logk, logs], axis=1), logh, w)
            print(f"  LOO mean: {list(mu)}")
            print(f"  LOO std : {list(sd)}")
        if args.bootstrap and args.bootstrap > 0:
            med, lo, hi = _bootstrap_coeffs(np.stack([logk, logs], axis=1), logh, w, nboot=args.bootstrap, seed=42)
            print(f"  Bootstrap medians: {list(med)}")
            print(f"  16–84 percentiles: low={list(lo)}  high={list(hi)}")

        # 輸出 CSV
        if args.out_csv:
            Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"log10_h": logh, "log10_kappa": logk, "log10_Sigma_tot": logs}).to_csv(args.out_csv, index=False)
            print(f"Saved -> {args.out_csv} (rows={len(logh)})")

        # 畫圖
        if args.out_plot:
            Path(args.out_plot).parent.mkdir(parents=True, exist_ok=True)
            # κ–h
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

            # color by Σ
            root = Path(args.out_plot)
            out_sigma = root.with_name(root.stem + "_sigma" + root.suffix) if root.suffix else root.with_name(root.stem.replace(".png", "") + "_sigma.png")
            plt.figure(figsize=(5, 4))
            sc = plt.scatter(logk, logh, s=28, c=logs, alpha=0.9)
            plt.xlabel(r"$\log_{10}\,\kappa$")
            plt.ylabel(r"$\log_{10}\,h\ \mathrm{(kpc)}$")
            cb = plt.colorbar(sc)
            cb.set_label(r"$\log_{10}\,\Sigma_{\rm tot}$")
            plt.tight_layout()
            plt.savefig(out_sigma, dpi=180)
            print(f"Saved plot -> {out_sigma}")

        # 距離不變（同一遮罩 m）
        dist_col = args.dist_col or "D_Mpc_h"
        use_dist = dist_col if dist_col in df.columns else ("D_Mpc_gal" if "D_Mpc_gal" in df.columns else None)
        if use_dist is None:
            print("[DIST-INV] skipped: no distance column found")
            md = None
            dist_report = {"dist_col_used": None, "beta": [np.nan, np.nan, np.nan], "bse": [np.nan, np.nan, np.nan], "R2": np.nan}
        else:
            D = np.maximum(df[use_dist].astype(float).to_numpy(), TINY)[m]
            y_d = np.log10(np.maximum(df["h_kpc"].astype(float).to_numpy()[m] / D, TINY))
            x1_d = np.log10(np.maximum(df["kappa"].astype(float).to_numpy()[m] * D, TINY))
            x2_d = logs
            md = _ols_wls(np.stack([x1_d, x2_d], axis=1), y_d, w)
            print(f"[DIST-INV] log(h/D) = {md.params[0]:.3f} + {md.params[1]:.3f} log(κD) + {md.params[2]:.3f} log Σ ; R^2={md.rsquared:.3f}")
            dist_report = {"dist_col_used": use_dist, "beta": md.params.tolist(), "bse": md.bse.tolist(), "R2": float(md.rsquared)}

        # 報告
        report = {
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
            "dist_n": int(len(y_d)) if use_dist is not None else 0,
            "dist_b_kappa": float(md.params[1]) if use_dist is not None else np.nan,
            "dist_c_sigma": float(md.params[2]) if use_dist is not None else np.nan,
            "dist_R2": float(md.rsquared) if use_dist is not None else np.nan,
            "aicc": {"kappa_only": float(aicc_k), "sigma_only": float(aicc_s), "kappa_sigma": float(aicc_ks)},
            "params": {
                "kappa_only": {"beta": mk.params.tolist(), "bse": mk.bse.tolist(), "R2": float(mk.rsquared)},
                "kappa_sigma": {"beta": m2.params.tolist(), "bse": m2.bse.tolist(), "R2": float(m2.rsquared)},
                "dist_inv": dist_report,
            },
            "n_used": int(n_used),
            "used_wls": bool(used_wls),
            "has_sigma": True,
            "used_fallback_sigma": False,
        }

        if args.report_json:
            Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
            with open(args.report_json, "w") as f:
                json.dump(report, f, indent=2)
            print(f"Saved report -> {args.report_json}")

        return report

    # ---- 路徑 2：沒有 per-radius Σ -> 啟用 galaxy-level Σ fallback ----
    print("[Σ] per-radius columns missing; building galaxy-level Σ fallback from L36/M_HI...")
    jj = _build_fallback_sigma_samples(df, args)

    if jj.empty or not np.isfinite(jj["Sigma_tot"]).any():
        # fallback 失敗：退回 κ-only（以 galaxy-level 样本跑，避免 per-radius 與 κ+Σ 比樣本不一致）
        print("[Σ-fallback] failed (no usable Σ). Falling back to κ-only model.")
        # 至少提供 κ-only 的 galaxy-level 回歸（κ★, h）
        xk = np.log10(np.maximum(jj["kappa_char"].to_numpy(), TINY)) if not jj.empty else np.array([])
        y = np.log10(np.maximum(jj["h_kpc"].to_numpy(), TINY)) if not jj.empty else np.array([])

        if len(y) < 3:
            raise RuntimeError("沒有可用樣本（Σ 缺失且無法合成；κ-only 樣本數不足）。")

        # 權重
        w, used_wls = (None, False)
        if args.wls:
            # galaxy-level 權重
            h_lin = np.maximum(jj["h_kpc"].to_numpy(), TINY)
            sig_lin = jj["h_sigma"].to_numpy()
            rel = np.where(np.isfinite(sig_lin) & (sig_lin > 0), sig_lin / h_lin, np.nan)
            sig_log = rel / np.log(10.0)
            w = 1.0 / np.square(sig_log)
            w[~np.isfinite(w)] = np.nan
            used_wls = np.isfinite(w).any()
            if not used_wls:
                w = None

        m = _finite_mask(xk, y)
        if w is not None:
            m &= np.isfinite(w)
            w = w[m]
        xk, y = xk[m], y[m]

        mk = _ols_wls(xk.reshape(-1, 1), y, w)
        aicc_k = _aicc(len(y), k=2, rss=_rss(mk))
        print(("Weighted " if used_wls else "") + "OLS (log10 h = a + b log10 κ):")
        print(f"  a = {mk.params[0]:.3f} ± {mk.bse[0]:.3f}")
        print(f"  b = {mk.params[1]:.3f} ± {mk.bse[1]:.3f}")
        print(f"  R^2 = {mk.rsquared:.3f}")
        print(f"[log h ~ log κ] N={len(y)}, p=2")
        print(f"  AICc (κ-only) = {aicc_k:.2f}")

        # 最小 JSON
        report = {
            "a": float(mk.params[0]),
            "a_se": float(mk.bse[0]),
            "b_kappa": float(mk.params[1]),
            "b_se": float(mk.bse[1]),
            "c_sigma": np.nan,
            "c_se": np.nan,
            "R2": float(mk.rsquared),
            "AICc": float(aicc_k),
            "dAIC_kappa_only": 0.0,
            "dAIC_sigma_only": np.nan,
            "dAIC_kappa_sigma": np.nan,
            "dist_n": 0,
            "dist_b_kappa": np.nan,
            "dist_c_sigma": np.nan,
            "dist_R2": np.nan,
            "aicc": {"kappa_only": float(aicc_k), "sigma_only": np.nan, "kappa_sigma": np.nan},
            "params": {"kappa_only": {"beta": mk.params.tolist(), "bse": mk.bse.tolist(), "R2": float(mk.rsquared)}, "kappa_sigma": None, "dist_inv": {}},
            "n_used": int(len(y)),
            "used_wls": bool(used_wls),
            "has_sigma": False,
            "used_fallback_sigma": False,
        }
        if args.report_json:
            Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
            with open(args.report_json, "w") as f:
                json.dump(report, f, indent=2)
            print(f"Saved report -> {args.report_json}")
        return report

    # --- 用 fallback 樣本做 κ-only / Σ-only / κ+Σ（與舊版一致，每星系一點） ---
    print(f"[Σ-fallback] built galaxy-level samples: N_galaxies={len(jj)} (valid Σ rows={(np.isfinite(jj['Sigma_tot']).sum())})")

    xk = np.log10(np.maximum(jj["kappa_char"].to_numpy(), TINY))
    xs = np.log10(np.maximum(jj["Sigma_tot"].to_numpy(), TINY))
    y = np.log10(np.maximum(jj["h_kpc"].to_numpy(), TINY))

    # 權重（WLS）
    w = None
    used_wls = False
    if args.wls and "h_sigma" in jj.columns:
        sig_lin = jj["h_sigma"].to_numpy()
        h_lin = np.maximum(jj["h_kpc"].to_numpy(), TINY)
        rel = np.where(np.isfinite(sig_lin) & (sig_lin > 0), sig_lin / h_lin, np.nan)
        sig_log = rel / np.log(10.0)
        w = 1.0 / np.square(sig_log)
        w[~np.isfinite(w)] = np.nan
        used_wls = np.isfinite(w).any()
        if not used_wls:
            w = None

    m = _finite_mask(xk, xs, y)
    if w is not None:
        m &= np.isfinite(w)
        w = w[m]
    xk, xs, y = xk[m], xs[m], y[m]

    n_used = len(y)
    if n_used < 3:
        raise RuntimeError("Σ-fallback 樣本不足以做多變量回歸（<3 個星系）。")

    # 統計量（κ vs h）
    r, _ = pearsonr(xk, y)
    rho, _ = spearmanr(xk, y)
    print(f"Sample size (used in fit): {n_used}")
    print(f"Pearson r (log–log): {r:.3f}")
    print(f"Spearman ρ (log–log): {rho:.3f}")

    # κ-only / Σ-only / κ+Σ（在 **同一組樣本** 上比較 AICc）
    mk = _ols_wls(xk.reshape(-1, 1), y, w)
    ms = _ols_wls(xs.reshape(-1, 1), y, w)
    m2 = _ols_wls(np.stack([xk, xs], axis=1), y, w)

    aicc_k = _aicc(n_used, k=2, rss=_rss(mk))
    aicc_s = _aicc(n_used, k=2, rss=_rss(ms))
    aicc_ks = _aicc(n_used, k=3, rss=_rss(m2))

    print(("Weighted " if used_wls else "") + "OLS (log10 h = a + b log10 κ):")
    print(f"  a = {mk.params[0]:.3f} ± {mk.bse[0]:.3f}")
    print(f"  b = {mk.params[1]:.3f} ± {mk.bse[1]:.3f}")
    print(f"  R^2 = {mk.rsquared:.3f}")
    print(f"[log h ~ log κ + log Σ_fallback] N={n_used}, p=3")
    print(f"  beta: {m2.params[0]:.3f}±{m2.bse[0]:.3f}, {m2.params[1]:.3f}±{m2.bse[1]:.3f}, {m2.params[2]:.3f}±{m2.bse[2]:.3f}")
    print(f"  R^2 = {m2.rsquared:.3f}, AICc = {aicc_ks:.2f}")
    print(f"  AICc (κ-only/Σ-only/κ+Σ) = {aicc_k:.2f} / {aicc_s:.2f} / {aicc_ks:.2f}")
    print(f"  ΔAICc = {{'kappa_only': {aicc_k - aicc_ks}, 'sigma_only': {aicc_s - aicc_ks}, 'kappa_sigma': 0.0}}")

    # LOO / Bootstrap（針對 fallback X=[κ★, Σ_fallback]）
    if args.loo:
        mu, sd = _loo_coeffs(np.stack([xk, xs], axis=1), y, w)
        print(f"  LOO mean: {list(mu)}")
        print(f"  LOO std : {list(sd)}")
    if args.bootstrap and args.bootstrap > 0:
        med, lo, hi = _bootstrap_coeffs(np.stack([xk, xs], axis=1), y, w, nboot=args.bootstrap, seed=42)
        print(f"  Bootstrap medians: {list(med)}")
        print(f"  16–84 percentiles: low={list(lo)}  high={list(hi)}")

    # 輸出 CSV
    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "log10_h": y,
                "log10_kappa_char": xk,
                "log10_Sigma_fallback": xs,
            }
        ).to_csv(args.out_csv, index=False)
        print(f"Saved -> {args.out_csv} (rows={len(y)})")

    # 畫圖（κ–h；以及以 X=Σ/κ^2 著色的第二張）
    if args.out_plot:
        Path(args.out_plot).parent.mkdir(parents=True, exist_ok=True)
        # κ–h
        plt.figure(figsize=(5, 4))
        plt.scatter(xk, y, s=28, alpha=0.85)
        xgrid = np.linspace(xk.min() - 0.1, xk.max() + 0.1, 200)
        ygrid = mk.params[0] + mk.params[1] * xgrid
        plt.plot(xgrid, ygrid, linewidth=2)
        plt.xlabel(r"$\log_{10}\,\kappa_\star$")
        plt.ylabel(r"$\log_{10}\,h\ \mathrm{(kpc)}$")
        plt.tight_layout()
        plt.savefig(args.out_plot, dpi=180)
        print(f"Saved plot -> {args.out_plot}")

        # 以 Σ_fallback 著色
        root = Path(args.out_plot)
        out_sigma = root.with_name(root.stem + "_sigma" + root.suffix) if root.suffix else root.with_name(root.stem.replace(".png", "") + "_sigma.png")
        plt.figure(figsize=(5, 4))
        sc = plt.scatter(xk, y, s=28, c=xs, alpha=0.9)
        plt.xlabel(r"$\log_{10}\,\kappa_\star$")
        plt.ylabel(r"$\log_{10}\,h\ \mathrm{(kpc)}$")
        cb = plt.colorbar(sc)
        cb.set_label(r"$\log_{10}\,\Sigma_{\rm fallback}$")
        plt.tight_layout()
        plt.savefig(out_sigma, dpi=180)
        print(f"Saved plot -> {out_sigma}")

    # 距離不變（用 fallback 樣本）
    jjD = jj[np.isfinite(jj["D_Mpc"])].copy()
    if len(jjD) >= 3:
        yD = np.log10(np.maximum(jjD["h_kpc"].to_numpy() / np.maximum(jjD["D_Mpc"].to_numpy(), TINY), TINY))
        xkD = np.log10(np.maximum(jjD["kappa_char"].to_numpy() * np.maximum(jjD["D_Mpc"].to_numpy(), TINY), TINY))
        xsD = np.log10(np.maximum(jjD["Sigma_tot"].to_numpy(), TINY))
        wD = None
        if args.wls and "h_sigma" in jjD.columns:
            sig_lin = jjD["h_sigma"].to_numpy()
            h_lin = np.maximum(jjD["h_kpc"].to_numpy(), TINY)
            rel = np.where(np.isfinite(sig_lin) & (sig_lin > 0), sig_lin / h_lin, np.nan)
            sig_log = rel / np.log(10.0)
            wD = 1.0 / np.square(sig_log)
            wD[~np.isfinite(wD)] = np.nan
            if not np.isfinite(wD).any():
                wD = None
        md = _ols_wls(np.stack([xkD, xsD], axis=1), yD, wD)
        print(f"[DIST-INV] log(h/D) = {md.params[0]:.3f} + {md.params[1]:.3f} log(κD) + {md.params[2]:.3f} log Σ ; R^2={md.rsquared:.3f}")
        dist_section = {"n": int(len(jjD)), "beta": md.params.tolist(), "bse": md.bse.tolist(), "R2": float(md.rsquared)}
    else:
        print("[DIST-INV] skipped: insufficient fallback distance rows")
        md = None
        dist_section = {"n": int(len(jjD)), "beta": [np.nan, np.nan, np.nan], "bse": [np.nan, np.nan, np.nan], "R2": np.nan}

    # 報告
    report = {
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
        "dist_n": int(dist_section["n"]),
        "dist_b_kappa": float(dist_section["beta"][1]) if md is not None else np.nan,
        "dist_c_sigma": float(dist_section["beta"][2]) if md is not None else np.nan,
        "dist_R2": float(dist_section["R2"]),
        "aicc": {"kappa_only": float(aicc_k), "sigma_only": float(aicc_s), "kappa_sigma": float(aicc_ks)},
        "params": {
            "kappa_only": {"beta": mk.params.tolist(), "bse": mk.bse.tolist(), "R2": float(mk.rsquared)},
            "kappa_sigma": {"beta": m2.params.tolist(), "bse": m2.bse.tolist(), "R2": float(m2.rsquared)},
            "dist_inv": dist_section,
        },
        "n_used": int(n_used),
        "used_wls": bool(used_wls),
        "has_sigma": False,
        "used_fallback_sigma": True,
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

    # Σ_tot 組合（若輸入檔沒有 per-radius Sigma 欄位時，會啟用 galaxy-level fallback）
    ap.add_argument("--use-total-sigma", action="store_true", help="Σ_tot = Σ_* + Σ_gas（fallback 時生效）")
    ap.add_argument("--use-exp", action="store_true", help="在 r★ 評估 Σ：乘上 exp(-r★/Rd)（gas 用 Rgas）")
    ap.add_argument("--ml36", type=float, default=0.5, help="M/L[3.6μm]（fallback 評估 Σ_* 用）")
    ap.add_argument("--rgas-mult", type=float, default=1.7, help="R_gas = rgas_mult * Rd（fallback 評估 Σ_gas 用）")
    ap.add_argument("--gas-helium", type=float, default=1.33, help="氦修正（fallback 評估 Σ_gas 用）")

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
