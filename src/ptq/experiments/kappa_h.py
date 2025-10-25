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

本版重點（論文等級）：
- 提供 **--per-galaxy** 模式：每星系只取一筆代表點（R★），避免同一星系多半徑重複樣本。
- **代表半徑 R★** 可選：2.2Rd / v_disk 峰值 / 平坦段 / 中位數半徑。
- 交叉驗證與 bootstrap 提供 **by-galaxy** 版本，避免群聚偏誤。
- **galaxy-equalization 權重**（per-radius 模式時可開啟），避免半徑點數多的星系主導擬合。
- 若每半徑 Σ(R) 缺失，僅在 R★ 以 L36/M_HI 做 **galaxy-level fallback**（不再把同一 Σ 灌到所有半徑）。

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
- Σ_tot 若不存在，先嘗試由每半徑星/氣體面密度欄位組合；若仍無，僅在代表半徑 R★ 啟用 fallback：
  用 L36/M_HI 估計圓內平均面密度（不需 Rd；透明標註）。

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
    """point-wise LOO（僅作為備用；主推 by-galaxy 版本）"""
    n = len(y)
    ps: List[np.ndarray] = []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        m = _ols_wls(X[mask], y[mask], None if w is None else w[mask])
        ps.append(m.params)
    P = np.stack(ps, axis=0)
    return P.mean(axis=0), P.std(axis=0)


def _loo_groups(X: np.ndarray, y: np.ndarray, w: np.ndarray | None, groups: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """by-galaxy LOO：每次拿掉整個星系"""
    uniq = np.unique(groups.astype(str))
    ps: List[np.ndarray] = []
    for g in uniq:
        keep = groups.astype(str) != g
        if keep.sum() < X.shape[1] + 1:
            continue
        m = _ols_wls(X[keep], y[keep], None if w is None else w[keep])
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


def _bootstrap_groups(
    X: np.ndarray, y: np.ndarray, w: np.ndarray | None, groups: np.ndarray, nboot: int = 2000, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """by-galaxy bootstrap：重抽星系集合"""
    rng = np.random.default_rng(seed)
    g = groups.astype(str)
    uniq = np.unique(g)
    B: List[np.ndarray] = []
    for _ in range(nboot):
        # 以星系為單位重抽
        pick = rng.integers(0, len(uniq), size=len(uniq))
        sel = np.concatenate([np.where(g == uniq[i])[0] for i in pick])
        m = _ols_wls(X[sel], y[sel], None if w is None else w[sel])
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
    優先使用每半徑 Σ 欄位（若存在），否則回傳 NaN（讓上層啟用 galaxy-level fallback）。
    """
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

        if w is not None:
            ww = w[i0:i1].copy()
            ww = np.where(np.isfinite(ww) & (ww > 0), ww, np.nan)
            good = np.isfinite(x) & np.isfinite(y) & np.isfinite(ww)
        else:
            good = np.isfinite(x) & np.isfinite(y)

        if good.sum() < min_points:
            continue

        xg, yg = x[good], y[good]
        X = np.column_stack([np.ones_like(xg), xg])

        try:
            if w is None:
                beta, *_ = np.linalg.lstsq(X, yg, rcond=None)
            else:
                wg = ww[good]
                XtW = X.T * wg
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
# 代表半徑 R★ 選擇與 per-galaxy 摘要
# ---------------------------
def _choose_rstar_index(gdf: pd.DataFrame, mode: str = "vdisk-peak", rd_col_candidates: List[str] | None = None) -> int:
    """回傳 gdf 內代表半徑索引（整數 index label）"""
    if rd_col_candidates is None:
        rd_col_candidates = ["Rd_kpc", "R_d_kpc", "r_d_kpc", "Rd", "R_d"]

    if mode.lower() == "2.2rd":
        rd = None
        for c in rd_col_candidates:
            if c in gdf.columns and np.isfinite(gdf[c]).any():
                rd = float(gdf[c].dropna().iloc[0])
                break
        if rd and rd > 0:
            Rstar = 2.2 * rd
            j = int((gdf["r_kpc"] - Rstar).abs().idxmin())
            return j

    if mode.lower() == "flat-rc":
        r = np.maximum(gdf["r_kpc"].to_numpy(), TINY)
        v = np.maximum(gdf["v_obs_kms"].to_numpy(), TINY)
        sl = np.abs(np.gradient(np.log(v), np.log(r)))
        ok = sl < 0.2
        if ok.any():
            jj = np.where(ok)[0]
            return int(gdf.index.to_numpy()[jj[len(jj)//2]])

    if mode.lower() == "median":
        order = np.argsort(gdf["r_kpc"].to_numpy())
        return int(gdf.index.to_numpy()[order[len(order)//2]])

    # default: vdisk-peak（若無 v_disk_kms 則改用 v_obs_kms）
    if "v_disk_kms" in gdf.columns and np.isfinite(gdf["v_disk_kms"]).any():
        return int(gdf["v_disk_kms"].astype(float).idxmax())
    return int(gdf["v_obs_kms"].astype(float).idxmax())


def _sigma_fallback_at_R(L36_disk: float | None, L36_tot: float | None, L36_bulge: float | None,
                         M_HI: float | None, R_kpc: float | None,
                         ml36: float | None, rgas_mult: float | None, gas_helium: float | None) -> float:
    """
    在半徑 R 上以 galaxy-level 量估計圓內平均 Σ（不需要 Rd）：
        Σ_* ≈ (ml36 * L36_disk)/ (2π R^2) ；若無 L36_disk 則用 L36_tot - L36_bulge（或 L36_tot）
        Σ_gas ≈ (M_HI * rgas_mult * gas_helium) / (2π R^2)
    以上僅作 fallback；若資料不足則回傳 NaN。
    """
    if R_kpc is None or not np.isfinite(R_kpc) or R_kpc <= 0:
        return np.nan

    area = 2.0 * np.pi * (float(R_kpc) ** 2)
    star_mass = np.nan
    gas_mass = np.nan

    if ml36 is not None:
        if L36_disk is not None and np.isfinite(L36_disk):
            star_mass = float(ml36) * float(L36_disk)
        elif L36_tot is not None and np.isfinite(L36_tot):
            bul = float(L36_bulge) if (L36_bulge is not None and np.isfinite(L36_bulge)) else 0.0
            star_mass = float(ml36) * (float(L36_tot) - bul)

    if M_HI is not None and np.isfinite(M_HI):
        gas_mass = float(M_HI)
        if rgas_mult is not None:
            gas_mass *= float(rgas_mult)
        if gas_helium is not None:
            gas_mass *= float(gas_helium)

    sig = 0.0
    ok = False
    if np.isfinite(star_mass):
        sig += star_mass / area
        ok = True
    if np.isfinite(gas_mass):
        sig += gas_mass / area
        ok = True
    return float(sig) if ok else np.nan


def _collapse_per_galaxy(
    df: pd.DataFrame,
    rstar_mode: str = "vdisk-peak",
    ml36: float | None = None,
    rgas_mult: float | None = None,
    gas_helium: float | None = None,
) -> pd.DataFrame:
    """
    針對每個星系挑代表半徑 R★，輸出一筆：
        log10_h, log10_kappa, log10_Sigma_tot(可空), w（來自 h 的不確定度）
    Σ 的來源優先序：每半徑 Σ -> 在 R★ 取值；否則 galaxy-level fallback 在 R★ 計算一次。
    """
    rows = []
    for g, gdf in df.groupby("gkey"):
        j = _choose_rstar_index(gdf, mode=rstar_mode)
        s = gdf.loc[j]
        logh = np.log10(max(float(s["h_kpc"]), TINY))
        logk = np.log10(max(float(s["kappa"]), TINY))

        # Σ at R★
        if "Sigma_tot" in gdf.columns and np.isfinite(gdf.loc[j, "Sigma_tot"]):
            logs = np.log10(max(float(gdf.loc[j, "Sigma_tot"]), TINY))
        else:
            logs_val = _sigma_fallback_at_R(
                L36_disk=float(s["L36_disk"]) if "L36_disk" in s.index and np.isfinite(s["L36_disk"]) else None,
                L36_tot=float(s["L36_tot"]) if "L36_tot" in s.index and np.isfinite(s["L36_tot"]) else None,
                L36_bulge=float(s["L36_bulge"]) if "L36_bulge" in s.index and np.isfinite(s["L36_bulge"]) else None,
                M_HI=float(s["M_HI"]) if "M_HI" in s.index and np.isfinite(s["M_HI"]) else None,
                R_kpc=float(s["r_kpc"]),
                ml36=ml36, rgas_mult=rgas_mult, gas_helium=gas_helium,
            )
            logs = np.log10(max(logs_val, TINY)) if np.isfinite(logs_val) else np.nan

        # 以 h 的線性不確定度換算 log 權重；若無則 NaN（回退 OLS）
        lo = s.get("h_kpc_err_lo", np.nan)
        hi = s.get("h_kpc_err_hi", np.nan)
        if np.isfinite(lo) and lo > 0 and np.isfinite(hi) and hi > 0:
            sig_lin = 0.5 * (float(lo) + float(hi))
        elif np.isfinite(lo) and lo > 0:
            sig_lin = float(lo)
        elif np.isfinite(hi) and hi > 0:
            sig_lin = float(hi)
        else:
            sig_lin = np.nan
        if np.isfinite(sig_lin):
            rel = sig_lin / max(float(s["h_kpc"]), TINY)
            sig_log = rel / np.log(10.0)
            w = 1.0 / (sig_log ** 2) if np.isfinite(sig_log) and sig_log > 0 else np.nan
        else:
            w = np.nan

        rows.append({
            "gkey": g,
            "galaxy": s["galaxy"],
            "log10_h": logh,
            "log10_kappa": logk,
            "log10_Sigma_tot": logs,
            "w": w,
        })

    out = pd.DataFrame(rows)
    return out


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

    # ======== per-galaxy（推薦主文）或 per-radius ========
    if args.per_galaxy:
        # 訊息：若每半徑 Σ 缺，將在 R★ 啟動 fallback（僅一次）
        if not np.isfinite(df["Sigma_tot"]).any():
            print("[Σ] per-radius columns missing; using galaxy-level Σ fallback at R★ (single-point per galaxy).")
        used = _collapse_per_galaxy(
            df, rstar_mode=args.rstar, ml36=args.ml36, rgas_mult=args.rgas_mult, gas_helium=args.gas_helium
        )
        logh = used["log10_h"].to_numpy()
        logk = used["log10_kappa"].to_numpy()
        logs = used["log10_Sigma_tot"].to_numpy()
        w = used["w"].to_numpy() if args.wls else None
        groups = used["gkey"].to_numpy()
    else:
        # per-radius 模式：只在「真的有 per-radius Σ」時使用 Σ
        has_sigma_per_radius = np.isfinite(df["Sigma_tot"].astype(float)).any()
        if not has_sigma_per_radius:
            print("[Σ] per-radius columns missing; κ-only for per-radius mode (to avoid replicating galaxy-level Σ across radii).")
        logk = np.log10(np.maximum(df["kappa"].astype(float).to_numpy(), TINY))
        logh = np.log10(np.maximum(df["h_kpc"].astype(float).to_numpy(), TINY))
        logs = np.log10(np.maximum(df["Sigma_tot"].astype(float).to_numpy(), TINY)) if has_sigma_per_radius else None

        # 權重（WLS）；若權重不可用則退回 OLS
        w = None
        if args.wls:
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

        groups = df["gkey"].to_numpy()

        # galaxy-equalization（避免多半徑星系主導）
        if args.equalize_galaxy_weights and (w is not None):
            # 僅對可用樣本計數（先不過濾，等下 m 時再第二次 equalize）
            pass  # 實際 equalize 會在濾掉缺值後進行

    # 是否有可用的 Σ？
    has_sigma = (logs is not None) and np.isfinite(logs).any()

    # ---- 過濾缺值/無效權重 ----
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
    groups = np.asarray(groups)[m]

    # per-radius + equalize：在「濾後集合」上做 equalization
    if (not args.per_galaxy) and args.equalize_galaxy_weights and (w is not None):
        counts = pd.Series(1, index=np.arange(len(groups))).groupby(groups).transform("count").to_numpy()
        counts = np.maximum(counts, 1)
        w = w / counts

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
    print(("Weighted " if w is not None else "") + "OLS (log10 h = a + b log10 κ):")
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

    # ---- LOO / Bootstrap ----
    do_group_cv = args.cv_by_galaxy or args.per_galaxy
    if args.loo:
        if has_sigma:
            X_for_cv = np.stack([logk, logs], axis=1)
        else:
            X_for_cv = logk.reshape(-1, 1)
        if do_group_cv:
            mu, sd = _loo_groups(X_for_cv, logh, w, groups)
        else:
            mu, sd = _loo_coeffs(X_for_cv, logh, w)
        print(f"  LOO mean: {list(mu)}")
        print(f"  LOO std : {list(sd)}")

    if args.bootstrap and args.bootstrap > 0:
        if has_sigma:
            X_for_bs = np.stack([logk, logs], axis=1)
        else:
            X_for_bs = logk.reshape(-1, 1)
        if do_group_cv:
            med, lo, hi = _bootstrap_groups(X_for_bs, logh, w, groups, nboot=args.bootstrap, seed=42)
        else:
            med, lo, hi = _bootstrap_coeffs(X_for_bs, logh, w, nboot=args.bootstrap, seed=42)
        print(f"  Bootstrap medians: {list(med)}")
        print(f"  16–84 percentiles: low={list(lo)}  high={list(hi)}")

    # ---- 輸出 CSV（用到的樣本與 log 值）----
    if args.out_csv:
        if has_sigma:
            df_used = pd.DataFrame({"log10_h": logh, "log10_kappa": logk, "log10_Sigma_tot": logs, "gkey": groups})
        else:
            df_used = pd.DataFrame({"log10_h": logh, "log10_kappa": logk, "gkey": groups})
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        df_used.to_csv(args.out_csv, index=False)
        print(f"Saved -> {args.out_csv} (rows={len(df_used)})")

    # ---- 畫圖（可選）----
    if args.out_plot:
        Path(args.out_plot).parent.mkdir(parents=True, exist_ok=True)
        # 圖1：log κ vs log h（含 κ-only 擬合線）
        plt.figure(figsize=(5, 4))
        plt.scatter(logk, logh, s=28, alpha=0.85)
        xgrid = np.linspace(logk.min() - 0.1, logk.max() + 0.1, 200)
        ygrid = (mk.params[0] + mk.params[1] * xgrid)
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

    # ---- 距離不變（DIST-INV）----
    dist_col = args.dist_col or "D_Mpc_h"
    use_dist = dist_col if dist_col in df.columns else ("D_Mpc_gal" if "D_Mpc_gal" in df.columns else None)
    if use_dist is None:
        print("[DIST-INV] skipped: no distance column found")
        dist_report = {"dist_col_used": None, "beta": [np.nan, np.nan, np.nan], "bse": [np.nan, np.nan, np.nan], "R2": np.nan}
        dist_n = 0
        md = None
    else:
        # 取與主擬合相同的樣本（同一遮罩 m）
        D_all = np.maximum(df[use_dist].astype(float).to_numpy(), TINY)
        D = D_all[m] if len(D_all) == len(m) or (not args.per_galaxy) else None

        if args.per_galaxy:
            # per-galaxy：以代表半徑所在列的距離（其實星系距離相同），從 groups 取對應星系
            # 這裡直接取該星系第一筆距離；D 不影響 log(h/D) 相對比較（距離常數）
            # 仍保留寫法完整性。
            g_to_D = df.groupby("gkey")[use_dist].first().to_dict()
            D = np.asarray([max(float(g_to_D.get(g, np.nan)), TINY) for g in groups])

        y_d = np.log10(np.maximum((10 ** logh) / D, TINY))  # log(h/D)
        x1_d = np.log10(np.maximum((10 ** logk) * D, TINY))  # log(κD)

        if has_sigma:
            x2_d = logs  # 已經 masked 過
            Xd = np.stack([x1_d, x2_d], axis=1)
            md = _ols_wls(Xd, y_d, w)
            beta = md.params.tolist()
            bse = md.bse.tolist()
            if len(beta) == 2:
                beta = [beta[0], beta[1], np.nan]
                bse = [bse[0], bse[1], np.nan]
        else:
            Xd = x1_d.reshape(-1, 1)
            md = _ols_wls(Xd, y_d, w)
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

    # ---- JSON 報告（在沒有 Σ 時，相關欄位以 NaN/None 補齊）----
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
        # 其他
        "n_used": int(n_used),
        "used_wls": bool(w is not None),
        "has_sigma": bool(has_sigma),
        "per_galaxy": bool(args.per_galaxy),
        "rstar_mode": args.rstar if args.per_galaxy else None,
        "cv_by_galaxy": bool(do_group_cv),
        "args": vars(args),
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

    # 模式切換
    ap.add_argument("--per-galaxy", action="store_true", help="每星系只取代表半徑 R★ 一筆（推薦主文使用）")
    ap.add_argument("--rstar", type=str, default="vdisk-peak",
                    choices=["vdisk-peak", "2.2Rd", "flat-rc", "median"],
                    help="代表半徑 R★ 定義（預設 vdisk 峰值；無 vdisk 用 v_obs 峰值）")
    ap.add_argument("--equalize-galaxy-weights", action="store_true",
                    help="per-radius 模式時啟用：每星系總權重約為 1（避免多半徑星系主導）")
    ap.add_argument("--cv-by-galaxy", action="store_true",
                    help="LOO/Bootstrap 以星系為單位（per-galaxy 模式自動啟用）")

    # 權重與健壯度分析
    ap.add_argument("--wls", action="store_true", help="使用 WLS（以 h 的線性不確定度推估 log-space 權重；若無效則退回 OLS）")
    ap.add_argument("--loo", action="store_true", help="交叉驗證（預設 by-galaxy，若未指定則 point-wise）")
    ap.add_argument("--bootstrap", type=int, default=0, help="bootstrap 次數（0 表示不執行）")

    # Σ_tot 組合（若輸入檔沒有 Sigma_tot 時才會用到這些）
    ap.add_argument("--use-total-sigma", action="store_true", help="相容旗標（實際自動偵測/建構 Sigma_tot）")
    ap.add_argument("--use-exp", action="store_true", help="相容旗標（不影響計算）")
    ap.add_argument("--ml36", type=float, default=None, help="M/L[3.6μm] 乘數（fallback 用於 L36_*）")
    ap.add_argument("--rgas-mult", type=float, default=None, help="氣體倍率（如需修正 HI->gas）")
    ap.add_argument("--gas-helium", type=float, default=None, help="氦修正倍率（如 1.33）")

    # κ 計算選項（當缺少 kappa 欄位時生效）
    ap.add_argument("--velocity-column", type=str, default="auto", help="用於 κ 計算的速度欄位（auto|v_obs_kms|v_circ_kms）")
    ap.add_argument("--deproject-velocity", action="store_true", help="以 i_deg 反投影速度（v/sin i）")
    ap.add_argument("--deriv-window", type=int, default=7, help="局部回歸視窗大小（總點數約 2*half+1；建議奇數≥5）")
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
