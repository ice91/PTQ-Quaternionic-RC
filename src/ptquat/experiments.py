# src/ptquat/experiments.py
from __future__ import annotations
import os, math, json, shutil, tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import yaml

from .data import load_tidy_sparc, GalaxyData
from .likelihood import build_covariance, gaussian_loglike
from .constants import H0_SI, KPC, KM
from .models import (
    model_v_baryon, model_v_mond, model_v_nfw1p,
    model_v_ptq, model_v_ptq_split, model_v_ptq_nu, model_v_ptq_screen,
    vbar_squared_kms2
)
from .fit_global import run as run_fit
from .plotting import plot_residual_plateau, plot_ppc_hist

# -------------------------------
# 共用：呼叫全域擬合並回傳 summary dict
# -------------------------------
def _call_fit(data_path: str,
              outdir: str,
              model: str,
              prior: str = "galaxies-only",
              sigma_sys: float = 4.0,
              H0_kms_mpc: Optional[float] = None,
              nwalkers: str = "4x",
              steps: int = 12000,
              seed: int = 42,
              a0_si: Optional[float] = None,
              a0_range: str = "5e-11,2e-10",
              logM200_range: str = "9,13",
              c0: float = 10.0,
              c_slope: float = -0.1,
              backend_hdf5: Optional[str] = None,
              thin_by: int = 10,
              resume: bool = False) -> Dict:
    argv = [
        f"--data-path={data_path}",
        f"--outdir={outdir}",
        f"--model={model}",
        f"--prior={prior}",
        f"--sigma-sys={sigma_sys}",
        f"--nwalkers={nwalkers}",
        f"--steps={steps}",
        f"--seed={seed}",
        f"--a0-range={a0_range}",
        f"--logM200-range={logM200_range}",
        f"--c0={c0}",
        f"--c-slope={c_slope}",
        f"--thin-by={thin_by}",
    ]
    if H0_kms_mpc is not None: argv.append(f"--H0-kms-mpc={H0_kms_mpc}")
    if a0_si is not None:      argv.append(f"--a0-si={a0_si}")
    if backend_hdf5 is not None: argv.append(f"--backend-hdf5={backend_hdf5}")
    if resume: argv.append("--resume")

    run_fit(argv)  # 會把結果寫到 outdir 下
    summ_path = Path(outdir) / "global_summary.yaml"
    if not summ_path.exists():
        raise RuntimeError(f"global_summary.yaml not found at {summ_path}")
    return yaml.safe_load(open(summ_path, "r"))


# -------------------------------
# S1 Posterior(-like) Predictive Check（以中位數參數 + 完整協方差）
# -------------------------------
def ppc_check(results_dir: str, data_path: str, out_prefix: str = "ppc") -> Dict:
    """
    使用 global_summary.yaml 的中位數參數，重建每個星系的模型 v_mod，
    用 build_covariance 產生 C，計算 standardized residuals 與 coverage。
    產出：
      - {results_dir}/{out_prefix}_per_point.csv
      - {results_dir}/{out_prefix}_summary.yaml
      - {results_dir}/{out_prefix}_z_hist.png
    """
    results = yaml.safe_load(open(Path(results_dir) / "global_summary.yaml"))
    model = results["model"]
    H0_si = float(results.get("H0_si", H0_SI))
    sig_med = float(results["sigma_sys_median"])
    eps = results.get("epsilon_median")
    q   = results.get("q_median")
    a0  = results.get("a0_median")
    per = pd.read_csv(Path(results_dir) / "per_galaxy_summary.csv").set_index("galaxy")

    # 載入資料
    gdict = load_tidy_sparc(data_path)
    rows = []
    # 建立模型函數（取決於 model）
    def make_vfun(gal_name: str):
        g = gdict[gal_name]
        if model == "ptq":
            U = float(per.loc[gal_name, "Upsilon_med"])
            return lambda rk: model_v_ptq(U, eps, rk, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_si)
        elif model == "ptq-nu":
            U = float(per.loc[gal_name, "Upsilon_med"])
            return lambda rk: model_v_ptq_nu(U, eps, rk, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_si)
        elif model == "ptq-screen":
            U = float(per.loc[gal_name, "Upsilon_med"])
            q_use = float(q) if q is not None else 1.0
            return lambda rk: model_v_ptq_screen(U, eps, q_use, rk, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_si)
        elif model == "ptq-split":
            Ud = float(per.loc[gal_name, "Upsilon_med"])
            Ub = float(per.loc[gal_name, "Upsilon_bulge_med"])
            return lambda rk: model_v_ptq_split(Ud, Ub, eps, rk, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_si)
        elif model == "baryon":
            U = float(per.loc[gal_name, "Upsilon_med"])
            return lambda rk: model_v_baryon(U, rk, g.v_disk, g.v_bulge, g.v_gas)
        elif model == "mond":
            U = float(per.loc[gal_name, "Upsilon_med"])
            return lambda rk: model_v_mond(U, a0, rk, g.v_disk, g.v_bulge, g.v_gas)
        elif model == "nfw1p":
            U  = float(per.loc[gal_name, "Upsilon_med"])
            lM = float(per.loc[gal_name, "log10_M200_med"])
            return lambda rk: model_v_nfw1p(U, lM, rk, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_si)
        else:
            raise ValueError(model)

    # 計算 standardized residuals
    for name, g in sorted(gdict.items()):
        if name not in per.index:  # 保險
            continue
        vfun = make_vfun(name)
        v_mod = vfun(g.r_kpc)
        C = build_covariance(v_mod, g.r_kpc, g.v_err, g.D_Mpc, g.D_err_Mpc,
                             g.i_deg, g.i_err_deg, vfun, sigma_sys_kms=sig_med)
        # 單點標準差
        sig = np.sqrt(np.clip(np.diag(C), 1e-30, np.inf))
        z   = (g.v_obs - v_mod) / sig
        for rk, vo, vm, zz, ss in zip(g.r_kpc, g.v_obs, v_mod, z, sig):
            rows.append(dict(galaxy=name, r_kpc=float(rk),
                             v_obs=float(vo), v_mod=float(vm),
                             z=float(zz), sigma=float(ss)))
    df = pd.DataFrame(rows)
    out_csv = Path(results_dir) / f"{out_prefix}_per_point.csv"
    df.to_csv(out_csv, index=False)

    # Coverage 指標
    cov68 = float(np.mean(np.abs(df["z"].values) <= 1.0))
    cov95 = float(np.mean(np.abs(df["z"].values) <= 2.0))
    summ = dict(model=model, N=len(df), coverage68=cov68, coverage95=cov95)
    out_yaml = Path(results_dir) / f"{out_prefix}_summary.yaml"
    with open(out_yaml, "w") as f:
        yaml.safe_dump(summ, f)

    # 直方圖
    out_png = Path(results_dir) / f"{out_prefix}_z_hist.png"
    plot_ppc_hist(df["z"].values, out_png)

    return summ


# -------------------------------
# S2 誤差壓力測試：倍增 i_err / D_err 並重跑
# -------------------------------
def stress_errors(data_path: str,
                  out_root: str,
                  model: str,
                  scale_i: float = 2.0,
                  scale_D: float = 2.0,
                  **fit_kwargs) -> Dict:
    """
    複製 CSV，將 i_err_deg 與 D_err_Mpc 乘上給定因子，重跑並回傳 summary。
    """
    base = pd.read_csv(data_path)
    for col in ["i_err_deg", "D_err_Mpc"]:
        if col not in base.columns:
            raise ValueError(f"Column '{col}' not in {data_path}")
    mod = base.copy()
    mod["i_err_deg"] = mod["i_err_deg"] * float(scale_i)
    mod["D_err_Mpc"] = mod["D_err_Mpc"] * float(scale_D)

    out_dir = Path(out_root); out_dir.mkdir(parents=True, exist_ok=True)
    tmp_csv = out_dir / f"tidy_scaled_i{scale_i}_D{scale_D}.csv"
    mod.to_csv(tmp_csv, index=False)

    outdir = out_dir / f"{model}_i{scale_i}_D{scale_D}"
    summary = _call_fit(str(tmp_csv), str(outdir), model=model, **fit_kwargs)
    return summary


# -------------------------------
# S3 內盤遮罩敏感度：移除 r < rmin_kpc 的資料點
# -------------------------------
def mask_inner(data_path: str,
               out_root: str,
               model: str,
               rmin_kpc: float = 2.0,
               **fit_kwargs) -> Dict:
    """
    製作一份新的 tidy CSV，把半徑小於 rmin_kpc 的觀測點去掉，重跑。
    """
    base = pd.read_csv(data_path)
    if "r_kpc" not in base.columns:
        raise ValueError("Need 'r_kpc' column in tidy csv.")
    mod = base.loc[base["r_kpc"] >= float(rmin_kpc)].copy()

    out_dir = Path(out_root); out_dir.mkdir(parents=True, exist_ok=True)
    tmp_csv = out_dir / f"tidy_rmin_{rmin_kpc:.2f}kpc.csv"
    mod.to_csv(tmp_csv, index=False)

    outdir = out_dir / f"{model}_rmin_{rmin_kpc:.2f}kpc"
    summary = _call_fit(str(tmp_csv), str(outdir), model=model, **fit_kwargs)
    return summary


# -------------------------------
# S4 H0 敏感度：掃過不同 H0 比較 ε/資訊準則
# -------------------------------
def scan_H0(data_path: str,
            out_root: str,
            model: str,
            H0_list: List[float],
            **fit_kwargs) -> pd.DataFrame:
    """
    對多個 H0 值進行擬合，輸出彙整表格（epsilon, AIC/BIC,...）。
    """
    out_dir = Path(out_root); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for H0 in H0_list:
        outdir = out_dir / f"{model}_H0_{H0:.1f}"
        summ = _call_fit(data_path, str(outdir), model=model, H0_kms_mpc=H0, **fit_kwargs)
        rows.append(dict(
            H0_kms_mpc=H0,
            model=summ["model"],
            epsilon_median=summ.get("epsilon_median"),
            q_median=summ.get("q_median"),
            AIC_full=summ["AIC_full"],
            BIC_full=summ["BIC_full"],
            chi2_total=summ["chi2_total"],
            k=summ["k_parameters"],
            N=summ["N_total"],
            outdir=str(outdir)
        ))
    df = pd.DataFrame(rows).sort_values("AIC_full")
    df.to_csv(out_dir / f"{model}_H0_scan.csv", index=False)
    return df


# -------------------------------
# 殘差加速度平臺（堆疊）
# -------------------------------
def residual_plateau(results_dir: str,
                     data_path: str,
                     nbins: int = 24,
                     out_prefix: str = "plateau") -> Tuple[pd.DataFrame, Path]:
    """
    計算每個資料點的 Δa = v^2/r - a_bar(r)，並以 r_kpc 分箱後取中位數與 16/84 百分位。
    產出：
      - {results_dir}/{out_prefix}_per_point.csv
      - {results_dir}/{out_prefix}_binned.csv
      - {results_dir}/{out_prefix}.png
    """
    results = yaml.safe_load(open(Path(results_dir) / "global_summary.yaml"))
    model = results["model"]
    H0_si = float(results.get("H0_si", H0_SI))
    eps = results.get("epsilon_median")
    q   = results.get("q_median")
    a0  = results.get("a0_median")
    per = pd.read_csv(Path(results_dir) / "per_galaxy_summary.csv").set_index("galaxy")

    gdict = load_tidy_sparc(data_path)
    pts = []

    # helper 取得 v_mod 與 vbar2
    def get_vmod_vbar2(gname: str, g: GalaxyData):
        if model == "ptq":
            U = float(per.loc[gname, "Upsilon_med"])
            vmod = model_v_ptq(U, eps, g.r_kpc, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_si)
            vbar2 = vbar_squared_kms2(U, g.v_disk, g.v_bulge, g.v_gas)
        elif model == "ptq-nu":
            U = float(per.loc[gname, "Upsilon_med"])
            vmod = model_v_ptq_nu(U, eps, g.r_kpc, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_si)
            vbar2 = vbar_squared_kms2(U, g.v_disk, g.v_bulge, g.v_gas)
        elif model == "ptq-screen":
            U = float(per.loc[gname, "Upsilon_med"])
            q_use = float(q) if q is not None else 1.0
            vmod = model_v_ptq_screen(U, eps, q_use, g.r_kpc, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_si)
            vbar2 = vbar_squared_kms2(U, g.v_disk, g.v_bulge, g.v_gas)
        elif model == "ptq-split":
            Ud = float(per.loc[gname, "Upsilon_med"])
            Ub = float(per.loc[gname, "Upsilon_bulge_med"])
            vmod = model_v_ptq_split(Ud, Ub, eps, g.r_kpc, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_si)
            vbar2 = (Ud*(g.v_disk**2) + Ub*(g.v_bulge**2) + g.v_gas**2)
        elif model == "baryon":
            U = float(per.loc[gname, "Upsilon_med"])
            vmod = model_v_baryon(U, g.r_kpc, g.v_disk, g.v_bulge, g.v_gas)
            vbar2 = vbar_squared_kms2(U, g.v_disk, g.v_bulge, g.v_gas)
        elif model == "mond":
            U = float(per.loc[gname, "Upsilon_med"])
            vmod = model_v_mond(U, a0, g.r_kpc, g.v_disk, g.v_bulge, g.v_gas)
            vbar2 = vbar_squared_kms2(U, g.v_disk, g.v_bulge, g.v_gas)
        elif model == "nfw1p":
            U  = float(per.loc[gname, "Upsilon_med"])
            lM = float(per.loc[gname, "log10_M200_med"])
            vmod = model_v_nfw1p(U, lM, g.r_kpc, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_si)
            vbar2 = vbar_squared_kms2(U, g.v_disk, g.v_bulge, g.v_gas)
        else:
            raise ValueError(model)
        return vmod, vbar2

    for name, g in sorted(gdict.items()):
        vmod, vbar2 = get_vmod_vbar2(name, g)
        r_m = g.r_kpc * KPC
        delta_a = ((vmod**2 - vbar2) * (KM**2)) / np.maximum(r_m, 1e-30)  # m/s^2
        for rk, da in zip(g.r_kpc, delta_a):
            pts.append(dict(galaxy=name, r_kpc=float(rk), delta_a=float(da)))
    df = pd.DataFrame(pts)
    per_csv = Path(results_dir) / f"{out_prefix}_per_point.csv"
    df.to_csv(per_csv, index=False)

    # 分箱統計
    rmin, rmax = float(df["r_kpc"].min()), float(df["r_kpc"].max())
    edges = np.linspace(rmin, rmax, nbins+1)
    mids  = 0.5*(edges[:-1]+edges[1:])
    q16, q50, q84, cnt = [], [], [], []
    for i in range(nbins):
        mask = (df["r_kpc"]>=edges[i]) & (df["r_kpc"]<edges[i+1])
        vals = df.loc[mask, "delta_a"].values
        if len(vals)==0:
            q16.append(np.nan); q50.append(np.nan); q84.append(np.nan); cnt.append(0)
        else:
            q16.append(float(np.percentile(vals,16)))
            q50.append(float(np.percentile(vals,50)))
            q84.append(float(np.percentile(vals,84)))
            cnt.append(int(len(vals)))
    binned = pd.DataFrame(dict(r_mid_kpc=mids, q16=q16, q50=q50, q84=q84, n=cnt))
    bin_csv = Path(results_dir) / f"{out_prefix}_binned.csv"
    binned.to_csv(bin_csv, index=False)

    # 畫圖
    out_png = Path(results_dir) / f"{out_prefix}.png"
    plot_residual_plateau(df, binned, out_png)
    return df, out_png


# -------------------------------
# 交叉尺度閉合檢驗：ε_cos v.s. ε_RC
# -------------------------------
def closure_test(results_dir: str,
                 epsilon_cos: Optional[float]=None,
                 omega_lambda: Optional[float]=None) -> Dict:
    """
    若給 ωΛ，則 ε_cos = sqrt(ωΛ/(1-ωΛ))；與 results_dir 的 ε_RC 做 3σ 比較。
    """
    results = yaml.safe_load(open(Path(results_dir)/"global_summary.yaml"))
    eps_rc, eps16, eps84 = results.get("epsilon_median"), results.get("epsilon_p16"), results.get("epsilon_p84")
    if eps_rc is None:
        raise ValueError("No epsilon in results; not a PTQ-family fit.")

    if epsilon_cos is None:
        if omega_lambda is None:
            raise ValueError("Provide epsilon_cos or omega_lambda.")
        if not (0.0 < omega_lambda < 1.0):
            raise ValueError("omega_lambda must be in (0,1).")
        epsilon_cos = math.sqrt(omega_lambda/(1.0-omega_lambda))

    # 取 1σ ≈ (p84 - p16)/2
    sig = 0.5*abs(float(eps84)-float(eps16)) if (eps16 is not None and eps84 is not None) else np.nan
    diff = float(eps_rc) - float(epsilon_cos)
    result = dict(
        epsilon_RC=float(eps_rc),
        epsilon_cos=float(epsilon_cos),
        sigma_RC=float(sig),
        diff=float(diff),
        pass_within_3sigma = (abs(diff) <= 3.0*sig if np.isfinite(sig) else None)
    )
    out = Path(results_dir) / "closure_test.yaml"
    with open(out,"w") as f:
        yaml.safe_dump(result, f)
    return result
