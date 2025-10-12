# src/ptquat/experiments.py
from __future__ import annotations
import os, math, json, shutil, tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import yaml

from .data import load_tidy_sparc, GalaxyData
from .likelihood import build_covariance, gaussian_loglike
from .constants import H0_SI, KPC, KM
from .models import (
    model_v_baryon, model_v_mond, model_v_nfw1p,
    model_v_ptq, model_v_ptq_split, model_v_ptq_nu, model_v_ptq_screen,
    vbar_squared_kms2, linear_term_kms2
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
              likelihood: str = "gauss",
              t_dof: float = 8.0,
              backend_hdf5: Optional[str] = None,  # 型別修正：字串路徑
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
        f"--likelihood={likelihood}",
        f"--t-dof={t_dof}",
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
def ppc_check(results_dir: str, data_path: str, out_prefix: str = "ppc") -> dict:
    """
    Posterior(-like) predictive coverage check using median params from results_dir.
    Uses Student-t critical values if the fitted run used t-likelihood.
    """
    res = Path(results_dir)
    summ = yaml.safe_load(open(res/"global_summary.yaml"))
    per  = pd.read_csv(res/"per_galaxy_summary.csv").set_index("galaxy")

    like = str(summ.get("likelihood", "gauss"))
    nu   = float(summ.get("t_dof", 8.0))
    H0_si = float(summ.get("H0_si", H0_SI))
    model = summ["model"]
    sig_med = float(summ["sigma_sys_median"])
    eps_med = summ.get("epsilon_median", None)
    q_med   = summ.get("q_median", None)
    a0_med  = summ.get("a0_median", None)

    # critical multipliers for 68/95
    k68, k95 = 1.0, 1.96
    if like == "t":
        try:
            from scipy.stats import t as _t
            k68 = float(_t.ppf(0.84, df=nu))
            k95 = float(_t.ppf(0.975, df=nu))
        except Exception:
            pass  # fallback to Gaussian multipliers

    gdict = load_tidy_sparc(data_path)
    galaxies = [gdict[k] for k in sorted(gdict.keys())]

    n = 0; hit68 = 0; hit95 = 0
    for g in galaxies:
        if model in ("ptq", "ptq-nu", "ptq-screen", "baryon", "mond", "nfw1p"):
            U = float(per.loc[g.name, "Upsilon_med"])
        if model == "ptq-split":
            Ud = float(per.loc[g.name, "Upsilon_med"])
            Ub = float(per.loc[g.name, "Upsilon_bulge_med"])
        if model == "nfw1p":
            lM = float(per.loc[g.name, "log10_M200_med"])

        if model == "ptq":
            def vfun(rk): return model_v_ptq(U, eps_med, rk, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_si)
        elif model == "ptq-nu":
            def vfun(rk): return model_v_ptq_nu(U, eps_med, rk, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_si)
        elif model == "ptq-screen":
            def vfun(rk): return model_v_ptq_screen(U, eps_med, q_med, rk, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_si)
        elif model == "ptq-split":
            def vfun(rk): return model_v_ptq_split(Ud, Ub, eps_med, rk, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_si)
        elif model == "baryon":
            def vfun(rk): return model_v_baryon(U, rk, g.v_disk, g.v_bulge, g.v_gas)
        elif model == "mond":
            def vfun(rk): return model_v_mond(U, a0_med, rk, g.v_disk, g.v_bulge, g.v_gas)
        elif model == "nfw1p":
            def vfun(rk): return model_v_nfw1p(U, lM, rk, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_SI)
        else:
            raise ValueError("unknown model")

        v_mod = vfun(g.r_kpc)
        Cg = build_covariance(v_mod, g.r_kpc, g.v_err, g.D_Mpc, g.D_err_Mpc,
                              g.i_deg, g.i_err_deg, vfun, sigma_sys_kms=sig_med)
        diag = np.clip(np.diag(Cg), 1e-12, None)
        sig  = np.sqrt(diag)
        r = np.abs(g.v_obs - v_mod)
        n  += r.size
        hit68 += int((r <= k68*sig).sum())
        hit95 += int((r <= k95*sig).sum())

    cov68 = hit68 / n
    cov95 = hit95 / n

    out = {"model": model, "N": n, "coverage68": cov68, "coverage95": cov95}
    with open(res/f"{out_prefix}_coverage.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


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


# ====== Kappa checks: per-galaxy & radius-resolved ======

def _get_Rd_kpc_safe(g: GalaxyData) -> float:
    """Try to read Rd from GalaxyData; fallback to r_peak(v_disk)/2.2."""
    for attr in ("Rd_kpc", "R_d_kpc", "R_d", "Rd"):
        if hasattr(g, attr):
            val = getattr(g, attr)
            try:
                v = float(val)
                if np.isfinite(v) and v > 0:
                    return v
            except Exception:
                pass
    # Fallback: exponential disk peaks ~ at 2.2 Rd
    i_pk = int(np.argmax(g.v_disk))
    rpk = float(g.r_kpc[i_pk]) if len(g.r_kpc) > 0 else 1.0
    return max(rpk / 2.2, 0.1)

def _make_vfun(model: str, H0_si: float, per_row: pd.Series, g: GalaxyData,
               eps: Optional[float], q: Optional[float], a0: Optional[float]):
    """Return (vfun, vbar2 array). vfun(r_kpc) -> model velocity [km/s]."""
    if model == "ptq":
        U = float(per_row["Upsilon_med"])
        vfun = lambda rk: model_v_ptq(U, eps, rk, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_si)
        vbar2 = vbar_squared_kms2(U, g.v_disk, g.v_bulge, g.v_gas)
    elif model == "ptq-nu":
        U = float(per_row["Upsilon_med"])
        vfun = lambda rk: model_v_ptq_nu(U, eps, rk, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_si)
        vbar2 = vbar_squared_kms2(U, g.v_disk, g.v_bulge, g.v_gas)
    elif model == "ptq-screen":
        U = float(per_row["Upsilon_med"])
        q_use = float(q) if q is not None else 1.0
        vfun = lambda rk: model_v_ptq_screen(U, eps, q_use, rk, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_si)
        vbar2 = vbar_squared_kms2(U, g.v_disk, g.v_bulge, g.v_gas)
    elif model == "ptq-split":
        Ud = float(per_row["Upsilon_med"])
        Ub = float(per_row["Upsilon_bulge_med"])
        vfun = lambda rk: model_v_ptq_split(Ud, Ub, eps, rk, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_si)
        vbar2 = Ud*(g.v_disk**2) + Ub*(g.v_bulge**2) + g.v_gas**2
    elif model == "baryon":
        U = float(per_row["Upsilon_med"])
        vfun = lambda rk: model_v_baryon(U, rk, g.v_disk, g.v_bulge, g.v_gas)
        vbar2 = vbar_squared_kms2(U, g.v_disk, g.v_bulge, g.v_gas)
    elif model == "mond":
        U = float(per_row["Upsilon_med"])
        vfun = lambda rk: model_v_mond(U, a0, rk, g.v_disk, g.v_bulge, g.v_gas)
        vbar2 = vbar_squared_kms2(U, g.v_disk, g.v_bulge, g.v_gas)
    elif model == "nfw1p":
        U  = float(per_row["Upsilon_med"])
        lM = float(per_row["log10_M200_med"])
        vfun = lambda rk: model_v_nfw1p(U, lM, rk, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_si)
        vbar2 = vbar_squared_kms2(U, g.v_disk, g.v_bulge, g.v_gas)
    else:
        raise ValueError(model)
    return vfun, vbar2

def _eps_cos_from_args(epsilon_cos: Optional[float], omega_lambda: Optional[float]) -> float:
    if epsilon_cos is not None:
        return float(epsilon_cos)
    if omega_lambda is not None:
        if not (0.0 < omega_lambda < 1.0):
            raise ValueError("omega_lambda must be in (0,1).")
        return float(np.sqrt(omega_lambda/(1.0-omega_lambda)))
    # default anchor if not provided
    return 1.47

def kappa_per_galaxy(results_dir: str,
                     data_path: str,
                     eta: float = 0.15,
                     frac_vmax: float = 0.9,
                     epsilon_cos: Optional[float] = None,
                     omega_lambda: Optional[float] = None,
                     nsamp: int = 300,
                     y_source: str = "model",      # "model" | "obs" | "obs-debias"
                     eps_norm: str = "fit",        # "fit" | "cos"
                     rstar_from: str = "model",    # "model" | "obs"
                     out_prefix: str = "kappa_gal") -> Dict[str, Any]:
    """
    檢核A：以特徵半徑 r* 做 y 對 x 的迴歸。
      x = kappa_pred = eta * Rd / r*
      y = eps_eff / eps_den
        - eps_den = epsilon_median (eps_norm=fit) 或 ε_cos (eps_norm=cos)
      y_source 決定 eps_eff 來源；rstar_from 決定 r* 用模型或觀測曲線選點。
    """
    assert y_source in ("model", "obs", "obs-debias")
    assert eps_norm in ("fit", "cos")
    assert rstar_from in ("model", "obs")

    res = yaml.safe_load(open(Path(results_dir)/"global_summary.yaml"))
    model  = res["model"]
    H0_si  = float(res.get("H0_si", H0_SI))
    sig_med = float(res["sigma_sys_median"])
    eps_fit_global = float(res.get("epsilon_median")) if res.get("epsilon_median") is not None else np.nan
    q    = res.get("q_median")
    a0   = res.get("a0_median")
    per  = pd.read_csv(Path(results_dir)/"per_galaxy_summary.csv").set_index("galaxy")

    # 分母選擇
    if eps_norm == "fit":
        eps_den = float(eps_fit_global)
    else:  # "cos"
        eps_den = _eps_cos_from_args(epsilon_cos, omega_lambda)
    if not np.isfinite(eps_den) or eps_den == 0.0:
        raise ValueError("Invalid epsilon denominator (eps_den). Check --eps-norm / omega-lambda / epsilon-cos.")

    gdict = load_tidy_sparc(data_path)
    rng = np.random.default_rng(1234)
    rows = []

    for name, g in sorted(gdict.items()):
        if name not in per.index:
            continue
        vfun, vbar2 = _make_vfun(model, H0_si, per.loc[name], g, res.get("epsilon_median"), q, a0)

        v_obs = g.v_obs
        r     = g.r_kpc
        if len(r) < 3:
            continue

        v_mod = vfun(r)

        # r* 的選擇與 y_source 無關，由 rstar_from 控制
        v_for_rstar = v_mod if (rstar_from == "model") else v_obs
        vmax  = float(np.nanmax(v_for_rstar))
        idxs  = np.where(v_for_rstar >= frac_vmax * vmax)[0]
        i_star = int(idxs[0]) if len(idxs) > 0 else int(np.nanargmax(v_for_rstar))
        i_star = max(0, min(i_star, len(r)-1))
        r_star = float(r[i_star])

        # 協方差：抽樣用完整 C；去偏時用只含測量變異的 C_meas
        C = build_covariance(v_mod, r, g.v_err, g.D_Mpc, g.D_err_Mpc,
                             g.i_deg, g.i_err_deg, vfun, sigma_sys_kms=sig_med)
        C_meas = build_covariance(v_mod, r, g.v_err, g.D_Mpc, g.D_err_Mpc,
                                  g.i_deg, g.i_err_deg, vfun, sigma_sys_kms=0.0)
        sig_i2_meas = float(np.clip(np.diag(C_meas)[i_star], 1e-30, np.inf))

        denom   = float(linear_term_kms2(1.0, np.asarray([r_star]), H0_si=H0_si)[0])   # (cH0) r*
        vbar2_i = float(vbar2[i_star])

        # y_source：決定 eps_eff 的估計方式
        if y_source == "model":
            eps_eff_med = (v_mod[i_star]**2 - vbar2_i) / denom
            eps_lo = np.nan; eps_hi = np.nan
        elif y_source == "obs":
            try:
                Vs = rng.multivariate_normal(mean=v_obs, cov=C, size=nsamp)
                eps_samp = (Vs[:, i_star]**2 - vbar2_i) / denom
                eps_eff_med = float(np.percentile(eps_samp, 50))
                eps_lo      = float(np.percentile(eps_samp, 16))
                eps_hi      = float(np.percentile(eps_samp, 84))
            except np.linalg.LinAlgError:
                sig_i = float(np.sqrt(np.clip(np.diag(C)[i_star], 1e-30, np.inf)))
                eps_eff_med = (v_obs[i_star]**2 - vbar2_i) / denom
                eps_lo  = eps_eff_med - (2*v_obs[i_star]*sig_i)/denom
                eps_hi  = eps_eff_med + (2*v_obs[i_star]*sig_i)/denom
        else:  # "obs-debias"：只扣測量變異
            try:
                Vs = rng.multivariate_normal(mean=v_obs, cov=C, size=nsamp)
                eps_samp = ((Vs[:, i_star]**2 - sig_i2_meas) - vbar2_i) / denom
                eps_eff_med = float(np.percentile(eps_samp, 50))
                eps_lo      = float(np.percentile(eps_samp, 16))
                eps_hi      = float(np.percentile(eps_samp, 84))
            except np.linalg.LinAlgError:
                eps_eff_med = ((v_obs[i_star]**2 - sig_i2_meas) - vbar2_i) / denom
                sig_i = float(np.sqrt(np.clip(np.diag(C)[i_star], 1e-30, np.inf)))
                base  = max(v_obs[i_star]**2 - sig_i2_meas, 0.0)**0.5
                eps_lo  = eps_eff_med - (2*base*sig_i)/denom
                eps_hi  = eps_eff_med + (2*base*sig_i)/denom

        Rd = _get_Rd_kpc_safe(g)
        kappa_pred = float(eta * Rd / r_star)

        rows.append(dict(
            galaxy=name,
            r_star_kpc=r_star,
            Rd_kpc=Rd,
            eps_eff_med=eps_eff_med,
            eps_eff_p16=eps_lo,
            eps_eff_p84=eps_hi,
            eps_eff_over_epsden=eps_eff_med/eps_den,
            kappa_pred=kappa_pred
        ))

    # 組表、輸出
    df = pd.DataFrame(rows)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["kappa_pred", "eps_eff_over_epsden"])

    out_csv = Path(results_dir)/f"{out_prefix}_per_galaxy.csv"
    df.to_csv(out_csv, index=False)

    # OLS: y = a x + b
    if len(df) >= 2:
        a, b = np.polyfit(df["kappa_pred"].values, df["eps_eff_over_epsden"].values, 1)
        yhat = a*df["kappa_pred"].values + b
        r2 = float(1.0 - np.sum((df["eps_eff_over_epsden"].values - yhat)**2) /
                          np.sum((df["eps_eff_over_epsden"].values - df["eps_eff_over_epsden"].mean())**2))
        k_est = a*eta
    else:
        a = b = r2 = k_est = np.nan

    # 圖
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=150)
    ax.scatter(df["kappa_pred"], df["eps_eff_over_epsden"], s=18, alpha=0.7, label="galaxies")
    xgrid = np.linspace(0.0, max(1.05*df["kappa_pred"].max(), 0.2), 200) if len(df)>0 else np.linspace(0.0, 0.2, 200)
    ax.plot(xgrid, xgrid, linestyle="--", linewidth=1.2, label="y = x")
    if np.isfinite(a):
        ax.plot(xgrid, a*xgrid + b, linewidth=1.4,
                label=f"fit: y={a:.2f}x+{b:.2f}, R²={r2:.2f}, k≈{k_est:.2f}")
    ax.set_xlabel(r"$\kappa_{\rm pred}=\eta\,R_d/r_\ast$")
    ylab_den = r"\varepsilon_{\rm fit}" if eps_norm=="fit" else r"\varepsilon_{\rm cos}"
    ax.set_ylabel(rf"$\varepsilon_{{\rm eff}}(r_\ast)/{ylab_den}$")
    ax.legend(); ax.grid(True, alpha=0.25)
    out_png = Path(results_dir)/f"{out_prefix}.png"
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)

    # 總結（沿用你之前的鍵值，方便對照）
    summ = dict(
        N=len(df),
        eta=eta, frac_vmax=frac_vmax, y_source=y_source,
        slope=a, intercept=b, R2=r2, k_est=k_est,
        eps_fit=eps_fit_global,
        eps_den=eps_den,
        ratio_epsfit_over_epsden=(float(eps_fit_global)/float(eps_den) if (np.isfinite(eps_fit_global) and np.isfinite(eps_den) and eps_den!=0) else np.nan),
        csv=str(out_csv), png=str(out_png)
    )
    with open(Path(results_dir)/f"{out_prefix}_summary.json","w") as f:
        json.dump(summ, f, indent=2)
    return summ


def kappa_radius_resolved(results_dir: str,
                          data_path: str,
                          eta: float = 0.15,
                          epsilon_cos: Optional[float] = None,
                          omega_lambda: Optional[float] = None,
                          nbins: int = 24,
                          min_per_bin: int = 20,
                          x_kind: str = "r_over_Rd",
                          eps_norm: str = "fit",         # 新增
                          out_prefix: str = "kappa_profile",
                          x_markers: Optional[List[float]] = None) -> Tuple[pd.DataFrame, Path]:
    assert eps_norm in ("fit","cos")

    res = yaml.safe_load(open(Path(results_dir)/"global_summary.yaml"))
    model = res["model"]
    H0_si = float(res.get("H0_si", H0_SI))
    sig_med = float(res["sigma_sys_median"])
    eps_fit_global = float(res.get("epsilon_median")) if res.get("epsilon_median") is not None else np.nan
    q   = res.get("q_median")
    a0  = res.get("a0_median")
    per = pd.read_csv(Path(results_dir) / "per_galaxy_summary.csv").set_index("galaxy")

    # 分母
    if eps_norm == "fit":
        eps_den = float(eps_fit_global)
    else:
        eps_den = _eps_cos_from_args(epsilon_cos, omega_lambda)
    if not np.isfinite(eps_den) or eps_den == 0.0:
        raise ValueError("Invalid epsilon denominator (eps_den) for kappa_profile.")

    gdict = load_tidy_sparc(data_path)
    pts = []
    for name, g in sorted(gdict.items()):
        if name not in per.index:
            continue
        vfun, vbar2 = _make_vfun(model, H0_si, per.loc[name], g, res.get("epsilon_median"), q, a0)
        v_mod = vfun(g.r_kpc)
        C = build_covariance(v_mod, g.r_kpc, g.v_err, g.D_Mpc, g.D_err_Mpc,
                             g.i_deg, g.i_err_deg, vfun, sigma_sys_kms=sig_med)
        sig_pt = np.sqrt(np.clip(np.diag(C), 1e-30, np.inf))

        denom = linear_term_kms2(1.0, g.r_kpc, H0_si=H0_si)  # (cH0) r
        eps_eff = (g.v_obs**2 - vbar2) / np.maximum(denom, 1e-30)
        Rd = _get_Rd_kpc_safe(g)
        x = (g.r_kpc / Rd) if x_kind == "r_over_Rd" else g.r_kpc
        y = eps_eff / eps_den

        for xi, yi, ri, si in zip(x, y, g.r_kpc, sig_pt):
            if np.isfinite(xi) and np.isfinite(yi) and si < 100.0:
                pts.append(dict(galaxy=name, r_kpc=float(ri), x=float(xi), y=float(yi), Rd_kpc=float(Rd)))



    df = pd.DataFrame(pts)
    per_csv = Path(results_dir)/f"{out_prefix}_per_point.csv"
    df.to_csv(per_csv, index=False)

    # 分箱
    if len(df) == 0:
        raise RuntimeError("No valid points for kappa profile.")
    x_min = float(np.quantile(df["x"], 0.02))
    x_max = float(np.quantile(df["x"], 0.98))
    edges = np.linspace(x_min, x_max, nbins+1)
    mids  = 0.5*(edges[:-1]+edges[1:])
    q16, q50, q84, cnt = [], [], [], []
    for i in range(nbins):
        m = (df["x"]>=edges[i]) & (df["x"]<edges[i+1])
        vals = df.loc[m, "y"].values
        if len(vals) < min_per_bin:
            q16.append(np.nan); q50.append(np.nan); q84.append(np.nan); cnt.append(int(len(vals)))
        else:
            q16.append(float(np.percentile(vals,16)))
            q50.append(float(np.percentile(vals,50)))
            q84.append(float(np.percentile(vals,84)))
            cnt.append(int(len(vals)))

    binned = pd.DataFrame(dict(x_mid=mids, q16=q16, q50=q50, q84=q84, n=cnt))
    bin_csv = Path(results_dir)/f"{out_prefix}_binned.csv"
    binned.to_csv(bin_csv, index=False)

    # 預測曲線 y_pred = eta / x  (若 x=r_kpc，則用樣本 Rd 的中位數換成 eta*Rd_med/x)
    if x_kind == "r_over_Rd":
        xgrid = np.linspace(max(x_min, 1e-3), x_max, 400)
        ypred = eta / np.maximum(xgrid, 1e-6)
        pred_label = r"$\eta/x$"
    else:
        Rd_med = float(np.median(df["Rd_kpc"]))
        xgrid = np.linspace(max(x_min, 1e-3), x_max, 400)
        ypred = (eta * Rd_med) / np.maximum(xgrid, 1e-6)
        pred_label = r"$\eta\,R_{d,{\rm med}}/r$"

    # 圖與 x=2.2 檢查
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.6, 4.0), dpi=150)
    ax.plot(xgrid, ypred, linestyle="--", linewidth=1.2, label=pred_label)
    ax.fill_between(binned["x_mid"], binned["q16"], binned["q84"], alpha=0.25, label="stacked 16–84%")
    ax.plot(binned["x_mid"], binned["q50"], linewidth=1.6, label="stacked median")
    ax.set_xlabel("x = r/Rd" if x_kind=="r_over_Rd" else "r [kpc]")
    #ax.set_ylabel(r"$\varepsilon_{\rm eff}(r)/\varepsilon_{\rm cos}$")
    ax.set_ylabel(rf"$\varepsilon_{{\rm eff}}(r)/{'\\varepsilon_{\\rm fit}' if eps_norm=='fit' else '\\varepsilon_{\\rm cos}'}$")
    ax.grid(True, alpha=0.25)

    xcheck = {}
    if x_kind == "r_over_Rd":
        # 線性內插 binned median 取得 y_med at 指定 x
        def _interp(xs, ys, x0):
            xs = np.asarray(xs); ys = np.asarray(ys)
            m = np.isfinite(xs) & np.isfinite(ys)
            if m.sum() < 2 or not (xs[m].min() <= x0 <= xs[m].max()):
                return np.nan
            return float(np.interp(x0, xs[m], ys[m]))
        for x0 in (x_markers or []):
            y_med = _interp(binned["x_mid"], binned["q50"], float(x0))
            y_th  = float(eta / max(x0, 1e-6))
            xcheck[str(x0)] = dict(y_median=y_med, y_pred=y_th, delta=y_med - y_th)
            # 視覺標記
            ax.axvline(x0, linestyle=":", linewidth=1.0, alpha=0.6)
            if np.isfinite(y_med):
                ax.scatter([x0],[y_med], s=28, zorder=5, label=f"median @ x={x0:g}")
            ax.scatter([x0],[y_th], s=28, marker="x", zorder=5, label=f"pred @ x={x0:g}")

    # 圖例與輸出
    ax.legend()
    out_png = Path(results_dir)/f"{out_prefix}.png"
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

    # 寫出 x=2.2 檢查摘要
    with open(Path(results_dir)/f"{out_prefix}_xcheck.json","w") as f:
        json.dump(dict(x_kind=x_kind, eta=eta, xcheck=xcheck), f, indent=2)

    return binned, out_png

# ====== 兩參數擬合與 bootstrap（從 kappa_profile_* 輸出檔就地完成） ======
def _ab_fit_from_xy(x_mid: np.ndarray, y_med: np.ndarray) -> dict:
    """線性最小平方法：y = A*(1/x) + B，回傳 A, B, R2, N。"""
    import numpy as np
    m = np.isfinite(x_mid) & np.isfinite(y_med) & (x_mid > 0)
    x = x_mid[m]; y = y_med[m]
    if x.size < 3:
        return dict(A=np.nan, B=np.nan, R2=np.nan, N=int(x.size))
    X = np.vstack([1.0/x, np.ones_like(x)]).T
    A, B = np.linalg.lstsq(X, y, rcond=None)[0]
    yhat = A*(1.0/x) + B
    ss_res = float(np.sum((y - yhat)**2))
    ss_tot = float(np.sum((y - y.mean())**2))
    R2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else np.nan)
    return dict(A=float(A), B=float(B), R2=float(R2), N=int(x.size))


def kappa_two_param_fit(results_dir: str,
                        prefix: str = "kappa_profile",
                        eps_norm: str = "cos",
                        epsilon_cos: float | None = None,
                        omega_lambda: float | None = None,
                        out_json: str | None = None) -> dict:
    """
    讀取 {results_dir}/{prefix}_binned.csv，對 q50 做 y=A/x+B 擬合。
    - 若 eps_norm="cos"：eta_hat=A、B_cos=B。
    - 若 eps_norm="fit"：需同時提供 epsilon_cos（或 omega_lambda），將
        eta_hat = A * (eps_fit/eps_cos)， B_cos = B * (eps_fit/eps_cos)。
    會把結果寫到 out_json（預設落在 results_dir 下，檔名 *_abfit.json）。
    """
    import json, numpy as np, pandas as pd, math
    from pathlib import Path
    res = Path(results_dir)
    binned_csv = res / f"{prefix}_binned.csv"
    if not binned_csv.exists():
        raise FileNotFoundError(f"{binned_csv} not found. Run `ptquat exp kappa-prof` first.")

    # 讀入 binned
    b = pd.read_csv(binned_csv)
    fit = _ab_fit_from_xy(b["x_mid"].values, b["q50"].values)

    # 取得 eps_fit 與 eps_cos
    summ = yaml.safe_load(open(res/"global_summary.yaml"))
    eps_fit = float(summ.get("epsilon_median")) if summ.get("epsilon_median") is not None else np.nan
    if epsilon_cos is None and omega_lambda is not None:
        if not (0.0 < float(omega_lambda) < 1.0):
            raise ValueError("omega_lambda must be in (0,1)")
        epsilon_cos = float(math.sqrt(float(omega_lambda)/(1.0-float(omega_lambda))))
    if epsilon_cos is None:
        # fallback：與 kappa_profile 預設一致
        epsilon_cos = 1.47

    if eps_norm == "cos":
        eta_hat = fit["A"]
        B_cos   = fit["B"]
        scale   = 1.0
    elif eps_norm == "fit":
        scale   = float(eps_fit) / float(epsilon_cos)
        eta_hat = fit["A"] * scale
        B_cos   = fit["B"] * scale
    else:
        raise ValueError("eps_norm must be 'fit' or 'cos'.")

    out = dict(
        prefix=prefix,
        eps_norm=eps_norm,
        A=fit["A"], B=fit["B"], R2=fit["R2"], N=fit["N"],
        epsilon_fit=float(eps_fit), epsilon_cos=float(epsilon_cos),
        scale_fit_to_cos=float(scale),
        eta_hat=float(eta_hat), B_cos=float(B_cos),
        binned_csv=str(binned_csv)
    )
    if out_json is None:
        out_json = str(res / f"{prefix}_abfit.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    return out


def kappa_two_param_bootstrap(results_dir: str,
                              prefix: str = "kappa_profile",
                              eps_norm: str = "cos",
                              epsilon_cos: float | None = None,
                              omega_lambda: float | None = None,
                              n_boot: int = 2000,
                              min_per_bin: int = 20,
                              seed: int = 1234,
                              out_json: str | None = None) -> dict:
    """
    從 {results_dir}/{prefix}_per_point.csv 作「按星系」bootstrap：
      1) 以 galaxy 為單位重抽樣（同權重），
      2) 依抽樣集重新分箱（與 kappa_profile 相同做法），取每箱 q50，
      3) 對 (x_mid, q50) 擬合 y=A/x+B，保留 A, B。
    產出 A, B 的 16/50/84 百分位，並轉成 (eta_hat, B_cos)。
    """
    import json, numpy as np, pandas as pd, math
    from pathlib import Path
    rng = np.random.default_rng(int(seed))
    res = Path(results_dir)
    per_csv = res / f"{prefix}_per_point.csv"
    if not per_csv.exists():
        raise FileNotFoundError(f"{per_csv} not found. Run `ptquat exp kappa-prof` first.")

    df = pd.read_csv(per_csv)
    # 只留必要欄位
    need = {"galaxy","x","y"}
    if not need.issubset(set(df.columns)):
        raise ValueError(f"{per_csv} needs columns {need}")
    # 取得 eps_fit / eps_cos
    summ = yaml.safe_load(open(res/"global_summary.yaml"))
    eps_fit = float(summ.get("epsilon_median")) if summ.get("epsilon_median") is not None else np.nan
    if epsilon_cos is None and omega_lambda is not None:
        if not (0.0 < float(omega_lambda) < 1.0):
            raise ValueError("omega_lambda must be in (0,1)")
        epsilon_cos = float(math.sqrt(float(omega_lambda)/(1.0-float(omega_lambda))))
    if epsilon_cos is None:
        epsilon_cos = 1.47

    A_list, B_list = [], []
    gals = np.array(sorted(df["galaxy"].unique()))
    G = len(gals)

    for _ in range(int(n_boot)):
        # 以星系為單位重抽樣
        pick = gals[rng.integers(0, G, size=G)]
        sdf = pd.concat([df.loc[df["galaxy"]==g] for g in pick], ignore_index=True)

        # 重新分箱（與 kappa_profile 相近）：用 bootstrap 樣本 x 的 2–98% 分位界線
        x = sdf["x"].values
        x_min = float(np.nanquantile(x, 0.02))
        x_max = float(np.nanquantile(x, 0.98))
        edges = np.linspace(x_min, x_max, 24+1)
        mids  = 0.5*(edges[:-1]+edges[1:])
        ymed = np.full_like(mids, np.nan, dtype=float)

        for i in range(len(edges)-1):
            m = (sdf["x"]>=edges[i]) & (sdf["x"]<edges[i+1])
            vals = sdf.loc[m, "y"].values
            if vals.size >= min_per_bin:
                ymed[i] = float(np.nanpercentile(vals, 50))

        fit = _ab_fit_from_xy(mids, ymed)
        A_list.append(fit["A"]); B_list.append(fit["B"])

    A_arr = np.asarray(A_list); B_arr = np.asarray(B_list)
    def _q(a): 
        return dict(p16=float(np.nanpercentile(a,16)),
                    p50=float(np.nanpercentile(a,50)),
                    p84=float(np.nanpercentile(a,84)))

    if eps_norm == "cos":
        scale = 1.0
        eta_q  = _q(A_arr);  Bc_q = _q(B_arr)
    elif eps_norm == "fit":
        scale = float(eps_fit) / float(epsilon_cos)
        eta_q = _q(A_arr*scale); Bc_q = _q(B_arr*scale)
    else:
        raise ValueError("eps_norm must be 'fit' or 'cos'.")

    out = dict(
        prefix=prefix, eps_norm=eps_norm,
        epsilon_fit=float(eps_fit), epsilon_cos=float(epsilon_cos),
        scale_fit_to_cos=float(scale), n_boot=int(n_boot), seed=int(seed),
        A=_q(A_arr), B=_q(B_arr),
        eta_hat=eta_q, B_cos=Bc_q,
        per_point_csv=str(per_csv)
    )
    if out_json is None:
        out_json = str(res / f"{prefix}_abfit_boot.json")
    with open(out_json, "w") as f:
        json.dump(out, f, indent=2)
    return out
