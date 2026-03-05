# src/ptquat/experiments.py
from __future__ import annotations
import math, json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
import yaml

from .data import load_tidy_sparc, GalaxyData
from .likelihood import build_covariance
from .constants import H0_SI, KPC, KM
from .models import (
    model_v_baryon, model_v_mond, model_v_nfw1p,
    model_v_ptq, model_v_ptq_split, model_v_ptq_nu, model_v_ptq_screen,
    vbar_squared_kms2, linear_term_kms2
)
from .fit_global import (
    run as run_fit,
    _layout,
    _unpack,
    log_likelihood_per_galaxy,
    log_likelihood_full_per_galaxy,
)
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
    res = Path(results_dir)
    summ = yaml.safe_load(open(res / "global_summary.yaml"))
    per = pd.read_csv(res / "per_galaxy_summary.csv").set_index("galaxy")

    like = str(summ.get("likelihood", "gauss"))
    nu = float(summ.get("t_dof", 8.0))
    H0_si = float(summ.get("H0_si", H0_SI))
    model = summ["model"]
    sig_med = float(summ.get("sigma_sys_median", 0.0))
    eps_med = summ.get("epsilon_median", None)
    q_med = summ.get("q_median", None)
    a0_med = summ.get("a0_median", None)

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

    n = 0
    hit68 = 0
    hit95 = 0
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
        Cg = build_covariance(
            v_mod, g.r_kpc, g.v_err, g.D_Mpc, g.D_err_Mpc,
            g.i_deg, g.i_err_deg, vfun, sigma_sys_kms=sig_med
        )
        diag = np.clip(np.diag(Cg), 1e-12, None)
        sig = np.sqrt(diag)
        r = np.abs(g.v_obs - v_mod)
        n += r.size
        hit68 += int((r <= k68 * sig).sum())
        hit95 += int((r <= k95 * sig).sum())

    cov68 = hit68 / n
    cov95 = hit95 / n

    out = {"model": model, "N": n, "coverage68": cov68, "coverage95": cov95}
    with open(res / f"{out_prefix}_coverage.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


# -------------------------------
# S2 誤差壓力測試：增 i_err / D_err 並重跑
# -------------------------------
def stress_errors(data_path: str,
                  out_root: str,
                  model: str,
                  scale_i: float = 2.0,
                  scale_D: float = 2.0,
                  **fit_kwargs) -> Dict:
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
# S3 內盤遮罩敏感度
# -------------------------------
def mask_inner(data_path: str,
               out_root: str,
               model: str,
               rmin_kpc: float = 2.0,
               **fit_kwargs) -> Dict:
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
# S4 H0 敏感度掃描
# -------------------------------
def scan_H0(data_path: str,
            out_root: str,
            model: str,
            H0_list: List[float],
            **fit_kwargs) -> pd.DataFrame:
    out_dir = Path(out_root); out_dir.mkdir(parents=True, exist_ok=True)
    rows_diag = []
    rows_full = []
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
# 殘差加速度平臺
# -------------------------------
def residual_plateau(results_dir: str,
                     data_path: str,
                     nbins: int = 24,
                     out_prefix: str = "plateau") -> Tuple[pd.DataFrame, Path]:
    results = yaml.safe_load(open(Path(results_dir) / "global_summary.yaml"))
    model = results["model"]
    H0_si = float(results.get("H0_si", H0_SI))
    eps = results.get("epsilon_median")
    q = results.get("q_median")
    a0 = results.get("a0_median")
    per = pd.read_csv(Path(results_dir) / "per_galaxy_summary.csv").set_index("galaxy")

    gdict = load_tidy_sparc(data_path)
    pts = []

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
            U = float(per.loc[gname, "Upsilon_med"])
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
    edges = np.linspace(rmin, rmax, nbins + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    q16, q50, q84, cnt = [], [], [], []
    for i in range(nbins):
        mask = (df["r_kpc"] >= edges[i]) & (df["r_kpc"] < edges[i + 1])
        vals = df.loc[mask, "delta_a"].values
        if len(vals) == 0:
            q16.append(np.nan); q50.append(np.nan); q84.append(np.nan); cnt.append(0)
        else:
            q16.append(float(np.percentile(vals, 16)))
            q50.append(float(np.percentile(vals, 50)))
            q84.append(float(np.percentile(vals, 84)))
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
                 epsilon_cos: Optional[float] = None,
                 omega_lambda: Optional[float] = None) -> Dict:
    results = yaml.safe_load(open(Path(results_dir) / "global_summary.yaml"))
    eps_rc, eps16, eps84 = results.get("epsilon_median"), results.get("epsilon_p16"), results.get("epsilon_p84")
    if eps_rc is None:
        raise ValueError("No epsilon in results; not a PTQ-family fit.")

    if epsilon_cos is None:
        if omega_lambda is None:
            raise ValueError("Provide epsilon_cos or omega_lambda.")
        if not (0.0 < omega_lambda < 1.0):
            raise ValueError("omega_lambda must be in (0,1).")
        epsilon_cos = math.sqrt(omega_lambda / (1.0 - omega_lambda))

    sig = 0.5 * abs(float(eps84) - float(eps16)) if (eps16 is not None and eps84 is not None) else np.nan
    diff = float(eps_rc) - float(epsilon_cos)
    result = dict(
        epsilon_RC=float(eps_rc),
        epsilon_cos=float(epsilon_cos),
        sigma_RC=float(sig),
        diff=float(diff),
        pass_within_3sigma=(abs(diff) <= 3.0 * sig if np.isfinite(sig) else None)
    )
    out = Path(results_dir) / "closure_test.yaml"
    with open(out, "w") as f:
        yaml.safe_dump(result, f)
    return result


# ====== Kappa checks ======

def _get_Rd_kpc_safe(g: GalaxyData) -> float:
    """Prefer explicit Rd_kpc from data; fallback to r_peak(v_disk)/2.2."""
    if getattr(g, "Rd_kpc", None) is not None:
        v = float(g.Rd_kpc)
        if np.isfinite(v) and v > 0:
            return v
    # Fallback: exponential disk peaks ~ at 2.2 Rd
    i_pk = int(np.argmax(g.v_disk)) if len(g.v_disk) > 0 else 0
    rpk = float(g.r_kpc[i_pk]) if len(g.r_kpc) > 0 else 1.0
    return max(rpk / 2.2, 0.1)

def _make_vfun(model: str, H0_si: float, per_row: pd.Series, g: GalaxyData,
               eps: Optional[float], q: Optional[float], a0: Optional[float]):
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
        U = float(per_row["Upsilon_med"])
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
        return float(np.sqrt(omega_lambda / (1.0 - omega_lambda)))
    return 1.47  # default anchor

def _rstar_linear_interp(r: np.ndarray, v: np.ndarray, frac_vmax: float) -> float:
    """
    Return r* where v crosses f*Vmax using linear interpolation between the first
    pair (i-1, i) with v[i-1] < fV <= v[i]. If no crossing, fall back to argmax.
    """
    r = np.asarray(r, float); v = np.asarray(v, float)
    if r.size < 2:
        return float(r[0]) if r.size else 1.0
    vmax = float(np.nanmax(v))
    fV = float(frac_vmax) * vmax
    idxs = np.where(v >= fV)[0]
    if idxs.size == 0:
        return float(r[int(np.nanargmax(v))])
    i = int(idxs[0])
    if i == 0:
        return float(r[0])
    v0, v1 = float(v[i-1]), float(v[i])
    r0, r1 = float(r[i-1]), float(r[i])
    if not np.isfinite(v0) or not np.isfinite(v1) or v1 == v0:
        return float(r[i])
    w = (fV - v0) / (v1 - v0)
    return float(r0 + w * (r1 - r0))

def _interp_rows_at_scalar(xp: np.ndarray, fp2d: np.ndarray, x0: float) -> np.ndarray:
    """
    對 2D 陣列的每一列，在同一個 x0 做線性內插。
    xp: (n,) 單調遞增
    fp2d: (nsamp, n)
    回傳: (nsamp,)
    """
    xp = np.asarray(xp, float)
    F  = np.asarray(fp2d, float)
    assert xp.ndim == 1 and F.ndim == 2 and F.shape[1] == xp.size, "shape mismatch for interpolation"

    n = xp.size
    # 找到右側索引 i，使得 xp[i-1] <= x0 <= xp[i]
    i = int(np.searchsorted(xp, float(x0)))
    if i <= 0:
        return F[:, 0].copy()
    if i >= n:
        return F[:, -1].copy()

    xL, xR = xp[i-1], xp[i]
    w = (float(x0) - xL) / (xR - xL) if xR != xL else 0.0
    return (1.0 - w) * F[:, i-1] + w * F[:, i]

def _deming_fit(x: np.ndarray, y: np.ndarray, lam: float = 1.0) -> Tuple[float, float]:
    """
    Deming regression (measurement error in both variables), returns (slope, intercept).
    lam = sigma_x^2 / sigma_y^2. Default 1.0 if unknown.
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    xbar, ybar = np.nanmean(x), np.nanmean(y)
    Sxx = np.nanmean((x - xbar) ** 2)
    Syy = np.nanmean((y - ybar) ** 2)
    Sxy = np.nanmean((x - xbar) * (y - ybar))
    if not np.isfinite(Sxy) or Sxy == 0.0:
        return np.nan, np.nan
    term = (Syy - lam * Sxx)
    beta = (term + np.sqrt(term * term + 4 * lam * Sxy * Sxy)) / (2 * Sxy)
    alpha = ybar - beta * xbar
    return float(beta), float(alpha)


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
                     regression: str = "ols",      # "ols" | "deming"
                     deming_lambda: float = 1.0,   # sigma_x^2 / sigma_y^2
                     interpolate_rstar: bool = True,
                     out_prefix: str = "kappa_gal") -> Dict[str, Any]:
    """
    檢核A：以特徵半徑 r* 做 y 對 x 的迴歸。
      x = kappa_pred = (eta/eps_den) * (Rd / r*)
      y = eps_eff / eps_den
    """
    assert y_source in ("model", "obs", "obs-debias")
    assert eps_norm in ("fit", "cos")
    assert rstar_from in ("model", "obs")
    assert regression in ("ols", "deming")

    res = yaml.safe_load(open(Path(results_dir) / "global_summary.yaml"))
    model = res["model"]
    H0_si = float(res.get("H0_si", H0_SI))
    sig_med = float(res.get("sigma_sys_median", 0.0))
    eps_fit_global = float(res.get("epsilon_median")) if res.get("epsilon_median") is not None else np.nan
    q = res.get("q_median")
    a0 = res.get("a0_median")
    per = pd.read_csv(Path(results_dir) / "per_galaxy_summary.csv").set_index("galaxy")

    # 分母
    if eps_norm == "fit":
        eps_den = float(eps_fit_global)
    else:
        eps_den = _eps_cos_from_args(epsilon_cos, omega_lambda)
    if not np.isfinite(eps_den) or eps_den == 0.0:
        raise ValueError("Invalid epsilon denominator (eps_den).")

    gdict = load_tidy_sparc(data_path)
    rng = np.random.default_rng(1234)
    rows = []

    for name, g in sorted(gdict.items()):
        if name not in per.index:
            continue
        vfun, vbar2 = _make_vfun(model, H0_si, per.loc[name], g, res.get("epsilon_median"), q, a0)

        r = g.r_kpc
        if len(r) < 3:
            continue
        v_obs = g.v_obs
        v_mod = vfun(r)

        # r*
        v_for_rstar = v_mod if (rstar_from == "model") else v_obs
        if interpolate_rstar:
            r_star = _rstar_linear_interp(r, v_for_rstar, frac_vmax)
        else:
            vmax = float(np.nanmax(v_for_rstar))
            idxs = np.where(v_for_rstar >= frac_vmax * vmax)[0]
            i_star = int(idxs[0]) if len(idxs) > 0 else int(np.nanargmax(v_for_rstar))
            i_star = max(0, min(i_star, len(r) - 1))
            r_star = float(r[i_star])

        # 找到 r* 對應索引（最近點，僅用於讀 vbar2 與誤差）
        i_near = int(np.nanargmin(np.abs(r - r_star)))
        vbar2_i = float(vbar2[i_near])
        denom = float(linear_term_kms2(1.0, np.asarray([r_star]), H0_si=H0_si)[0])   # (cH0) r*

        # y_source
        if y_source == "model":
            v_pred2 = float(np.interp(r_star, r, v_mod) ** 2)  # 內插 v_mod 到 r*
            eps_eff_med = (v_pred2 - vbar2_i) / denom
            eps_lo = np.nan; eps_hi = np.nan

        elif y_source == "obs":
            C_full = build_covariance(
                v_mod, r, g.v_err, g.D_Mpc, g.D_err_Mpc,
                g.i_deg, g.i_err_deg, vfun, sigma_sys_kms=sig_med
            )
            try:
                Vs = rng.multivariate_normal(mean=v_obs, cov=C_full, size=nsamp)
                v_star = _interp_rows_at_scalar(r, Vs, r_star)
                eps_samp = (v_star**2 - vbar2_i) / denom
                eps_eff_med = float(np.percentile(eps_samp, 50))
                eps_lo = float(np.percentile(eps_samp, 16))
                eps_hi = float(np.percentile(eps_samp, 84))
            except np.linalg.LinAlgError:
                sig_i = float(np.sqrt(np.clip(np.diag(C_full)[i_near], 1e-30, np.inf)))
                v0 = float(np.interp(r_star, r, v_obs))
                eps_eff_med = (v0**2 - vbar2_i) / denom
                eps_lo = eps_eff_med - (2 * v0 * sig_i) / denom
                eps_hi = eps_eff_med + (2 * v0 * sig_i) / denom

        else:  # "obs-debias"
            sig_i2_meas = float(g.v_err[i_near] ** 2)
            v0 = float(np.interp(r_star, r, v_obs))
            eps_eff_med = ((v0**2 - sig_i2_meas) - vbar2_i) / denom
            try:
                C_meas_diag = np.diag(np.asarray(g.v_err, float) ** 2)
                Vs = rng.multivariate_normal(mean=v_obs, cov=C_meas_diag, size=nsamp)
                v_star = _interp_rows_at_scalar(r, Vs, r_star)
                eps_samp = ((v_star**2 - sig_i2_meas) - vbar2_i) / denom
                eps_lo = float(np.percentile(eps_samp, 16))
                eps_hi = float(np.percentile(eps_samp, 84))
            except np.linalg.LinAlgError:
                sig_i = float(np.sqrt(max(sig_i2_meas, 1e-30)))
                base = max(v0**2 - sig_i2_meas, 0.0) ** 0.5
                eps_lo = eps_eff_med - (2 * base * sig_i) / denom
                eps_hi = eps_eff_med + (2 * base * sig_i) / denom

        Rd = _get_Rd_kpc_safe(g)
        kappa_pred = float((eta / eps_den) * Rd / r_star)

        rows.append(dict(
            galaxy=name,
            r_star_kpc=r_star,
            Rd_kpc=Rd,
            eps_eff_med=eps_eff_med,
            eps_eff_p16=eps_lo,
            eps_eff_p84=eps_hi,
            eps_eff_over_epsden=eps_eff_med / eps_den,
            kappa_pred=kappa_pred
        ))

    df = pd.DataFrame(rows)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["kappa_pred", "eps_eff_over_epsden"])
    out_csv = Path(results_dir) / f"{out_prefix}_per_galaxy.csv"
    df.to_csv(out_csv, index=False)

    # 回歸
    if len(df) >= 2:
        x = df["kappa_pred"].values
        y = df["eps_eff_over_epsden"].values
        if regression == "deming":
            a, b = _deming_fit(x, y, lam=float(deming_lambda))
        else:
            a, b = np.polyfit(x, y, 1)
        yhat = a * x + b
        denom_var = np.sum((y - y.mean())**2)
        r2 = float(1.0 - np.sum((y - yhat)**2) / denom_var) if denom_var > 0 else np.nan
        k_est = a * eta
    else:
        a = b = r2 = k_est = np.nan

    # 圖
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=150)
    if len(df) > 0:
        ax.scatter(df["kappa_pred"], df["eps_eff_over_epsden"], s=18, alpha=0.7, label="galaxies")
    xgrid = np.linspace(0.0, max(1.05 * df["kappa_pred"].max(), 0.2) if len(df) > 0 else 0.2, 200)
    ax.plot(xgrid, xgrid, linestyle="--", linewidth=1.2, label="y = x")
    if np.isfinite(a):
        ax.plot(xgrid, a * xgrid + b, linewidth=1.4,
                label=f"{regression} fit: y={a:.2f}x+{b:.2f}, R²={r2:.2f}")
    ylab_den = r"\varepsilon_{\rm fit}" if eps_norm == "fit" else r"\varepsilon_{\rm cos}"
    ax.set_xlabel(r"$\kappa_{\rm pred}=(\eta/\varepsilon_{\rm den})\,R_d/r_\ast$")
    ax.set_ylabel(rf"$\varepsilon_{{\rm eff}}(r_\ast)/{ylab_den}$")
    ax.legend(); ax.grid(True, alpha=0.25)
    out_png = Path(results_dir) / f"{out_prefix}.png"
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)

    # 總結
    summ = dict(
        N=int(len(df)),
        eta=float(eta),
        frac_vmax=float(frac_vmax),
        y_source=y_source,
        eps_norm=eps_norm,
        rstar_from=rstar_from,
        interpolate_rstar=bool(interpolate_rstar),
        regression=regression,
        deming_lambda=float(deming_lambda),
        slope=float(a), intercept=float(b), R2=float(r2), k_est=float(k_est),
        eps_fit=float(eps_fit_global),
        eps_den=float(eps_den),
        ratio_epsfit_over_epsden=(float(eps_fit_global) / float(eps_den)
                                  if (np.isfinite(eps_fit_global) and np.isfinite(eps_den) and eps_den != 0)
                                  else np.nan),
        csv=str(out_csv), png=str(out_png)
    )
    with open(Path(results_dir) / f"{out_prefix}_summary.json", "w") as f:
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
                          eps_norm: str = "fit",
                          out_prefix: str = "kappa_profile",
                          x_markers: Optional[List[float]] = None) -> Tuple[pd.DataFrame, Path]:
    assert eps_norm in ("fit", "cos")

    res = yaml.safe_load(open(Path(results_dir) / "global_summary.yaml"))
    model = res["model"]
    H0_si = float(res.get("H0_si", H0_SI))
    sig_med = float(res.get("sigma_sys_median", 0.0))
    eps_fit_global = float(res.get("epsilon_median")) if res.get("epsilon_median") is not None else np.nan
    q = res.get("q_median")
    a0 = res.get("a0_median")
    per = pd.read_csv(Path(results_dir) / "per_galaxy_summary.csv").set_index("galaxy")

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
    per_csv = Path(results_dir) / f"{out_prefix}_per_point.csv"
    df.to_csv(per_csv, index=False)

    if len(df) == 0:
        raise RuntimeError("No valid points for kappa profile.")
    x_min = float(np.quantile(df["x"], 0.02))
    x_max = float(np.quantile(df["x"], 0.98))
    edges = np.linspace(x_min, x_max, nbins + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    q16, q50, q84, cnt = [], [], [], []
    for i in range(nbins):
        m = (df["x"] >= edges[i]) & (df["x"] < edges[i + 1])
        vals = df.loc[m, "y"].values
        if len(vals) < min_per_bin:
            q16.append(np.nan); q50.append(np.nan); q84.append(np.nan); cnt.append(int(len(vals)))
        else:
            q16.append(float(np.percentile(vals, 16)))
            q50.append(float(np.percentile(vals, 50)))
            q84.append(float(np.percentile(vals, 84)))
            cnt.append(int(len(vals)))

    binned = pd.DataFrame(dict(x_mid=mids, q16=q16, q50=q50, q84=q84, n=cnt))
    bin_csv = Path(results_dir) / f"{out_prefix}_binned.csv"
    binned.to_csv(bin_csv, index=False)

    # 參考曲線 A = eta/eps_den
    if x_kind == "r_over_Rd":
        xgrid = np.linspace(max(x_min, 1e-3), x_max, 400)
        ypred = (eta / eps_den) / np.maximum(xgrid, 1e-6)
        pred_label = r"$({\eta}/{\varepsilon_{\rm den}})/x$"
    else:
        Rd_med = float(np.median(df["Rd_kpc"]))
        xgrid = np.linspace(max(x_min, 1e-3), x_max, 400)
        ypred = (eta / eps_den) * (Rd_med) / np.maximum(xgrid, 1e-6)
        pred_label = r"$({\eta}/{\varepsilon_{\rm den}})\,R_{d,{\rm med}}/r$"

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.6, 4.0), dpi=150)
    ax.plot(xgrid, ypred, linestyle="--", linewidth=1.2, label=pred_label)
    ax.fill_between(binned["x_mid"], binned["q16"], binned["q84"], alpha=0.25, label="stacked 16–84%")
    ax.plot(binned["x_mid"], binned["q50"], linewidth=1.6, label="stacked median")
    ax.set_xlabel("x = r/Rd" if x_kind == "r_over_Rd" else "r [kpc]")
    ylab_den = r"\varepsilon_{\rm fit}" if eps_norm == "fit" else r"\varepsilon_{\rm cos}"
    ax.set_ylabel(rf"$\varepsilon_{{\rm eff}}(r)/{ylab_den}$")
    ax.grid(True, alpha=0.25)

    xcheck = {}
    if x_kind == "r_over_Rd":
        def _interp(xs, ys, x0):
            xs = np.asarray(xs); ys = np.asarray(ys)
            m = np.isfinite(xs) & np.isfinite(ys)
            if m.sum() < 2 or not (xs[m].min() <= x0 <= xs[m].max()):
                return np.nan
            return float(np.interp(x0, xs[m], ys[m]))
        for x0 in (x_markers or []):
            y_med = _interp(binned["x_mid"], binned["q50"], float(x0))
            y_th = float((eta / eps_den) / max(x0, 1e-6))
            xcheck[str(x0)] = dict(y_median=y_med, y_pred=y_th, delta=y_med - y_th)
            ax.axvline(x0, linestyle=":", linewidth=1.0, alpha=0.6)
            if np.isfinite(y_med):
                ax.scatter([x0], [y_med], s=28, zorder=5, label=f"median @ x={x0:g}")
            ax.scatter([x0], [y_th], s=28, marker="x", zorder=5, label=f"pred @ x={x0:g}")

    ax.legend()
    out_png = Path(results_dir) / f"{out_prefix}.png"
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

    with open(Path(results_dir) / f"{out_prefix}_xcheck.json", "w") as f:
        json.dump(dict(x_kind=x_kind, eta=eta, eps_den=float(eps_den), xcheck=xcheck), f, indent=2)

    return binned, out_png

# ====== 兩參數擬合與 bootstrap ======
def _ab_fit_from_xy(x_mid: np.ndarray, y_med: np.ndarray) -> dict:
    m = np.isfinite(x_mid) & np.isfinite(y_med) & (x_mid > 0)
    x = x_mid[m]; y = y_med[m]
    if x.size < 3:
        return dict(A=np.nan, B=np.nan, R2=np.nan, N=int(x.size))
    X = np.vstack([1.0 / x, np.ones_like(x)]).T
    A, B = np.linalg.lstsq(X, y, rcond=None)[0]
    yhat = A * (1.0 / x) + B
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    R2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else np.nan)
    return dict(A=float(A), B=float(B), R2=float(R2), N=int(x.size))

def kappa_two_param_fit(results_dir: str,
                        prefix: str = "kappa_profile",
                        eps_norm: str = "cos",
                        epsilon_cos: float | None = None,
                        omega_lambda: float | None = None,
                        out_json: str | None = None) -> dict:
    import math, json
    from pathlib import Path
    res = Path(results_dir)
    binned_csv = res / f"{prefix}_binned.csv"
    if not binned_csv.exists():
        raise FileNotFoundError(f"{binned_csv} not found. Run `ptquat exp kappa-prof` first.")

    b = pd.read_csv(binned_csv)
    fit = _ab_fit_from_xy(b["x_mid"].values, b["q50"].values)

    summ = yaml.safe_load(open(res / "global_summary.yaml"))
    eps_fit = float(summ.get("epsilon_median")) if summ.get("epsilon_median") is not None else np.nan
    if epsilon_cos is None and omega_lambda is not None:
        if not (0.0 < float(omega_lambda) < 1.0):
            raise ValueError("omega_lambda must be in (0,1)")
        epsilon_cos = float(math.sqrt(float(omega_lambda) / (1.0 - float(omega_lambda))))
    if epsilon_cos is None:
        epsilon_cos = 1.47

    if eps_norm == "cos":
        eta_hat = fit["A"]
        B_cos = fit["B"]
        scale = 1.0
    elif eps_norm == "fit":
        scale = float(eps_fit) / float(epsilon_cos)
        eta_hat = fit["A"] * scale
        B_cos = fit["B"] * scale
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
    import math, json
    from pathlib import Path
    rng = np.random.default_rng(int(seed))
    res = Path(results_dir)
    per_csv = res / f"{prefix}_per_point.csv"
    if not per_csv.exists():
        raise FileNotFoundError(f"{per_csv} not found. Run `ptquat exp kappa-prof` first.")

    df = pd.read_csv(per_csv)
    need = {"galaxy", "x", "y"}
    if not need.issubset(set(df.columns)):
        raise ValueError(f"{per_csv} needs columns {need}")

    summ = yaml.safe_load(open(res / "global_summary.yaml"))
    eps_fit = float(summ.get("epsilon_median")) if summ.get("epsilon_median") is not None else np.nan
    if epsilon_cos is None and omega_lambda is not None:
        if not (0.0 < float(omega_lambda) < 1.0):
            raise ValueError("omega_lambda must be in (0,1)")
        epsilon_cos = float(math.sqrt(float(omega_lambda) / (1.0 - float(omega_lambda))))
    if epsilon_cos is None:
        epsilon_cos = 1.47

    A_list, B_list = [], []
    gals = np.array(sorted(df["galaxy"].unique()))
    G = len(gals)

    for _ in range(int(n_boot)):
        pick = gals[rng.integers(0, G, size=G)]
        sdf = pd.concat([df.loc[df["galaxy"] == g] for g in pick], ignore_index=True)

        x = sdf["x"].values
        x_min = float(np.nanquantile(x, 0.02))
        x_max = float(np.nanquantile(x, 0.98))
        edges = np.linspace(x_min, x_max, 24 + 1)
        mids = 0.5 * (edges[:-1] + edges[1:])
        ymed = np.full_like(mids, np.nan, dtype=float)

        for i in range(len(edges) - 1):
            m = (sdf["x"] >= edges[i]) & (sdf["x"] < edges[i + 1])
            vals = sdf.loc[m, "y"].values
            if vals.size >= min_per_bin:
                ymed[i] = float(np.nanpercentile(vals, 50))

        fit = _ab_fit_from_xy(mids, ymed)
        A_list.append(fit["A"]); B_list.append(fit["B"])

    A_arr = np.asarray(A_list); B_arr = np.asarray(B_list)

    def _q(a):
        return dict(p16=float(np.nanpercentile(a, 16)),
                    p50=float(np.nanpercentile(a, 50)),
                    p84=float(np.nanpercentile(a, 84)))

    if eps_norm == "cos":
        scale = 1.0
        eta_q = _q(A_arr); Bc_q = _q(B_arr)
    elif eps_norm == "fit":
        scale = float(eps_fit) / float(epsilon_cos)
        eta_q = _q(A_arr * scale); Bc_q = _q(B_arr * scale)
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


def kappa_profile_fit(results_dir: str,
                      prefix: str,
                      eps_norm: str = "cos",                 # "cos" or "fit"
                      epsilon_cos: Optional[float] = None,
                      omega_lambda: Optional[float] = None,
                      bootstrap: int = 0,
                      seed: int = 1234) -> Dict[str, Any]:
    """
    讀取 {results_dir}/{prefix}_binned.csv 的 (x_mid, q50)，以 y = A*(1/x) + B 做 OLS。
    A 為圖上幅度（已除 eps_den）；物理幅度 ≈ A * eps_den。
    """
    import json
    from pathlib import Path

    res = Path(results_dir)
    binned_csv = res / f"{prefix}_binned.csv"
    if not binned_csv.exists():
        raise FileNotFoundError(f"{binned_csv} not found. Run kappa-prof first with the same prefix.")

    summ = yaml.safe_load(open(res / "global_summary.yaml"))
    eps_fit = float(summ.get("epsilon_median", np.nan))
    if eps_norm not in ("cos", "fit"):
        raise ValueError("eps_norm must be 'cos' or 'fit'.")

    if eps_norm == "cos":
        eps_cos = _eps_cos_from_args(epsilon_cos, omega_lambda)
    else:
        eps_cos = float(_eps_cos_from_args(None, 0.685))  # 僅用於輸出換算

    df = pd.read_csv(binned_csv)
    m = np.isfinite(df.q50) & np.isfinite(df.x_mid) & (df.x_mid > 0)
    x = df.loc[m, "x_mid"].to_numpy()
    y = df.loc[m, "q50"].to_numpy()
    if x.size < 3:
        raise RuntimeError("Not enough bins to fit (need >=3).")
    X = np.vstack([1.0 / x, np.ones_like(x)]).T
    A, B = np.linalg.lstsq(X, y, rcond=None)[0]

    yhat = X @ np.array([A, B])
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    R2 = 1.0 - (ss_res / ss_tot if ss_tot > 0 else np.nan)

    scale = float(eps_fit / eps_cos) if np.isfinite(eps_fit) and eps_cos else 1.0
    eta_hat = (A * scale) if (eps_norm == "fit") else A
    B_cos = (B * scale) if (eps_norm == "fit") else B

    out_main = dict(
        prefix=prefix, eps_norm=eps_norm,
        A=float(A), B=float(B), R2=float(R2), N=int(x.size),
        epsilon_fit=float(eps_fit), epsilon_cos=float(eps_cos),
        scale_fit_to_cos=(float(scale) if eps_norm == "fit" else 1.0),
        eta_hat=float(eta_hat), B_cos=float(B_cos),
        binned_csv=str(binned_csv)
    )
    with open(res / f"{prefix}_fit_summary.json", "w") as f:
        json.dump(out_main, f, indent=2)

    if bootstrap and bootstrap > 0:
        rng = np.random.default_rng(seed)
        A_s, B_s = [], []
        idx = np.arange(x.size)
        for _ in range(int(bootstrap)):
            bs = rng.choice(idx, size=idx.size, replace=True)
            Xb = np.vstack([1.0 / x[bs], np.ones_like(x[bs])]).T
            Ab, Bb = np.linalg.lstsq(Xb, y[bs], rcond=None)[0]
            A_s.append(float(Ab)); B_s.append(float(Bb))

        def qnt(v):
            v = np.asarray(v)
            q16, q50, q84 = np.percentile(v, [16, 50, 84])
            return dict(p16=float(q16), p50=float(q50), p84=float(q84))

        out_boot = dict(
            prefix=prefix, eps_norm=eps_norm,
            epsilon_fit=float(eps_fit), epsilon_cos=float(eps_cos),
            scale_fit_to_cos=(float(scale) if eps_norm == "fit" else 1.0),
            n_boot=int(bootstrap), seed=int(seed),
            A=qnt(A_s), B=qnt(B_s),
            eta_hat=(qnt(np.asarray(A_s) * scale) if eps_norm == "fit" else qnt(A_s)),
            B_cos=(qnt(np.asarray(B_s) * scale) if eps_norm == "fit" else qnt(B_s)),
            per_point_csv=str(res / f"{prefix}_per_point.csv")
        )
        with open(res / f"{prefix}_fit_bootstrap.json", "w") as f:
            json.dump(out_boot, f, indent=2)
        out_main["bootstrap_json"] = str(res / f"{prefix}_fit_bootstrap.json")

    return out_main


# ====== Geometry (h) variants for Appendix G.2 ======
def _build_h_map(geom_path: str, h_col: str) -> Dict[str, float]:
    """
    從幾何 CSV 建立 {galaxy -> h_kpc} 對照；使用中位數當代表值。
    需求欄位：['galaxy', h_col]；會過濾非有限/<=0 的 h。
    """
    df = pd.read_csv(geom_path)
    need = {"galaxy", h_col}
    if not need.issubset(set(df.columns)):
        raise ValueError(f"{geom_path} must contain columns {need}")
    s = (df[["galaxy", h_col]]
         .dropna(subset=[h_col])
         .groupby("galaxy")[h_col].median())
    hmap = {str(k): float(v) for k, v in s.items() if np.isfinite(v) and v > 0.0}
    if len(hmap) == 0:
        raise RuntimeError(f"No valid {h_col}>0 in {geom_path}")
    return hmap


def kappa_geom_per_galaxy(results_dir: str,
                          data_path: str,
                          geom_path: str,
                          h_col: str = "h_kpc",
                          eta: float = 0.15,
                          frac_vmax: float = 0.9,
                          epsilon_cos: Optional[float] = None,
                          omega_lambda: Optional[float] = None,
                          nsamp: int = 300,
                          y_source: str = "obs-debias",
                          eps_norm: str = "cos",
                          rstar_from: str = "obs",
                          regression: str = "deming",
                          deming_lambda: float = 1.0,
                          interpolate_rstar: bool = True,
                          out_prefix: str = "kappa_geom_h") -> Dict[str, Any]:
    """
    幾何版 per-galaxy：以 r* 做回歸；以 h 取代 Rd：
      x = kappa_pred = (eta/eps_den) * (h / r*)
      y = eps_eff(r*)/eps_den
    """
    assert y_source in ("model", "obs", "obs-debias")
    assert eps_norm in ("fit", "cos")
    assert rstar_from in ("model", "obs")
    assert regression in ("ols", "deming")

    hmap = _build_h_map(geom_path, h_col)

    res = yaml.safe_load(open(Path(results_dir) / "global_summary.yaml"))
    model = res["model"]
    H0_si = float(res.get("H0_si", H0_SI))
    sig_med = float(res.get("sigma_sys_median", 0.0))
    eps_fit_global = float(res.get("epsilon_median")) if res.get("epsilon_median") is not None else np.nan
    q = res.get("q_median"); a0 = res.get("a0_median")
    per = pd.read_csv(Path(results_dir) / "per_galaxy_summary.csv").set_index("galaxy")

    # 分母
    if eps_norm == "fit":
        eps_den = float(eps_fit_global)
    else:
        eps_den = _eps_cos_from_args(epsilon_cos, omega_lambda)
    if not np.isfinite(eps_den) or eps_den == 0.0:
        raise ValueError("Invalid epsilon denominator (eps_den).")

    gdict = load_tidy_sparc(data_path)
    rng = np.random.default_rng(1234)
    rows = []

    for name, g in sorted(gdict.items()):
        if (name not in per.index) or (name not in hmap):
            continue
        h = float(hmap[name])
        if not (np.isfinite(h) and h > 0.0):
            continue

        vfun, vbar2 = _make_vfun(model, H0_si, per.loc[name], g, res.get("epsilon_median"), q, a0)
        r = g.r_kpc
        if len(r) < 3: continue
        v_obs = g.v_obs; v_mod = vfun(r)

        # r*
        v_for_rstar = v_mod if (rstar_from == "model") else v_obs
        if interpolate_rstar:
            r_star = _rstar_linear_interp(r, v_for_rstar, frac_vmax)
        else:
            vmax = float(np.nanmax(v_for_rstar))
            idxs = np.where(v_for_rstar >= frac_vmax * vmax)[0]
            i_star = int(idxs[0]) if len(idxs) > 0 else int(np.nanargmax(v_for_rstar))
            i_star = max(0, min(i_star, len(r) - 1))
            r_star = float(r[i_star])

        i_near = int(np.nanargmin(np.abs(r - r_star)))
        vbar2_i = float(vbar2[i_near])
        denom = float(linear_term_kms2(1.0, np.asarray([r_star]), H0_si=H0_si)[0])

        if y_source == "model":
            v0 = float(np.interp(r_star, r, v_mod))
            eps_eff_med = (v0**2 - vbar2_i) / denom
            eps_lo = eps_hi = np.nan
        elif y_source == "obs":
            C_full = build_covariance(
                v_mod, r, g.v_err, g.D_Mpc, g.D_err_Mpc,
                g.i_deg, g.i_err_deg, vfun, sigma_sys_kms=sig_med
            )
            try:
                Vs = rng.multivariate_normal(mean=v_obs, cov=C_full, size=nsamp)
                v_star = _interp_rows_at_scalar(r, Vs, r_star)
                eps_s = (v_star**2 - vbar2_i) / denom
                eps_eff_med = float(np.nanpercentile(eps_s, 50))
                eps_lo = float(np.nanpercentile(eps_s, 16))
                eps_hi = float(np.nanpercentile(eps_s, 84))
            except np.linalg.LinAlgError:
                sig_i = float(np.sqrt(np.clip(np.diag(C_full)[i_near], 1e-30, np.inf)))
                v0 = float(np.interp(r_star, r, v_obs))
                eps_eff_med = (v0**2 - vbar2_i) / denom
                eps_lo = eps_eff_med - (2 * v0 * sig_i) / denom
                eps_hi = eps_eff_med + (2 * v0 * sig_i) / denom
        else:
            sig2_meas = float(g.v_err[i_near] ** 2)
            v0 = float(np.interp(r_star, r, v_obs))
            eps_eff_med = ((v0**2 - sig2_meas) - vbar2_i) / denom
            try:
                C_meas = np.diag(np.asarray(g.v_err, float) ** 2)
                Vs = rng.multivariate_normal(mean=v_obs, cov=C_meas, size=nsamp)
                v_star = _interp_rows_at_scalar(r, Vs, r_star)
                eps_s = ((v_star**2 - sig2_meas) - vbar2_i) / denom
                eps_lo = float(np.nanpercentile(eps_s, 16))
                eps_hi = float(np.nanpercentile(eps_s, 84))
            except np.linalg.LinAlgError:
                sig_i = float(np.sqrt(max(sig2_meas, 1e-30)))
                base = max(v0**2 - sig2_meas, 0.0) ** 0.5
                eps_lo = eps_eff_med - (2 * base * sig_i) / denom
                eps_hi = eps_eff_med + (2 * base * sig_i) / denom

        kappa_pred = float((eta / eps_den) * (h / r_star))
        rows.append(dict(
            galaxy=name,
            r_star_kpc=r_star,
            h_kpc=h,
            eps_eff_med=eps_eff_med,
            eps_eff_p16=eps_lo,
            eps_eff_p84=eps_hi,
            eps_eff_over_epsden=eps_eff_med / eps_den,
            kappa_pred=kappa_pred
        ))

    df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["kappa_pred", "eps_eff_over_epsden"])
    out_csv = Path(results_dir) / f"{out_prefix}_per_galaxy.csv"
    df.to_csv(out_csv, index=False)

    # 回歸
    if len(df) >= 2:
        x = df["kappa_pred"].values; y = df["eps_eff_over_epsden"].values
        if regression == "deming":
            a, b = _deming_fit(x, y, lam=float(deming_lambda))
        else:
            a, b = np.polyfit(x, y, 1)
        yhat = a * x + b
        denom_var = np.sum((y - y.mean())**2)
        r2 = float(1.0 - np.sum((y - yhat)**2) / denom_var) if denom_var > 0 else np.nan
        k_est = a * eta
    else:
        a = b = r2 = k_est = np.nan

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=150)
    if len(df) > 0:
        ax.scatter(df["kappa_pred"], df["eps_eff_over_epsden"], s=18, alpha=0.7, label="galaxies")
    xgrid = np.linspace(0.0, max(1.05 * df["kappa_pred"].max(), 0.2) if len(df) > 0 else 0.2, 200)
    ax.plot(xgrid, xgrid, linestyle="--", linewidth=1.2, label="y = x")
    if np.isfinite(a):
        ax.plot(xgrid, a * xgrid + b, linewidth=1.4,
                label=f"{regression} fit: y={a:.2f}x+{b:.2f}, R²={r2:.2f}")
    ylab_den = r"\varepsilon_{\rm fit}" if eps_norm == "fit" else r"\varepsilon_{\rm cos}"
    ax.set_xlabel(r"$\kappa_{\rm pred}=(\eta/\varepsilon_{\rm den})\,h/r_\ast$")
    ax.set_ylabel(rf"$\varepsilon_{{\rm eff}}(r_\ast)/{ylab_den}$")
    ax.legend(); ax.grid(True, alpha=0.25)
    out_png = Path(results_dir) / f"{out_prefix}.png"
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)

    summ = dict(
        N=int(len(df)), eta=float(eta), frac_vmax=float(frac_vmax),
        y_source=y_source, eps_norm=eps_norm, rstar_from=rstar_from,
        interpolate_rstar=bool(interpolate_rstar),
        regression=regression, deming_lambda=float(deming_lambda),
        slope=float(a), intercept=float(b), R2=float(r2), k_est=float(k_est),
        eps_fit=float(eps_fit_global), eps_den=float(eps_den),
        ratio_epsfit_over_epsden=(
            float(eps_fit_global) / float(eps_den)
            if (np.isfinite(eps_fit_global) and np.isfinite(eps_den) and eps_den != 0) else np.nan
        ),
        csv=str(out_csv), png=str(out_png)
    )
    with open(Path(results_dir) / f"{out_prefix}_summary.json", "w") as f:
        json.dump(summ, f, indent=2)
    return summ


def kappa_radius_resolved_geom(results_dir: str,
                               data_path: str,
                               geom_path: str,
                               h_col: str = "h_kpc",
                               eta: float = 0.15,
                               epsilon_cos: Optional[float] = None,
                               omega_lambda: Optional[float] = None,
                               nbins: int = 24,
                               min_per_bin: int = 20,
                               eps_norm: str = "cos",
                               out_prefix: str = "kappa_profile_h") -> Tuple[pd.DataFrame, Path]:
    """
    幾何版半徑堆疊：x=r/h，y=ε_eff(r)/ε_den；與 y=(η/ε_den)/x 比較。
    """
    assert eps_norm in ("fit", "cos")
    hmap = _build_h_map(geom_path, h_col)

    summ = yaml.safe_load(open(Path(results_dir) / "global_summary.yaml"))
    model = summ["model"]
    H0_si = float(summ.get("H0_si", H0_SI))
    sig_med = float(summ.get("sigma_sys_median", 0.0))
    eps_fit_global = float(summ.get("epsilon_median")) if summ.get("epsilon_median") is not None else np.nan
    q = summ.get("q_median"); a0 = summ.get("a0_median")
    per = pd.read_csv(Path(results_dir) / "per_galaxy_summary.csv").set_index("galaxy")

    eps_den = float(eps_fit_global) if eps_norm == "fit" else _eps_cos_from_args(epsilon_cos, omega_lambda)
    if not np.isfinite(eps_den) or eps_den == 0.0:
        raise ValueError("Invalid eps_den for kappa_profile_h.")

    gdict = load_tidy_sparc(data_path)
    pts = []
    for name, g in sorted(gdict.items()):
        if (name not in per.index) or (name not in hmap):
            continue
        h = float(hmap[name])
        if not (np.isfinite(h) and h > 0): continue

        vfun, vbar2 = _make_vfun(model, H0_si, per.loc[name], g, summ.get("epsilon_median"), q, a0)
        v_mod = vfun(g.r_kpc)
        C = build_covariance(v_mod, g.r_kpc, g.v_err, g.D_Mpc, g.D_err_Mpc,
                             g.i_deg, g.i_err_deg, vfun, sigma_sys_kms=sig_med)
        _ = np.sqrt(np.clip(np.diag(C), 1e-30, np.inf))  # 僅為過濾可疑點時參考

        denom = linear_term_kms2(1.0, g.r_kpc, H0_si=H0_si)
        eps_eff = (g.v_obs**2 - vbar2) / np.maximum(denom, 1e-30)
        x = g.r_kpc / h
        y = eps_eff / eps_den

        for xi, yi, ri in zip(x, y, g.r_kpc):
            if np.isfinite(xi) and np.isfinite(yi):
                pts.append(dict(galaxy=name, r_kpc=float(ri), x=float(xi), y=float(yi), h_kpc=h))

    df = pd.DataFrame(pts)
    per_csv = Path(results_dir) / f"{out_prefix}_per_point.csv"
    df.to_csv(per_csv, index=False)
    if len(df) == 0:
        raise RuntimeError("No valid points for kappa_profile_h.")

    x_min = float(np.nanquantile(df["x"], 0.02))
    x_max = float(np.nanquantile(df["x"], 0.98))
    edges = np.linspace(x_min, x_max, nbins + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    q16, q50, q84, cnt = [], [], [], []
    for i in range(nbins):
        m = (df["x"] >= edges[i]) & (df["x"] < edges[i + 1])
        vals = df.loc[m, "y"].values
        if len(vals) < min_per_bin:
            q16.append(np.nan); q50.append(np.nan); q84.append(np.nan); cnt.append(int(len(vals)))
        else:
            q16.append(float(np.nanpercentile(vals, 16)))
            q50.append(float(np.nanpercentile(vals, 50)))
            q84.append(float(np.nanpercentile(vals, 84)))
            cnt.append(int(len(vals)))

    binned = pd.DataFrame(dict(x_mid=mids, q16=q16, q50=q50, q84=q84, n=cnt))
    bin_csv = Path(results_dir) / f"{out_prefix}_binned.csv"
    binned.to_csv(bin_csv, index=False)

    # 參考曲線 A = eta/eps_den
    xgrid = np.linspace(max(x_min, 1e-3), x_max, 400)
    ypred = (eta / eps_den) / np.maximum(xgrid, 1e-6)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.6, 4.0), dpi=150)
    ax.plot(xgrid, ypred, linestyle="--", linewidth=1.2, label=r"$({\eta}/{\varepsilon_{\rm den}})/x$")
    ax.fill_between(binned["x_mid"], binned["q16"], binned["q84"], alpha=0.25, label="stacked 16–84%")
    ax.plot(binned["x_mid"], binned["q50"], linewidth=1.6, label="stacked median")
    ax.set_xlabel("x = r/h")
    ylab_den = r"\varepsilon_{\rm fit}" if eps_norm == "fit" else r"\varepsilon_{\rm cos}"
    ax.set_ylabel(rf"$\varepsilon_{{\rm eff}}(r)/{ylab_den}$")
    ax.grid(True, alpha=0.25); ax.legend()
    out_png = Path(results_dir) / f"{out_prefix}.png"
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)

    with open(Path(results_dir) / f"{out_prefix}_xcheck.json", "w") as f:
        json.dump(dict(eta=eta, eps_den=float(eps_den)), f, indent=2)

    return binned, out_png


# -------------------------------
# Model compare: WAIC from posterior samples
# -------------------------------
def _waic_from_chain(log_lik_mat: np.ndarray) -> Tuple[float, float, float]:
    """
    log_lik_mat: (n_samples, n_data_points) per-galaxy log-likelihoods.
    Returns (waic, p_waic).
    WAIC = -2 * (lppd - p_waic)
    lppd = sum_i log(mean_s(exp(ll_i)))
    p_waic = sum_i var_s(ll_i)
    """
    n_samples = log_lik_mat.shape[0]
    # lppd_i = log(mean(exp(ll_i))), use logsumexp for stability
    ll_max = np.nanmax(log_lik_mat, axis=0, keepdims=True)
    ll_shift = log_lik_mat - ll_max
    lppd = np.log(np.nanmean(np.exp(np.clip(ll_shift, -700, 700)), axis=0)) + np.squeeze(ll_max)
    lppd = np.where(np.isfinite(lppd), lppd, -np.inf)
    p_waic = np.nanvar(log_lik_mat, axis=0)
    p_waic = np.where(np.isfinite(p_waic), p_waic, 0.0)
    lppd_sum = float(np.sum(lppd))
    p_waic_sum = float(np.sum(p_waic))
    waic = -2.0 * (lppd_sum - p_waic_sum)
    return lppd_sum, p_waic_sum, waic


def compare_models_waic(
    data_path: str,
    fit_root: str,
    models: List[str],
    outdir: str,
    seed: int = 0,
    burn_frac: float = 1.0 / 3.0,
    thin: int = 10,
    run_fits_if_missing: bool = False,
    fit_steps: int = 100,
    fit_nwalkers: str = "2x",
) -> Dict[str, Any]:
    """
    Compute WAIC for each model from posterior samples (chain.h5), write compare_table.csv,
    compare_table.tex, rank_plot.png/pdf, manifest.yaml.
    If run_fits_if_missing, run short fits with backend when chain.h5 is absent.
    """
    import subprocess

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    gdict = load_tidy_sparc(data_path)
    galaxies = [gdict[k] for k in sorted(gdict.keys())]
    G = len(galaxies)
    N_total = sum(len(g.r_kpc) for g in galaxies)

    a0_range = (5e-11, 2e-10)
    logM_range = (9.0, 13.0)
    fit_root_p = Path(fit_root)

    rows_diag: list[dict] = []
    rows_full: list[dict] = []
    for model in models:
        res_dir = fit_root_p / f"{model}_gauss"
        chain_path = res_dir / "chain.h5"
        summ_path = res_dir / "global_summary.yaml"

        if (not chain_path.exists()) or (run_fits_if_missing and not summ_path.exists()):
            if run_fits_if_missing:
                res_dir.mkdir(parents=True, exist_ok=True)
                run_fit([
                    f"--data-path={data_path}",
                    f"--outdir={res_dir}",
                    f"--model={model}",
                    f"--prior=galaxies-only",
                    f"--sigma-sys=4.0",
                    f"--steps={fit_steps}",
                    f"--nwalkers={fit_nwalkers}",
                    f"--seed={seed}",
                    f"--backend-hdf5={chain_path}",
                    f"--thin-by={thin}",
                ])
            else:
                raise FileNotFoundError(
                    f"chain.h5 not found at {chain_path}. Run fits with --backend-hdf5 {chain_path} first."
                )

        if not summ_path.exists():
            raise FileNotFoundError(f"global_summary.yaml not found at {summ_path}")

        summ = yaml.safe_load(open(summ_path, encoding="utf-8"))
        H0_si = float(summ.get("H0_si", H0_SI))
        like_kind = str(summ.get("likelihood", "gauss"))
        t_dof = float(summ.get("t_dof", 8.0))
        sigma_sys = float(summ.get("sigma_sys_median", 4.0))

        from emcee.backends import HDFBackend
        backend = HDFBackend(str(chain_path))
        n_steps = backend.iteration
        burn = int(n_steps * burn_frac)
        chain = backend.get_chain(discard=burn, flat=True, thin=thin)
        n_samples = chain.shape[0]

        L = _layout(model, G, sigma_sys_learn=True, a0_fixed=None)
        k_params = L["k"]

        log_lik_list_diag = []
        log_lik_list_full = []
        for s in range(n_samples):
            theta = chain[s]
            ll_diag = log_likelihood_per_galaxy(
                theta, galaxies, sigma_sys, H0_si, L,
                c0=10.0, c_slope=-0.1, a0_fixed=None,
                like_kind=like_kind, t_dof=t_dof
            )
            ll_full = log_likelihood_full_per_galaxy(
                theta, galaxies, sigma_sys, H0_si, L,
                c0=10.0, c_slope=-0.1, a0_fixed=None,
                like_kind=like_kind, t_dof=t_dof
            )
            log_lik_list_diag.append(ll_diag)
            log_lik_list_full.append(ll_full)
        log_lik_mat = np.array(log_lik_list_diag)       # (S, N_total)
        log_lik_full_mat = np.array(log_lik_list_full)  # (S, G)

        # Diagnostics
        ll_shape = log_lik_mat.shape
        ll_mean = float(np.nanmean(log_lik_mat))
        lppd, p_waic, waic = _waic_from_chain(log_lik_mat)
        lppd_full, p_waic_full, waic_full = _waic_from_chain(log_lik_full_mat)
        p_waic_per_point = float(p_waic / max(N_total, 1))
        print(f"[compare] {model}: log_lik shape={ll_shape}, "
              f"mean log_lik per-point={ll_mean:.3f}, "
              f"mean p_waic per-point={p_waic_per_point:.3f}")

        rows_diag.append({
            "model": model,
            "lppd": lppd,
            "waic": waic,
            "p_waic": p_waic,
            "n_params": k_params,
            "n_data": N_total,
            "mean_loglik_per_point": ll_mean,
            "_n_samples": n_samples,
        })

        rows_full.append({
            "model": model,
            "waic": waic_full,
            "p_waic": p_waic_full,
            "n_params": k_params,
            "n_data": G,
        })

    df = pd.DataFrame(rows_diag)
    n_samp = int(df["_n_samples"].iloc[0]) if len(df) else 0
    df["mean_lppd_per_point"] = df["lppd"] / df["n_data"].clip(lower=1)
    df = df.drop(columns=["_n_samples"])
    waic_min = df["waic"].min()
    df["delta_waic"] = df["waic"] - waic_min
    df["rank"] = df["waic"].rank().astype(int)
    df = df.sort_values("waic").reset_index(drop=True)

    # Primary compare table (diag + per_point)
    df.to_csv(out_path / "compare_table.csv", index=False)

    # Breakdown table relative to best (smallest WAIC) for diag+per_point
    if len(df):
        best_idx = df["waic"].idxmin()
        best = df.loc[best_idx]
        d_lppd = df["lppd"] - float(best["lppd"])
        d_pwaic = df["p_waic"] - float(best["p_waic"])
        d_waic = df["waic"] - float(best["waic"])
        breakdown = pd.DataFrame({
            "model": df["model"],
            "lppd": df["lppd"],
            "p_waic": df["p_waic"],
            "waic": df["waic"],
            "delta_lppd_vs_best": d_lppd,
            "delta_pwaic_vs_best": d_pwaic,
            "delta_waic_vs_best": d_waic,
        })
        breakdown.to_csv(out_path / "breakdown.csv", index=False)

        # Print ΔWAIC decomposition: ΔWAIC = -2[(Δlppd) - (ΔpWAIC)]
        print("[compare] ΔWAIC decomposition (diag+per_point vs best WAIC model):")
        for _, r in breakdown.iterrows():
            if r["model"] == best["model"]:
                continue
            dl = float(r["delta_lppd_vs_best"])
            dp = float(r["delta_pwaic_vs_best"])
            dwaic = float(r["delta_waic_vs_best"])
            check = -2.0 * (dl - dp)
            print(f"  {r['model']}: Δlppd={dl:.3f}, ΔpWAIC={dp:.3f}, "
                  f"ΔWAIC={dwaic:.3f}, -2[(Δlppd)-(ΔpWAIC)]={check:.3f}")

    # Full-covariance, per-galaxy WAIC table
    df_full = pd.DataFrame(rows_full)
    if len(df_full):
        waic_min_full = df_full["waic"].min()
        df_full["delta_waic"] = df_full["waic"] - waic_min_full
        df_full = df_full.sort_values("waic").reset_index(drop=True)
        df_full["cov_mode"] = "full"
        df_full["unit"] = "per_gal"
        df_full.to_csv(out_path / "compare_table_covfull_pergal.csv", index=False)

        # Combined table of all modes
        df_diag_for_all = df[["model", "waic", "delta_waic", "p_waic", "n_data"]].copy()
        df_diag_for_all["cov_mode"] = "diag"
        df_diag_for_all["unit"] = "per_point"

        df_full_for_all = df_full[["model", "waic", "delta_waic", "p_waic", "n_data", "cov_mode", "unit"]].copy()
        df_all = pd.concat([df_diag_for_all, df_full_for_all], ignore_index=True)
        df_all.to_csv(out_path / "compare_table_all_modes.csv", index=False)

    # LaTeX table
    tex = "\\begin{tabular}{lrrrrrr}\n"
    tex += "\\toprule\nModel & WAIC & $\\Delta$WAIC & $p_{\\mathrm{WAIC}}$ & $k$ & $N$ & Rank \\\\\n\\midrule\n"
    for _, r in df.iterrows():
        tex += f"{r['model']} & {r['waic']:.1f} & {r['delta_waic']:.1f} & {r['p_waic']:.1f} & {int(r['n_params'])} & {int(r['n_data'])} & {int(r['rank'])} \\\\\n"
    tex += "\\bottomrule\n\\end{tabular}\n"
    (out_path / "compare_table.tex").write_text(tex, encoding="utf-8")

    # Bar plot (ΔWAIC)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=150)
    x = np.arange(len(df))
    ax.barh(x, df["delta_waic"].values, color="steelblue", alpha=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels(df["model"].values)
    ax.set_xlabel(r"$\Delta$WAIC")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path / "rank_plot.png")
    fig.savefig(out_path / "rank_plot.pdf")
    plt.close(fig)

    # manifest
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True
        ).strip()
    except Exception:
        git_hash = "unknown"

    cmd_str = "ptquat exp compare --models " + " ".join(models) + f" --data {data_path} --fit-root {fit_root} --outdir {outdir} --seed {seed}"
    manifest = {
        "git_hash": git_hash,
        "command": cmd_str,
        "n_samples": n_samp,
        "seed": seed,
        "models": models,
    }
    with open(out_path / "manifest.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)

    return {"compare_table": str(out_path / "compare_table.csv"), "df": df}


# -------------------------------
# LOO: PSIS-LOO via ArviZ
# -------------------------------
def loo_compare_models(
    data_path: str,
    fit_root: str,
    models: List[str],
    outdir: str,
    seed: int = 0,
    burn_frac: float = 1.0 / 3.0,
    thin: int = 10,
    run_fits_if_missing: bool = False,
    fit_steps: int = 100,
    fit_nwalkers: str = "2x",
) -> Dict[str, Any]:
    """
    Compare models using PSIS-LOO (via ArviZ).

    - Uses diag-covariance, per-point log-likelihood (same as WAIC diag mode).
    - Builds an InferenceData with log_likelihood and calls az.loo(pointwise=True).
    - Writes loo_table.csv, pareto_k.csv, pareto_k_hist.png, elpd_difference_plot.png.
    """
    try:
        import arviz as az  # type: ignore[import]
    except Exception as e:  # pragma: no cover - runtime dependency
        raise SystemExit(
            "ArviZ (arviz) is required for `ptquat exp loo` but is not importable.\n"
            "Please install it in your environment (e.g. `pip install arviz`) and retry."
        ) from e

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    gdict = load_tidy_sparc(data_path)
    galaxies = [gdict[k] for k in sorted(gdict.keys())]
    G = len(galaxies)
    N_total = sum(len(g.r_kpc) for g in galaxies)

    fit_root_p = Path(fit_root)

    # Flattened observation metadata (shared across models)
    obs_gal: List[str] = []
    obs_r: List[float] = []
    for g in galaxies:
        for r in g.r_kpc:
            obs_gal.append(g.name)
            obs_r.append(float(r))

    rows: List[dict] = []
    pareto_rows: List[dict] = []

    for model in models:
        res_dir = fit_root_p / f"{model}_gauss"
        chain_path = res_dir / "chain.h5"
        summ_path = res_dir / "global_summary.yaml"

        if (not chain_path.exists()) or (run_fits_if_missing and not summ_path.exists()):
            if run_fits_if_missing:
                res_dir.mkdir(parents=True, exist_ok=True)
                run_fit([
                    f"--data-path={data_path}",
                    f"--outdir={res_dir}",
                    f"--model={model}",
                    f"--prior=galaxies-only",
                    f"--sigma-sys=4.0",
                    f"--steps={fit_steps}",
                    f"--nwalkers={fit_nwalkers}",
                    f"--seed={seed}",
                    f"--backend-hdf5={chain_path}",
                    f"--thin-by={thin}",
                ])
            else:
                raise FileNotFoundError(
                    f"chain.h5 not found at {chain_path}. Run fits with --backend-hdf5 {chain_path} first."
                )

        if not summ_path.exists():
            raise FileNotFoundError(f"global_summary.yaml not found at {summ_path}")

        summ = yaml.safe_load(open(summ_path, encoding="utf-8"))
        H0_si = float(summ.get("H0_si", H0_SI))
        like_kind = str(summ.get("likelihood", "gauss"))
        t_dof = float(summ.get("t_dof", 8.0))
        sigma_sys = float(summ.get("sigma_sys_median", 4.0))

        from emcee.backends import HDFBackend
        backend = HDFBackend(str(chain_path))
        n_steps = backend.iteration
        burn = int(n_steps * burn_frac)
        chain = backend.get_chain(discard=burn, flat=True, thin=thin)
        n_samples = chain.shape[0]

        L = _layout(model, G, sigma_sys_learn=True, a0_fixed=None)

        log_lik_list: List[np.ndarray] = []
        for s in range(n_samples):
            theta = chain[s]
            ll = log_likelihood_per_galaxy(
                theta, galaxies, sigma_sys, H0_si, L,
                c0=10.0, c_slope=-0.1, a0_fixed=None,
                like_kind=like_kind, t_dof=t_dof
            )
            log_lik_list.append(ll)
        log_lik_mat = np.array(log_lik_list)  # (S, N_total)

        # Build InferenceData with shape (chain, draw, obs) = (1, S, N)
        idata = az.from_dict(log_likelihood={"y": log_lik_mat[None, :, :]})
        loo_res = az.loo(idata, pointwise=True)

        elpd_loo = float(loo_res.loo)
        p_loo = float(loo_res.p_loo)
        looic = float(loo_res.looic)
        pareto_k = np.asarray(loo_res.pareto_k)

        rows.append({
            "model": model,
            "elpd_loo": elpd_loo,
            "p_loo": p_loo,
            "looic": looic,
            "n_data": N_total,
        })

        if pareto_k.size != N_total:
            raise RuntimeError(f"pareto_k size {pareto_k.size} != N_total={N_total}")
        frac_good = float(np.mean(pareto_k < 0.7))
        print(f"[loo] {model}: pareto_k<0.7 fraction = {frac_good:.3f}")

        for idx, k in enumerate(pareto_k):
            pareto_rows.append(dict(
                model=model,
                obs_index=int(idx),
                galaxy=obs_gal[idx],
                radius=obs_r[idx],
                pareto_k=float(k),
            ))

    # LOO table (model-level)
    df = pd.DataFrame(rows)
    if len(df):
        looic_min = df["looic"].min()
        df["delta_looic"] = df["looic"] - looic_min
        df["rank"] = df["looic"].rank().astype(int)
        df = df.sort_values("looic").reset_index(drop=True)
    df.to_csv(out_path / "loo_table.csv", index=False)

    # pareto_k diagnostics
    df_k = pd.DataFrame(pareto_rows)
    df_k.to_csv(out_path / "pareto_k.csv", index=False)

    if not df_k.empty:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        for model in df_k["model"].unique():
            kk = df_k.loc[df_k["model"] == model, "pareto_k"].values
            ax.hist(kk, bins=20, alpha=0.5, label=model, range=(0.0, max(1.0, float(kk.max()))))
        ax.set_xlabel("Pareto k")
        ax.set_ylabel("count")
        ax.set_title("Pareto k diagnostics (PSIS-LOO)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_path / "pareto_k_hist.png")
        plt.close(fig)

    # elpd difference plot (relative to best LOOIC)
    if len(df) and len(df) > 1:
        best_idx = df["looic"].idxmin()
        best = df.loc[best_idx]
        df["delta_elpd"] = df["elpd_loo"] - float(best["elpd_loo"])

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3.5), dpi=150)
        x = np.arange(len(df))
        ax.axhline(0.0, color="k", linewidth=1.0)
        ax.bar(x, df["delta_elpd"].values, color="steelblue", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(df["model"].values, rotation=45, ha="right")
        ax.set_ylabel(r"$\Delta \mathrm{elpd}_{\mathrm{LOO}}$ vs best")
        ax.set_title("ELPD-LOO differences")
        fig.tight_layout()
        fig.savefig(out_path / "elpd_difference_plot.png")
        plt.close(fig)

    return {"loo_table": str(out_path / "loo_table.csv"), "df": df}


# ---- 追加到檔尾：z-space zero-parameter tests ----

def _cH0_SI_from_summary(H0_si: float) -> float:
    val = float(linear_term_kms2(1.0, np.asarray([1.0]), H0_si=H0_si)[0])
    return val * (KM**2) / KPC

def _a0_SI_from_summary(eps_fit: float, H0_si: float) -> float:
    return float(eps_fit) * _cH0_SI_from_summary(H0_si)

def _nu_q_of_z(z: np.ndarray, q: float) -> np.ndarray:
    zz = np.clip(np.asarray(z, dtype=float), 1e-12, np.inf)
    return 0.5 + np.sqrt(0.25 + np.power(zz, -float(q)))

def _f_q(z: np.ndarray, q: float) -> np.ndarray:
    return np.asarray(z, float) * (_nu_q_of_z(z, q) - 1.0)

def _build_y_eps_eff_over_epsden(g: GalaxyData,
                                 vfun, vbar2: np.ndarray,
                                 eps_den: float,
                                 H0_si: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_kpc = g.r_kpc
    r_m = np.maximum(r_kpc * KPC, 1e-30)
    vbar2_kms2 = np.asarray(vbar2, float)
    gN = (vbar2_kms2 * (KM**2)) / r_m
    denom = np.maximum(linear_term_kms2(1.0, r_kpc, H0_si=H0_si), 1e-30)
    y = (g.v_obs**2 - vbar2_kms2) / denom
    y /= float(eps_den)
    sig_v = np.asarray(g.v_err, float)
    sigma_y = (2.0 * np.abs(g.v_obs) * sig_v) / (denom * float(eps_den))
    return gN, y, sigma_y

def z_profile(results_dir: str,
              data_path: str,
              nbins: int = 24,
              min_per_bin: int = 20,
              eps_norm: str = "cos",             # "cos" or "fit"
              epsilon_cos: Optional[float] = None,
              omega_lambda: Optional[float] = None,
              out_prefix: str = "z_profile",
              z_quantile_clip: Tuple[float, float] = (0.01, 0.99),
              do_theory: bool = True) -> Tuple[pd.DataFrame, Path]:
    """
    零自由度堆疊：x=z=g_N/a0，y=ε_eff/ε_den。若 model=ptq-screen 且有 q_median，疊上 y_th(z)=S f_q(z)。
    產物：
      - {results_dir}/{out_prefix}_per_point.csv（含每點 z, y, sigma_y）
      - {results_dir}/{out_prefix}_binned.csv（z 中位數統計）
      - {results_dir}/{out_prefix}.png
      - {results_dir}/{out_prefix}_summary.json（覆蓋率/指標）
    """
    res = Path(results_dir)
    summ = yaml.safe_load(open(res / "global_summary.yaml"))
    model = str(summ["model"])
    H0_si = float(summ.get("H0_si", H0_SI))
    eps_fit = float(summ.get("epsilon_median", np.nan))
    q_med = summ.get("q_median", None)
    sig_med = float(summ.get("sigma_sys_median", 0.0))  # 僅在建 C 時可能用到

    # 分母 ε_den
    if eps_norm == "fit":
        eps_den = float(eps_fit)
    else:
        eps_den = _eps_cos_from_args(epsilon_cos, omega_lambda)
    if not (np.isfinite(eps_den) and eps_den > 0):
        raise ValueError("Invalid eps_den for z_profile.")

    # a0（SI）
    a0_SI = _a0_SI_from_summary(eps_fit, H0_si)

    # 準備星系資料
    per = pd.read_csv(res / "per_galaxy_summary.csv").set_index("galaxy")
    gdict = load_tidy_sparc(data_path)

    rows = []
    for name, g in sorted(gdict.items()):
        if name not in per.index:
            continue
        vfun, vbar2 = _make_vfun(model, H0_si, per.loc[name], g,
                                 summ.get("epsilon_median"), summ.get("q_median"), summ.get("a0_median"))
        v_mod = vfun(g.r_kpc)
        # 用完整 C 僅為了保守濾出離群 σ；實際 σy 用測量誤差線性化（避免 D/i 關聯）
        try:
            C = build_covariance(v_mod, g.r_kpc, g.v_err, g.D_Mpc, g.D_err_Mpc,
                                 g.i_deg, g.i_err_deg, vfun, sigma_sys_kms=sig_med)
            diag_ok = np.sqrt(np.clip(np.diag(C), 1e-30, np.inf)) < 1e5
        except Exception:
            diag_ok = np.ones_like(g.r_kpc, dtype=bool)

        gN, y, sigma_y = _build_y_eps_eff_over_epsden(g, vfun, vbar2, eps_den, H0_si)
        z = gN / np.maximum(a0_SI, 1e-30)

        m = np.isfinite(z) & np.isfinite(y) & np.isfinite(sigma_y) & diag_ok
        for zi, yi, si in zip(z[m], y[m], sigma_y[m]):
            rows.append(dict(galaxy=name, z=float(zi), y=float(yi), sigma_y=float(si)))

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No valid points for z_profile.")
    per_csv = res / f"{out_prefix}_per_point.csv"
    df.to_csv(per_csv, index=False)

    # 分箱（依 z 的分位裁切，避免極端值）
    z_lo, z_hi = np.nanquantile(df["z"], z_quantile_clip[0]), np.nanquantile(df["z"], z_quantile_clip[1])
    mkeep = (df["z"] >= z_lo) & (df["z"] <= z_hi)
    dfx = df.loc[mkeep].copy()
    edges = np.linspace(float(z_lo), float(z_hi), nbins + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    q16, q50, q84, npts = [], [], [], []
    for i in range(nbins):
        m = (dfx["z"] >= edges[i]) & (dfx["z"] < edges[i + 1])
        vals = dfx.loc[m, "y"].values
        if vals.size < min_per_bin:
            q16.append(np.nan); q50.append(np.nan); q84.append(np.nan); npts.append(int(vals.size))
        else:
            q16.append(float(np.nanpercentile(vals, 16)))
            q50.append(float(np.nanpercentile(vals, 50)))
            q84.append(float(np.nanpercentile(vals, 84)))
            npts.append(int(vals.size))
    binned = pd.DataFrame(dict(z_mid=mids, q16=q16, q50=q50, q84=q84, n=npts))
    bin_csv = res / f"{out_prefix}_binned.csv"
    binned.to_csv(bin_csv, index=False)

    # 覆蓋率（若能計算理論曲線）
    cov68 = cov95 = np.nan
    yth = None
    S = float(eps_fit / eps_den) if np.isfinite(eps_fit) else np.nan
    if do_theory and (model == "ptq-screen") and (q_med is not None) and np.isfinite(S):
        q_use = float(q_med)
        # 理論：yth(z)=S * f_q(z)
        yth = S * _f_q(df["z"].values, q_use)
        # 用 σy 計覆蓋
        dy = np.abs(df["y"].values - yth)
        m = np.isfinite(dy) & np.isfinite(df["sigma_y"].values)
        if m.any():
            k68, k95 = 1.0, 1.96
            cov68 = float((dy[m] <= k68 * df["sigma_y"].values[m]).mean())
            cov95 = float((dy[m] <= k95 * df["sigma_y"].values[m]).mean())

    # 畫圖
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.8, 4.1), dpi=150)
    ax.fill_between(binned["z_mid"], binned["q16"], binned["q84"], alpha=0.25, label="stacked 16–84%")
    ax.plot(binned["z_mid"], binned["q50"], linewidth=1.6, label="stacked median")
    if yth is not None:
        # 在同樣的 z 範圍畫理論曲線
        zgrid = np.linspace(float(z_lo), float(z_hi), 400)
        ax.plot(zgrid, S * _f_q(zgrid, float(q_med)), linestyle="--", linewidth=1.3,
                label=rf"zero-param $S f_q(z)$ (q={float(q_med):.2f})")
    ax.set_xlabel(r"$z=g_N/a_0$")
    ylab_den = r"\varepsilon_{\rm cos}" if eps_norm == "cos" else r"\varepsilon_{\rm fit}"
    ax.set_ylabel(rf"$\varepsilon_{{\rm eff}}(r)/{ylab_den}$")
    ax.grid(True, alpha=0.25); ax.legend()
    out_png = res / f"{out_prefix}.png"
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)

    summ = dict(
        model=model, eps_norm=eps_norm, eps_den=float(eps_den),
        epsilon_fit=float(eps_fit), H0_si=float(H0_si),
        a0_SI=float(a0_SI),
        nbins=int(nbins), min_per_bin=int(min_per_bin),
        z_clip=[float(z_lo), float(z_hi)],
        coverage68=float(cov68) if np.isfinite(cov68) else None,
        coverage95=float(cov95) if np.isfinite(cov95) else None,
        per_point_csv=str(per_csv), binned_csv=str(bin_csv), png=str(out_png)
    )
    with open(res / f"{out_prefix}_summary.json", "w") as f:
        json.dump(summ, f, indent=2)
    return binned, out_png


def z_per_galaxy(results_dir: str,
                 data_path: str,
                 frac_vmax: float = 0.9,
                 y_source: str = "obs-debias",
                 rstar_from: str = "obs",
                 eps_norm: str = "cos",
                 epsilon_cos: Optional[float] = None,
                 omega_lambda: Optional[float] = None,
                 nsamp: int = 300,
                 interpolate_rstar: bool = True,
                 out_prefix: str = "z_gal") -> Dict[str, Any]:
    """
    每星系在 r* 的單點：x=z(r*)，y=ε_eff(r*)/ε_den；r* 可用內插。
    會輸出 per-galaxy CSV 與散點圖；若 model=ptq-screen 且有 q_median，疊上 y_th(z)=S f_q(z)。
    """
    assert y_source in ("model", "obs", "obs-debias")
    assert rstar_from in ("model", "obs")
    assert eps_norm in ("fit", "cos")

    res = Path(results_dir)
    summ = yaml.safe_load(open(res / "global_summary.yaml"))
    model = str(summ["model"])
    H0_si = float(summ.get("H0_si", H0_SI))
    eps_fit = float(summ.get("epsilon_median", np.nan))
    q_med = summ.get("q_median", None)
    sig_med = float(summ.get("sigma_sys_median", 0.0))

    # 分母 ε_den 與 a0
    eps_den = float(eps_fit) if eps_norm == "fit" else _eps_cos_from_args(epsilon_cos, omega_lambda)
    if not (np.isfinite(eps_den) and eps_den > 0):
        raise ValueError("Invalid eps_den for z_gal.")
    a0_SI = _a0_SI_from_summary(eps_fit, H0_si)

    per = pd.read_csv(res / "per_galaxy_summary.csv").set_index("galaxy")
    gdict = load_tidy_sparc(data_path)
    rng = np.random.default_rng(1234)

    rows = []
    for name, g in sorted(gdict.items()):
        if name not in per.index:
            continue
        vfun, vbar2 = _make_vfun(model, H0_si, per.loc[name], g,
                                 summ.get("epsilon_median"), summ.get("q_median"), summ.get("a0_median"))
        r = g.r_kpc
        if len(r) < 3:
            continue
        v_obs = g.v_obs
        v_mod = vfun(r)
        v_ref = v_mod if rstar_from == "model" else v_obs

        # r*（可內插）
        if interpolate_rstar:
            r_star = _rstar_linear_interp(r, v_ref, frac_vmax)
        else:
            vmax = float(np.nanmax(v_ref)); idxs = np.where(v_ref >= frac_vmax * vmax)[0]
            i_star = int(idxs[0]) if len(idxs) > 0 else int(np.nanargmax(v_ref))
            i_star = max(0, min(i_star, len(r) - 1))
            r_star = float(r[i_star])

        # 最近點作為誤差參考
        i_near = int(np.nanargmin(np.abs(r - r_star)))
        vbar2_i = float(vbar2[i_near])

        # y(r*)
        denom = float(linear_term_kms2(1.0, np.asarray([r_star]), H0_si=H0_si)[0])
        if y_source == "model":
            v0 = float(np.interp(r_star, r, v_mod))
            eps_eff_med = (v0**2 - vbar2_i) / denom
            eps_lo = eps_hi = np.nan
            sig_y = np.nan
        elif y_source == "obs":
            C_full = build_covariance(
                v_mod, r, g.v_err, g.D_Mpc, g.D_err_Mpc,
                g.i_deg, g.i_err_deg, vfun, sigma_sys_kms=sig_med
            )
            try:
                Vs = rng.multivariate_normal(mean=v_obs, cov=C_full, size=nsamp)
                v_star = _interp_rows_at_scalar(r, Vs, r_star)
                eps_s = (v_star**2 - vbar2_i) / denom
                eps_eff_med = float(np.nanpercentile(eps_s, 50))
                eps_lo = float(np.nanpercentile(eps_s, 16))
                eps_hi = float(np.nanpercentile(eps_s, 84))
            except np.linalg.LinAlgError:
                sig_i = float(np.sqrt(np.clip(np.diag(C_full)[i_near], 1e-30, np.inf)))
                v0 = float(np.interp(r_star, r, v_obs))
                eps_eff_med = (v0**2 - vbar2_i) / denom
                eps_lo = eps_eff_med - (2 * v0 * sig_i) / denom
                eps_hi = eps_eff_med + (2 * v0 * sig_i) / denom
            # 近似 σy（僅用於覆蓋率統計）
            v0 = float(np.interp(r_star, r, v_obs))
            sig_v = float(g.v_err[i_near])
            sig_y = (2.0 * abs(v0) * sig_v) / (denom * float(eps_den))
        else:  # obs-debias
            sig2_meas = float(g.v_err[i_near] ** 2)
            v0 = float(np.interp(r_star, r, v_obs))
            eps_eff_med = ((v0**2 - sig2_meas) - vbar2_i) / denom
            try:
                C_meas = np.diag(np.asarray(g.v_err, float) ** 2)
                Vs = rng.multivariate_normal(mean=v_obs, cov=C_meas, size=nsamp)
                v_star = _interp_rows_at_scalar(r, Vs, r_star)
                eps_s = ((v_star**2 - sig2_meas) - vbar2_i) / denom
                eps_lo = float(np.nanpercentile(eps_s, 16))
                eps_hi = float(np.nanpercentile(eps_s, 84))
            except np.linalg.LinAlgError:
                sig_i = float(np.sqrt(max(sig2_meas, 1e-30)))
                base = max(v0**2 - sig2_meas, 0.0) ** 0.5
                eps_lo = eps_eff_med - (2 * base * sig_i) / denom
                eps_hi = eps_eff_med + (2 * base * sig_i) / denom
            # 近似 σy（僅用於覆蓋率）
            sig_v = float(g.v_err[i_near])
            base = max(v0**2 - sig2_meas, 0.0) ** 0.5
            sig_y = (2.0 * base * sig_v) / (denom * float(eps_den)) if base > 0 else np.nan

        # x = z(r*) = g_N(r*)/a0
        r_m = max(r_star * KPC, 1e-30)
        gN_star = (vbar2_i * (KM**2)) / r_m
        z_star = gN_star / max(a0_SI, 1e-30)

        rows.append(dict(
            galaxy=name,
            r_star_kpc=r_star,
            z_star=z_star,
            y_eps_over_epsden=eps_eff_med / float(eps_den),
            y_p16=(eps_lo / float(eps_den)) if np.isfinite(eps_lo) else np.nan,
            y_p84=(eps_hi / float(eps_den)) if np.isfinite(eps_hi) else np.nan,
            sigma_y=float(sig_y) if np.isfinite(sig_y) else np.nan
        ))

    df = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["z_star", "y_eps_over_epsden"])
    out_csv = res / f"{out_prefix}_per_galaxy.csv"
    df.to_csv(out_csv, index=False)

    # 可能的理論曲線與覆蓋率
    S = float(eps_fit / eps_den) if np.isfinite(eps_fit) else np.nan
    cov68 = cov95 = None
    yth = None
    if (model == "ptq-screen") and (q_med is not None) and np.isfinite(S):
        zvals = df["z_star"].values
        yth = S * _f_q(zvals, float(q_med))
        if "sigma_y" in df.columns:
            dy = np.abs(df["y_eps_over_epsden"].values - yth)
            m = np.isfinite(dy) & np.isfinite(df["sigma_y"].values)
            if m.any():
                k68, k95 = 1.0, 1.96
                cov68 = float((dy[m] <= k68 * df["sigma_y"].values[m]).mean())
                cov95 = float((dy[m] <= k95 * df["sigma_y"].values[m]).mean())

    # 圖
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.4, 4.0), dpi=150)
    if len(df) > 0:
        ax.scatter(df["z_star"], df["y_eps_over_epsden"], s=20, alpha=0.8, label="galaxies @ r*")
    if yth is not None:
        zgrid = np.linspace(max(1e-4, float(df["z_star"].min())), float(df["z_star"].max()), 400)
        ax.plot(zgrid, S * _f_q(zgrid, float(q_med)), linestyle="--", linewidth=1.3,
                label=rf"zero-param $S f_q(z)$ (q={float(q_med):.2f})")
    ax.set_xlabel(r"$z_\ast=g_N(r_\ast)/a_0$")
    ylab_den = r"\varepsilon_{\rm cos}" if eps_norm == "cos" else r"\varepsilon_{\rm fit}"
    ax.set_ylabel(rf"$\varepsilon_{{\rm eff}}(r_\ast)/{ylab_den}$")
    ax.grid(True, alpha=0.25); ax.legend()
    out_png = res / f"{out_prefix}.png"
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)

    out = dict(
        N=int(len(df)),
        eps_norm=eps_norm,
        eps_den=float(eps_den),
        epsilon_fit=float(eps_fit),
        S=float(S) if np.isfinite(S) else None,
        coverage68=cov68, coverage95=cov95,
        csv=str(out_csv), png=str(out_png)
    )
    with open(res / f"{out_prefix}_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    return out
