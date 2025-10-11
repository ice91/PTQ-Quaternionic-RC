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
from typing import Any
from .models import linear_term_kms2  # (cH0) r in (km/s)^2


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
def ppc_check(results_dir: str, data_path: str, out_prefix: str = "ppc") -> dict:
    """
    Posterior(-like) predictive coverage check using median params from results_dir.
    Uses Student-t critical values if the fitted run used t-likelihood.
    """
    import os, json
    import numpy as np
    import pandas as pd
    import yaml
    from pathlib import Path
    from .data import load_tidy_sparc
    from .models import (
        model_v_ptq, model_v_ptq_nu, model_v_ptq_screen, model_v_ptq_split,
        model_v_baryon, model_v_mond, model_v_nfw1p
    )
    from .constants import H0_SI
    from .likelihood import build_covariance

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

# ====== Kappa checks: per-galaxy & radius-resolved (append to end of experiments.py) ======


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
                     y_source: str = "model",  # "model" | "obs" | "obs-debias"
                     out_prefix: str = "kappa_gal") -> Dict[str, Any]:
    """
    檢核A：以特徵半徑 r* 做 y 對 x 的迴歸。
      x = kappa_pred = eta * Rd / r*
      y = eps_eff/eps_cos；依 y_source 決定來源：
        - "model":    y = (v_mod^2 - v_bar^2)/((cH0) r*) / eps_cos   ← 論文主圖建議
        - "obs":      y = (v_obs^2 - v_bar^2)/((cH0) r*) / eps_cos    （含噪平方偏置）
        - "obs-debias": 同上但扣掉點位變異 sigma_v^2（近似去偏）
    迴歸 y = a x + b；圖上同時標示 k = a*eta 方便閱讀。
    """
    assert y_source in ("model", "obs", "obs-debias")
    res = yaml.safe_load(open(Path(results_dir)/"global_summary.yaml"))
    model = res["model"]
    H0_si = float(res.get("H0_si", H0_SI))
    sig_med = float(res["sigma_sys_median"])
    eps = res.get("epsilon_median")
    q   = res.get("q_median")
    a0  = res.get("a0_median")
    per = pd.read_csv(Path(results_dir)/"per_galaxy_summary.csv").set_index("galaxy")

    eps_cos = _eps_cos_from_args(epsilon_cos, omega_lambda)
    gdict = load_tidy_sparc(data_path)
    rng = np.random.default_rng(1234)

    rows = []
    for name, g in sorted(gdict.items()):
        if name not in per.index:
            continue
        vfun, vbar2 = _make_vfun(model, H0_si, per.loc[name], g, eps, q, a0)

        v_obs = g.v_obs
        r     = g.r_kpc
        if len(r) < 3:
            continue
        vmax = float(np.nanmax(v_obs))
        idxs = np.where(v_obs >= frac_vmax * vmax)[0]
        i_star = int(idxs[0]) if len(idxs)>0 else int(np.floor(0.8*(len(r)-1)))
        i_star = max(0, min(i_star, len(r)-1))
        r_star = float(r[i_star])

        # 協方差與各定量
        v_mod = vfun(r)
        C = build_covariance(v_mod, r, g.v_err, g.D_Mpc, g.D_err_Mpc,
                             g.i_deg, g.i_err_deg, vfun, sigma_sys_kms=sig_med)
        sig_i2 = float(np.clip(np.diag(C)[i_star], 1e-30, np.inf))
        denom  = float(linear_term_kms2(1.0, np.asarray([r_star]), H0_si=H0_si)[0])
        vbar2_i = float(vbar2[i_star])

        # y 的來源
        if y_source == "model":
            # 模型驗證：直接用模型預測，不做 MC；不受噪音平方偏置影響
            eps_med = (v_mod[i_star]**2 - vbar2_i) / denom
            eps_lo = np.nan; eps_hi = np.nan

        elif y_source == "obs":
            # 完全資料驅動：以觀測為中心抽樣（會含 +sigma^2 偏置）
            try:
                Vs = rng.multivariate_normal(mean=v_obs, cov=C, size=nsamp)
                eps_samp = (Vs[:, i_star]**2 - vbar2_i) / denom
                eps_med = float(np.percentile(eps_samp, 50))
                eps_lo  = float(np.percentile(eps_samp, 16))
                eps_hi  = float(np.percentile(eps_samp, 84))
            except np.linalg.LinAlgError:
                sig_i = float(np.sqrt(sig_i2))
                eps_med = (v_obs[i_star]**2 - vbar2_i) / denom
                eps_lo  = eps_med - (2*v_obs[i_star]*sig_i)/denom
                eps_hi  = eps_med + (2*v_obs[i_star]*sig_i)/denom

        else:  # "obs-debias"
            # 觀測為中心，但扣掉單點變異近似去偏：E[v_obs^2]≈v_true^2+sigma^2
            # 中位數用點估；區間用 MC 再對每次樣本扣掉對角變異近似
            try:
                Vs = rng.multivariate_normal(mean=v_obs, cov=C, size=nsamp)
                # 對每次樣本近似扣一個固定的 sig_i2
                eps_samp = ((Vs[:, i_star]**2 - sig_i2) - vbar2_i) / denom
                eps_med = float(np.percentile(eps_samp, 50))
                eps_lo  = float(np.percentile(eps_samp, 16))
                eps_hi  = float(np.percentile(eps_samp, 84))
            except np.linalg.LinAlgError:
                eps_med = ((v_obs[i_star]**2 - sig_i2) - vbar2_i) / denom
                # 粗略區間：仍用導數法但以 debias 後的中心
                sig_i = float(np.sqrt(sig_i2))
                base  = max(v_obs[i_star]**2 - sig_i2, 0.0)**0.5
                eps_lo  = eps_med - (2*base*sig_i)/denom
                eps_hi  = eps_med + (2*base*sig_i)/denom

        Rd = _get_Rd_kpc_safe(g)
        kappa_pred = float(eta * Rd / r_star)
        rows.append(dict(
            galaxy=name,
            r_star_kpc=r_star,
            Rd_kpc=Rd,
            eps_eff_med=eps_med,
            eps_eff_p16=eps_lo,
            eps_eff_p84=eps_hi,
            eps_eff_over_epscos=eps_med/eps_cos,
            kappa_pred=kappa_pred
        ))

    df = pd.DataFrame(rows).dropna()
    out_csv = Path(results_dir)/f"{out_prefix}_per_galaxy.csv"
    df.to_csv(out_csv, index=False)

    # y = a x + b
    if len(df) >= 2:
        a, b = np.polyfit(df["kappa_pred"].values, df["eps_eff_over_epscos"].values, 1)
        yhat = a*df["kappa_pred"].values + b
        r2 = float(1.0 - np.sum((df["eps_eff_over_epscos"].values - yhat)**2) /
                          np.sum((df["eps_eff_over_epscos"].values - df["eps_eff_over_epscos"].mean())**2))
    else:
        a = b = r2 = np.nan

    # 畫圖（同時標註 k = a*eta）
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=150)
    ax.scatter(df["kappa_pred"], df["eps_eff_over_epscos"], s=18, alpha=0.7, label="galaxies")
    xgrid = np.linspace(0.0, max(1.05*df["kappa_pred"].max(), 0.2), 200)
    ax.plot(xgrid, xgrid, linestyle="--", linewidth=1.2, label="y = x")
    if np.isfinite(a):
        k_est = a*eta
        ax.plot(xgrid, a*xgrid + b, linewidth=1.4,
                label=f"fit: y={a:.2f}x+{b:.2f}, R²={r2:.2f}, k≈{k_est:.2f}")
    ax.set_xlabel(r"$\kappa_{\rm pred}=\eta\,R_d/r_\ast$")
    ax.set_ylabel(r"$\varepsilon_{\rm eff}(r_\ast)/\varepsilon_{\rm cos}$")
    ax.legend()
    ax.grid(True, alpha=0.25)
    out_png = Path(results_dir)/f"{out_prefix}.png"
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)

    summ = dict(
        N=len(df),
        eps_cos=eps_cos, eta=eta, frac_vmax=frac_vmax,
        slope=a, intercept=b, R2=r2, k_est=a*eta,
        y_source=y_source,
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
                          out_prefix: str = "kappa_profile") -> Tuple[pd.DataFrame, Path]:
    """
    檢核 B：半徑解析的疊圖。
      y(r) = eps_eff(r)/eps_cos,  其中 eps_eff(r) = [v_obs^2 - v_bar^2]/[(cH0) r]
      x = r/Rd (預設；也可選 x=r_kpc)
    產出 per-point 與 binned 的 CSV 與圖。
    """
    res = yaml.safe_load(open(Path(results_dir)/"global_summary.yaml"))
    model = res["model"]
    H0_si = float(res.get("H0_si", H0_SI))
    sig_med = float(res["sigma_sys_median"])
    eps = res.get("epsilon_median")
    q   = res.get("q_median")
    a0  = res.get("a0_median")
    per = pd.read_csv(Path(results_dir)/"per_galaxy_summary.csv").set_index("galaxy")

    eps_cos = _eps_cos_from_args(epsilon_cos, omega_lambda)
    gdict = load_tidy_sparc(data_path)

    pts = []
    for name, g in sorted(gdict.items()):
        if name not in per.index:
            continue
        vfun, vbar2 = _make_vfun(model, H0_si, per.loc[name], g, eps, q, a0)
        v_mod = vfun(g.r_kpc)  # only used for C; eps_eff 用觀測值
        C = build_covariance(v_mod, g.r_kpc, g.v_err, g.D_Mpc, g.D_err_Mpc,
                             g.i_deg, g.i_err_deg, vfun, sigma_sys_kms=sig_med)
        # 單點標準差（用於濾除極端不確定點）
        sig_pt = np.sqrt(np.clip(np.diag(C), 1e-30, np.inf))

        denom = linear_term_kms2(1.0, g.r_kpc, H0_si=H0_si)  # (cH0) r
        eps_eff = (g.v_obs**2 - vbar2) / np.maximum(denom, 1e-30)
        Rd = _get_Rd_kpc_safe(g)
        x = (g.r_kpc / Rd) if x_kind == "r_over_Rd" else g.r_kpc
        y = eps_eff / eps_cos

        for xi, yi, ri, si in zip(x, y, g.r_kpc, sig_pt):
            if np.isfinite(xi) and np.isfinite(yi) and si < 100.0:  # 粗略剔除病態點
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

    # 畫圖
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.6, 4.0), dpi=150)
    ax.plot(xgrid, ypred, linestyle="--", linewidth=1.2, label=pred_label)
    ax.fill_between(binned["x_mid"], binned["q16"], binned["q84"], alpha=0.25, label="stacked 16–84%")
    ax.plot(binned["x_mid"], binned["q50"], linewidth=1.6, label="stacked median")
    ax.set_xlabel("x = r/Rd" if x_kind=="r_over_Rd" else "r [kpc]")
    ax.set_ylabel(r"$\varepsilon_{\rm eff}(r)/\varepsilon_{\rm cos}$")
    ax.grid(True, alpha=0.25)
    ax.legend()
    out_png = Path(results_dir)/f"{out_prefix}.png"
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

    return binned, out_png
