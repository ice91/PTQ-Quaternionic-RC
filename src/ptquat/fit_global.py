from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import emcee
import yaml

from .data import load_tidy_sparc, GalaxyData
from .likelihood import build_covariance, gaussian_loglike
from .constants import H0_SI
from .models import (
    model_v_ptq, model_v_ptq_split, model_v_baryon, model_v_mond, model_v_nfw1p
)

# ------------------------- 參數版型 -------------------------

def _layout(model: str, G: int, sigma_sys_learn: bool,
            a0_fixed: Optional[float]) -> dict:
    """
    回傳每種模型下 theta 的結構（順序與切片）。
    """
    L = {"model": model, "G": G, "sigma_learn": sigma_sys_learn, "a0_fixed": a0_fixed}
    s = 0
    if model == "ptq":
        # [eps] + [U_g]*G + [ln_sigma]?
        L["i_eps"] = s; s += 1
        L["sl_U"]  = slice(s, s+G); s += G
    elif model == "ptq-split":
        # [eps] + [Ud_g]*G + [Ub_g]*G + [ln_sigma]?
        L["i_eps"] = s; s += 1
        L["sl_Ud"] = slice(s, s+G); s += G
        L["sl_Ub"] = slice(s, s+G); s += G
    elif model == "baryon":
        # [U_g]*G + [ln_sigma]?
        L["sl_U"] = slice(s, s+G); s += G
    elif model == "mond":
        # [log10a0]? + [U_g]*G + [ln_sigma]?
        if a0_fixed is None:
            L["i_log10a0"] = s; s += 1
        L["sl_U"] = slice(s, s+G); s += G
    elif model == "nfw1p":
        # [U_g]*G + [log10M200_g]*G + [ln_sigma]?
        L["sl_U"]   = slice(s, s+G); s += G
        L["sl_lM"]  = slice(s, s+G); s += G
    else:
        raise ValueError("Unknown model: "+model)

    if sigma_sys_learn:
        L["i_lnsig"] = s; s += 1
    L["k"] = s
    return L


def _unpack(theta: np.ndarray, L: dict) -> dict:
    """
    將 theta 依版型解析；回傳 dict.
    """
    out = {"model": L["model"]}
    if "i_eps" in L:        out["eps"] = float(theta[L["i_eps"]])
    if "sl_U" in L:         out["U"]   = np.asarray(theta[L["sl_U"]])
    if "sl_Ud" in L:        out["Ud"]  = np.asarray(theta[L["sl_Ud"]])
    if "sl_Ub" in L:        out["Ub"]  = np.asarray(theta[L["sl_Ub"]])
    if "i_log10a0" in L:    out["log10a0"] = float(theta[L["i_log10a0"]])
    if "sl_lM" in L:        out["lM"]  = np.asarray(theta[L["sl_lM"]])
    if "i_lnsig" in L:      out["sigma"] = float(np.exp(theta[L["i_lnsig"]]))
    return out


# ---------------------------- Priors ----------------------------

def log_prior(theta: np.ndarray,
              galaxies: List[GalaxyData],
              prior_kind: str,
              L: dict,
              logM_range: Tuple[float,float],
              a0_range: Tuple[float,float]) -> float:
    """
    針對每種模型套用合適的先驗。
    - PTQ: epsilon ~ flat(0,4) 或 N(1.47,0.05^2)（planck-anchored）
    - Upsilon: 每星系 Gaussian N(0.5,0.1^2) 且截在 [0.05,1.5]
    - MOND: 若 a0 需抽樣，log10 a0 在 [log10(lo), log10(hi)] 上一律常數
    - NFW-1p: log10 M200_g 在 [lo,hi] 上一律常數
    - sigma_sys（若學習）: 對 σ 使用半常態 prior，並以 lnσ 參數化加入 Jacobian
    """
    G = len(galaxies)
    pars = _unpack(theta, L)
    lp = 0.0
    m  = L["model"]

    # epsilon prior
    if m in ("ptq", "ptq-split"):
        eps = pars["eps"]
        if prior_kind == "galaxies-only":
            if not (0.0 < eps < 4.0):
                return -np.inf
        elif prior_kind == "planck-anchored":
            if not (0.0 < eps < 4.0):
                return -np.inf
            mu, sig = 1.47, 0.05
            lp += -0.5*((eps-mu)/sig)**2 - np.log(sig*np.sqrt(2*np.pi))
        else:
            return -np.inf

    # Upsilon priors
    if m in ("ptq","baryon","mond","nfw1p"):
        U = pars["U"]
        for u in U:
            if not (0.05 <= u <= 1.5):
                return -np.inf
            mu_u, sig_u = 0.5, 0.1
            lp += -0.5*((u-mu_u)/sig_u)**2 - np.log(sig_u*np.sqrt(2*np.pi))
    elif m == "ptq-split":
        Ud, Ub = pars["Ud"], pars["Ub"]
        for u in np.concatenate([Ud, Ub]):
            if not (0.05 <= u <= 1.5):
                return -np.inf
            mu_u, sig_u = 0.5, 0.1
            lp += -0.5*((u-mu_u)/sig_u)**2 - np.log(sig_u*np.sqrt(2*np.pi))

    # MOND a0 prior (if sampled)
    if m == "mond" and ("log10a0" in pars):
        lo, hi = a0_range
        loglo, loghi = np.log10(lo), np.log10(hi)
        if not (loglo <= pars["log10a0"] <= loghi):
            return -np.inf
        # uniform in log10(a0) -> 常數項即可

    # NFW-1p halo mass prior
    if m == "nfw1p":
        lo, hi = logM_range
        lM = pars["lM"]
        if np.any((lM < lo) | (lM > hi)):
            return -np.inf
        # uniform in log10 M200 -> 常數

    # sigma_sys prior（若學習）
    if "sigma" in pars:
        s = pars["sigma"]
        if not (np.isfinite(s) and s > 0):
            return -np.inf
        s0 = 5.0  # 弱先驗 scale，可視需要做為 CLI 參數（這裡採固定）
        lp += np.log(s) - 0.5*(s/s0)**2 - np.log(s0*np.sqrt(2*np.pi))

    return lp


# ------------------------- Likelihood -------------------------

def log_likelihood(theta: np.ndarray,
                   galaxies: List[GalaxyData],
                   sigma_sys_fixed: float,
                   H0_si: float,
                   L: dict,
                   c0: float,
                   c_slope: float,
                   a0_fixed: Optional[float]) -> float:
    pars = _unpack(theta, L)
    m = L["model"]

    sigma_use = pars.get("sigma", sigma_sys_fixed)
    total = 0.0

    for gi, g in enumerate(galaxies):
        if m == "ptq":
            U   = float(pars["U"][gi])
            eps = float(pars["eps"])
            def vfun(rk):
                return model_v_ptq(U, eps, rk, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_si)

        elif m == "ptq-split":
            Ud  = float(pars["Ud"][gi])
            Ub  = float(pars["Ub"][gi])
            eps = float(pars["eps"])
            def vfun(rk):
                return model_v_ptq_split(Ud, Ub, eps, rk, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_si)

        elif m == "baryon":
            U = float(pars["U"][gi])
            def vfun(rk):
                return model_v_baryon(U, rk, g.v_disk, g.v_bulge, g.v_gas)

        elif m == "mond":
            U = float(pars["U"][gi])
            a0 = a0_fixed if a0_fixed is not None else (10.0**pars["log10a0"])
            def vfun(rk):
                return model_v_mond(U, a0, rk, g.v_disk, g.v_bulge, g.v_gas)

        elif m == "nfw1p":
            U  = float(pars["U"][gi])
            lM = float(pars["lM"][gi])
            def vfun(rk):
                return model_v_nfw1p(U, lM, rk, g.v_disk, g.v_bulge, g.v_gas,
                                     H0_si=H0_si, c0=c0, c_slope=c_slope)

        else:
            raise ValueError("Unknown model: "+m)

        v_mod = vfun(g.r_kpc)
        Cg = build_covariance(v_mod, g.r_kpc, g.v_err, g.D_Mpc, g.D_err_Mpc,
                              g.i_deg, g.i_err_deg, vfun, sigma_sys_kms=sigma_use)
        total += gaussian_loglike(g.v_obs, v_mod, Cg)
    return total


def log_posterior(theta: np.ndarray,
                  galaxies: List[GalaxyData],
                  sigma_sys_fixed: float,
                  H0_si: float,
                  prior_kind: str,
                  L: dict,
                  c0: float,
                  c_slope: float,
                  logM_range: Tuple[float,float],
                  a0_range: Tuple[float,float],
                  a0_fixed: Optional[float]) -> float:
    lp = log_prior(theta, galaxies, prior_kind, L, logM_range, a0_range)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, galaxies, sigma_sys_fixed, H0_si, L, c0, c_slope, a0_fixed)
    return lp + ll


# ----------------------------- CLI -----------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Global fits for PTQ / Baryon / MOND / NFW-1p with full likelihood and AIC/BIC."
    )
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--model", choices=["ptq","ptq-split","baryon","mond","nfw1p"], default="ptq")

    # cosmology / noise
    ap.add_argument("--sigma-sys", type=float, default=4.0, help="Velocity floor [km/s].")
    ap.add_argument("--H0-kms-mpc", type=float, default=None, help="Override H0 [km/s/Mpc].")

    # priors
    ap.add_argument("--prior", choices=["galaxies-only","planck-anchored"], default="galaxies-only")
    ap.add_argument("--logM200-range", type=str, default="9,13", help="Uniform prior bounds for log10 M200 [Msun] (nfw1p).")
    ap.add_argument("--a0-range", type=str, default="5e-11,2e-10", help="Uniform prior on a0 [m/s^2] (mond).")
    ap.add_argument("--a0-si", type=float, default=None, help="Fix MOND a0 (if given).")

    # NFW c–M relation
    ap.add_argument("--c0", type=float, default=10.0, help="c(M) normalization at 1e12 Msun (nfw1p).")
    ap.add_argument("--c-slope", type=float, default=-0.1, help="c(M) slope beta (nfw1p).")

    # sampler
    ap.add_argument("--nwalkers", type=str, default="4x")
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=42)

    # backend
    ap.add_argument("--backend-hdf5", type=str, default=None)
    ap.add_argument("--thin-by", type=int, default=10)
    ap.add_argument("--resume", action="store_true")
    return ap.parse_args(argv)


# ---------------------------- 主程式 ----------------------------

def run(argv=None):
    args = parse_args(argv)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # data
    gdict = load_tidy_sparc(args.data_path)
    galaxies = [gdict[k] for k in sorted(gdict.keys())]
    G = len(galaxies)
    print(f"Loaded {G} galaxies.")

    # H0
    H0_si = H0_SI
    if args.H0_kms_mpc is not None:
        from .constants import KM, MPC
        H0_si = (args.H0_kms_mpc * KM) / MPC

    # model layout
    a0_range = tuple(map(float, args.a0_range.split(",")))
    logM_range = tuple(map(float, args.logM200_range.split(",")))
    L = _layout(args.model, G, sigma_sys_learn=True, a0_fixed=args.a0_si)
    k = L["k"]

    # walkers
    if args.nwalkers.endswith("x"):
        mult = int(args.nwalkers[:-1])
        nwalkers = max(2*k, mult*k)
    else:
        nwalkers = int(args.nwalkers)

    # init positions
    rng = np.random.default_rng(args.seed)
    p0 = np.empty((nwalkers, k))

    s = 0
    if args.model in ("ptq","ptq-split"):
        # epsilon
        if args.prior == "planck-anchored":
            p0[:, s] = rng.normal(1.47, 0.05, size=nwalkers)
        else:
            p0[:, s] = rng.uniform(0.2, 2.5, size=nwalkers)
        s += 1

    if args.model == "ptq":
        p0[:, s:s+G] = np.clip(rng.normal(0.5, 0.05, size=(nwalkers, G)), 0.1, 1.2); s += G
    elif args.model == "ptq-split":
        p0[:, s:s+G] = np.clip(rng.normal(0.5, 0.06, size=(nwalkers, G)), 0.1, 1.5); s += G
        p0[:, s:s+G] = np.clip(rng.normal(0.5, 0.06, size=(nwalkers, G)), 0.1, 1.5); s += G
    elif args.model == "baryon":
        p0[:, s:s+G] = np.clip(rng.normal(0.5, 0.05, size=(nwalkers, G)), 0.1, 1.5); s += G
    elif args.model == "mond":
        if args.a0_si is None:
            loglo, loghi = np.log10(a0_range[0]), np.log10(a0_range[1])
            p0[:, s] = rng.uniform(loglo, loghi, size=nwalkers); s += 1
        p0[:, s:s+G] = np.clip(rng.normal(0.5, 0.05, size=(nwalkers, G)), 0.1, 1.5); s += G
    elif args.model == "nfw1p":
        p0[:, s:s+G] = np.clip(rng.normal(0.5, 0.05, size=(nwalkers, G)), 0.1, 1.5); s += G
        lo, hi = logM_range
        p0[:, s:s+G] = rng.uniform(lo, hi, size=(nwalkers, G)); s += G

    # ln sigma (always learn; we give weak prior in log_prior)
    sig0 = max(args.sigma_sys, 1e-3)
    p0[:, s] = np.log(sig0); s += 1
    assert s == k

    # backend
    backend = None
    p0_init = p0
    if args.backend_hdf5:
        from emcee.backends import HDFBackend
        backend_path = Path(args.backend_hdf5); backend_path.parent.mkdir(parents=True, exist_ok=True)
        backend = HDFBackend(str(backend_path))
        if args.resume and backend.iteration > 0:
            print(f"Resuming from backend {backend_path} at iteration={backend.iteration}")
            p0_init = None
        else:
            backend.reset(nwalkers, k)

    # sampler
    sampler = emcee.EnsembleSampler(
        nwalkers, k, log_posterior,
        args=(galaxies, args.sigma_sys, H0_si, args.prior, L,
              args.c0, args.c_slope, logM_range, a0_range, args.a0_si),
        backend=backend
    )
    print(f"Running emcee with nwalkers={nwalkers}, steps={args.steps}, k={k}")
    sampler.run_mcmc(p0_init, args.steps, progress=True)

    # -------- Summaries --------
    burn = args.steps // 3
    thin = max(int(args.thin_by), 1)
    chain = (backend.get_chain(discard=burn, flat=True, thin=thin)
             if backend is not None else
             sampler.get_chain(discard=burn, flat=True, thin=thin))

    # 中位數參數
    med = np.median(chain, axis=0)
    P = _unpack(med, L)

    # 後驗抽樣的 epsilon / a0 / sigma
    eps_stats = None
    a0_stats  = None
    if args.model in ("ptq","ptq-split"):
        eps_samples = (backend.get_chain(discard=burn, flat=True, thin=thin)[:, L["i_eps"]]
                       if backend is not None else
                       sampler.get_chain(discard=burn, flat=True, thin=thin)[:, L["i_eps"]])
        q = np.percentile(eps_samples, [16,50,84]); eps_stats = (float(q[1]), float(q[0]), float(q[2]))
    if args.model == "mond" and ("i_log10a0" in L):
        a0_samp = (backend.get_chain(discard=burn, flat=True, thin=thin)[:, L["i_log10a0"]]
                   if backend is not None else
                   sampler.get_chain(discard=burn, flat=True, thin=thin)[:, L["i_log10a0"]])
        a0_lin = 10.0**a0_samp
        q = np.percentile(a0_lin, [16,50,84]); a0_stats = (float(q[1]), float(q[0]), float(q[2]))

    sig_samp = (backend.get_chain(discard=burn, flat=True, thin=thin)[:, L["i_lnsig"]]
                if backend is not None else
                sampler.get_chain(discard=burn, flat=True, thin=thin)[:, L["i_lnsig"]])
    sig_lin = np.exp(sig_samp)
    q = np.percentile(sig_lin, [16,50,84]); sig_stats = (float(q[1]), float(q[0]), float(q[2]))

    # 用中位數參數評估模型並計算分星系與全域指標
    rows = []
    N_total = 0
    chi2_tot = 0.0
    logL_tot = 0.0

    # 取 per-galaxy 中位數 U / logM（若有）
    if args.model == "ptq":
        U_med = P["U"]
    elif args.model == "ptq-split":
        Ud_med, Ub_med = P["Ud"], P["Ub"]
    elif args.model == "baryon":
        U_med = P["U"]
    elif args.model == "mond":
        U_med = P["U"]
    elif args.model == "nfw1p":
        U_med  = P["U"]
        lM_med = P["lM"]

    sigma_use = float(P["sigma"])
    eps_use   = float(P["eps"]) if "eps" in P else None
    a0_use    = (args.a0_si if args.a0_si is not None
                 else (10.0**P["log10a0"] if "log10a0" in P else None))

    for gi, g in enumerate(galaxies):
        if args.model == "ptq":
            U = float(U_med[gi])
            def vfun(rk):
                return model_v_ptq(U, eps_use, rk, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_si)
        elif args.model == "ptq-split":
            Ud, Ub = float(Ud_med[gi]), float(Ub_med[gi])
            def vfun(rk):
                return model_v_ptq_split(Ud, Ub, eps_use, rk, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_si)
        elif args.model == "baryon":
            U = float(U_med[gi])
            def vfun(rk):
                return model_v_baryon(U, rk, g.v_disk, g.v_bulge, g.v_gas)
        elif args.model == "mond":
            U = float(U_med[gi])
            def vfun(rk):
                return model_v_mond(U, a0_use, rk, g.v_disk, g.v_bulge, g.v_gas)
        elif args.model == "nfw1p":
            U  = float(U_med[gi])
            lM = float(lM_med[gi])
            def vfun(rk):
                return model_v_nfw1p(U, lM, rk, g.v_disk, g.v_bulge, g.v_gas,
                                     H0_si=H0_si, c0=args.c0, c_slope=args.c_slope)
        else:
            raise ValueError

        v_mod = vfun(g.r_kpc)
        Cg = build_covariance(v_mod, g.r_kpc, g.v_err, g.D_Mpc, g.D_err_Mpc,
                              g.i_deg, g.i_err_deg, vfun, sigma_sys_kms=sigma_use)

        r = g.v_obs - v_mod
        alpha = np.linalg.solve(Cg, r)
        chi2_g = float(r @ alpha)
        nu_g = max(len(g.r_kpc) - (2 if args.model=="ptq-split" else 1), 1)
        chi2_red_g = chi2_g / nu_g
        ll_g = gaussian_loglike(g.v_obs, v_mod, Cg)

        N_total += len(g.r_kpc)
        chi2_tot += chi2_g
        logL_tot += ll_g

        rows.append(dict(
            galaxy=g.name,
            n=len(g.r_kpc),
            Upsilon_med=(float(Ud_med[gi]) if args.model=="ptq-split" else float(U_med[gi])),
            Upsilon_bulge_med=(float(Ub_med[gi]) if args.model=="ptq-split" else None),
            log10_M200_med=(float(lM_med[gi]) if args.model=="nfw1p" else None),
            chi2=chi2_g,
            chi2_red=chi2_red_g,
            loglike=ll_g
        ))

        from .plotting import plot_rc
        plot_rc(g.name, g.r_kpc, g.v_obs, g.v_err, v_mod, outdir / f"plot_{g.name}.png")

    # AIC/BIC（兩種版本）
    k_eff = L["k"]   # 已包含 ln sigma
    AIC_quad = chi2_tot + 2 * k_eff
    BIC_quad = chi2_tot + k_eff * np.log(N_total)
    AIC_full = -2.0 * logL_tot + 2 * k_eff
    BIC_full = -2.0 * logL_tot + k_eff * np.log(N_total)

    # 存檔
    df_pg = pd.DataFrame(rows).sort_values("galaxy")
    df_pg.to_csv(outdir / "per_galaxy_summary.csv", index=False)

    summary = dict(
        model=args.model,
        n_galaxies=G, N_total=N_total,
        prior=args.prior, H0_si=H0_si,
        sigma_sys_median=sig_stats[0], sigma_sys_p16=sig_stats[1], sigma_sys_p84=sig_stats[2],
        epsilon_median=(eps_stats[0] if eps_stats else None),
        epsilon_p16=(eps_stats[1] if eps_stats else None),
        epsilon_p84=(eps_stats[2] if eps_stats else None),
        a0_median=(a0_stats[0] if a0_stats else (args.a0_si if args.model=="mond" else None)),
        a0_p16=(a0_stats[1] if a0_stats else None),
        a0_p84=(a0_stats[2] if a0_stats else None),
        chi2_total=float(chi2_tot), logL_total_full=float(logL_tot),
        AIC_quad=float(AIC_quad), BIC_quad=float(BIC_quad),
        AIC_full=float(AIC_full), BIC_full=float(BIC_full),
        k_parameters=int(k_eff),
        steps=args.steps, nwalkers=nwalkers, burn_in=burn, thin=thin,
        backend_hdf5=str(args.backend_hdf5) if args.backend_hdf5 else None,
        resumed=bool(args.backend_hdf5 and args.resume)
    )
    with open(outdir / "global_summary.yaml", "w") as f:
        yaml.safe_dump(summary, f)

    print("\n=== Global summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved: {outdir/'per_galaxy_summary.csv'}, {outdir/'global_summary.yaml'} and plots/*.png")
