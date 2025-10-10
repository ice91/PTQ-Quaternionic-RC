from __future__ import annotations
import argparse, json, math
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import emcee
import yaml

from .data import load_tidy_sparc, GalaxyData
#from .likelihood import build_covariance, gaussian_loglike  # keep existing gaussian impl
from .likelihood import build_covariance, gaussian_loglike, student_t_loglike  # ← 加入 student_t_loglike
from .constants import H0_SI
from .models import (
    model_v_ptq, model_v_ptq_split, model_v_baryon, model_v_mond, model_v_nfw1p,
    model_v_ptq_nu, model_v_ptq_screen
)

# ---------- local Student-t loglike (multivariate) ----------
def student_t_loglike(y_obs: np.ndarray, y_mod: np.ndarray, C: np.ndarray, nu: float) -> float:
    """
    Multivariate Student-t log-likelihood with location=y_mod, scale=C, dof=nu.
    Density ~ Gamma((nu+d)/2)/(Gamma(nu/2)*(nu*pi)^{d/2}*|C|^{1/2}) * (1 + z2/nu)^{-(nu+d)/2}
    where z2 = (y-μ)^T C^{-1} (y-μ).
    """
    r = y_obs - y_mod
    d = r.size
    if nu <= 2.0:  # keep numerically safe & reasonable tails
        nu = 2.0001
    sign, logdet = np.linalg.slogdet(C)
    if not np.isfinite(logdet) or sign <= 0:
        return -np.inf
    alpha = np.linalg.solve(C, r)
    z2 = float(r @ alpha)
    # log constants
    from math import lgamma, log, pi
    c0 = lgamma(0.5*(nu + d)) - lgamma(0.5*nu) - 0.5*d*log(nu*pi) - 0.5*logdet
    return c0 - 0.5*(nu + d)*np.log1p(z2/nu)


# ------------------------- 參數版型 -------------------------

def _layout(model: str, G: int, sigma_sys_learn: bool,
            a0_fixed: Optional[float]) -> dict:
    L = {"model": model, "G": G, "sigma_learn": sigma_sys_learn, "a0_fixed": a0_fixed}
    s = 0
    if model in ("ptq", "ptq-nu"):
        L["i_eps"] = s; s += 1
        L["sl_U"]  = slice(s, s+G); s += G
    elif model == "ptq-screen":
        L["i_eps"] = s; s += 1
        L["i_lq"]  = s; s += 1
        L["sl_U"]  = slice(s, s+G); s += G
    elif model == "ptq-split":
        L["i_eps"] = s; s += 1
        L["sl_Ud"] = slice(s, s+G); s += G
        L["sl_Ub"] = slice(s, s+G); s += G
    elif model == "baryon":
        L["sl_U"] = slice(s, s+G); s += G
    elif model == "mond":
        if a0_fixed is None:
            L["i_log10a0"] = s; s += 1
        L["sl_U"] = slice(s, s+G); s += G
    elif model == "nfw1p":
        L["sl_U"]   = slice(s, s+G); s += G
        L["sl_lM"]  = slice(s, s+G); s += G
    else:
        raise ValueError("Unknown model: "+model)

    if sigma_sys_learn:
        L["i_lnsig"] = s; s += 1
    L["k"] = s
    return L


def _unpack(theta: np.ndarray, L: dict) -> dict:
    out = {"model": L["model"]}
    if "i_eps" in L:        out["eps"] = float(theta[L["i_eps"]])
    if "sl_U" in L:         out["U"]   = np.asarray(theta[L["sl_U"]])
    if "sl_Ud" in L:        out["Ud"]  = np.asarray(theta[L["sl_Ud"]])
    if "sl_Ub" in L:        out["Ub"]  = np.asarray(theta[L["sl_Ub"]])
    if "i_log10a0" in L:    out["log10a0"] = float(theta[L["i_log10a0"]])
    if "sl_lM" in L:        out["lM"]  = np.asarray(theta[L["sl_lM"]])
    if "i_lq" in L:         out["lq"]  = float(theta[L["i_lq"]])  # q = exp(lq)
    if "i_lnsig" in L:      out["sigma"] = float(np.exp(theta[L["i_lnsig"]]))
    return out


# ---------------------------- Priors ----------------------------

def log_prior(theta: np.ndarray,
              galaxies: List[GalaxyData],
              prior_kind: str,
              L: dict,
              logM_range: Tuple[float,float],
              a0_range: Tuple[float,float]) -> float:
    G = len(galaxies)
    pars = _unpack(theta, L)
    lp = 0.0
    m  = L["model"]

    # epsilon prior
    if m in ("ptq", "ptq-nu", "ptq-screen", "ptq-split"):
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

    # screen: prior on q (via lq)
    if m == "ptq-screen":
        lq = pars["lq"]
        q  = float(np.exp(lq))
        if not (0.25 <= q <= 8.0):
            return -np.inf
        mu_lq, sig_lq = 0.0, 0.35
        lp += -0.5*((lq-mu_lq)/sig_lq)**2 - np.log(sig_lq*np.sqrt(2*np.pi))

    # Upsilon priors
    if m in ("ptq","ptq-nu","ptq-screen","baryon","mond","nfw1p"):
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

    # NFW-1p halo mass prior
    if m == "nfw1p":
        lo, hi = logM_range
        lM = pars["lM"]
        if np.any((lM < lo) | (lM > hi)):
            return -np.inf

    # sigma_sys prior（若學習）
    if "sigma" in pars:
        s = pars["sigma"]
        if not (np.isfinite(s) and s > 0):
            return -np.inf
        s0 = 5.0
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
                   a0_fixed: Optional[float],
                   like_kind: str,
                   t_dof: float) -> float:
    pars = _unpack(theta, L)
    m = L["model"]
    sigma_use = pars.get("sigma", sigma_sys_fixed)
    total = 0.0

    # choose pointwise loglike
    def _ll(y_obs, y_mod, C):
        if like_kind == "t":
            return student_t_loglike(y_obs, y_mod, C, t_dof)
        else:
            return gaussian_loglike(y_obs, y_mod, C)

    for gi, g in enumerate(galaxies):
        if m == "ptq":
            U   = float(pars["U"][gi]); eps = float(pars["eps"])
            def vfun(rk): return model_v_ptq(U, eps, rk, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_si)
        elif m == "ptq-nu":
            U   = float(pars["U"][gi]); eps = float(pars["eps"])
            def vfun(rk): return model_v_ptq_nu(U, eps, rk, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_si)
        elif m == "ptq-screen":
            U   = float(pars["U"][gi]); eps = float(pars["eps"]); q = float(np.exp(pars["lq"]))
            def vfun(rk): return model_v_ptq_screen(U, eps, q, rk, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_si)
        elif m == "ptq-split":
            Ud  = float(pars["Ud"][gi]); Ub  = float(pars["Ub"][gi]); eps = float(pars["eps"])
            def vfun(rk): return model_v_ptq_split(Ud, Ub, eps, rk, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_si)
        elif m == "baryon":
            U = float(pars["U"][gi])
            def vfun(rk): return model_v_baryon(U, rk, g.v_disk, g.v_bulge, g.v_gas)
        elif m == "mond":
            U = float(pars["U"][gi]); a0 = a0_fixed if a0_fixed is not None else (10.0**pars["log10a0"])
            def vfun(rk): return model_v_mond(U, a0, rk, g.v_disk, g.v_bulge, g.v_gas)
        elif m == "nfw1p":
            U  = float(pars["U"][gi]); lM = float(pars["lM"][gi])
            def vfun(rk): return model_v_nfw1p(U, lM, rk, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_si, c0=c0, c_slope=c_slope)
        else:
            raise ValueError("Unknown model: "+m)

        v_mod = vfun(g.r_kpc)
        Cg = build_covariance(v_mod, g.r_kpc, g.v_err, g.D_Mpc, g.D_err_Mpc,
                              g.i_deg, g.i_err_deg, vfun, sigma_sys_kms=sigma_use)
        total += _ll(g.v_obs, v_mod, Cg)
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
                  a0_fixed: Optional[float],
                  like_kind: str,
                  t_dof: float) -> float:
    lp = log_prior(theta, galaxies, prior_kind, L, logM_range, a0_range)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, galaxies, sigma_sys_fixed, H0_si, L, c0, c_slope, a0_fixed, like_kind, t_dof)
    return lp + ll


# ----------------------------- CLI -----------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Global fits for PTQ / PTQ-ν / PTQ-screen / Baryon / MOND / NFW-1p with full likelihood and AIC/BIC."
    )
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--model", choices=["ptq","ptq-nu","ptq-screen","ptq-split","baryon","mond","nfw1p"], default="ptq")

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

    # likelihood (NEW)
    ap.add_argument("--likelihood", choices=["gauss","t"], default="gauss", help="Gaussian or Student-t likelihood.")
    ap.add_argument("--t-dof", type=float, default=8.0, help="DoF ν for Student-t (ν>2).")

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
    if args.model in ("ptq","ptq-nu","ptq-screen","ptq-split"):
        if args.prior == "planck-anchored":
            p0[:, s] = rng.normal(1.47, 0.05, size=nwalkers)
        else:
            p0[:, s] = rng.uniform(0.2, 2.5, size=nwalkers)
        s += 1

    if args.model == "ptq-screen":
        p0[:, s] = rng.normal(0.0, 0.2, size=nwalkers); s += 1

    if args.model in ("ptq","ptq-nu","ptq-screen"):
        p0[:, s:s+G] = np.clip(rng.normal(0.5, 0.05, size=(nwalkers, G)), 0.1, 1.5); s += G
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
              args.c0, args.c_slope, logM_range, a0_range, args.a0_si,
              args.likelihood, float(args.t_dof)),
        backend=backend
    )
    print(f"Running emcee with nwalkers={nwalkers}, steps={args.steps}, k={k}, likelihood={args.likelihood}, t_dof={args.t_dof}")
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

    # 後驗抽樣的 epsilon / a0 / sigma / q
    eps_stats = None; a0_stats  = None; q_stats   = None
    if args.model in ("ptq","ptq-nu","ptq-screen","ptq-split"):
        col = L["i_eps"]
        eps_samples = (backend.get_chain(discard=burn, flat=True, thin=thin)[:, col]
                       if backend is not None else
                       sampler.get_chain(discard=burn, flat=True, thin=thin)[:, col])
        qnt = np.percentile(eps_samples, [16,50,84]); eps_stats = (float(qnt[1]), float(qnt[0]), float(qnt[2]))
    if args.model == "ptq-screen":
        col = L["i_lq"]
        lq_samples = (backend.get_chain(discard=burn, flat=True, thin=thin)[:, col]
                      if backend is not None else
                      sampler.get_chain(discard=burn, flat=True, thin=thin)[:, col])
        q_lin = np.exp(lq_samples)
        qnt = np.percentile(q_lin, [16,50,84]); q_stats = (float(qnt[1]), float(qnt[0]), float(qnt[2]))
    if args.model == "mond" and ("i_log10a0" in L):
        a0_samp = (backend.get_chain(discard=burn, flat=True, thin=thin)[:, L["i_log10a0"]]
                   if backend is not None else
                   sampler.get_chain(discard=burn, flat=True, thin=thin)[:, L["i_log10a0"]])
        a0_lin = 10.0**a0_samp
        qnt = np.percentile(a0_lin, [16,50,84]); a0_stats = (float(qnt[1]), float(qnt[0]), float(qnt[2]))

    sig_samp = (backend.get_chain(discard=burn, flat=True, thin=thin)[:, L["i_lnsig"]]
                if backend is not None else
                sampler.get_chain(discard=burn, flat=True, thin=thin)[:, L["i_lnsig"]])
    sig_lin = np.exp(sig_samp)
    qnt = np.percentile(sig_lin, [16,50,84]); sig_stats = (float(qnt[1]), float(qnt[0]), float(qnt[2]))

    # 用中位數參數評估模型並計算分星系與全域指標
    rows = []
    N_total = 0
    chi2_tot = 0.0
    logL_tot = 0.0

    if args.model in ("ptq","ptq-nu","ptq-screen","baryon","mond","nfw1p"):
        U_med = P["U"]
    elif args.model == "ptq-split":
        Ud_med, Ub_med = P["Ud"], P["Ub"]
    if args.model == "nfw1p":
        lM_med = P["lM"]

    sigma_use = float(P["sigma"])
    eps_use   = float(P["eps"]) if "eps" in P else None
    a0_use    = (args.a0_si if args.a0_si is not None else (10.0**P["log10a0"] if "log10a0" in P else None))
    q_use     = float(np.exp(P["lq"])) if "lq" in P else None

    def _ll(y_obs, y_mod, C):
        if args.likelihood == "t":
            return student_t_loglike(y_obs, y_mod, C, float(args.t_dof))
        else:
            return gaussian_loglike(y_obs, y_mod, C)

    for gi, g in enumerate(galaxies):
        if args.model == "ptq":
            U = float(U_med[gi])
            def vfun(rk): return model_v_ptq(U, eps_use, rk, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_si)
        elif args.model == "ptq-nu":
            U = float(U_med[gi])
            def vfun(rk): return model_v_ptq_nu(U, eps_use, rk, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_si)
        elif args.model == "ptq-screen":
            U = float(U_med[gi])
            def vfun(rk): return model_v_ptq_screen(U, eps_use, q_use, rk, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_si)
        elif args.model == "ptq-split":
            Ud, Ub = float(Ud_med[gi]), float(Ub_med[gi])
            def vfun(rk): return model_v_ptq_split(Ud, Ub, eps_use, rk, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_SI)
        elif args.model == "baryon":
            U = float(U_med[gi])
            def vfun(rk): return model_v_baryon(U, rk, g.v_disk, g.v_bulge, g.v_gas)
        elif args.model == "mond":
            U = float(U_med[gi])
            def vfun(rk): return model_v_mond(U, a0_use, rk, g.v_disk, g.v_bulge, g.v_gas)
        elif args.model == "nfw1p":
            U  = float(U_med[gi]); lM = float(lM_med[gi])
            def vfun(rk): return model_v_nfw1p(U, lM, rk, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_si, c0=args.c0, c_slope=args.c_slope)
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
        ll_g = _ll(g.v_obs, v_mod, Cg)

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

    # AIC/BIC
    k_eff = L["k"]
    AIC_quad = chi2_tot + 2 * k_eff
    BIC_quad = chi2_tot + k_eff * np.log(N_total)
    AIC_full = -2.0 * logL_tot + 2 * k_eff
    BIC_full = -2.0 * logL_tot + k_eff * np.log(N_total)

    # 存檔
    df_pg = pd.DataFrame(rows).sort_values("galaxy")
    df_pg.to_csv(outdir / "per_galaxy_summary.csv", index=False)

    summary = dict(
        model=args.model,
        likelihood=args.likelihood, t_dof=float(args.t_dof),
        n_galaxies=G, N_total=N_total,
        prior=args.prior, H0_si=H0_si,
        sigma_sys_median=float(sig_stats[0]), sigma_sys_p16=float(sig_stats[1]), sigma_sys_p84=float(sig_stats[2]),
        epsilon_median=(float(eps_stats[0]) if eps_stats else None),
        epsilon_p16=(float(eps_stats[1]) if eps_stats else None),
        epsilon_p84=(float(eps_stats[2]) if eps_stats else None),
        q_median=(float(q_stats[0]) if q_stats else None),
        q_p16=(float(q_stats[1]) if q_stats else None),
        q_p84=(float(q_stats[2]) if q_stats else None),
        a0_median=(float(a0_stats[0]) if a0_stats else (float(args.a0_si) if args.model=="mond" and args.a0_si is not None else None)),
        a0_p16=(float(a0_stats[1]) if a0_stats else None),
        a0_p84=(float(a0_stats[2]) if a0_stats else None),
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
