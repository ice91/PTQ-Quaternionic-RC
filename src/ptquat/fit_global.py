from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import emcee
import yaml

from .data import load_tidy_sparc, GalaxyData
from .models import model_v_kms, model_v_kms_split
from .likelihood import build_covariance, gaussian_loglike
from .constants import H0_SI

# ---------- Helpers to unpack theta given options ----------

def _unpack_theta(theta: np.ndarray, G: int,
                  split_ml: bool,
                  epsilon_fixed: Optional[float],
                  sigma_sys_learn: bool) -> Tuple[float, np.ndarray, Optional[np.ndarray], float]:
    """
    回傳 (eps, Ups_or_Ud, Ub_or_None, sigma_sys_kms)
    - 若 split_ml=False: Ups_or_Ud shape=(G,), Ub_or_None=None
    - 若 split_ml=True : Ups_or_Ud=U_disk(G,), Ub_or_None=U_bulge(G,)
    - 若 epsilon_fixed is not None: 用該值，theta 不含 epsilon
    - 若 sigma_sys_learn=True: theta 末端為 ln_sigma，sigma=exp(ln_sigma)
    """
    idx = 0
    if epsilon_fixed is None:
        eps = float(theta[idx]); idx += 1
    else:
        eps = float(epsilon_fixed)

    if split_ml:
        U_disk = theta[idx:idx+G]; idx += G
        U_bulge = theta[idx:idx+G]; idx += G
    else:
        U = theta[idx:idx+G]; idx += G
        U_disk, U_bulge = U, None

    if sigma_sys_learn:
        ln_sigma = float(theta[idx]); idx += 1
        sigma_sys_kms = float(np.exp(ln_sigma))
    else:
        sigma_sys_kms = np.nan  # 占位，外部會覆寫為固定值

    return eps, np.asarray(U_disk), (None if U_bulge is None else np.asarray(U_bulge)), sigma_sys_kms


# ------------------------ Priors ------------------------ #

def log_prior(theta: np.ndarray,
              galaxies: List[GalaxyData],
              prior_kind: str,
              split_ml: bool,
              epsilon_fixed: Optional[float],
              sigma_sys_learn: bool,
              sigma_sys_prior_scale: float,
              sigma_sys_fixed: float) -> float:
    G = len(galaxies)
    eps, U1, U2, sigma_learn = _unpack_theta(theta, G, split_ml, epsilon_fixed, sigma_sys_learn)
    lp = 0.0

    # epsilon prior（只有在抽樣時才套用）
    if epsilon_fixed is None:
        if prior_kind == "galaxies-only":
            if not (0.0 < eps < 4.0):
                return -np.inf
        elif prior_kind == "planck-anchored":
            mu, sig = 1.47, 0.05
            if not (0.0 < eps < 4.0):
                return -np.inf
            lp += -0.5 * ((eps - mu) / sig)**2 - np.log(sig*np.sqrt(2*np.pi))
        else:
            return -np.inf

    # Upsilon priors（拆分則兩組都套用）
    if split_ml:
        Ups_all = np.concatenate([U1, U2])
    else:
        Ups_all = U1
    for u in Ups_all:
        if not (0.05 <= u <= 1.5):
            return -np.inf
        mu_u, sig_u = 0.5, 0.1
        lp += -0.5 * ((u - mu_u) / sig_u)**2 - np.log(sig_u*np.sqrt(2*np.pi))

    # sigma_sys prior（若學習）
    if sigma_sys_learn:
        s0 = float(sigma_sys_prior_scale)
        sigma = sigma_learn
        if not np.isfinite(sigma) or sigma <= 0:
            return -np.inf
        # half-normal(s0) on sigma; parameterized by ln_sigma with Jacobian
        # log p = ln(sigma) - 0.5 * (sigma/s0)^2 - ln(s0*sqrt(2π))
        lp += np.log(sigma) - 0.5 * (sigma / s0)**2 - np.log(s0*np.sqrt(2*np.pi))

    return lp


# -------------------- Log-likelihood -------------------- #

def log_likelihood(theta: np.ndarray,
                   galaxies: List[GalaxyData],
                   sigma_sys_kms: float,
                   H0_si: float,
                   split_ml: bool,
                   epsilon_fixed: Optional[float],
                   sigma_sys_learn: bool) -> float:
    G = len(galaxies)
    eps, U1, U2, sigma_learn = _unpack_theta(theta, G, split_ml, epsilon_fixed, sigma_sys_learn)

    if sigma_sys_learn:
        sigma_use = sigma_learn
    else:
        sigma_use = sigma_sys_kms

    total = 0.0
    for gi, g in enumerate(galaxies):
        if split_ml:
            Ud, Ub = float(U1[gi]), float(U2[gi])
            def vfun(rk):
                return model_v_kms_split(Ud, Ub, eps, rk, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_si)
        else:
            U = float(U1[gi])
            def vfun(rk):
                return model_v_kms(U, eps, rk, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_si)

        v_mod = vfun(g.r_kpc)
        Cg = build_covariance(v_mod, g.r_kpc, g.v_err, g.D_Mpc, g.D_err_Mpc,
                              g.i_deg, g.i_err_deg, vfun, sigma_sys_kms=sigma_use)
        total += gaussian_loglike(g.v_obs, v_mod, Cg)
    return total


def log_posterior(theta: np.ndarray,
                  galaxies: List[GalaxyData],
                  sigma_sys_kms: float,
                  H0_si: float,
                  prior_kind: str,
                  split_ml: bool,
                  epsilon_fixed: Optional[float],
                  sigma_sys_learn: bool,
                  sigma_sys_prior_scale: float) -> float:
    lp = log_prior(theta, galaxies, prior_kind, split_ml, epsilon_fixed,
                   sigma_sys_learn, sigma_sys_prior_scale, sigma_sys_kms)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, galaxies, sigma_sys_kms, H0_si,
                        split_ml, epsilon_fixed, sigma_sys_learn)
    return lp + ll


# ------------------------- CLI -------------------------- #
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Global-epsilon SPARC fit with full likelihood AIC/BIC and optional sigma_sys learning.")
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--prior", choices=["galaxies-only","planck-anchored"], default="galaxies-only")
    ap.add_argument("--sigma-sys", type=float, default=4.0)
    ap.add_argument("--H0-kms-mpc", type=float, default=None)
    ap.add_argument("--nwalkers", type=str, default="4x")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)
    # 新增選項
    ap.add_argument("--epsilon-fixed", type=float, default=None)
    ap.add_argument("--split-ml", action="store_true")
    ap.add_argument("--sigma-sys-learn", action="store_true")
    ap.add_argument("--sigma-sys-prior-scale", type=float, default=5.0)
    # HDF5 backend
    ap.add_argument("--backend-hdf5", type=str, default=None)
    ap.add_argument("--thin-by", type=int, default=10)
    ap.add_argument("--resume", action="store_true")
    # add model
    ap.add_argument("--model", choices=["ptq","mond","baryon"], default="ptq",
                help="ptq: epsilon global; mond: a0 global; baryon: epsilon fixed 0")
    ap.add_argument("--a0-si", type=float, default=None,
                    help="Fix MOND a0 in SI (m/s^2); if omitted, a0 is sampled globally.")
    ap.add_argument("--a0-prior-uniform", type=str, default="5e-11,2e-10",
                    help="If sampling a0: uniform prior [lo,hi] in SI. Default 5e-11–2e-10.")
    #nfw1p
    ap.add_argument("--model", choices=["ptq","baryon","mond","nfw1p"], default="ptq")
    ap.add_argument("--c0", type=float, default=10.0, help="c(M) normalization at 1e12 Msun")
    ap.add_argument("--c-slope", type=float, default=-0.1, help="c(M) slope beta")
    ap.add_argument("--logM200-range", type=str, default="9,13", help="Uniform prior bounds for log10 M200 [Msun]")
# （若你還沒加 MOND，可暫時忽略 mond 相關 CLI 選項）

    return ap.parse_args(argv)


def run(argv=None):
    args = parse_args(argv)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    gdict = load_tidy_sparc(args.data_path)
    galaxies = [gdict[k] for k in sorted(gdict.keys())]
    G = len(galaxies)
    print(f"Loaded {G} galaxies.")

    H0_si = H0_SI
    if args.H0_kms_mpc is not None:
        from .constants import KM, MPC
        H0_si = (args.H0_kms_mpc * KM) / MPC

    # ---- parameter dimension ----
    n_eps = 0 if args.epsilon_fixed is not None else 1
    n_up  = (2*G) if args.split_ml else G
    n_sig = 1 if args.sigma_sys_learn else 0
    k = n_eps + n_up + n_sig

    if args.nwalkers.endswith("x"):
        mult = int(args.nwalkers[:-1])
        nwalkers = max(2*k, mult*k)
    else:
        nwalkers = int(args.nwalkers)

    # ---- initial positions ----
    rng = np.random.default_rng(args.seed)
    p0 = np.empty((nwalkers, k)); idx = 0
    if n_eps == 1:
        if args.prior == "planck-anchored":
            p0[:, idx] = rng.normal(1.47, 0.05, size=nwalkers)
        else:
            p0[:, idx] = rng.uniform(0.2, 2.5, size=nwalkers)
        idx += 1
    # Ups
    if args.split_ml:
        p0[:, idx:idx+G] = np.clip(rng.normal(0.5, 0.05, size=(nwalkers, G)), 0.1, 1.2); idx += G
        p0[:, idx:idx+G] = np.clip(rng.normal(0.5, 0.05, size=(nwalkers, G)), 0.1, 1.2); idx += G
    else:
        p0[:, idx:idx+G] = np.clip(rng.normal(0.5, 0.05, size=(nwalkers, G)), 0.1, 1.2); idx += G
    # ln sigma
    if args.sigma_sys_learn:
        sig0 = float(args.sigma_sys) if args.sigma_sys > 0 else 4.0
        p0[:, idx] = np.log(sig0); idx += 1

    # ---- backend (optional) ----
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

    sampler = emcee.EnsembleSampler(
        nwalkers, k, log_posterior,
        args=(galaxies, args.sigma_sys, H0_si, args.prior,
              args.split_ml, args.epsilon_fixed,
              args.sigma_sys_learn, args.sigma_sys_prior_scale),
        backend=backend
    )
    print(f"Running emcee with nwalkers={nwalkers}, steps={args.steps}, k={k}")
    sampler.run_mcmc(p0_init, args.steps, progress=True)

    # ---- Postprocess ----
    burn = args.steps // 3
    thin = max(int(args.thin_by), 1)
    chain = (backend.get_chain(discard=burn, flat=True, thin=thin)
             if backend is not None else
             sampler.get_chain(discard=burn, flat=True, thin=thin))

    # 取樣拆解
    def split_chain_row(row):
        eps, U1, U2, sig = _unpack_theta(row, G, args.split_ml, args.epsilon_fixed, args.sigma_sys_learn)
        return eps, U1, U2, (sig if args.sigma_sys_learn else args.sigma_sys)

    # 取 ε 與（學習）σ_sys 的後驗摘要
    eps_samples = []
    sig_samples = []
    for row in chain:
        eps, _, _, sig = split_chain_row(row)
        eps_samples.append(eps)
        sig_samples.append(sig)
    eps_samples = np.asarray(eps_samples)
    sig_samples = np.asarray(sig_samples)

    def med_lo_hi(x):
        q = np.percentile(x, [16,50,84])
        return float(q[1]), float(q[0]), float(q[2])

    eps_med, eps_lo, eps_hi = med_lo_hi(eps_samples)
    sig_med, sig_lo, sig_hi = med_lo_hi(sig_samples)

    # 用後驗中位數評估模型；同時計算 chi2 與 完整 logL（含 logdet）
    rows = []
    N_total = 0
    chi2_tot = 0.0
    logL_tot = 0.0

    # 構造「中位數」參數向量
    if args.epsilon_fixed is None:
        # 取中位數 ε
        pass
    eps_use = eps_med
    # Ups 中位數
    if args.split_ml:
        # 需要從 chain 中取每個 U 的中位數
        Ups_all = chain[:, (0 if args.epsilon_fixed is not None else 1):(0 if args.epsilon_fixed is not None else 1)+2*G]
        Ud_med = np.median(Ups_all[:, :G], axis=0)
        Ub_med = np.median(Ups_all[:, G:],  axis=0)
    else:
        Ups_all = chain[:, (0 if args.epsilon_fixed is not None else 1):(0 if args.epsilon_fixed is not None else 1)+G]
        U_med = np.median(Ups_all, axis=0)

    sigma_use = sig_med

    for gi, g in enumerate(galaxies):
        if args.split_ml:
            Ud, Ub = float(Ud_med[gi]), float(Ub_med[gi])
            def vfun(rk):
                return model_v_kms_split(Ud, Ub, eps_use, rk, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_si)
        else:
            U = float(U_med[gi])
            def vfun(rk):
                return model_v_kms(U, eps_use, rk, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_si)

        v_mod = vfun(g.r_kpc)
        Cg = build_covariance(v_mod, g.r_kpc, g.v_err, g.D_Mpc, g.D_err_Mpc,
                              g.i_deg, g.i_err_deg, vfun, sigma_sys_kms=sigma_use)

        # chi2_g（二次項）
        r = g.v_obs - v_mod
        alpha = np.linalg.solve(Cg, r)
        chi2_g = float(r @ alpha)
        nu_g = max(len(g.r_kpc) - (2 if args.split_ml else 1), 1)
        chi2_red_g = chi2_g / nu_g

        # 完整 logL_g（含 logdet）
        ll_g = gaussian_loglike(g.v_obs, v_mod, Cg)

        N_total += len(g.r_kpc)
        chi2_tot += chi2_g
        logL_tot += ll_g

        rows.append(dict(
            galaxy=g.name,
            n=len(g.r_kpc),
            Upsilon_med=(float(Ud) if args.split_ml else float(U)),
            Upsilon_bulge_med=(float(Ub) if args.split_ml else None),
            chi2=chi2_g,
            chi2_red=chi2_red_g,
            loglike=ll_g
        ))

        from .plotting import plot_rc
        plot_rc(g.name, g.r_kpc, g.v_obs, g.v_err, v_mod, outdir / f"plot_{g.name}.png")

    # AIC/BIC（兩種版本都報）
    k_eff = n_eps + n_up + (1 if args.sigma_sys_learn else 0)
    AIC_quad = chi2_tot + 2 * k_eff
    BIC_quad = chi2_tot + k_eff * np.log(N_total)

    AIC_full = -2.0 * logL_tot + 2 * k_eff
    BIC_full = -2.0 * logL_tot + k_eff * np.log(N_total)

    # 保存 per-galaxy CSV
    df_pg = pd.DataFrame(rows).sort_values("galaxy")
    df_pg.to_csv(outdir / "per_galaxy_summary.csv", index=False)

    # 後驗樣本（thinned）
    np.savez_compressed(outdir / "posterior_samples.npz",
                        epsilon=eps_samples, sigma_sys=sig_samples,
                        steps=args.steps, burn_in=burn, thin=thin, nwalkers=nwalkers)

    # 接受率（用 sampler 屬性，不要向 backend 拿）
    acc = getattr(sampler, "acceptance_fraction", None)
    acc_mean = float(np.mean(acc)) if acc is not None else None
    acc_median = float(np.median(acc)) if acc is not None else None

    summary = dict(
        n_galaxies=G, N_total=N_total,
        prior=args.prior,
        split_ml=bool(args.split_ml),
        epsilon_fixed=args.epsilon_fixed,
        sigma_sys_learn=bool(args.sigma_sys_learn),
        sigma_sys_prior_scale=float(args.sigma_sys_prior_scale),
        sigma_sys_fixed=(None if args.sigma_sys_learn else float(args.sigma_sys)),
        sigma_sys_median=sig_med, sigma_sys_p16=sig_lo, sigma_sys_p84=sig_hi,
        H0_si=H0_si,
        epsilon_median=eps_med, epsilon_p16=eps_lo, epsilon_p84=eps_hi,
        # quad-only 指標（傳統）
        chi2_total=float(chi2_tot), AIC_quad=float(AIC_quad), BIC_quad=float(BIC_quad),
        # full likelihood 指標（建議報）
        logL_total_full=float(logL_tot), AIC_full=float(AIC_full), BIC_full=float(BIC_full),
        k_parameters=int(k_eff),
        steps=args.steps, nwalkers=nwalkers, burn_in=burn, thin=thin,
        acceptance_mean=acc_mean, acceptance_median=acc_median,
        backend_hdf5=str(args.backend_hdf5) if args.backend_hdf5 else None,
        resumed=bool(args.backend_hdf5 and args.resume)
    )
    with open(outdir / "global_summary.yaml", "w") as f:
        yaml.safe_dump(summary, f)

    print("\n=== Global summary ===")
    print(json.dumps(summary, indent=2))
    print(f"\nSaved: {outdir/'per_galaxy_summary.csv'}, {outdir/'global_summary.yaml'}, "
          f"{outdir/'posterior_samples.npz'} and plots/*.png")

if __name__ == "__main__":
    run()
