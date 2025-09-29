# src/ptquat/fit_global.py
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
import emcee
from tqdm import tqdm
import yaml

from .data import load_tidy_sparc, GalaxyData
from .models import model_v_kms
from .likelihood import build_covariance, gaussian_loglike
from .constants import H0_SI


# ------------------------ Priors ------------------------ #
def log_prior(theta: np.ndarray,
              galaxies: List[GalaxyData],
              prior_kind: str) -> float:
    """
    theta = [epsilon, Upsilon_1, ..., Upsilon_G]
    prior_kind: 'galaxies-only' (flat on epsilon) OR 'planck-anchored' (Gaussian on epsilon).
    Upsilon_* Gaussian broad prior: N(0.5, 0.1^2) truncated to [0.05, 1.5].
    """
    eps = theta[0]
    Ups = theta[1:]

    # epsilon prior
    if prior_kind == "galaxies-only":
        if not (0.0 < eps < 4.0):
            return -np.inf
        lp = 0.0
    elif prior_kind == "planck-anchored":
        mu, sig = 1.47, 0.05
        lp = -0.5 * ((eps - mu) / sig)**2 - np.log(sig*np.sqrt(2*np.pi))
        if not (0.0 < eps < 4.0):
            return -np.inf
    else:
        raise ValueError("Unknown prior_kind")

    # Upsilon_* priors
    for u in Ups:
        if not (0.05 <= u <= 1.5):
            return -np.inf
        mu_u, sig_u = 0.5, 0.1
        lp += -0.5 * ((u - mu_u) / sig_u)**2 - np.log(sig_u*np.sqrt(2*np.pi))

    return lp


# -------------------- Log-likelihood -------------------- #
def log_likelihood(theta: np.ndarray,
                   galaxies: List[GalaxyData],
                   sigma_sys_kms: float,
                   H0_si: float) -> float:
    eps = float(theta[0])
    Ups = theta[1:]

    total = 0.0
    for g, U in zip(galaxies, Ups):
        # v_model and a closure for grad computation
        def vfun(rk):
            # Interpolate baryons to new radii if needed (we assume same radii here)
            return model_v_kms(U, eps, rk, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_si)

        v_mod = vfun(g.r_kpc)
        Cg = build_covariance(v_mod, g.r_kpc, g.v_err, g.D_Mpc, g.D_err_Mpc,
                              g.i_deg, g.i_err_deg, vfun, sigma_sys_kms=sigma_sys_kms)
        total += gaussian_loglike(g.v_obs, v_mod, Cg)
    return total


def log_posterior(theta: np.ndarray,
                  galaxies: List[GalaxyData],
                  sigma_sys_kms: float,
                  H0_si: float,
                  prior_kind: str) -> float:
    lp = log_prior(theta, galaxies, prior_kind)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, galaxies, sigma_sys_kms, H0_si)
    return lp + ll


# ------------------------- CLI -------------------------- #
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Global-epsilon SPARC mini/full fit with HDF5 backend support.")
    ap.add_argument("--data-path", required=True, help="Tidy SPARC subset CSV.")
    ap.add_argument("--outdir", default="results", help="Output directory.")
    ap.add_argument("--prior", choices=["galaxies-only","planck-anchored"], default="galaxies-only")
    ap.add_argument("--sigma-sys", type=float, default=4.0, help="Velocity floor [km/s].")
    ap.add_argument("--H0-kms-mpc", type=float, default=None, help="Override H0 [km/s/Mpc].")
    ap.add_argument("--nwalkers", type=str, default="4x", help="Num walkers or '4x' for 4*k.")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=42)

    # New: HDF5 backend + thinning + resume
    ap.add_argument("--backend-hdf5", type=str, default=None,
                    help="Path to emcee HDF5 backend (store chain on disk). Example: results/sparc90/chain.h5")
    ap.add_argument("--thin-by", type=int, default=10,
                    help="Thin factor when loading samples for summaries (>=1).")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from existing HDF5 backend (do not reset).")
    return ap.parse_args(argv)


def run(argv=None):
    args = parse_args(argv)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    gdict = load_tidy_sparc(args.data_path)
    galaxies = [gdict[k] for k in sorted(gdict.keys())]
    G = len(galaxies)
    print(f"Loaded {G} galaxies.")

    # H0 in SI
    H0_si = H0_SI
    if args.H0_kms_mpc is not None:
        from .constants import KM, MPC
        H0_si = (args.H0_kms_mpc * KM) / MPC

    # Parameter vector: [epsilon, Upsilon_1, ..., Upsilon_G]
    k = 1 + G
    if args.nwalkers.endswith("x"):
        mult = int(args.nwalkers[:-1])
        nwalkers = max(2 * k, mult * k)
    else:
        nwalkers = int(args.nwalkers)

    # Init positions
    rng = np.random.default_rng(args.seed)
    p0 = np.empty((nwalkers, k))
    # init epsilon
    if args.prior == "planck-anchored":
        p0[:, 0] = rng.normal(1.47, 0.05, size=nwalkers)
    else:
        p0[:, 0] = rng.uniform(0.2, 2.5, size=nwalkers)
    # init Upsilons
    p0[:, 1:] = rng.normal(0.5, 0.05, size=(nwalkers, G))
    p0[:, 1:] = np.clip(p0[:, 1:], 0.1, 1.2)

    # Optional HDF5 backend
    backend = None
    if args.backend_hdf5:
        try:
            from emcee.backends import HDFBackend
        except Exception as e:
            raise RuntimeError(
                "HDF5 backend requires emcee>=3 and h5py. Please install with: pip install h5py"
            ) from e

        backend_path = Path(args.backend_hdf5)
        backend_path.parent.mkdir(parents=True, exist_ok=True)
        backend = HDFBackend(str(backend_path))

        if args.resume and backend.iteration > 0:
            print(f"Resuming from backend {backend_path} at iteration={backend.iteration}")
            p0_init = None  # continue from last state
        else:
            backend.reset(nwalkers, k)
            p0_init = p0
    else:
        p0_init = p0

    sampler = emcee.EnsembleSampler(
        nwalkers, k, log_posterior,
        args=(galaxies, args.sigma_sys, H0_si, args.prior),
        backend=backend
    )

    print(f"Running emcee with nwalkers={nwalkers}, steps={args.steps}, k={k}")
    sampler.run_mcmc(p0_init, args.steps, progress=True)

    # -------- Postprocess (memory-safe with thinning) -------- #
    burn = args.steps // 3
    thin = max(int(args.thin_by), 1)

    if backend is not None:
        chain = backend.get_chain(discard=burn, flat=True, thin=thin)
    else:
        chain = sampler.get_chain(discard=burn, flat=True, thin=thin)

    eps_samples = chain[:, 0]
    Ups_samples = chain[:, 1:]

    # Medians/68%
    def med_lo_hi(x):
        q = np.percentile(x, [16, 50, 84])
        return float(q[1]), float(q[0]), float(q[2])

    eps_med, eps_lo, eps_hi = med_lo_hi(eps_samples)

    # Evaluate best (median) model and chi2, AIC, BIC
    theta_med = np.concatenate([[eps_med], np.median(Ups_samples, axis=0)])

    rows = []
    N_total = 0
    chi2_tot = 0.0

    for gi, g in enumerate(galaxies):
        U_med = theta_med[1 + gi]

        def vfun(rk):
            return model_v_kms(U_med, eps_med, rk, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_si)

        v_mod = vfun(g.r_kpc)
        Cg = build_covariance(v_mod, g.r_kpc, g.v_err, g.D_Mpc, g.D_err_Mpc,
                              g.i_deg, g.i_err_deg, vfun, sigma_sys_kms=args.sigma_sys)
        # chi2_g = r^T C^{-1} r
        r = g.v_obs - v_mod
        alpha = np.linalg.solve(Cg, r)
        chi2_g = float(r @ alpha)
        nu_g = max(len(g.r_kpc) - 1, 1)  # dof proxy per galaxy (M/L only)
        chi2_red_g = chi2_g / nu_g
        N_total += len(g.r_kpc)
        chi2_tot += chi2_g

        rows.append(dict(
            galaxy=g.name,
            n=len(g.r_kpc),
            Upsilon_med=U_med,
            chi2=chi2_g,
            chi2_red=chi2_red_g,
        ))

        # Save plot
        from .plotting import plot_rc
        plot_rc(g.name, g.r_kpc, g.v_obs, g.v_err, v_mod, outdir / f"plot_{g.name}.png")

    # Model selection counts (propagated D,i -> k = 1 + G)
    AIC = chi2_tot + 2 * (1 + G)
    BIC = chi2_tot + (1 + G) * np.log(N_total)

    # Save per-galaxy CSV
    df_pg = pd.DataFrame(rows).sort_values("galaxy")
    df_pg.to_csv(outdir / "per_galaxy_summary.csv", index=False)

    # Save posterior (thinned)
    np.savez_compressed(outdir / "posterior_samples.npz",
                        epsilon=eps_samples, Upsilons=Ups_samples, theta_med=theta_med,
                        burn_in=burn, thin=thin, nwalkers=nwalkers, steps=args.steps)

    # Global summary
    summary = dict(
        n_galaxies=G,
        N_total=N_total,
        prior=args.prior,
        sigma_sys_kms=args.sigma_sys,
        H0_si=H0_si,
        epsilon_median=eps_med,
        epsilon_p16=eps_lo,
        epsilon_p84=eps_hi,
        chi2_total=float(chi2_tot),
        AIC=float(AIC),
        BIC=float(BIC),
        k_parameters=int(1 + G),
        steps=args.steps,
        nwalkers=nwalkers,
        burn_in=burn,
        thin=thin,
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
