"""True maximum-likelihood layer for scientific hardening R1B.

Objective: maximize log L(theta) with the production full-covariance Gaussian
likelihood. Priors never enter the objective. Hard physical domains may be
enforced as optimizer bounds only.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize

from .constants import C_LIGHT, H0_SI
from .data import GalaxyData, load_tidy_sparc
from .fit_global import _layout, _unpack, log_likelihood

# Hard physical domains (bounds only — not prior densities)
UPSILON_BOUNDS = (0.05, 1.5)
Q_BOUNDS = (0.25, 8.0)
SIGMA_BOUNDS = (1e-3, 200.0)
LOGM_BOUNDS = (9.0, 13.0)
# Historical MOND a0 prior window — used as matched acceleration-scale domain
A0_MATCHED_BOUNDS = (5.0e-11, 2.0e-10)
EPS_WIDE_BOUNDS = (1.0e-6, 3.999)  # historical PTQ hard domain (sensitivity only)


def a0_to_epsilon(a0_si: float, H0_si: float = H0_SI) -> float:
    return float(a0_si) / (C_LIGHT * float(H0_si))


def epsilon_to_a0(eps: float, H0_si: float = H0_SI) -> float:
    return float(eps) * C_LIGHT * float(H0_si)


def matched_epsilon_bounds(H0_si: float = H0_SI) -> Tuple[float, float]:
    lo, hi = A0_MATCHED_BOUNDS
    return a0_to_epsilon(lo, H0_si), a0_to_epsilon(hi, H0_si)


def parameter_domain_manifest(H0_si: float = H0_SI) -> dict:
    eps_lo, eps_hi = matched_epsilon_bounds(H0_si)
    return {
        "H0_si": H0_si,
        "c_light": C_LIGHT,
        "c_H0": C_LIGHT * H0_si,
        "mapping": "a0 = epsilon * c * H0",
        "matched_a0_bounds_si": list(A0_MATCHED_BOUNDS),
        "matched_epsilon_bounds": [eps_lo, eps_hi],
        "historical_epsilon_wide_bounds": list(EPS_WIDE_BOUNDS),
        "upsilon_bounds": list(UPSILON_BOUNDS),
        "q_bounds": list(Q_BOUNDS),
        "sigma_sys_bounds_kms": list(SIGMA_BOUNDS),
        "log10_M200_bounds": list(LOGM_BOUNDS),
        "note": (
            "Canonical PTQ-nu/MOND and PTQ-screen/MOND-screen MLE equivalence "
            "uses matched_a0_bounds mapped bijectively to epsilon."
        ),
    }


def theta_bounds(model: str, G: int, H0_si: float = H0_SI,
                 domain: str = "matched") -> List[Tuple[float, float]]:
    """Bounds in the native theta coordinates used by fit_global._layout."""
    L = _layout(model, G, sigma_sys_learn=True, a0_fixed=None)
    bounds: List[Optional[Tuple[float, float]]] = [None] * L["k"]
    # Acceleration-scale models that share the MOND-like a0 map use matched bounds.
    # Linear PTQ (epsilon*c*H0*r) has a historically smaller epsilon optimum and uses
    # the wide historical epsilon domain unless domain == "matched_force".
    use_matched_eps = model in ("ptq-nu", "ptq-screen") and domain == "matched"
    if use_matched_eps:
        eps_lo, eps_hi = matched_epsilon_bounds(H0_si)
    else:
        eps_lo, eps_hi = EPS_WIDE_BOUNDS
    a0_lo, a0_hi = A0_MATCHED_BOUNDS

    if "i_eps" in L:
        bounds[L["i_eps"]] = (eps_lo, eps_hi)
    if "i_log10a0" in L:
        bounds[L["i_log10a0"]] = (math.log10(a0_lo), math.log10(a0_hi))
    if "i_lq" in L:
        bounds[L["i_lq"]] = (math.log(Q_BOUNDS[0]), math.log(Q_BOUNDS[1]))
    if "sl_U" in L:
        for i in range(L["sl_U"].start, L["sl_U"].stop):
            bounds[i] = UPSILON_BOUNDS
    if "sl_lM" in L:
        for i in range(L["sl_lM"].start, L["sl_lM"].stop):
            bounds[i] = LOGM_BOUNDS
    if "i_lnsig" in L:
        bounds[L["i_lnsig"]] = (math.log(SIGMA_BOUNDS[0]), math.log(SIGMA_BOUNDS[1]))

    out: List[Tuple[float, float]] = []
    for b in bounds:
        if b is None:
            raise RuntimeError("Incomplete bounds for layout")
        out.append(b)
    return out


def objective_neg_loglike(theta: np.ndarray,
                          galaxies: List[GalaxyData],
                          model: str,
                          H0_si: float,
                          c0: float = 10.0,
                          c_slope: float = -0.1) -> float:
    """Minimize this: -log L. No prior terms."""
    L = _layout(model, len(galaxies), sigma_sys_learn=True, a0_fixed=None)
    theta = np.asarray(theta, float)
    # Soft domain rejection without prior density
    for i, (lo, hi) in enumerate(theta_bounds(model, len(galaxies), H0_si)):
        if not (lo <= theta[i] <= hi):
            return 1.0e50
    try:
        ll = log_likelihood(
            theta, galaxies, sigma_sys_fixed=4.0, H0_si=H0_si, L=L,
            c0=c0, c_slope=c_slope, a0_fixed=None, like_kind="gauss", t_dof=8.0,
        )
    except Exception:
        return 1.0e50
    if not np.isfinite(ll):
        return 1.0e50
    return float(-ll)


def evaluate_loglike(theta: np.ndarray,
                     galaxies: List[GalaxyData],
                     model: str,
                     H0_si: float = H0_SI,
                     c0: float = 10.0,
                     c_slope: float = -0.1) -> float:
    L = _layout(model, len(galaxies), sigma_sys_learn=True, a0_fixed=None)
    return float(log_likelihood(
        np.asarray(theta, float), galaxies, 4.0, H0_si, L, c0, c_slope, None, "gauss", 8.0,
    ))


def galaxy_loglike(theta: np.ndarray,
                   galaxies: List[GalaxyData],
                   gi: int,
                   model: str,
                   H0_si: float = H0_SI,
                   c0: float = 10.0,
                   c_slope: float = -0.1) -> float:
    """Full-covariance logL contribution of a single galaxy (no prior)."""
    from .models import (
        model_v_baryon, model_v_mond, model_v_mond_screen, model_v_nfw1p,
        model_v_ptq, model_v_ptq_nu, model_v_ptq_screen,
    )
    from .likelihood import build_covariance, gaussian_loglike

    L = _layout(model, len(galaxies), True, None)
    pars = _unpack(np.asarray(theta, float), L)
    g = galaxies[gi]
    sigma_use = pars.get("sigma", 4.0)
    m = model

    if m == "ptq":
        U = float(pars["U"][gi]); eps = float(pars["eps"])
        def vfun(rk): return model_v_ptq(U, eps, rk, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_si)
    elif m == "ptq-nu":
        U = float(pars["U"][gi]); eps = float(pars["eps"])
        def vfun(rk): return model_v_ptq_nu(U, eps, rk, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_si)
    elif m == "ptq-screen":
        U = float(pars["U"][gi]); eps = float(pars["eps"]); q = float(np.exp(pars["lq"]))
        def vfun(rk): return model_v_ptq_screen(U, eps, q, rk, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_si)
    elif m == "baryon":
        U = float(pars["U"][gi])
        def vfun(rk): return model_v_baryon(U, rk, g.v_disk, g.v_bulge, g.v_gas)
    elif m == "mond":
        U = float(pars["U"][gi]); a0 = 10.0 ** pars["log10a0"]
        def vfun(rk): return model_v_mond(U, a0, rk, g.v_disk, g.v_bulge, g.v_gas)
    elif m == "mond-screen":
        U = float(pars["U"][gi]); a0 = 10.0 ** pars["log10a0"]; q = float(np.exp(pars["lq"]))
        def vfun(rk): return model_v_mond_screen(U, a0, q, rk, g.v_disk, g.v_bulge, g.v_gas)
    elif m == "nfw1p":
        U = float(pars["U"][gi]); lM = float(pars["lM"][gi])
        def vfun(rk): return model_v_nfw1p(U, lM, rk, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_si, c0=c0, c_slope=c_slope)
    else:
        raise ValueError(m)

    v_mod = vfun(g.r_kpc)
    Cg = build_covariance(v_mod, g.r_kpc, g.v_err, g.D_Mpc, g.D_err_Mpc,
                          g.i_deg, g.i_err_deg, vfun, sigma_sys_kms=sigma_use)
    return float(gaussian_loglike(g.v_obs, v_mod, Cg))


def map_theta_ptqnu_to_mond(theta_ptqnu: np.ndarray, G: int, H0_si: float = H0_SI) -> np.ndarray:
    Lp = _layout("ptq-nu", G, True, None)
    Lm = _layout("mond", G, True, None)
    p = _unpack(theta_ptqnu, Lp)
    out = np.zeros(Lm["k"])
    out[Lm["i_log10a0"]] = math.log10(epsilon_to_a0(p["eps"], H0_si))
    out[Lm["sl_U"]] = p["U"]
    out[Lm["i_lnsig"]] = math.log(p["sigma"])
    return out


def map_theta_mond_to_ptqnu(theta_mond: np.ndarray, G: int, H0_si: float = H0_SI) -> np.ndarray:
    Lm = _layout("mond", G, True, None)
    Lp = _layout("ptq-nu", G, True, None)
    p = _unpack(theta_mond, Lm)
    out = np.zeros(Lp["k"])
    a0 = 10.0 ** p["log10a0"]
    out[Lp["i_eps"]] = a0_to_epsilon(a0, H0_si)
    out[Lp["sl_U"]] = p["U"]
    out[Lp["i_lnsig"]] = math.log(p["sigma"])
    return out


def map_theta_ptqscreen_to_mondscreen(theta: np.ndarray, G: int, H0_si: float = H0_SI) -> np.ndarray:
    Lp = _layout("ptq-screen", G, True, None)
    Lm = _layout("mond-screen", G, True, None)
    p = _unpack(theta, Lp)
    out = np.zeros(Lm["k"])
    out[Lm["i_log10a0"]] = math.log10(epsilon_to_a0(p["eps"], H0_si))
    out[Lm["i_lq"]] = p["lq"]
    out[Lm["sl_U"]] = p["U"]
    out[Lm["i_lnsig"]] = math.log(p["sigma"])
    return out


def map_theta_mondscreen_to_ptqscreen(theta: np.ndarray, G: int, H0_si: float = H0_SI) -> np.ndarray:
    Lm = _layout("mond-screen", G, True, None)
    Lp = _layout("ptq-screen", G, True, None)
    p = _unpack(theta, Lm)
    out = np.zeros(Lp["k"])
    a0 = 10.0 ** p["log10a0"]
    out[Lp["i_eps"]] = a0_to_epsilon(a0, H0_si)
    out[Lp["i_lq"]] = p["lq"]
    out[Lp["sl_U"]] = p["U"]
    out[Lp["i_lnsig"]] = math.log(p["sigma"])
    return out


HISTORICAL_SUMMARY_PATHS = {
    "baryon": "results/revision_baseline_sixmodels_12000/baryon_gauss",
    "mond": "results/revision_baseline_sixmodels_12000/mond_gauss",
    "ptq": "results/revision_baseline_sixmodels_12000/ptq_gauss",
    "ptq-nu": "results/revision_baseline_sixmodels_12000/ptq-nu_gauss",
    "ptq-screen": "results/revision_baseline_sixmodels_12000/ptq-screen_gauss",
    "nfw1p": "results/revision_baseline_sixmodels_12000/nfw1p_gauss",
    "mond-screen": "results/revision_prod_equal12000_fresh/mond-screen_gauss",
}


def load_historical_median_theta(impl_root: Path,
                                 model: str,
                                 galaxies: List[GalaxyData],
                                 H0_si: float = H0_SI) -> Tuple[np.ndarray, dict]:
    rel = HISTORICAL_SUMMARY_PATHS[model]
    base = impl_root / rel
    summary = yaml.safe_load((base / "global_summary.yaml").read_text())
    pg = pd.read_csv(base / "per_galaxy_summary.csv")
    names = [g.name for g in galaxies]
    pg = pg.set_index("galaxy").loc[names]
    G = len(galaxies)
    L = _layout(model, G, True, None)
    theta = np.zeros(L["k"])

    if "i_eps" in L:
        theta[L["i_eps"]] = float(summary["epsilon_median"])
    if "i_log10a0" in L:
        theta[L["i_log10a0"]] = math.log10(float(summary["a0_median"]))
    if "i_lq" in L:
        theta[L["i_lq"]] = math.log(float(summary["q_median"]))
    if "sl_U" in L:
        theta[L["sl_U"]] = pg["Upsilon_med"].to_numpy(float)
    if "sl_lM" in L:
        theta[L["sl_lM"]] = pg["log10_M200_med"].to_numpy(float)
    if "i_lnsig" in L:
        theta[L["i_lnsig"]] = math.log(float(summary["sigma_sys_median"]))

    # Clip into matched domain if needed (historical medians should already be interior)
    b = theta_bounds(model, G, H0_si, domain="matched")
    for i, (lo, hi) in enumerate(b):
        theta[i] = float(np.clip(theta[i], lo, hi))

    meta = {
        "source_dir": str(base),
        "historical_logL_total_full": summary.get("logL_total_full"),
        "historical_AIC_full": summary.get("AIC_full"),
        "historical_BIC_full": summary.get("BIC_full"),
        "historical_k": summary.get("k_parameters"),
        "epsilon_median": summary.get("epsilon_median"),
        "a0_median": summary.get("a0_median"),
        "q_median": summary.get("q_median"),
        "sigma_sys_median": summary.get("sigma_sys_median"),
    }
    return theta, meta


def boundary_status(theta: np.ndarray, bounds: Sequence[Tuple[float, float]],
                    rtol: float = 1e-4) -> dict:
    near = []
    for i, ((lo, hi), x) in enumerate(zip(bounds, theta)):
        span = max(hi - lo, abs(lo), abs(hi), 1e-30)
        if abs(x - lo) <= rtol * span or abs(x - hi) <= rtol * span:
            near.append({"index": i, "value": float(x), "lo": lo, "hi": hi})
    return {"n_near_boundary": len(near), "near_boundary": near}


@dataclass
class StartResult:
    start_id: str
    initial_logL: float
    final_logL: float
    success: bool
    message: str
    nit: int
    nfev: int
    boundary: dict


def _global_indices(L: dict) -> List[int]:
    idx = []
    for key in ("i_eps", "i_log10a0", "i_lq", "i_lnsig"):
        if key in L:
            idx.append(int(L[key]))
    return idx


def _set_galaxy_nuisance(theta: np.ndarray, L: dict, gi: int, value: float,
                         which: str = "U") -> None:
    if which == "U" and "sl_U" in L:
        theta[L["sl_U"].start + gi] = value
    elif which == "lM" and "sl_lM" in L:
        theta[L["sl_lM"].start + gi] = value


def _block_coordinate_mle(model: str,
                          galaxies: List[GalaxyData],
                          theta0: np.ndarray,
                          H0_si: float,
                          bounds: List[Tuple[float, float]],
                          n_outer: int = 25) -> Tuple[np.ndarray, Any, float, float]:
    """Block-coordinate ascent on log L: alternate globals and per-galaxy nuisances.

    This targets a stationary point of the true full-covariance likelihood without
    priors. Much faster than dense finite-difference L-BFGS-B in ~100D.
    """
    from scipy.optimize import minimize_scalar

    L = _layout(model, len(galaxies), True, None)
    gidx = _global_indices(L)
    theta = np.asarray(theta0, float).copy()
    for i, (lo, hi) in enumerate(bounds):
        theta[i] = float(np.clip(theta[i], lo, hi))
    ll0 = evaluate_loglike(theta, galaxies, model, H0_si)
    nfev = 1
    G = len(galaxies)

    def neg_full(th):
        nonlocal nfev
        nfev += 1
        return objective_neg_loglike(th, galaxies, model, H0_si)

    for _outer in range(n_outer):
        ll_before = evaluate_loglike(theta, galaxies, model, H0_si)
        nfev += 1

        # --- globals ---
        if gidx:
            x0 = theta[gidx].copy()
            gb = [bounds[i] for i in gidx]

            def neg_g(x):
                th = theta.copy()
                th[gidx] = x
                return neg_full(th)

            res_g = minimize(
                neg_g, x0, method="L-BFGS-B", bounds=gb,
                options={"maxiter": 80, "ftol": 1e-14, "gtol": 1e-9},
            )
            theta[gidx] = np.asarray(res_g.x, float)
            nfev += int(getattr(res_g, "nfev", 0))

        # --- per-galaxy Upsilon (single-galaxy likelihood only) ---
        if "sl_U" in L and "sl_lM" not in L:
            ulo, uhi = UPSILON_BOUNDS
            for gi in range(G):
                def neg_u(u, gi=gi):
                    th = theta.copy()
                    th[L["sl_U"].start + gi] = float(u)
                    nfev_local = galaxy_loglike(th, galaxies, gi, model, H0_si)
                    return -nfev_local

                r = minimize_scalar(neg_u, bounds=(ulo, uhi), method="bounded",
                                    options={"xatol": 1e-6, "maxiter": 40})
                theta[L["sl_U"].start + gi] = float(r.x)
                nfev += int(getattr(r, "nfev", 0))

        # --- NFW: joint (U, logM) per galaxy ---
        if "sl_U" in L and "sl_lM" in L:
            ulo, uhi = UPSILON_BOUNDS
            mlo, mhi = LOGM_BOUNDS
            for gi in range(G):
                def neg_um(x, gi=gi):
                    th = theta.copy()
                    th[L["sl_U"].start + gi] = float(x[0])
                    th[L["sl_lM"].start + gi] = float(x[1])
                    return -galaxy_loglike(th, galaxies, gi, model, H0_si)

                x0 = np.array([
                    theta[L["sl_U"].start + gi],
                    theta[L["sl_lM"].start + gi],
                ], float)
                r = minimize(
                    neg_um, x0, method="L-BFGS-B",
                    bounds=[(ulo, uhi), (mlo, mhi)],
                    options={"maxiter": 40, "ftol": 1e-12},
                )
                theta[L["sl_U"].start + gi] = float(r.x[0])
                theta[L["sl_lM"].start + gi] = float(r.x[1])
                nfev += int(getattr(r, "nfev", 0))

        ll_after = evaluate_loglike(theta, galaxies, model, H0_si)
        nfev += 1
        if abs(ll_after - ll_before) < 1e-6:
            break

    # Final joint polish on globals only (nuisances frozen at coordinate optimum)
    if gidx:
        x0 = theta[gidx].copy()
        gb = [bounds[i] for i in gidx]

        def neg_g2(x):
            th = theta.copy()
            th[gidx] = x
            return neg_full(th)

        res_g2 = minimize(
            neg_g2, x0, method="L-BFGS-B", bounds=gb,
            options={"maxiter": 100, "ftol": 1e-14, "gtol": 1e-10},
        )
        theta[gidx] = np.asarray(res_g2.x, float)
        nfev += int(getattr(res_g2, "nfev", 0))

    ll_f = evaluate_loglike(theta, galaxies, model, H0_si)
    nfev += 1

    class _Res:
        pass

    res = _Res()
    res.success = bool(ll_f >= ll0 - 1e-8)
    res.message = f"block_coordinate_ascent outer_converged ll0={ll0:.6f} ll={ll_f:.6f}"
    res.nit = n_outer
    res.nfev = nfev
    return theta, res, ll0, ll_f


def optimize_mle(model: str,
                 galaxies: List[GalaxyData],
                 starts: List[Tuple[str, np.ndarray]],
                 H0_si: float = H0_SI,
                 domain: str = "matched",
                 maxiter: int = 25) -> dict:
    """maxiter here means block-coordinate outer iterations."""
    G = len(galaxies)
    L = _layout(model, G, True, None)
    bounds = theta_bounds(model, G, H0_si, domain=domain)
    N_points = int(sum(len(g.r_kpc) for g in galaxies))
    k = int(L["k"])

    n_outer = maxiter if model != "nfw1p" else max(maxiter, 15)

    best = None
    best_theta = None
    best_start = None
    best_msg = ""
    best_success = False
    start_rows = []
    traces = []

    for start_id, theta0 in starts:
        theta_f, res, ll0, ll_f = _block_coordinate_mle(
            model, galaxies, theta0, H0_si, bounds, n_outer=n_outer
        )
        bstat = boundary_status(theta_f, bounds)
        row = StartResult(
            start_id=start_id,
            initial_logL=ll0,
            final_logL=ll_f,
            success=bool(res.success),
            message=str(res.message),
            nit=int(res.nit),
            nfev=int(res.nfev),
            boundary=bstat,
        )
        start_rows.append(asdict(row))
        traces.append({
            "start_id": start_id,
            "initial_logL": ll0,
            "final_logL": ll_f,
            "success": bool(res.success),
            "message": str(res.message),
            "nit": int(res.nit),
            "nfev": int(res.nfev),
        })
        if best is None or ll_f > best:
            best = ll_f
            best_theta = theta_f
            best_start = start_id
            best_msg = str(res.message)
            best_success = bool(res.success)

    finals = sorted([t["final_logL"] for t in traces], reverse=True)
    cross_start_delta = float(finals[0] - finals[1]) if len(finals) > 1 else 0.0

    AIC = -2.0 * best + 2 * k
    BIC = -2.0 * best + k * math.log(N_points)
    pars = _unpack(best_theta, L)

    param_vector = {
        "eps": pars.get("eps"),
        "log10a0": pars.get("log10a0"),
        "a0": (10.0 ** pars["log10a0"]) if "log10a0" in pars else None,
        "q": float(np.exp(pars["lq"])) if "lq" in pars else None,
        "sigma_sys": pars.get("sigma"),
        "Upsilon_mean": float(np.mean(pars["U"])) if "U" in pars else None,
        "log10_M200_mean": float(np.mean(pars["lM"])) if "lM" in pars else None,
    }

    return {
        "model": model,
        "max_loglike": float(best),
        "AIC": float(AIC),
        "BIC_NPOINTS": float(BIC),
        "k": k,
        "N_data": N_points,
        "N_gal": G,
        "optimizer": "block-coordinate ascent on logL (globals + per-galaxy; no priors)",
        "converged": bool(best_success),
        "termination_status": best_msg,
        "best_start": best_start,
        "cross_start_delta_logL": cross_start_delta,
        "boundary": boundary_status(best_theta, bounds),
        "parameter_vector": param_vector,
        "theta": best_theta.tolist(),
        "starts": start_rows,
        "domain": domain,
        "H0_si": H0_si,
    }


def build_multistart(model: str,
                     galaxies: List[GalaxyData],
                     impl_root: Path,
                     H0_si: float = H0_SI,
                     n_perturb: int = 2,
                     seed: int = 42) -> List[Tuple[str, np.ndarray]]:
    G = len(galaxies)
    theta_hist, _ = load_historical_median_theta(impl_root, model, galaxies, H0_si)
    starts: List[Tuple[str, np.ndarray]] = [("historical_posterior_median", theta_hist)]
    rng = np.random.default_rng(seed)
    bounds = theta_bounds(model, G, H0_si, domain="matched")

    for j in range(n_perturb):
        th = theta_hist.copy()
        noise = rng.normal(0.0, 0.02, size=th.shape)
        for i, (lo, hi) in enumerate(bounds):
            span = hi - lo
            th[i] = float(np.clip(th[i] + noise[i] * span, lo, hi))
        starts.append((f"perturb_{j}", th))

    L = _layout(model, G, True, None)
    if "i_eps" in L and model in ("ptq-nu", "ptq-screen"):
        th = theta_hist.copy()
        lo, hi = bounds[L["i_eps"]]
        th[L["i_eps"]] = 0.5 * (lo + hi)
        starts.append(("mid_epsilon", th.copy()))
    if "i_log10a0" in L:
        lo, hi = bounds[L["i_log10a0"]]
        th2 = theta_hist.copy()
        th2[L["i_log10a0"]] = 0.5 * (lo + hi)
        starts.append(("mid_log10a0", th2))

    if model == "ptq-nu":
        th_m, _ = load_historical_median_theta(impl_root, "mond", galaxies, H0_si)
        starts.append(("mapped_from_mond_median", map_theta_mond_to_ptqnu(th_m, G, H0_si)))
    elif model == "mond":
        th_p, _ = load_historical_median_theta(impl_root, "ptq-nu", galaxies, H0_si)
        starts.append(("mapped_from_ptqnu_median", map_theta_ptqnu_to_mond(th_p, G, H0_si)))
    elif model == "ptq-screen":
        th_m, _ = load_historical_median_theta(impl_root, "mond-screen", galaxies, H0_si)
        starts.append(("mapped_from_mondscreen_median", map_theta_mondscreen_to_ptqscreen(th_m, G, H0_si)))
    elif model == "mond-screen":
        th_p, _ = load_historical_median_theta(impl_root, "ptq-screen", galaxies, H0_si)
        starts.append(("mapped_from_ptqscreen_median", map_theta_ptqscreen_to_mondscreen(th_p, G, H0_si)))

    return starts


def write_model_outputs(outdir: Path, result: dict, hist_meta: dict) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    theta = result["theta"]
    summary = {k: v for k, v in result.items() if k != "theta"}
    summary["historical"] = hist_meta
    summary["delta_logL_vs_historical_median"] = float(
        result["max_loglike"] - hist_meta["historical_logL_total_full"]
    )
    summary["delta_AIC_vs_historical_median"] = float(
        result["AIC"] - hist_meta["historical_AIC_full"]
    )
    (outdir / "mle_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(result["starts"]).to_csv(outdir / "optimization_trace.csv", index=False)
    init_rows = [
        {
            "start_id": s["start_id"],
            "initial_logL": s["initial_logL"],
            "final_logL": s["final_logL"],
            "success": s["success"],
            "nit": s["nit"],
            "nfev": s["nfev"],
        }
        for s in result["starts"]
    ]
    pd.DataFrame(init_rows).to_csv(outdir / "initialization_summary.csv", index=False)
    np.save(outdir / "theta_mle.npy", np.asarray(theta, float))


def model_dirname(model: str) -> str:
    return model.replace("-", "_")
