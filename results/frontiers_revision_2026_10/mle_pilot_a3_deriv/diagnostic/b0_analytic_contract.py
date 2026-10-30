"""Non-activated B0 analytic-derivative prototype.

Calls the tested integrator for the objective. Does not replace an optimizer callback.
Unsupported coordinates are omitted, not filled with zero.
"""
import hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp

CAND = Path("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/mle_pilot_a/candidate/src")
sys.path.insert(0, str(CAND))
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from ptquat.constants import C_LIGHT, H0_SI, KPC, KM
from ptquat.marginal_likelihood import PRIMARY_B0, integrate, nuisance_nodes, velocity
from ptquat.models import model_v_ptq, model_v_ptq_nu, vbar_squared_kms2

SUPPORTED = {
    "ptq": ("epsilon", "ln_sigma_sys"),
    "ptq-nu": ("epsilon", "ln_sigma_sys"),
}
IDENTITY_LIMIT = 1e-9


def as_g(g):
    from types import SimpleNamespace
    return SimpleNamespace(
        name=g.name, r=g.r_kpc, y=g.v_obs, e=g.v_err,
        disk=g.v_disk, bulge=g.v_bulge, gas=g.v_gas,
        D=float(g.D_Mpc), sD=float(g.D_err_Mpc),
        inc=float(np.deg2rad(g.i_deg)), si=float(np.deg2rad(g.i_err_deg)),
    )


def _dvd(model, gg, pp, gi, ds, vd):
    """Smooth-branch dv/depsilon on the distance-scaled catalog grid. Flags nonsmooth events."""
    scale = ds / gg.D
    radius = gg.r * scale[:, None]
    disk = gg.disk * np.sqrt(scale)[:, None]
    bulge = gg.bulge * np.sqrt(scale)[:, None]
    gas = gg.gas * np.sqrt(scale)[:, None]
    U = float(pp["U"][gi])
    eps = float(pp["eps"])
    flags = []
    if np.any(vd <= 0) or not np.isfinite(vd).all():
        flags.append("nonpositive_velocity")
    if model == "ptq":
        formula = model_v_ptq(U, eps, radius, disk, bulge, gas, H0_si=H0_SI)
        if np.max(np.abs(formula - vd)) > 1e-8 * max(1.0, float(np.max(np.abs(vd)))):
            flags.append("velocity_formula_mismatch")
        # v^2 = v_bar^2 + (epsilon*c*H0*r_m)/KM^2, so dv/depsilon = that coefficient / (2v).
        additive = (C_LIGHT * H0_SI * (radius * KPC)) / (KM ** 2)
        return additive / (2.0 * vd), flags
    if model == "ptq-nu":
        formula = model_v_ptq_nu(U, eps, radius, disk, bulge, gas, H0_si=H0_SI)
        if np.max(np.abs(formula - vd)) > 1e-8 * max(1.0, float(np.max(np.abs(vd)))):
            flags.append("velocity_formula_mismatch")
        a0 = eps * C_LIGHT * H0_SI
        if a0 <= 1e-30:
            flags.append("a0_floor")
        vN2 = vbar_squared_kms2(U, disk, bulge, gas)
        r_m = radius * KPC
        if np.any(r_m <= 1e-12):
            flags.append("radius_floor")
        r_m = np.maximum(r_m, 1e-12)
        gN = (vN2 * (KM ** 2)) / r_m
        y_raw = gN / max(a0, 1e-30)
        if np.any(y_raw < 1e-11):
            flags.append("y_near_clip")
        y = np.maximum(y_raw, 1e-12)
        root = np.sqrt(0.25 + 1.0 / y)
        nu = 0.5 + root
        # dnu/dy = -0.5 / (sqrt(0.25 + 1/y) * y^2); dy/da0 = -y/a0; da0/depsilon = c*H0.
        dnu_dy = -0.5 / (root * y ** 2)
        dv = (vd / (2.0 * nu)) * dnu_dy * (-y / max(a0, 1e-30)) * (C_LIGHT * H0_SI)
        return dv, flags
    raise ValueError(model)


def galaxy_b0_derivatives(g, pars, gi, model, order=192):
    """Official integrate logL plus analytic d logL / d epsilon and d logL / d ln_sigma.

    Quadrature nodes and weights depend on galaxy geometry and order, not on theta.
    """
    if model not in SUPPORTED:
        raise ValueError("model outside this partial contract: " + model)
    gg = as_g(g)
    pp = dict(pars, model=model, _H0=H0_SI, _c0=10.0, _c_slope=-0.1)
    official = integrate(gg, pp, gi, "segmented_quantile", order, floors=("B0",))
    ds, wd, incs, wi, _, _ = nuisance_nodes(gg, "segmented_quantile", order)
    if abs(float(wd.sum() * wi.sum()) - 1.0) > 1e-12:
        raise RuntimeError("weights are not normalized")
    sigma = float(pp["sigma"])
    vd = velocity(gg, pp, gi, ds[:, None])
    dvd, flags = _dvd(model, gg, pp, gi, ds, vd)
    b = np.sin(incs) / np.sin(gg.inc)
    var = gg.e ** 2 + sigma ** 2
    inv = 1.0 / var
    inv2 = inv / var
    norm = float(np.sum(np.log(2.0 * np.pi * var)))
    Y2 = float(np.sum((gg.y ** 2) * inv))
    Y2q = float(np.sum((gg.y ** 2) * inv2))
    sum_inv = float(np.sum(inv))
    lls, d_sigma, d_eps, logw = [], [], [], []
    for start in range(0, len(ds), 128):
        end = min(start + 128, len(ds))
        v = vd[start:end]
        dv = dvd[start:end]
        cross = v @ (gg.y * inv)
        vv = (v * v) @ inv
        Q = Y2 - 2.0 * b[None, :] * cross[:, None] + (b ** 2)[None, :] * vv[:, None]
        ll = -0.5 * (Q + norm)
        # d ell / d sigma includes both the quadratic term and d(log det)/d sigma.
        crossq = v @ (gg.y * inv2)
        vvq = (v * v) @ inv2
        resid_over = Y2q - 2.0 * b[None, :] * crossq[:, None] + (b ** 2)[None, :] * vvq[:, None]
        dll_sigma = sigma * (resid_over - sum_inv)
        # mean = b * v; d ell / d v_k = b * (y_k - b v_k) / var_k.
        cy = dv @ (gg.y * inv)
        cv = (dv * v) @ inv
        dll_eps = b[None, :] * cy[:, None] - (b ** 2)[None, :] * cv[:, None]
        lls.append(ll)
        d_sigma.append(dll_sigma)
        d_eps.append(dll_eps)
        logw.append(np.log(wd[start:end])[:, None] + np.log(wi)[None, :])
    ll = np.concatenate([a.ravel() for a in lls])
    ds_ = np.concatenate([a.ravel() for a in d_sigma])
    de = np.concatenate([a.ravel() for a in d_eps])
    lw = np.concatenate([a.ravel() for a in logw])
    score = ll + lw
    logL = float(logsumexp(score))
    posterior = np.exp(score - logL)
    dlogL_dsigma = float(np.sum(posterior * ds_))
    dlogL_depsilon = float(np.sum(posterior * de))
    # sigma = exp(ln_sigma); chain rule at fixed other parameters.
    dlogL_dlnsigma = sigma * dlogL_dsigma
    official_logL = float(official["logL"]["B0"])
    domain = "INTERIOR_ANALYTICALLY_DIFFERENTIABLE" if not flags else "CLIPPED_OR_NONSMOOTH"
    return {
        "official_logL": official_logL,
        "rederived_logL": logL,
        "abs_gap": abs(logL - official_logL),
        "dlogL_depsilon": dlogL_depsilon,
        "dlogL_dln_sigma": dlogL_dlnsigma,
        "flags": flags,
        "domain": domain,
        "weight_sum": float(wd.sum() * wi.sum()),
    }


def evaluate_fixed_b0(theta, model, galaxies, order=192, charge=None):
    """Partial fixed-point evaluation. Not an optimizer Jacobian."""
    from ptquat.fit_global import _layout, _unpack
    if order != 192:
        raise ValueError("this prototype validates the scientific order 192 only")
    if model not in SUPPORTED:
        return {
            "objective": None,
            "analytic_partial_gradients": {},
            "unsupported": ["all"],
            "validation_metadata": {"reason": "model not in the partial contract", "activated": False},
        }
    layout = _layout(model, 91, True, None)
    pars = _unpack(np.asarray(theta, float), layout)
    rows = []
    for gi, g in enumerate(galaxies):
        if charge is not None:
            charge(model, gi, order)
        rows.append(galaxy_b0_derivatives(g, pars, gi, model, order))
    logL = float(sum(r["official_logL"] for r in rows))
    objective = -logL
    # F = -sum log L, so the optimizer derivative is the negative sum.
    dF_eps = -float(sum(r["dlogL_depsilon"] for r in rows))
    dF_sig = -float(sum(r["dlogL_dln_sigma"] for r in rows))
    gap = max(r["abs_gap"] for r in rows)
    flags = sorted({flag for r in rows for flag in r["flags"]})
    return {
        "objective": objective,
        "rederived_objective": -float(sum(r["rederived_logL"] for r in rows)),
        "max_abs_galaxy_logL_gap": gap,
        "identity_limit": IDENTITY_LIMIT,
        "identity_pass": gap <= IDENTITY_LIMIT,
        "analytic_partial_gradients": {
            "epsilon": {"native_dF": dF_eps, "units": "objective per native epsilon", "supported": True},
            "ln_sigma_sys": {"native_dF": dF_sig, "units": "objective per native ln_sigma", "supported": True},
        },
        "unsupported_not_returned_as_zero": [
            "galaxy_upsilon", "screen_epsilon", "screen_ln_q", "mond_log10_a0", "nfw_log10_M200",
        ],
        "validation_metadata": {
            "activated_as_optimizer_jacobian": False,
            "likelihood_version": PRIMARY_B0,
            "order": order,
            "n_galaxies": len(galaxies),
            "domain_flags": flags,
            "domain": "CLIPPED_OR_NONSMOOTH" if flags else "INTERIOR_ANALYTICALLY_DIFFERENTIABLE",
            "weights_depend_on_theta": False,
            "complete_jacobian": False,
        },
    }
