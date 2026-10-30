"""Isolated B0 derivatives for galaxy Upsilon and smooth screen epsilon, ln q.

Not an optimizer callback. Unsupported coordinates are omitted, never filled with zero.
The objective is the tested integrate() B0 marginal. Analytic pieces are cross-checked
against that integrator by the runner; this module does not call scipy.optimize.minimize.
"""
import os
import sys
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
from ptquat.models import (
    linear_term_kms2,
    model_v_mond_screen,
    model_v_ptq,
    model_v_ptq_nu,
    model_v_ptq_screen,
    nu_q,
    vbar_squared_kms2,
)

IDENTITY_LIMIT = 1e-9
# Predeclared before any fixed-point comparison. This is not a derivative tolerance.
# A direction is called analytic-FD consistent only when the observed gap is within
# max(1e-4, 1e-3*|analytic|). gtol=1e-6 is a separate, stricter question.
CONSISTENT_ABS_FLOOR = 1e-4
CONSISTENT_REL = 1e-3


def as_g(g):
    from types import SimpleNamespace
    return SimpleNamespace(
        name=g.name, r=g.r_kpc, y=g.v_obs, e=g.v_err,
        disk=g.v_disk, bulge=g.v_bulge, gas=g.v_gas,
        D=float(g.D_Mpc), sD=float(g.D_err_Mpc),
        inc=float(np.deg2rad(g.i_deg)), si=float(np.deg2rad(g.i_err_deg)),
    )


def _scaled(gg, ds):
    scale = np.asarray(ds, float) / gg.D
    radius = gg.r * scale[:, None]
    root = np.sqrt(scale)[:, None]
    disk = gg.disk * root
    bulge = gg.bulge * root
    gas = gg.gas * root
    return radius, disk, bulge, gas


def _nu_derivatives(y, kind, q):
    """Smooth-branch nu and partials. y must already be the value the code evaluates."""
    if kind == "simple":
        root = np.sqrt(0.25 + 1.0 / y)
        nu = 0.5 + root
        dnu_dy = -0.5 / (root * y ** 2)
        return nu, dnu_dy, None
    if kind != "screen":
        raise ValueError(kind)
    logy = np.log(y)
    # nu_q uses np.power(y, -q). exp(-q log y) is the same smooth value when finite.
    yq = np.exp((-float(q)) * logy)
    root = np.sqrt(0.25 + yq)
    nu = 0.5 + root
    # d/dy of y^{-q} is -q y^{-q-1}; d nu/dy = that / (2 root).
    dnu_dy = (-0.5 * float(q) * np.exp((-(float(q) + 1.0)) * logy)) / root
    # d nu/dq = -ln(y) * y^{-q} / (2 root). Chain to ln q multiplies by q.
    dnu_dq = (-0.5 * logy * yq) / root
    dnu_dlnq = float(q) * dnu_dq
    return nu, dnu_dy, dnu_dlnq


def analytic_dv(model, gg, pars, gi, ds, vd):
    """Partial dv on the distance-scaled grid, plus branch flags.

    Gas enters only as v_gas^2, so its catalog sign does not flip the baryonic term.
    Upsilon multiplies disk^2+bulge^2 only. MOND/screen y depends on that v_N^2.
    """
    radius, disk, bulge, gas = _scaled(gg, ds)
    U = float(pars["U"][gi])
    v_star2 = disk ** 2 + bulge ** 2
    vN2 = vbar_squared_kms2(U, disk, bulge, gas)
    flags = []
    if np.any(vd <= 0) or not np.isfinite(vd).all():
        flags.append("nonpositive_velocity")
    if np.any(gas < 0):
        flags.append("negative_catalog_gas_squared")
    if np.any(bulge < 0):
        flags.append("negative_catalog_bulge_squared")
    if not np.any(v_star2 > 0):
        flags.append("zero_stellar_template")
    H0 = float(pars["_H0"])
    dvds = {}

    def check(formula):
        scale = max(1.0, float(np.max(np.abs(vd))))
        if np.max(np.abs(formula - vd)) > 1e-8 * scale:
            flags.append("velocity_formula_mismatch")

    if model == "ptq":
        eps = float(pars["eps"])
        formula = model_v_ptq(U, eps, radius, disk, bulge, gas, H0_si=H0)
        check(formula)
        additive = (C_LIGHT * H0 * (radius * KPC)) / (KM ** 2)
        dvds["epsilon"] = additive / (2.0 * vd)
        dvds["upsilon"] = v_star2 / (2.0 * vd)
        return dvds, flags

    if model == "ptq-nu":
        eps = float(pars["eps"])
        formula = model_v_ptq_nu(U, eps, radius, disk, bulge, gas, H0_si=H0)
        check(formula)
        a0_raw = eps * C_LIGHT * H0
        kind = "simple"
        q = None
        da0_deps = C_LIGHT * H0
        native_accel = "epsilon"
    elif model == "ptq-screen":
        eps = float(pars["eps"])
        q = float(np.exp(pars["lq"]))
        formula = model_v_ptq_screen(U, eps, q, radius, disk, bulge, gas, H0_si=H0)
        check(formula)
        a0_raw = eps * C_LIGHT * H0
        kind = "screen"
        da0_deps = C_LIGHT * H0
        native_accel = "epsilon"
    elif model == "mond-screen":
        a0_raw = 10.0 ** float(pars["log10a0"])
        q = float(np.exp(pars["lq"]))
        formula = model_v_mond_screen(U, a0_raw, q, radius, disk, bulge, gas)
        check(formula)
        kind = "screen"
        da0_deps = None
        native_accel = "log10_a0"
    else:
        raise ValueError("model outside this isolated contract: " + model)

    if a0_raw <= 1e-30:
        flags.append("a0_floor")
    a0_used = max(float(a0_raw), 1e-30)
    r_m = radius * KPC
    if np.any(r_m <= 1e-12):
        flags.append("radius_floor")
    r_used = np.maximum(r_m, 1e-12)
    gN = (vN2 * (KM ** 2)) / r_used
    y_raw = gN / a0_used
    if np.any(y_raw < 1e-12):
        flags.append("y_floor")
    y = np.maximum(y_raw, 1e-12)
    nu, dnu_dy, dnu_dlnq = _nu_derivatives(y, kind, q)
    if kind == "screen":
        nu_ref = nu_q(y, q)
        if np.max(np.abs(nu - nu_ref)) > 1e-8 * max(1.0, float(np.max(np.abs(nu_ref)))):
            flags.append("nu_q_formula_mismatch")
        if not np.isfinite(nu).all() or not np.isfinite(dnu_dlnq).all():
            flags.append("q_power_nonfinite")
    # y = gN/a0 and gN ∝ vN2, so dy/dvN2 = y/vN2 on the unclipped side and 0 on the y floor.
    unclipped = y_raw >= 1e-12
    dy_dvN2 = np.where(unclipped & (vN2 > 0), y / np.maximum(vN2, 1e-300), 0.0)
    if np.any((~unclipped) & (vN2 <= 0)):
        flags.append("nonpositive_vN2_on_floor")
    if np.any(unclipped & (vN2 <= 0)):
        flags.append("nonpositive_vN2_off_floor")
    dv2_dvN2 = nu + vN2 * dnu_dy * dy_dvN2
    dvds["upsilon"] = dv2_dvN2 * v_star2 / (2.0 * vd)
    # Acceleration parameter. y is not held fixed.
    if native_accel == "epsilon":
        da0 = 0.0 if a0_raw <= 1e-30 else da0_deps
        dy_da = np.where(unclipped, -(y / a0_used) * da0, 0.0)
        dvds["epsilon"] = (vN2 * dnu_dy * dy_da) / (2.0 * vd)
    else:
        # a0 = 10^x, da0/dx = a0 ln 10, away from the a0 floor.
        da0 = 0.0 if a0_raw <= 1e-30 else a0_used * np.log(10.0)
        dy_da = np.where(unclipped, -(y / a0_used) * da0, 0.0)
        dvds["log10_a0"] = (vN2 * dnu_dy * dy_da) / (2.0 * vd)
    if kind == "screen":
        # q does not enter gN or a0. The ln q factor is the exp(ln q) chain rule.
        dvds["ln_q"] = (vN2 * dnu_dlnq) / (2.0 * vd)
    return dvds, flags


def _accumulate(gg, vd, dvds, sigma, ds, wd, incs, wi):
    b = np.sin(incs) / np.sin(gg.inc)
    var = gg.e ** 2 + sigma ** 2
    inv = 1.0 / var
    inv2 = inv / var
    norm = float(np.sum(np.log(2.0 * np.pi * var)))
    Y2 = float(np.sum((gg.y ** 2) * inv))
    Y2q = float(np.sum((gg.y ** 2) * inv2))
    sum_inv = float(np.sum(inv))
    names = list(dvds)
    blocks = {name: [] for name in names}
    blocks["ln_sigma_sys"] = []
    lls, logw = [], []
    for start in range(0, len(ds), 128):
        end = min(start + 128, len(ds))
        v = vd[start:end]
        cross = v @ (gg.y * inv)
        vv = (v * v) @ inv
        Q = Y2 - 2.0 * b[None, :] * cross[:, None] + (b ** 2)[None, :] * vv[:, None]
        ll = -0.5 * (Q + norm)
        crossq = v @ (gg.y * inv2)
        vvq = (v * v) @ inv2
        resid_over = Y2q - 2.0 * b[None, :] * crossq[:, None] + (b ** 2)[None, :] * vvq[:, None]
        # d ell/d sigma, including the log-determinant. Chain to ln sigma is another sigma.
        dll_sigma = sigma * (resid_over - sum_inv)
        lls.append(ll)
        blocks["ln_sigma_sys"].append(dll_sigma)
        logw.append(np.log(wd[start:end])[:, None] + np.log(wi)[None, :])
        for name in names:
            dv = dvds[name][start:end]
            cy = dv @ (gg.y * inv)
            cv = (dv * v) @ inv
            blocks[name].append(b[None, :] * cy[:, None] - (b ** 2)[None, :] * cv[:, None])
    ll = np.concatenate([a.ravel() for a in lls])
    lw = np.concatenate([a.ravel() for a in logw])
    score = ll + lw
    logL = float(logsumexp(score))
    posterior = np.exp(score - logL)
    out = {}
    for name, parts in blocks.items():
        flat = np.concatenate([a.ravel() for a in parts])
        dlogL = float(np.sum(posterior * flat))
        if name == "ln_sigma_sys":
            dlogL = sigma * dlogL
        out[name] = dlogL
    return logL, out


def galaxy_partials(g, pars, gi, model, order=192, charge=None):
    """Official B0 logL and d logL for the coordinates this model actually implements."""
    if model not in ("ptq", "ptq-nu", "ptq-screen", "mond-screen"):
        raise ValueError("model outside this isolated contract: " + model)
    if order != 192:
        raise ValueError("scientific order is 192; diagnostic 384 is not this function")
    gg = as_g(g)
    official = None
    if charge is not None:
        charge(model, gi, order)
    official = integrate(gg, pars, gi, "segmented_quantile", order, floors=("B0",))
    ds, wd, incs, wi, _, _ = nuisance_nodes(gg, "segmented_quantile", order)
    if abs(float(wd.sum() * wi.sum()) - 1.0) > 1e-12:
        raise RuntimeError("weights are not normalized")
    vd = velocity(gg, pars, gi, ds[:, None])
    dvds, flags = analytic_dv(model, gg, pars, gi, ds, vd)
    fatal = [f for f in flags if f in ("nonpositive_velocity", "velocity_formula_mismatch", "nu_q_formula_mismatch", "q_power_nonfinite", "nonpositive_vN2_off_floor")]
    rederived, dlogL = _accumulate(gg, vd, dvds, float(pars["sigma"]), ds, wd, incs, wi)
    official_logL = float(official["logL"]["B0"])
    if fatal:
        dlogL = {key: None for key in dlogL}
    domain = "INTERIOR_ANALYTICALLY_DIFFERENTIABLE"
    if any(f in flags for f in ("y_floor", "a0_floor", "radius_floor", "nonpositive_velocity", "q_power_nonfinite")):
        domain = "CLIPPED_OR_NONSMOOTH"
    if fatal:
        domain = "DERIVATIVE_NOT_VALIDATED"
    return {
        "galaxy": g.name,
        "official_logL": official_logL,
        "rederived_logL": rederived,
        "abs_gap": abs(rederived - official_logL),
        "dlogL": dlogL,
        "flags": flags,
        "domain": domain,
        "weights_depend_on_theta": False,
        "likelihood_version": PRIMARY_B0,
    }


def formula_self_check():
    """Uncharged finite-difference check of dv on a smooth synthetic grid."""
    r = np.array([2.0, 6.0, 12.0])
    disk = np.array([40.0, 70.0, 90.0])
    bulge = np.array([0.0, 15.0, 25.0])
    gas = np.array([10.0, -8.0, 30.0])
    U, eps, q = 0.6, 0.15, 1.35
    H0 = H0_SI

    def rel(a, b):
        return float(np.max(np.abs(a - b)) / max(1.0, float(np.max(np.abs(b)))))

    h = 1e-6
    vstar = disk ** 2 + bulge ** 2
    vN_p = vbar_squared_kms2(U + h, disk, bulge, gas)
    vN_m = vbar_squared_kms2(U - h, disk, bulge, gas)
    if rel((vN_p - vN_m) / (2 * h), vstar) > 1e-8:
        raise RuntimeError("gas/bulge Upsilon convention failed the synthetic check")
    # Squaring removes the catalog sign of gas.
    if rel(vbar_squared_kms2(U, disk, bulge, gas), vbar_squared_kms2(U, disk, bulge, np.abs(gas))) > 0:
        raise RuntimeError("gas sign entered v_bar^2")

    class G:
        pass
    gg = G()
    gg.r, gg.disk, gg.bulge, gg.gas, gg.D = r, disk, bulge, gas, 10.0
    ds = np.array([10.0])
    rows = []
    common = dict(sigma=8.0, _H0=H0, _c0=10.0, _c_slope=-0.1)
    cases = {
        "ptq": dict(U=np.array([U]), eps=eps, **common),
        "ptq-nu": dict(U=np.array([U]), eps=eps, **common),
        "ptq-screen": dict(U=np.array([U]), eps=eps, lq=np.log(q), **common),
        "mond-screen": dict(U=np.array([U]), log10a0=np.log10(eps * C_LIGHT * H0), lq=np.log(q), **common),
    }
    for model, pars in cases.items():
        pars = dict(pars, model=model)
        vd = velocity(gg, pars, 0, ds[:, None])
        dvds, flags = analytic_dv(model, gg, pars, 0, ds, vd)
        fatal = [f for f in flags if f not in ("negative_catalog_gas_squared", "negative_catalog_bulge_squared", "zero_stellar_template")]
        if fatal:
            raise RuntimeError(model + " synthetic flags " + ",".join(fatal))
        specs = {
            "upsilon": ("U", U),
            "epsilon": ("eps", eps),
            "ln_q": ("lq", np.log(q)),
            "log10_a0": ("log10a0", pars.get("log10a0")),
        }
        for name, dv in dvds.items():
            key, value = specs[name]
            pars_p = dict(pars)
            pars_m = dict(pars)
            if key == "U":
                pars_p["U"] = np.array([value + h])
                pars_m["U"] = np.array([value - h])
            else:
                pars_p[key] = value + h
                pars_m[key] = value - h
            vp = velocity(gg, pars_p, 0, ds[:, None])
            vm = velocity(gg, pars_m, 0, ds[:, None])
            fd = (vp - vm) / (2 * h)
            gap = rel(dv, fd)
            rows.append({"model": model, "coordinate": name, "max_rel": gap})
            if gap > 1e-6:
                raise RuntimeError(model + " " + name + " synthetic dv gap " + str(gap))
        # q does not change the Newtonian piece.
        if model == "ptq-screen":
            y = np.array([0.2, 1.0, 3.0])
            nu, dnu_dy, dnu_dlnq = _nu_derivatives(y, "screen", q)
            hq = 1e-6
            num = (nu_q(y, q * np.exp(hq)) - nu_q(y, q * np.exp(-hq))) / (2 * hq)
            if rel(dnu_dlnq, num) > 1e-6:
                raise RuntimeError("ln q chain failed")
            hy = 1e-6
            numy = (nu_q(y + hy, q) - nu_q(y - hy, q)) / (2 * hy)
            if rel(dnu_dy, numy) > 1e-6:
                raise RuntimeError("screen dnu/dy failed")
    # Confirm the additive linear term matches the catalog helper.
    lin = linear_term_kms2(eps, r, H0_si=H0)
    manual = eps * C_LIGHT * H0 * (r * KPC) / (KM ** 2)
    if rel(lin, manual) > 0:
        raise RuntimeError("linear term units")
    return rows
