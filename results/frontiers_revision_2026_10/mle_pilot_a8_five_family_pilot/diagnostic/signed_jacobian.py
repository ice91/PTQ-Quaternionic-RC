"""Isolated analytic Jacobian for the A5 signed-SPARC B0 objective.

The variance token remains the existing B0 implementation token.
The scientific target name is stored separately and is not canonical.
Unsupported coordinates are omitted, never replaced by zero.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
from scipy.special import logsumexp

from ptquat.constants import C_LIGHT, G_SI, H0_SI, KM, KPC, MSUN
from ptquat.marginal_likelihood import PRIMARY_B0, integrate, nuisance_nodes, velocity
from ptquat.models import (
    c_powerlaw,
    model_v_baryon,
    model_v_mond,
    model_v_mond_screen,
    model_v_nfw1p,
    model_v_ptq,
    model_v_ptq_nu,
    model_v_ptq_screen,
    nfw_v_kms_M200_c,
    nu_q,
    signed_square,
    vbar_squared_kms2,
)

SCIENTIFIC_TARGET = "REV26-P1E-UNIFORM-EXPLICIT-B0-SIGNED-SPARC-CANDIDATE-v2"
VARIANCE_TOKEN = PRIMARY_B0
IDENTITY_LIMIT = 1e-9
# Engineering consistency only. Distinct from gtol=1e-6.
CONSISTENT_ABS_FLOOR = 1e-4
CONSISTENT_REL = 1e-3
LN10 = math.log(10.0)


def as_g(g):
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
    return radius, gg.disk * root, gg.bulge * root, gg.gas * root


def _nu_derivatives(y, kind, q):
    if kind == "simple":
        root = np.sqrt(0.25 + 1.0 / y)
        nu = 0.5 + root
        dnu_dy = -0.5 / (root * y ** 2)
        return nu, dnu_dy, None
    logy = np.log(y)
    yq = np.exp((-float(q)) * logy)
    root = np.sqrt(0.25 + yq)
    nu = 0.5 + root
    dnu_dy = (-0.5 * float(q) * np.exp((-(float(q) + 1.0)) * logy)) / root
    dnu_dq = (-0.5 * logy * yq) / root
    return nu, dnu_dy, float(q) * dnu_dq


def halo_speed_and_mass_derivative(r_kpc, log10_m, h0_si=H0_SI):
    """Halo speed and d(v_halo^2)/d log10(M200), through c(M)=10*(M/1e12)^(-0.1)."""
    m_sun = 10.0 ** float(log10_m)
    c = c_powerlaw(m_sun)
    speed = nfw_v_kms_M200_c(np.asarray(r_kpc, float), m_sun, c, H0_si=h0_si)
    flags = []
    if c <= 1e-6:
        flags.append("concentration_floor")
    m_kg = m_sun * MSUN
    rho_crit = 3.0 * h0_si ** 2 / (8.0 * math.pi * G_SI)
    r200 = (3.0 * m_kg / (4.0 * math.pi * 200.0 * rho_crit)) ** (1.0 / 3.0)
    rs = r200 / c
    x = (np.asarray(r_kpc, float) * KPC) / rs
    g_enc = np.log1p(x) - x / (1.0 + x)
    gc = math.log1p(c) - c / (1.0 + c)
    if gc <= 1e-20:
        flags.append("gc_floor")
    if np.any(g_enc <= 0) or not np.isfinite(speed).all() or np.any(speed <= 0):
        flags.append("nonpositive_halo_speed")
    dc = c * (-0.1) * LN10
    dln_rs = LN10 / 3.0 - dc / c
    dx = x * (-dln_rs)
    dg = (x / (1.0 + x) ** 2) * dx
    dgc = (c / (1.0 + c) ** 2) * dc
    dln_mr = LN10 + dg / g_enc - dgc / gc
    v2 = np.asarray(speed, float) ** 2
    return speed, v2 * dln_mr, flags


def analytic_dv(model, gg, pars, gi, ds, vd):
    radius, disk, bulge, gas = _scaled(gg, ds)
    upsilon = float(pars["U"][gi])
    stellar = signed_square(disk) + signed_square(bulge)
    vn2 = vbar_squared_kms2(upsilon, disk, bulge, gas)
    flags = []
    if np.any(vd <= 0) or not np.isfinite(vd).all():
        flags.append("nonpositive_velocity")
    if np.any(gas < 0):
        flags.append("negative_catalog_gas_signed")
    if np.any(disk < 0) or np.any(bulge < 0):
        flags.append("negative_stellar_component")
    dvds = {}

    def check(formula):
        scale = max(1.0, float(np.max(np.abs(vd))))
        if np.max(np.abs(formula - vd)) > 1e-8 * scale:
            flags.append("velocity_formula_mismatch")

    if model == "baryon":
        check(model_v_baryon(upsilon, radius, disk, bulge, gas))
        dvds["upsilon"] = stellar / (2.0 * vd)
        return dvds, flags
    if model == "ptq":
        eps = float(pars["eps"])
        check(model_v_ptq(upsilon, eps, radius, disk, bulge, gas, H0_si=float(pars["_H0"])))
        additive = (C_LIGHT * float(pars["_H0"]) * (radius * KPC)) / (KM ** 2)
        dvds["epsilon"] = additive / (2.0 * vd)
        dvds["upsilon"] = stellar / (2.0 * vd)
        return dvds, flags
    if model == "nfw1p":
        logm = float(pars["lM"][gi])
        check(model_v_nfw1p(
            upsilon, logm, radius, disk, bulge, gas,
            H0_si=float(pars["_H0"]), c0=float(pars["_c0"]), c_slope=float(pars["_c_slope"]),
        ))
        _, dvhalo2, halo_flags = halo_speed_and_mass_derivative(radius, logm, float(pars["_H0"]))
        flags.extend(halo_flags)
        dvds["upsilon"] = stellar / (2.0 * vd)
        dvds["log10_M200"] = dvhalo2 / (2.0 * vd)
        return dvds, flags

    h0 = float(pars["_H0"])
    q = None
    if model == "mond":
        a0_raw = 10.0 ** float(pars["log10a0"])
        check(model_v_mond(upsilon, a0_raw, radius, disk, bulge, gas))
        kind = "simple"
        native = "log10_a0"
        a0_floor = 1e-16
    elif model == "ptq-nu":
        eps = float(pars["eps"])
        a0_raw = eps * C_LIGHT * h0
        check(model_v_ptq_nu(upsilon, eps, radius, disk, bulge, gas, H0_si=h0))
        kind = "simple"
        native = "epsilon"
        a0_floor = 1e-30
    elif model == "ptq-screen":
        eps = float(pars["eps"])
        q = float(math.exp(pars["lq"]))
        a0_raw = eps * C_LIGHT * h0
        check(model_v_ptq_screen(upsilon, eps, q, radius, disk, bulge, gas, H0_si=h0))
        kind = "screen"
        native = "epsilon"
        a0_floor = 1e-30
    elif model == "mond-screen":
        a0_raw = 10.0 ** float(pars["log10a0"])
        q = float(math.exp(pars["lq"]))
        check(model_v_mond_screen(upsilon, a0_raw, q, radius, disk, bulge, gas))
        kind = "screen"
        native = "log10_a0"
        a0_floor = 1e-30
    else:
        raise ValueError(model)
    if a0_raw <= a0_floor:
        flags.append("a0_floor")
    a0_used = max(float(a0_raw), a0_floor)
    r_m = radius * KPC
    if np.any(r_m <= 1e-12):
        flags.append("radius_floor")
    g_n = (vn2 * (KM ** 2)) / np.maximum(r_m, 1e-12)
    y_raw = g_n / a0_used
    if np.any(y_raw < 1e-12):
        flags.append("y_floor")
    if np.any(y_raw <= 0):
        flags.append("nonpositive_interpolation_argument")
    y = np.maximum(y_raw, 1e-12)
    nu, dnu_dy, dnu_dlnq = _nu_derivatives(y, kind, q)
    if kind == "screen":
        nu_ref = nu_q(y, q)
        if np.max(np.abs(nu - nu_ref)) > 1e-8 * max(1.0, float(np.max(np.abs(nu_ref)))):
            flags.append("nu_q_formula_mismatch")
        if not np.isfinite(nu).all() or not np.isfinite(dnu_dlnq).all():
            flags.append("q_power_nonfinite")
    unclipped = y_raw >= 1e-12
    dy_dvn2 = np.where(unclipped & (vn2 > 0), y / np.maximum(vn2, 1e-300), 0.0)
    if np.any(unclipped & (vn2 <= 0)):
        flags.append("nonpositive_vN2_off_floor")
    dv2_dvn2 = nu + vn2 * dnu_dy * dy_dvn2
    dvds["upsilon"] = dv2_dvn2 * stellar / (2.0 * vd)
    if native == "epsilon":
        da0 = 0.0 if a0_raw <= a0_floor else C_LIGHT * h0
        dy = np.where(unclipped, -(y / a0_used) * da0, 0.0)
        dvds["epsilon"] = (vn2 * dnu_dy * dy) / (2.0 * vd)
    else:
        da0 = 0.0 if a0_raw <= a0_floor else a0_used * LN10
        dy = np.where(unclipped, -(y / a0_used) * da0, 0.0)
        dvds["log10_a0"] = (vn2 * dnu_dy * dy) / (2.0 * vd)
    if kind == "screen":
        dvds["ln_q"] = (vn2 * dnu_dlnq) / (2.0 * vd)
    return dvds, flags


def _accumulate(gg, vd, dvds, sigma, ds, wd, incs, wi):
    b = np.sin(incs) / np.sin(gg.inc)
    var = gg.e ** 2 + sigma ** 2
    inv = 1.0 / var
    inv2 = inv / var
    norm = float(np.sum(np.log(2.0 * math.pi * var)))
    y2 = float(np.sum((gg.y ** 2) * inv))
    y2q = float(np.sum((gg.y ** 2) * inv2))
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
        quad = y2 - 2.0 * b[None, :] * cross[:, None] + (b ** 2)[None, :] * vv[:, None]
        lls.append(-0.5 * (quad + norm))
        crossq = v @ (gg.y * inv2)
        vvq = (v * v) @ inv2
        resid = y2q - 2.0 * b[None, :] * crossq[:, None] + (b ** 2)[None, :] * vvq[:, None]
        blocks["ln_sigma_sys"].append(sigma * (resid - sum_inv))
        logw.append(np.log(wd[start:end])[:, None] + np.log(wi)[None, :])
        for name in names:
            dv = dvds[name][start:end]
            cy = dv @ (gg.y * inv)
            cv = (dv * v) @ inv
            blocks[name].append(b[None, :] * cy[:, None] - (b ** 2)[None, :] * cv[:, None])
    ll = np.concatenate([a.ravel() for a in lls])
    lw = np.concatenate([a.ravel() for a in logw])
    score = ll + lw
    logl = float(logsumexp(score))
    posterior = np.exp(score - logl)
    out = {}
    for name, parts in blocks.items():
        flat = np.concatenate([a.ravel() for a in parts])
        value = float(np.sum(posterior * flat))
        if name == "ln_sigma_sys":
            value = sigma * value
        out[name] = value
    return logl, out


FATAL = {
    "nonpositive_velocity", "velocity_formula_mismatch", "nu_q_formula_mismatch",
    "q_power_nonfinite", "nonpositive_vN2_off_floor", "nonpositive_halo_speed",
    "negative_stellar_component", "nonpositive_interpolation_argument",
}


def galaxy_jacobian_row(g, pars, gi, model, order=192):
    if order != 192:
        raise ValueError("scientific quadrature order is 192")
    gg = as_g(g)
    pp = dict(pars, model=model, _H0=H0_SI, _c0=10.0, _c_slope=-0.1)
    official = integrate(gg, pp, gi, "segmented_quantile", order, floors=("B0",))
    ds, wd, incs, wi, _, _ = nuisance_nodes(gg, "segmented_quantile", order)
    vd = velocity(gg, pp, gi, ds[:, None])
    dvds, flags = analytic_dv(model, gg, pp, gi, ds, vd)
    rederived, dlogl = _accumulate(gg, vd, dvds, float(pp["sigma"]), ds, wd, incs, wi)
    if any(flag in FATAL for flag in flags):
        dlogl = {key: None for key in dlogl}
        domain = "DERIVATIVE_NOT_VALIDATED"
    elif any(flag in flags for flag in ("y_floor", "a0_floor", "radius_floor", "gc_floor", "concentration_floor")):
        domain = "CLIPPED_OR_NONSMOOTH"
    else:
        domain = "INTERIOR_ANALYTICALLY_DIFFERENTIABLE"
    official_logl = float(official["logL"]["B0"])
    return {
        "galaxy": g.name,
        "gi": int(gi),
        "official_logL": official_logl,
        "rederived_logL": rederived,
        "abs_gap": abs(rederived - official_logl),
        "dlogL": dlogl,
        "flags": flags,
        "domain": domain,
        "scientific_target": SCIENTIFIC_TARGET,
        "variance_token": VARIANCE_TOKEN,
    }
