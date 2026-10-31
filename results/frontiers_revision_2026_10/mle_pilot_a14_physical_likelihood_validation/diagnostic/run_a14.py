"""Fixed-parameter distance and inclination validation. No optimizer."""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

NEW = Path("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/mle_pilot_a14_physical_likelihood_validation")
A10 = Path("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/mle_pilot_a10_four_family_science_pilot")
IMPL = Path("/home/rocky/PTQ-Quaternionic-RC")
import sys
sys.path.insert(0, str(A10 / "candidate/src"))

from ptquat.constants import C_LIGHT, G_SI, H0_SI, KM, KPC, MSUN
from ptquat.data import load_tidy_sparc
from ptquat.marginal_likelihood import PRIMARY_B0, galaxy_log_marginal
from ptquat.models import (
    catalog_velocity_at_distance,
    c_powerlaw,
    model_v_baryon,
    model_v_nfw1p,
    model_v_ptq,
    model_v_ptq_nu,
    nfw_v_kms_M200_c,
    vbar_squared_kms2,
)

BUDGET_CAP = 1200
T0 = time.perf_counter()
MARGINAL_CALLS = 0
MODELS = ("baryon", "ptq-nu", "ptq", "nfw1p")
THETA = {
    "baryon": A10 / "state/baryon/accepted_3_theta.npy",
    "ptq-nu": A10 / "state/ptq-nu/accepted_7_theta.npy",
    "ptq": A10 / "state/ptq/accepted_8_theta.npy",
    "nfw1p": A10 / "state/nfw1p/accepted_8_theta.npy",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unpack(theta, model):
    theta = np.asarray(theta, float)
    slot = 0
    out = {"model": model}
    if model in ("ptq", "ptq-nu"):
        out["eps"] = float(theta[slot]); slot += 1
    out["U"] = np.asarray(theta[slot:slot + 91], float); slot += 91
    if model == "nfw1p":
        out["lM"] = np.asarray(theta[slot:slot + 91], float); slot += 91
    out["sigma"] = float(math.exp(theta[slot])); slot += 1
    if slot != theta.size:
        raise RuntimeError(f"theta length {theta.size} != {slot} for {model}")
    return out


def call_model(model, upsilon, radius, disk, bulge, gas, pars, gi):
    if model == "baryon":
        return model_v_baryon(upsilon, radius, disk, bulge, gas)
    if model == "ptq":
        return model_v_ptq(upsilon, pars["eps"], radius, disk, bulge, gas, H0_si=H0_SI)
    if model == "ptq-nu":
        return model_v_ptq_nu(upsilon, pars["eps"], radius, disk, bulge, gas, H0_si=H0_SI)
    if model == "nfw1p":
        return model_v_nfw1p(
            upsilon, float(pars["lM"][gi]), radius, disk, bulge, gas,
            H0_si=H0_SI, c0=10.0, c_slope=-0.1,
        )
    raise ValueError(model)


def velocity_transformed(galaxy, pars, gi, model, distance, mode):
    scale = float(distance) / float(galaxy.D_Mpc)
    if scale <= 0:
        raise ValueError("nonpositive distance scale")
    radius = galaxy.r_kpc * scale
    if mode == "full":
        gg = SimpleNamespace(
            D=galaxy.D_Mpc, r=galaxy.r_kpc, disk=galaxy.v_disk,
            bulge=galaxy.v_bulge, gas=galaxy.v_gas,
        )
        return np.asarray(catalog_velocity_at_distance(
            gg, pars, gi, model, distance, H0_si=H0_SI, c0=10.0, c_slope=-0.1,
        ), float)
    if mode == "radial_only":
        return np.asarray(call_model(
            model, float(pars["U"][gi]), radius, galaxy.v_disk, galaxy.v_bulge, galaxy.v_gas, pars, gi,
        ), float)
    raise ValueError(mode)


def fd_distance(galaxy, pars, gi, model, mode, relative_step):
    d0 = float(galaxy.D_Mpc)
    h = relative_step * d0
    vp = velocity_transformed(galaxy, pars, gi, model, d0 + h, mode)
    vm = velocity_transformed(galaxy, pars, gi, model, d0 - h, mode)
    return (vp - vm) / (2.0 * h)


def chain_rule(galaxy, pars, gi, model):
    """Partial numerical chain rule at the nominal catalog distance."""
    u = float(pars["U"][gi])
    r = np.asarray(galaxy.r_kpc, float)
    disk = np.asarray(galaxy.v_disk, float)
    bulge = np.asarray(galaxy.v_bulge, float)
    gas = np.asarray(galaxy.v_gas, float)
    base = call_model(model, u, r, disk, bulge, gas, pars, gi)
    hr = 1e-6 * np.maximum(r, 1e-8)
    dv_dr = (
        call_model(model, u, r + hr, disk, bulge, gas, pars, gi)
        - call_model(model, u, r - hr, disk, bulge, gas, pars, gi)
    ) / (2.0 * hr)
    pieces = {"radius": dv_dr * (r / galaxy.D_Mpc)}
    total = pieces["radius"].copy()
    for name, comp in (("disk", disk), ("bulge", bulge), ("gas", gas)):
        step = 1e-5 * np.maximum(np.abs(comp), 1.0)
        up = comp + step
        dn = comp - step
        if name == "disk":
            dv = call_model(model, u, r, up, bulge, gas, pars, gi) - call_model(model, u, r, dn, bulge, gas, pars, gi)
        elif name == "bulge":
            dv = call_model(model, u, r, disk, up, gas, pars, gi) - call_model(model, u, r, disk, dn, gas, pars, gi)
        else:
            dv = call_model(model, u, r, disk, bulge, up, pars, gi) - call_model(model, u, r, disk, bulge, dn, pars, gi)
        partial = dv / (2.0 * step)
        term = partial * (comp / (2.0 * galaxy.D_Mpc))
        pieces[name] = term
        total = total + term
    return base, total, pieces


def nfw_eta_derivative(galaxy, pars, gi):
    u = float(pars["U"][gi])
    r = np.asarray(galaxy.r_kpc, float)
    vbar2 = vbar_squared_kms2(u, galaxy.v_disk, galaxy.v_bulge, galaxy.v_gas)
    m_sun = 10.0 ** float(pars["lM"][gi])
    c = c_powerlaw(m_sun, c0=10.0, beta=-0.1)
    vhalo = nfw_v_kms_M200_c(r, m_sun, c, H0_si=H0_SI)
    rho = 3.0 * H0_SI ** 2 / (8.0 * math.pi * G_SI)
    r200 = (3.0 * m_sun * MSUN / (4.0 * math.pi * 200.0 * rho)) ** (1.0 / 3.0)
    rs = r200 / c
    x = (r * KPC) / rs
    f = np.log1p(x) - x / (1.0 + x)
    eta = (x ** 2) / ((1.0 + x) ** 2 * f) - 1.0
    v = np.sqrt(np.maximum(vbar2 + vhalo ** 2, 0.0))
    jd = (vbar2 + eta * (vhalo ** 2)) / (2.0 * v * galaxy.D_Mpc)
    return v, jd


def max_abs(a, b, mask):
    diff = np.abs(np.asarray(a, float) - np.asarray(b, float))
    diff = diff[mask]
    if diff.size == 0:
        return None
    return float(np.max(diff))


def rel_max(reference, diff, mask):
    scale = np.maximum(1.0, np.abs(np.asarray(reference, float)))
    value = np.abs(np.asarray(diff, float)) / scale
    value = value[mask]
    if value.size == 0:
        return None
    return float(np.max(value))


def gaussian_pair(y, e, sigma, mu, b):
    var = e * e + sigma * sigma
    resid_a = y - b * mu
    log_a = -0.5 * np.sum(resid_a ** 2 / var + np.log(2.0 * math.pi * var))
    var_b = var / (b * b)
    resid_b = y / b - mu
    log_b = -0.5 * np.sum(resid_b ** 2 / var_b + np.log(2.0 * math.pi * var_b))
    # Inconsistent control: data-side residual with the catalog variance.
    log_inconsistent = -0.5 * np.sum(resid_b ** 2 / var + np.log(2.0 * math.pi * var))
    return {
        "logL_catalog_frame": float(log_a),
        "logL_corrected_frame_scaled_variance": float(log_b),
        "normalization_gap_logL_corrected_minus_catalog": float(log_b - log_a),
        "expected_n_log_abs_b": float(y.size * math.log(abs(b))),
        "normalization_identity_abs": float(abs((log_b - log_a) - y.size * math.log(abs(b)))),
        "logL_inconsistent_uncorrected_variance": float(log_inconsistent),
    }


def main():
    global MARGINAL_CALLS
    selected_path = NEW / "SELECTED_GALAXIES.json"
    selected = json.loads(selected_path.read_text())
    if selected.get("likelihood_evaluated"):
        raise SystemExit("selection file already marked evaluated; refusing to reuse a post-hoc sample")
    filtered = A10 / "logs/s_domain13_filtered_tidy.csv"
    loaded = load_tidy_sparc(str(filtered))
    galaxies = [loaded[name] for name in sorted(loaded)]
    if len(galaxies) != 91 or sum(len(g.v_obs) for g in galaxies) != 1769:
        raise SystemExit("sample identity failed")
    thetas = {}
    theta_meta = {}
    for model, path in THETA.items():
        theta = np.load(path)
        thetas[model] = unpack(theta, model)
        theta_meta[model] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "length": int(theta.size),
            "label": "FIXED_NONSTATIONARY_TEST_VECTOR",
        }
    distance_rows = []
    inclination_rows = []
    marginal_rows = []
    curve_rows = []
    for item in selected["selected"]:
        galaxy = galaxies[item["sorted_index"]]
        if galaxy.name != item["galaxy"]:
            raise SystemExit("index mismatch " + galaxy.name)
        for model in MODELS:
            pars = thetas[model]
            base, chain, pieces = chain_rule(galaxy, pars, item["sorted_index"], model)
            positive = np.isfinite(base) & (base > 0)
            fd = {step: fd_distance(galaxy, pars, item["sorted_index"], model, "full", step) for step in (1e-3, 1e-4, 1e-5)}
            fd_radial = fd_distance(galaxy, pars, item["sorted_index"], model, "radial_only", 1e-4)
            scaling = base / (2.0 * galaxy.D_Mpc)
            record = {
                "galaxy": galaxy.name,
                "role": item["role"],
                "model": model,
                "n_positive_speed": int(positive.sum()),
                "n_radii": int(base.size),
                "upsilon": float(pars["U"][item["sorted_index"]]),
                "max_abs_fd_full_minus_chain_1e-4": max_abs(fd[1e-4], chain, positive),
                "max_rel_fd_full_minus_chain_1e-4": rel_max(chain, fd[1e-4] - chain, positive),
                "max_abs_fd_1e-3_minus_1e-4": max_abs(fd[1e-3], fd[1e-4], positive),
                "max_abs_fd_1e-4_minus_1e-5": max_abs(fd[1e-4], fd[1e-5], positive),
                "max_abs_fd_full_minus_half_scaling": max_abs(fd[1e-4], scaling, positive),
                "max_abs_radial_only_minus_radius_piece": max_abs(fd_radial, pieces["radius"], positive),
                "max_abs_full_minus_radial_only": max_abs(fd[1e-4], fd_radial, positive),
                "median_abs_template_piece": float(np.median(np.abs(
                    pieces["disk"][positive] + pieces["bulge"][positive] + pieces["gas"][positive]
                ))) if positive.any() else None,
            }
            if model == "nfw1p":
                _, eta_jd = nfw_eta_derivative(galaxy, pars, item["sorted_index"])
                record["max_abs_fd_full_minus_nfw_eta"] = max_abs(fd[1e-4], eta_jd, positive)
            if "eps" in pars:
                record["epsilon"] = pars["eps"]
            if model == "nfw1p":
                record["log10_M200"] = float(pars["lM"][item["sorted_index"]])
            distance_rows.append(record)
            if model == "baryon":
                for i, radius in enumerate(galaxy.r_kpc):
                    curve_rows.append({
                        "galaxy": galaxy.name,
                        "r_kpc": float(radius),
                        "v_obs": float(galaxy.v_obs[i]),
                        "v_err": float(galaxy.v_err[i]),
                        "dV_dD_full": float(fd[1e-4][i]),
                        "dV_dD_radial_only": float(fd_radial[i]),
                        "dV_dD_half_scaling": float(scaling[i]),
                        "v_baryon": float(base[i]),
                    })
        # Inclination identities at baryon and ptq-nu nominal speeds.
        for model in ("baryon", "ptq-nu", "ptq"):
            pars = thetas[model]
            mu = velocity_transformed(galaxy, pars, item["sorted_index"], model, galaxy.D_Mpc, "full")
            i0 = math.radians(float(galaxy.i_deg))
            for delta_deg in (1.0, -2.0):
                ip = math.radians(float(galaxy.i_deg) + delta_deg)
                if not 0.0 < ip < math.pi / 2.0:
                    inclination_rows.append({
                        "galaxy": galaxy.name, "model": model, "delta_deg": delta_deg,
                        "status": "SKIPPED_OUTSIDE_OPEN_INCLINATION_INTERVAL",
                    })
                    continue
                b = math.sin(ip) / math.sin(i0)
                stats = gaussian_pair(galaxy.v_obs, galaxy.v_err, pars["sigma"], mu, b)
                # Derivative of catalog mean at nominal inclination.
                db = 1e-6
                bp = math.sin(i0 + db) / math.sin(i0)
                bm = math.sin(i0 - db) / math.sin(i0)
                numerical_di = ((bp - bm) * mu) / (2.0 * db)
                analytic_di = mu * math.cos(i0) / math.sin(i0)
                stats.update({
                    "galaxy": galaxy.name,
                    "model": model,
                    "delta_deg": delta_deg,
                    "b": b,
                    "status": "EVALUATED",
                    "max_abs_inclination_derivative_gap": float(np.max(np.abs(numerical_di - analytic_di))),
                })
                inclination_rows.append(stats)
        # Official marginals. NFW at i=90 may fail; record and continue.
        for model in MODELS:
            if time.perf_counter() - T0 > 25 * 60:
                raise SystemExit("wall budget")
            if MARGINAL_CALLS >= BUDGET_CAP:
                raise SystemExit("marginal budget")
            pars = dict(thetas[model])
            try:
                MARGINAL_CALLS += 1
                record = galaxy_log_marginal(
                    galaxy, pars, item["sorted_index"], model,
                    likelihood_version=PRIMARY_B0, order=192,
                )
                marginal_rows.append({
                    "galaxy": galaxy.name,
                    "model": model,
                    "status": "FINITE",
                    "logL_B0_fine192": float(record["logL"]["B0"]),
                    "order": record["order"],
                    "likelihood_version": record["likelihood_version"],
                })
            except Exception as exc:
                marginal_rows.append({
                    "galaxy": galaxy.name,
                    "model": model,
                    "status": "FAILED",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
    # Counterfactual control on one distance finite difference, baryon only.
    counter = []
    for item in selected["selected"]:
        galaxy = galaxies[item["sorted_index"]]
        pars = thetas["baryon"]
        d0 = float(galaxy.D_Mpc)
        h = 1e-4 * d0
        def resid(distance, scale_obs):
            s = distance / d0
            v = velocity_transformed(galaxy, pars, item["sorted_index"], "baryon", distance, "full")
            y = galaxy.v_obs * s if scale_obs else galaxy.v_obs
            return y - v
        d_standard = (resid(d0 + h, False) - resid(d0 - h, False)) / (2.0 * h)
        d_counter = (resid(d0 + h, True) - resid(d0 - h, True)) / (2.0 * h)
        counter.append({
            "galaxy": galaxy.name,
            "label": "COUNTERFACTUAL_SENSITIVITY_CONTROL",
            "max_abs_gap_between_standard_and_vobs_times_s": float(np.max(np.abs(d_standard - d_counter))),
        })
    payload = {
        "optimizer_calls": 0,
        "marginal_evaluations": MARGINAL_CALLS,
        "wall_seconds": time.perf_counter() - T0,
        "theta": theta_meta,
        "distance": distance_rows,
        "inclination": inclination_rows,
        "marginal": marginal_rows,
        "counterfactual_vobs_times_s": counter,
        "curves_baryon_derivative": curve_rows,
    }
    (NEW / "logs/fixed_parameter_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    selected["likelihood_evaluated"] = True
    selected["evaluation_note"] = "Set only after the predeclared names were read back. The selected list was not edited."
    selected_path.write_text(json.dumps(selected, indent=2) + "\n")
    print("marginals", MARGINAL_CALLS, "wall", round(payload["wall_seconds"], 3))
    for row in distance_rows:
        print(row["galaxy"], row["model"], "chain", row["max_abs_fd_full_minus_chain_1e-4"], "half", row["max_abs_fd_full_minus_half_scaling"], "eta", row.get("max_abs_fd_full_minus_nfw_eta"))
    for row in marginal_rows:
        print("L", row["galaxy"], row["model"], row["status"], row.get("logL_B0_fine192"), row.get("error_type"))


if __name__ == "__main__":
    main()
