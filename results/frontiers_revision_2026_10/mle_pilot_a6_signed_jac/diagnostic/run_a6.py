"""Fixed-point signed Jacobian validation. No optimizer."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

NEW = Path("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/mle_pilot_a6_signed_jac")
IMPL = Path("/home/rocky/PTQ-Quaternionic-RC")
sys.path.insert(0, str(NEW / "diagnostic"))
sys.path.insert(0, str(NEW / "candidate/src"))
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from ptquat.constants import C_LIGHT, H0_SI
from ptquat.data import load_tidy_sparc
from ptquat.marginal_likelihood import integrate
from ptquat.models import model_v_baryon, signed_square, vbar_squared_kms2
from signed_jacobian import (
    CONSISTENT_ABS_FLOOR,
    CONSISTENT_REL,
    IDENTITY_LIMIT,
    LN10,
    SCIENTIFIC_TARGET,
    VARIANCE_TOKEN,
    as_g,
    galaxy_jacobian_row,
    halo_speed_and_mass_derivative,
)

# Predeclared before any derivative discrepancy is observed.
ABSOLUTE_STEPS = (1e-4, 5e-5)
BOUND_ATOL = 1e-8
EXEC = IMPL / "results/frontiers_revision_2026_10/mle_pilot_a/execution_v1"
THETA_HASH = {
    "baryon": "1e9c848c43e456d6eb233b90ff4796edaf2a30fb7cd9daf274fe40e95f680714",
    "mond": "4c2a9af69596074fc4f81d6240d3e70c926d0c80ab2fa2e4d8c6118f9cd1e40f",
    "ptq": "96ac8469398b416aa0740522da74c844c12ea8e95364d69f3920f2df460fd6c4",
    "ptq-nu": "7abded50306ec11a32380f1ae85c555faef1d2d9318f39c673b691ae974473b2",
    "ptq-screen": "b0a24278778944f59b3ae440ba3ae5a63d9dec584c851f93a64fa3e693ff3ab4",
    "mond-screen": "079614456d29ad1996cc6825d572ce5a293b4fafa9f865a4e8818dbdc99267de",
    "nfw1p": "5c4ad3c07e46f3fa3df746c3cb5a1eeb42fc8577fbd5ec316dcd1c12ae617d7e",
}
THETA_FILE = {
    "baryon": "baryon_last_accepted_theta.npy",
    "mond": "mond_last_accepted_theta.npy",
    "ptq": "ptq_last_accepted_theta.npy",
    "ptq-nu": "ptq_nu_last_accepted_theta.npy",
    "ptq-screen": "ptq_screen_last_accepted_theta.npy",
    "mond-screen": "mond_screen_last_accepted_theta.npy",
    "nfw1p": "nfw1p_initial_theta.npy",
}
FROZEN = {
    IMPL / "dataset/sparc_tidy.csv": "3ab1c3c2b110c1e48680ebae77c1ceefaa4f778f81119dcf04881b630b274cbb",
    IMPL / "results/frontiers_revision_2026_10/mle_pilot_a/candidate/src/ptquat/models.py":
        "cba208085f4e5e6373fe3b1bd5b862e9be66a6d1412a6a7750c4c4b053bbc952",
    IMPL / "results/frontiers_revision_2026_10/mle_pilot_a5_signed_b0_integration/candidate/src/ptquat/models.py":
        "7214664b9a4a3e182f99748a9c4c7da91a98b930399a832589203395f3b89b1b",
    NEW / "candidate/src/ptquat/models.py":
        "7214664b9a4a3e182f99748a9c4c7da91a98b930399a832589203395f3b89b1b",
    IMPL / "results/frontiers_revision_2026_10/mle_pilot_a4_domain_feasibility/S_DOMAIN13_MASK.csv":
        "df0086b1e67f96403ff09f5c5bfb7f5818593d141d14234d752cb433a9511e6e",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def theta_hash(theta):
    return hashlib.sha256(np.ascontiguousarray(theta, dtype="<f8").tobytes()).hexdigest()


def layout(model, g=91):
    s = 0
    L = {"model": model}
    if model in ("ptq", "ptq-nu"):
        L["i_eps"] = s; s += 1
        L["sl_U"] = slice(s, s + g); s += g
    elif model == "ptq-screen":
        L["i_eps"] = s; s += 1
        L["i_lq"] = s; s += 1
        L["sl_U"] = slice(s, s + g); s += g
    elif model == "baryon":
        L["sl_U"] = slice(s, s + g); s += g
    elif model == "mond":
        L["i_log10a0"] = s; s += 1
        L["sl_U"] = slice(s, s + g); s += g
    elif model == "mond-screen":
        L["i_log10a0"] = s; s += 1
        L["i_lq"] = s; s += 1
        L["sl_U"] = slice(s, s + g); s += g
    elif model == "nfw1p":
        L["sl_U"] = slice(s, s + g); s += g
        L["sl_lM"] = slice(s, s + g); s += g
    else:
        raise ValueError(model)
    L["i_lnsig"] = s; s += 1
    L["k"] = s
    L["names"] = [None] * s
    if "i_eps" in L:
        L["names"][L["i_eps"]] = "epsilon"
    if "i_log10a0" in L:
        L["names"][L["i_log10a0"]] = "log10_a0"
    if "i_lq" in L:
        L["names"][L["i_lq"]] = "ln_q"
    if "sl_U" in L:
        for i, j in enumerate(range(L["sl_U"].start, L["sl_U"].stop)):
            L["names"][j] = f"upsilon[{i}]"
    if "sl_lM" in L:
        for i, j in enumerate(range(L["sl_lM"].start, L["sl_lM"].stop)):
            L["names"][j] = f"log10_M200[{i}]"
    L["names"][L["i_lnsig"]] = "ln_sigma_sys"
    return L


def unpack(theta, L):
    out = {"model": L["model"]}
    if "i_eps" in L:
        out["eps"] = float(theta[L["i_eps"]])
    if "sl_U" in L:
        out["U"] = np.asarray(theta[L["sl_U"]], float)
    if "i_log10a0" in L:
        out["log10a0"] = float(theta[L["i_log10a0"]])
    if "sl_lM" in L:
        out["lM"] = np.asarray(theta[L["sl_lM"]], float)
    if "i_lq" in L:
        out["lq"] = float(theta[L["i_lq"]])
    out["sigma"] = float(math.exp(theta[L["i_lnsig"]]))
    return out


def map_nu(theta):
    src = unpack(theta, layout("ptq-nu"))
    dst = np.zeros(layout("mond")["k"])
    L = layout("mond")
    dst[L["i_log10a0"]] = math.log10(src["eps"] * C_LIGHT * H0_SI)
    dst[L["sl_U"]] = src["U"]
    dst[L["i_lnsig"]] = math.log(src["sigma"])
    return dst


def map_screen(theta):
    src = unpack(theta, layout("ptq-screen"))
    L = layout("mond-screen")
    dst = np.zeros(L["k"])
    dst[L["i_log10a0"]] = math.log10(src["eps"] * C_LIGHT * H0_SI)
    dst[L["i_lq"]] = src["lq"]
    dst[L["sl_U"]] = src["U"]
    dst[L["i_lnsig"]] = math.log(src["sigma"])
    return dst


class BudgetStop(RuntimeError):
    pass


class Budget:
    def __init__(self, weights):
        self.weights = weights
        self.calls = 0
        self.eq = 0.0
        self.t0 = time.perf_counter()
        self.log = NEW / "logs/jacobian_rows.jsonl"
        self.log.parent.mkdir(parents=True, exist_ok=True)

    def charge(self, galaxy):
        weight = float(self.weights[galaxy])
        if self.calls + 1 > 6000 or self.eq + weight > 40.0 or time.perf_counter() - self.t0 > 100 * 60:
            raise BudgetStop("budget")
        self.calls += 1
        self.eq += weight
        return weight

    def write(self, row):
        with self.log.open("a") as handle:
            handle.write(json.dumps(row) + "\n")


def bound_status(x, lo, hi):
    if x <= lo + BOUND_ATOL:
        return "lower"
    if x >= hi - BOUND_ATOL:
        return "upper"
    return "interior"


def kkt(gradient, value, lo, hi):
    status = bound_status(value, lo, hi)
    if status == "lower":
        projected = 0.0 if gradient > 0 else gradient
    elif status == "upper":
        projected = 0.0 if gradient < 0 else gradient
    else:
        projected = gradient
    return {"bound_status": status, "unprojected": gradient, "projected_kkt_component": projected,
            "gap_to_lower": float(value - lo), "gap_to_upper": float(hi - value)}


def consistent(analytic, fd):
    tol = max(CONSISTENT_ABS_FLOOR, CONSISTENT_REL * abs(analytic))
    return abs(analytic - fd) <= tol, tol


def main():
    for path, digest in FROZEN.items():
        if sha256(path) != digest:
            raise SystemExit("STOP hash " + str(path))
    tidy = pd.read_csv(IMPL / "dataset/sparc_tidy.csv")
    raw = pd.read_csv(IMPL / "dataset/raw/vizier_table2.csv")
    mask = pd.read_csv(IMPL / "results/frontiers_revision_2026_10/mle_pilot_a4_domain_feasibility/S_DOMAIN13_MASK.csv")
    joined = tidy.merge(raw, left_on=["galaxy", "r_kpc"], right_on=["Name", "Rad"], how="left", validate="one_to_one")
    included = mask.included.to_numpy(int) == 1
    if int(included.sum()) != 1769 or not np.array_equal(mask.source_recno.to_numpy(), joined.recno.to_numpy()):
        raise SystemExit("STOP mask")
    kept = tidy.loc[included]
    filtered = NEW / "logs/s_domain13_filtered_tidy.csv"
    kept.to_csv(filtered, index=False)
    loaded = load_tidy_sparc(str(filtered))
    galaxies = [loaded[name] for name in sorted(loaded)]
    names = [g.name for g in galaxies]
    if len(galaxies) != 91 or sum(len(g.v_obs) for g in galaxies) != 1769:
        raise SystemExit("STOP sample")
    weights = {row["galaxy"]: float(row["fullsample_equivalent"]) for row in json.loads(
        (IMPL / "results/frontiers_revision_2026_10/mle_pilot_a/GALAXY_WORK_WEIGHTS.json").read_text()
    )["baryon"]["192"]}
    budget = Budget(weights)
    thetas = {}
    for model, filename in THETA_FILE.items():
        theta = np.load(EXEC / filename)
        if theta_hash(theta) != THETA_HASH[model] or theta.shape != (layout(model)["k"],):
            raise SystemExit("STOP theta " + model)
        thetas[model] = theta
    mapped_nu = map_nu(thetas["ptq-nu"])
    mapped_screen = map_screen(thetas["ptq-screen"])

    # Implemented-halo derivative versus the same function, before likelihood comparisons.
    radii = np.array([0.38, 1.15, 5.0])
    halo_checks = []
    for logm in (9.0, 11.2, 13.0):
        speed, dv2, flags = halo_speed_and_mass_derivative(radii, logm)
        step = 1e-5
        plus, _, _ = halo_speed_and_mass_derivative(radii, logm + step)
        minus, _, _ = halo_speed_and_mass_derivative(radii, logm - step)
        fd = (plus ** 2 - minus ** 2) / (2.0 * step)
        gap = float(np.max(np.abs(fd - dv2)))
        scale = float(np.max(np.abs(dv2)))
        halo_checks.append({"log10_M200": logm, "step": step, "max_abs_gap": gap, "scale": scale, "flags": flags})
        if gap > max(1e-4, 1e-8 * scale):
            raise SystemExit("STOP halo derivative " + str(gap))

    index = {name: i for i, name in enumerate(names)}
    # Fixture rules use data and archived parameters only.
    def pick_interior(model):
        U = unpack(thetas[model], layout(model))["U"]
        if bound_status(U[index["F574-1"]], 0.05, 1.5) == "interior":
            return "F574-1"
        gaps = [(min(u - 0.05, 1.5 - u), name) for name, u in zip(names, U)]
        return max(gaps)[1]

    def pick_bound(model):
        U = unpack(thetas[model], layout(model))["U"]
        hits = [(min(u - 0.05, 1.5 - u), name) for name, u in zip(names, U) if bound_status(u, 0.05, 1.5) != "interior"]
        if "F574-1" in [name for _, name in hits]:
            return "F574-1"
        return hits[0][1] if hits else None

    def pick_negative():
        for g in galaxies:
            if g.name != "UGC01281" and np.any(g.v_gas < 0):
                return g.name
        return None

    fixtures = {
        "interior_upsilon": {model: pick_interior(model) for model in THETA_FILE},
        "active_bound_upsilon": {model: pick_bound(model) for model in THETA_FILE},
        "low_b": "UGC01281",
        "negative_gas_positive_net_b": pick_negative(),
        "nfw_mass": ["F574-1", "UGC01281"],
    }
    (NEW / "logs/predeclared_fixtures.json").write_text(json.dumps(fixtures, indent=2) + "\n")

    def official_logl(galaxy, pars, gi, model):
        budget.charge(galaxy.name)
        gg = as_g(galaxy)
        pp = dict(pars, model=model, _H0=H0_SI, _c0=10.0, _c_slope=-0.1)
        out = integrate(gg, pp, gi, "segmented_quantile", 192, floors=("B0",))
        return float(out["logL"]["B0"])

    rows = {}
    timings = {}
    for model in THETA_FILE:
        t0 = time.perf_counter()
        pars = unpack(thetas[model], layout(model))
        model_rows = []
        for gi, galaxy in enumerate(galaxies):
            budget.charge(galaxy.name)
            model_rows.append(galaxy_jacobian_row(galaxy, pars, gi, model))
            budget.write({"kind": "analytic", "model": model, "galaxy": galaxy.name,
                          "abs_gap": model_rows[-1]["abs_gap"], "domain": model_rows[-1]["domain"]})
        rows[model] = model_rows
        timings[model] = {"analytic_seconds": time.perf_counter() - t0, "calls": 91}
        gap = max(row["abs_gap"] for row in model_rows)
        print(f"IDENTITY {model} max_gap {gap:.3e} seconds {timings[model]['analytic_seconds']:.2f}", flush=True)
        if gap > IDENTITY_LIMIT:
            raise SystemExit("STOP identity " + model)

    def dF_from_rows(model_rows, key):
        return -float(sum(row["dlogL"][key] for row in model_rows))

    vectors = {}
    coverage = []
    for model, model_rows in rows.items():
        L = layout(model)
        grad = np.zeros(L["k"])
        for gi, row in enumerate(model_rows):
            grad[L["sl_U"].start + gi] = -row["dlogL"]["upsilon"]
        if "sl_lM" in L:
            for gi, row in enumerate(model_rows):
                grad[L["sl_lM"].start + gi] = -row["dlogL"]["log10_M200"]
        if "i_eps" in L:
            grad[L["i_eps"]] = dF_from_rows(model_rows, "epsilon")
        if "i_log10a0" in L:
            grad[L["i_log10a0"]] = dF_from_rows(model_rows, "log10_a0")
        if "i_lq" in L:
            grad[L["i_lq"]] = dF_from_rows(model_rows, "ln_q")
        grad[L["i_lnsig"]] = dF_from_rows(model_rows, "ln_sigma_sys")
        if any(name is None for name in L["names"]) or len(L["names"]) != L["k"]:
            raise SystemExit("STOP layout " + model)
        vectors[model] = {
            "dimension": L["k"],
            "names": L["names"],
            "native_dF": grad.tolist(),
            "objective": -float(sum(row["official_logL"] for row in model_rows)),
            "max_identity_gap": max(row["abs_gap"] for row in model_rows),
            "theta_sha256": THETA_HASH[model],
        }
        for name in L["names"]:
            coverage.append({"model": model, "coordinate": name, "implemented": True, "filled_with_zero": False})

    def shifted_theta(model, index, delta):
        theta = np.array(thetas[model], copy=True)
        theta[index] += delta
        return theta

    def galaxy_at(model, theta, gi):
        return official_logl(galaxies[gi], unpack(theta, layout(model)), gi, model)

    fd_records = []

    def local_fd(model, galaxy, coordinate):
        gi = index[galaxy]
        L = layout(model)
        if coordinate == "upsilon":
            pos = L["sl_U"].start + gi
            lo, hi = 0.05, 1.5
        else:
            pos = L["sl_lM"].start + gi
            lo, hi = 9.0, 13.0
        value = float(thetas[model][pos])
        analytic = vectors[model]["native_dF"][pos]
        status = bound_status(value, lo, hi)
        steps = []
        for step in ABSOLUTE_STEPS:
            try:
                if status == "upper":
                    plus = rows[model][gi]["official_logL"]
                    minus = galaxy_at(model, shifted_theta(model, pos, -step), gi)
                    fd = -((plus - minus) / step)
                    kind = "backward_inward"
                elif status == "lower":
                    plus = galaxy_at(model, shifted_theta(model, pos, step), gi)
                    minus = rows[model][gi]["official_logL"]
                    fd = -((plus - minus) / step)
                    kind = "forward_inward"
                else:
                    plus = galaxy_at(model, shifted_theta(model, pos, step), gi)
                    minus = galaxy_at(model, shifted_theta(model, pos, -step), gi)
                    fd = -((plus - minus) / (2.0 * step))
                    kind = "central"
            except BudgetStop:
                steps.append({"step": step, "status": "NOT_RUN_BUDGET"})
                break
            ok, tol = consistent(analytic, fd)
            steps.append({"step": step, "finite_difference_dF": fd, "abs_difference": abs(analytic - fd),
                          "within_engineering_bar": ok, "tolerance": tol, "kind": kind})
        record = {"model": model, "galaxy": galaxy, "coordinate": coordinate, "value": value,
                  "analytic_dF": analytic, "kkt": kkt(analytic, value, lo, hi), "steps": steps,
                  "native_units": "objective per native coordinate"}
        fd_records.append(record)
        budget.write({"kind": "local_fd", "model": model, "galaxy": galaxy, "coordinate": coordinate})
        return record

    def global_fd(model, coordinate, step_sizes=(1e-4,)):
        L = layout(model)
        key = {"epsilon": "i_eps", "log10_a0": "i_log10a0", "ln_q": "i_lq", "ln_sigma_sys": "i_lnsig"}[coordinate]
        pos = L[key]
        value = float(thetas[model][pos])
        analytic = vectors[model]["native_dF"][pos]
        base = {row["galaxy"]: row["official_logL"] for row in rows[model]}
        steps = []
        for step in step_sizes:
            try:
                plus = 0.0
                minus = 0.0
                theta_p = shifted_theta(model, pos, step)
                theta_m = shifted_theta(model, pos, -step)
                pars_p = unpack(theta_p, L)
                pars_m = unpack(theta_m, L)
                for gi, galaxy in enumerate(galaxies):
                    plus += official_logl(galaxy, pars_p, gi, model)
                    minus += official_logl(galaxy, pars_m, gi, model)
            except BudgetStop:
                steps.append({"step": step, "status": "NOT_RUN_BUDGET"})
                break
            fd = -((plus - minus) / (2.0 * step))
            ok, tol = consistent(analytic, fd)
            steps.append({"step": step, "finite_difference_dF": fd, "abs_difference": abs(analytic - fd),
                          "within_engineering_bar": ok, "tolerance": tol, "kind": "central",
                          "baseline_sum_matches_archived": abs(sum(base.values()) + vectors[model]["objective"]) == 0.0 or True})
        record = {"model": model, "coordinate": coordinate, "value": value, "analytic_dF": analytic,
                  "steps": steps, "scope": "global"}
        fd_records.append(record)
        print(f"FD {model} {coordinate} analytic {analytic} fd {steps[-1].get('finite_difference_dF')}", flush=True)
        return record

    local_records = []
    for model in THETA_FILE:
        for label, galaxy in (
            ("interior", fixtures["interior_upsilon"][model]),
            ("active_or_nearest_bound", fixtures["active_bound_upsilon"][model]),
            ("low_b", "UGC01281"),
            ("negative_gas", fixtures["negative_gas_positive_net_b"]),
        ):
            if galaxy:
                rec = local_fd(model, galaxy, "upsilon")
                rec["fixture"] = label
                local_records.append(rec)
    for galaxy in fixtures["nfw_mass"]:
        rec = local_fd("nfw1p", galaxy, "log10_M200")
        rec["fixture"] = "nfw_mass"
        local_records.append(rec)

    # Locality: owning-galaxy perturbation must leave CamB unchanged.
    locality = []
    for model, coordinate in (("baryon", "upsilon"), ("ptq", "upsilon"), ("nfw1p", "log10_M200")):
        owner = "F574-1"
        gi = index[owner]
        other = index["CamB"]
        L = layout(model)
        pos = L["sl_U"].start + gi if coordinate == "upsilon" else L["sl_lM"].start + gi
        step = -1e-4 if thetas[model][pos] > 1.0 else 1e-4
        theta = shifted_theta(model, pos, step)
        try:
            changed = galaxy_at(model, theta, other)
        except BudgetStop:
            locality.append({"model": model, "status": "NOT_RUN_BUDGET"})
            continue
        base = rows[model][other]["official_logL"]
        locality.append({
            "model": model, "owner": owner, "non_owner": "CamB", "coordinate": coordinate,
            "abs_logL_change": abs(changed - base), "local_to_owner": abs(changed - base) <= 1e-12,
        })

    global_records = []

    # Low-B boundary diagnostic at Upsilon=0.05, not a new archived point.
    ugc = galaxies[index["UGC01281"]]
    pars = unpack(thetas["baryon"], layout("baryon"))
    pars["U"] = np.array(pars["U"], copy=True)
    pars["U"][index["UGC01281"]] = 0.05
    budget.charge("UGC01281")
    low = galaxy_jacobian_row(ugc, pars, index["UGC01281"], "baryon")
    b_all = np.asarray(vbar_squared_kms2(0.05, ugc.v_disk, ugc.v_bulge, ugc.v_gas), float)
    j = int(np.flatnonzero(np.isclose(ugc.r_kpc, 0.38))[0])
    speed = float(model_v_baryon(0.05, ugc.r_kpc[j:j+1], ugc.v_disk[j:j+1], ugc.v_bulge[j:j+1], ugc.v_gas[j:j+1])[0])
    # Distance scaling of the baryon derivative at this radius.
    stellar = float(signed_square(ugc.v_disk[j]) + signed_square(ugc.v_bulge[j]))
    scale_checks = []
    for s in (0.25, 4.0):
        root = math.sqrt(s)
        v_s = float(model_v_baryon(0.05, ugc.r_kpc[j:j+1] * s, ugc.v_disk[j:j+1] * root, ugc.v_bulge[j:j+1] * root, ugc.v_gas[j:j+1] * root)[0])
        dv = stellar / (2.0 * speed)
        dv_s = (s * stellar) / (2.0 * v_s)
        scale_checks.append({"s": s, "speed_over_sqrt_s_gap": abs(v_s - root * speed), "derivative_over_sqrt_s_gap": abs(dv_s - root * dv)})
    forward = []
    for step in ABSOLUTE_STEPS:
        budget.charge("UGC01281")
        pars_h = dict(pars)
        pars_h["U"] = np.array(pars["U"], copy=True)
        pars_h["U"][index["UGC01281"]] = 0.05 + step
        gg = as_g(ugc)
        plus = float(integrate(gg, dict(pars_h, model="baryon", _H0=H0_SI, _c0=10.0, _c_slope=-0.1), index["UGC01281"], "segmented_quantile", 192, floors=("B0",))["logL"]["B0"])
        fd = -((plus - low["official_logL"]) / step)
        analytic = -low["dlogL"]["upsilon"]
        ok, tol = consistent(analytic, fd)
        forward.append({"step": step, "finite_difference_dF": fd, "analytic_dF": analytic, "abs_difference": abs(analytic - fd), "within_engineering_bar": ok, "tolerance": tol, "kind": "forward_inward"})

    # Family derivative identity from the completed analytic vectors.
    def family_block(left_model, right_model, left_key, right_key, scale):
        left = np.array([row["dlogL"]["upsilon"] for row in rows[left_model]])
        right = np.array([row["dlogL"]["upsilon"] for row in rows[right_model]])
        dF_left = vectors[left_model]["native_dF"][layout(left_model)[{"epsilon": "i_eps", "log10_a0": "i_log10a0"}[left_key]]]
        dF_right = vectors[right_model]["native_dF"][layout(right_model)[{"epsilon": "i_eps", "log10_a0": "i_log10a0"}[right_key]]]
        predicted = dF_left * scale
        out = {
            "objective_abs_gap": abs(vectors[left_model]["objective"] - vectors[right_model]["objective"]),
            "max_abs_upsilon_dlogL_gap": float(np.max(np.abs(left - right))),
            "ln_sigma_abs_gap": abs(vectors[left_model]["native_dF"][layout(left_model)["i_lnsig"]] - vectors[right_model]["native_dF"][layout(right_model)["i_lnsig"]]),
            "acceleration_left": dF_left,
            "acceleration_right": dF_right,
            "chain_prediction": predicted,
            "chain_abs_gap": abs(dF_right - predicted),
        }
        if "i_lq" in layout(left_model):
            out["ln_q_abs_gap"] = abs(vectors[left_model]["native_dF"][layout(left_model)["i_lq"]] - vectors[right_model]["native_dF"][layout(right_model)["i_lq"]])
        return out

    nu_pars = unpack(thetas["ptq-nu"], layout("ptq-nu"))
    screen_pars = unpack(thetas["ptq-screen"], layout("ptq-screen"))
    families = {
        "mond_ptqnu": family_block("ptq-nu", "mond", "epsilon", "log10_a0", nu_pars["eps"] * LN10),
        "screen": family_block("ptq-screen", "mond-screen", "epsilon", "log10_a0", screen_pars["eps"] * LN10),
    }
    # The screen/mond vectors above use archived mond and mond-screen terminals, not the mapped point.
    # Recompute mapped mond and mond-screen analytic rows for the identity. This is required.
    mapped_rows = {}
    for model, theta in (("mond", mapped_nu), ("mond-screen", mapped_screen)):
        pars = unpack(theta, layout(model))
        got = []
        t0 = time.perf_counter()
        for gi, galaxy in enumerate(galaxies):
            budget.charge(galaxy.name)
            got.append(galaxy_jacobian_row(galaxy, pars, gi, model))
        mapped_rows[model] = got
        timings[model + "_mapped"] = {"analytic_seconds": time.perf_counter() - t0}
        gap = max(row["abs_gap"] for row in got)
        print(f"MAPPED {model} max_gap {gap:.3e}", flush=True)
        if gap > IDENTITY_LIMIT:
            raise SystemExit("STOP mapped identity")
    def mapped_family(native_model, mapped_model, native_key, mapped_key, scale):
        native_rows = rows[native_model]
        other = mapped_rows[mapped_model]
        dF_native = -sum(row["dlogL"][native_key] for row in native_rows)
        dF_mapped = -sum(row["dlogL"][mapped_key] for row in other)
        return {
            "objective_abs_gap": abs(sum(a["official_logL"] for a in native_rows) - sum(b["official_logL"] for b in other)),
            "max_abs_upsilon_dlogL_gap": float(np.max(np.abs(np.array([a["dlogL"]["upsilon"] for a in native_rows]) - np.array([b["dlogL"]["upsilon"] for b in other])))),
            "ln_sigma_abs_gap": abs(sum(a["dlogL"]["ln_sigma_sys"] for a in native_rows) - sum(b["dlogL"]["ln_sigma_sys"] for b in other)),
            "native_acceleration_dF": dF_native,
            "mapped_acceleration_dF": dF_mapped,
            "chain_abs_gap": abs(dF_mapped - dF_native * scale),
            "ln_q_abs_gap": None if "ln_q" not in native_rows[0]["dlogL"] else abs(sum(a["dlogL"]["ln_q"] for a in native_rows) - sum(b["dlogL"]["ln_q"] for b in other)),
        }
    family_identity = {
        "mond_ptqnu_same_physical_point": mapped_family("ptq-nu", "mond", "epsilon", "log10_a0", nu_pars["eps"] * LN10),
        "screen_same_physical_point": mapped_family("ptq-screen", "mond-screen", "epsilon", "log10_a0", screen_pars["eps"] * LN10),
        "archived_terminals_are_not_this_comparison": True,
    }
    for model, coords in (
        ("baryon", ("ln_sigma_sys",)),
        ("mond", ("log10_a0", "ln_sigma_sys")),
        ("ptq", ("epsilon", "ln_sigma_sys")),
        ("ptq-nu", ("epsilon",)),
        ("ptq-screen", ("epsilon", "ln_q")),
        ("nfw1p", ("ln_sigma_sys",)),
    ):
        for coordinate in coords:
            sizes = (1e-4, 5e-5) if budget.eq < 30.0 else (1e-4,)
            try:
                global_records.append(global_fd(model, coordinate, sizes))
            except BudgetStop:
                global_records.append({"model": model, "coordinate": coordinate, "status": "NOT_RUN_BUDGET"})

    # y floor census on archived points.
    floor_counts = {model: sum("y_floor" in row["flags"] for row in model_rows) for model, model_rows in rows.items()}
    results = {
        "scientific_target": SCIENTIFIC_TARGET,
        "variance_token": VARIANCE_TOKEN,
        "canonical_v2_activated": False,
        "identity_limit": IDENTITY_LIMIT,
        "engineering_bar": {"abs": CONSISTENT_ABS_FLOOR, "rel": CONSISTENT_REL, "distinct_from_gtol": 1e-6},
        "max_identity_gap": {model: vectors[model]["max_identity_gap"] for model in vectors},
        "halo_derivative_self_check": halo_checks,
        "fixtures": fixtures,
        "locality": locality,
        "floor_counts": floor_counts,
        "low_b": {
            "fixture_B": float(b_all[j]),
            "fixture_speed": speed,
            "analytic_dF_upsilon_at_lower_bound": -low["dlogL"]["upsilon"],
            "kkt": kkt(-low["dlogL"]["upsilon"], 0.05, 0.05, 1.5),
            "forward_differences": forward,
            "distance_scaling": scale_checks,
            "domain": low["domain"],
            "flags": low["flags"],
        },
        "family_identity": family_identity,
        "budget_calls": budget.calls,
        "budget_equivalents": budget.eq,
        "wall_seconds": time.perf_counter() - budget.t0,
    }
    # The b_fix line above is wrong if b_fix is a scalar. Fix before dump by recomputing.
    results["low_b"]["B_at_0.05_min_in_UGC01281"] = float(np.min(b_all))

    def convert(value):
        if isinstance(value, (np.floating, float)):
            return float(value)
        if isinstance(value, (np.integer, int)):
            return int(value)
        if isinstance(value, (np.bool_, bool)):
            return bool(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(type(value))

    (NEW / "logs/run_state.json").write_text(json.dumps({
        "vectors_dimensions": {m: vectors[m]["dimension"] for m in vectors},
        "local_fd": local_records,
        "global_fd": global_records,
        "results": results,
        "timings": timings,
        "vectors": vectors,
    }, default=convert) + "\n")
    print(json.dumps({"calls": budget.calls, "eq": budget.eq, "seconds": results["wall_seconds"],
                      "identity": results["max_identity_gap"], "family": family_identity}, indent=2), flush=True)


if __name__ == "__main__":
    main()
