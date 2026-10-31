"""Replay reported nodes, A5/A6 identity, and targeted fine384. fine192 stays primary."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

TEST = Path("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/a15_conditional_sparc_evidence")
A10 = Path("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/mle_pilot_a10_four_family_science_pilot")
sys.path.insert(0, str(A10 / "candidate/src"))
sys.path.insert(0, str(A10 / "diagnostic"))

from ptquat.data import load_tidy_sparc
from ptquat.marginal_likelihood import PRIMARY_B0, galaxy_log_marginal, integrate
from ptquat.models import catalog_velocity_at_distance, vbar_squared_kms2, linear_term_kms2
from signed_jacobian import galaxy_jacobian_row

CAP_EXTRA = 2300


def gsimple(galaxy):
    from types import SimpleNamespace
    return SimpleNamespace(
        name=galaxy.name, r=galaxy.r_kpc, y=galaxy.v_obs, e=galaxy.v_err,
        disk=galaxy.v_disk, bulge=galaxy.v_bulge, gas=galaxy.v_gas,
        D=float(galaxy.D_Mpc), sD=float(galaxy.D_err_Mpc),
        inc=float(np.deg2rad(galaxy.i_deg)), si=float(np.deg2rad(galaxy.i_err_deg)),
    )


def main():
    t0 = time.perf_counter()
    study = json.loads((TEST / "PREDECLARED_STUDY.json").read_text())
    profiles = json.loads((TEST / "CONDITIONAL_PROFILE_RESULTS.json").read_text())
    store = np.load(TEST / "PER_GALAXY_LOGL_CACHE.npz")
    logL = store["logL"]
    loaded = load_tidy_sparc(str(A10 / "logs/s_domain13_filtered_tidy.csv"))
    galaxies = [loaded[name] for name in study["galaxies"]]
    calls = 0
    replay_gaps = []
    identity = []
    fine384 = []
    outer = []

    def budget(n=1):
        nonlocal calls
        if calls + n > CAP_EXTRA:
            raise SystemExit("validation budget")
        calls += n

    # fine192 replay of every reported grid minimum
    for im, model in enumerate(study["families"]):
        for iu, U in enumerate(study["nuisance"]["upsilon"]):
            for iss, sigma in enumerate(study["nuisance"]["sigma_sys_kms"]):
                prof = next(p for p in profiles if p["model"] == model and p["upsilon"] == U and p["sigma_sys_kms"] == sigma)
                node = prof["best_node"]
                pars = {"U": np.full(91, float(U)), "sigma": float(sigma), "eps": prof["best_epsilon"], "model": model}
                fresh = np.empty(91)
                for gi, galaxy in enumerate(galaxies):
                    budget()
                    record = galaxy_log_marginal(galaxy, pars, gi, model, likelihood_version=PRIMARY_B0, order=192)
                    fresh[gi] = float(record["logL"]["B0"])
                gap = float(np.max(np.abs(fresh - logL[im, iu, iss, node])))
                replay_gaps.append({"model": model, "upsilon": U, "sigma": sigma, "node": node, "max_abs_gap": gap})
                print("replay", model, U, sigma, gap, flush=True)

    # A5/A6 identity at the two primary minima
    for model in study["families"]:
        prof = next(p for p in profiles if p["model"] == model and p["upsilon"] == 0.5 and p["sigma_sys_kms"] == 5.0)
        pars = {"U": np.full(91, 0.5), "sigma": 5.0, "eps": prof["best_epsilon"]}
        gaps = []
        for gi, galaxy in enumerate(galaxies):
            budget()
            row = galaxy_jacobian_row(galaxy, pars, gi, model, order=192)
            gaps.append(row["abs_gap"])
            if row["abs_gap"] > 1e-9:
                identity.append({"galaxy": galaxy.name, "model": model, "gap": row["abs_gap"], "domain": row["domain"]})
        identity.append({
            "model": model, "upsilon": 0.5, "sigma": 5.0,
            "max_abs_gap": float(np.max(gaps)),
            "n_above_1e-9": int(np.sum(np.asarray(gaps) > 1e-9)),
        })
        print("identity", model, float(np.max(gaps)), flush=True)

    # fine384: primary best and neighbors, all galaxies; other minima at the best node only
    for im, model in enumerate(study["families"]):
        eps = np.asarray(study["grids"][model]["epsilon"], float)
        for iu, U in enumerate(study["nuisance"]["upsilon"]):
            for iss, sigma in enumerate(study["nuisance"]["sigma_sys_kms"]):
                prof = next(p for p in profiles if p["model"] == model and p["upsilon"] == U and p["sigma_sys_kms"] == sigma)
                nodes = [prof["best_node"]]
                if U == 0.5 and sigma == 5.0:
                    nodes = sorted({max(0, prof["best_node"] - 1), prof["best_node"], min(24, prof["best_node"] + 1)})
                for node in nodes:
                    pars = {"U": np.full(91, float(U)), "sigma": float(sigma), "eps": float(eps[node]), "model": model}
                    total = 0.0
                    max_gap = 0.0
                    for gi, galaxy in enumerate(galaxies):
                        budget()
                        from ptquat.constants import H0_SI
                        g = gsimple(galaxy)
                        p = dict(pars, _H0=H0_SI, _c0=10.0, _c_slope=-0.1)
                        value = float(integrate(g, p, gi, "segmented_quantile", 384, floors=("B0",))["logL"]["B0"])
                        total += value
                        max_gap = max(max_gap, abs(value - float(logL[im, iu, iss, node, gi])))
                    fine384.append({
                        "model": model, "upsilon": U, "sigma": sigma, "node": node,
                        "fine384_logL": total, "fine192_logL": float(logL[im, iu, iss, node].sum()),
                        "max_abs_galaxy_gap": max_gap,
                        "sum_gap": total - float(logL[im, iu, iss, node].sum()),
                    })
                    print("fine384", model, U, sigma, node, max_gap, flush=True)

    # outer-radius visibility at the primary linear node, nominal distance
    lin = next(p for p in profiles if p["model"] == "ptq" and p["upsilon"] == 0.5 and p["sigma_sys_kms"] == 5.0)
    nu = next(p for p in profiles if p["model"] == "ptq-nu" and p["upsilon"] == 0.5 and p["sigma_sys_kms"] == 5.0)
    n_linear_dominates = 0
    for gi, galaxy in enumerate(galaxies):
        vbar2 = vbar_squared_kms2(0.5, galaxy.v_disk, galaxy.v_bulge, galaxy.v_gas)
        lin_term = linear_term_kms2(lin["best_epsilon"], galaxy.r_kpc)
        ratio = float(lin_term[-1] / vbar2[-1]) if vbar2[-1] != 0 else float("nan")
        if np.isfinite(ratio) and ratio > 1:
            n_linear_dominates += 1
        if galaxy.name in study["representative_galaxies"]:
            outer.append({
                "galaxy": galaxy.name, "r_last_kpc": float(galaxy.r_kpc[-1]),
                "linear_over_vbar2_at_last": ratio,
                "n_negative_gas": int(np.sum(galaxy.v_gas < 0)),
                "conditional_linear_epsilon": lin["best_epsilon"],
                "conditional_ptq_nu_epsilon": nu["best_epsilon"],
            })
    outer.append({"n_galaxies_linear_term_exceeds_vbar2_at_last_radius": n_linear_dominates, "upsilon": 0.5, "sigma": 5.0})

    (TEST / "NUMERICAL_VALIDATION.json").write_text(json.dumps({
        "fine192_replay": replay_gaps,
        "a5_a6_identity": identity,
        "fine384": fine384,
        "outer_radius": outer,
        "validation_calls": calls,
        "wall_seconds": time.perf_counter() - t0,
        "primary_objective": "fine192",
    }, indent=2) + "\n")
    print("VALIDATION_DONE", calls, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
