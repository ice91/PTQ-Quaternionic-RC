"""Two bounded fine384 checks: shallow neighbor, and per-galaxy gaps at the linear minimum."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

TEST = Path("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/a15_conditional_sparc_evidence")
A10 = Path("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/mle_pilot_a10_four_family_science_pilot")
sys.path.insert(0, str(A10 / "candidate/src"))

from ptquat.constants import H0_SI
from ptquat.data import load_tidy_sparc
from ptquat.marginal_likelihood import integrate


def gsimple(galaxy):
    return SimpleNamespace(
        name=galaxy.name, r=galaxy.r_kpc, y=galaxy.v_obs, e=galaxy.v_err,
        disk=galaxy.v_disk, bulge=galaxy.v_bulge, gas=galaxy.v_gas,
        D=float(galaxy.D_Mpc), sD=float(galaxy.D_err_Mpc),
        inc=float(np.deg2rad(galaxy.i_deg)), si=float(np.deg2rad(galaxy.i_err_deg)),
    )


def main():
    t0 = time.perf_counter()
    study = json.loads((TEST / "PREDECLARED_STUDY.json").read_text())
    logL = np.load(TEST / "PER_GALAXY_LOGL_CACHE.npz")["logL"]
    loaded = load_tidy_sparc(str(A10 / "logs/s_domain13_filtered_tidy.csv"))
    galaxies = [loaded[name] for name in study["galaxies"]]
    calls = 0

    def eval_node(model, iu, iss, node, eps):
        nonlocal calls
        pars = {
            "U": np.full(91, float(study["nuisance"]["upsilon"][iu])),
            "sigma": float(study["nuisance"]["sigma_sys_kms"][iss]),
            "eps": float(eps), "model": model,
            "_H0": H0_SI, "_c0": 10.0, "_c_slope": -0.1,
        }
        gaps = []
        for gi, galaxy in enumerate(galaxies):
            calls += 1
            value = float(integrate(gsimple(galaxy), pars, gi, "segmented_quantile", 384, floors=("B0",))["logL"]["B0"])
            gaps.append(value - float(logL[iu and 0, iu, iss, node, gi] if False else logL[
                0 if model == "ptq-nu" else 1, iu, iss, node, gi
            ]))
        return np.asarray(gaps, float)

    # Shallow simple-nu neighbor: U=0.5 is iu=0, sigma=10 is iss=1, node 11.
    nu_eps = np.asarray(study["grids"]["ptq-nu"]["epsilon"], float)
    gaps_shallow = eval_node("ptq-nu", 0, 1, 11, nu_eps[11])
    # Linear minimum per galaxy: U=0.5, sigma=5, node 16.
    lin_eps = np.asarray(study["grids"]["ptq"]["epsilon"], float)
    gaps_lin = eval_node("ptq", 0, 0, 16, lin_eps[16])
    order = np.argsort(-np.abs(gaps_lin))
    out = {
        "calls": calls,
        "wall_seconds": time.perf_counter() - t0,
        "ptq_nu_U0.5_s10_node11": {
            "sum_gap": float(gaps_shallow.sum()),
            "max_abs_galaxy_gap": float(np.max(np.abs(gaps_shallow))),
            "purpose": "left neighbor of the shallowest simple-nu grid minimum",
        },
        "linear_U0.5_s5_node16_largest_abs_gaps": [
            {"galaxy": study["galaxies"][int(i)], "gap_fine384_minus_fine192": float(gaps_lin[int(i)])}
            for i in order[:8]
        ],
    }
    (TEST / "logs/fine384_galaxy.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
