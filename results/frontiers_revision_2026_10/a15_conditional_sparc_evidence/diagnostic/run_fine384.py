"""Targeted fine384 only. Replay and identity were already measured."""
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

from ptquat.constants import H0_SI
from ptquat.data import load_tidy_sparc
from ptquat.marginal_likelihood import integrate
from ptquat.models import linear_term_kms2, vbar_squared_kms2
from types import SimpleNamespace

PARTIAL = TEST / "logs/fine384_partial.json"


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
    profiles = json.loads((TEST / "CONDITIONAL_PROFILE_RESULTS.json").read_text())
    logL = np.load(TEST / "PER_GALAXY_LOGL_CACHE.npz")["logL"]
    loaded = load_tidy_sparc(str(A10 / "logs/s_domain13_filtered_tidy.csv"))
    galaxies = [loaded[name] for name in study["galaxies"]]
    done = json.loads(PARTIAL.read_text()) if PARTIAL.exists() else []
    seen = {(row["model"], row["upsilon"], row["sigma"], row["node"]) for row in done}
    calls = 0
    for im, model in enumerate(study["families"]):
        eps = np.asarray(study["grids"][model]["epsilon"], float)
        for iu, U in enumerate(study["nuisance"]["upsilon"]):
            for iss, sigma in enumerate(study["nuisance"]["sigma_sys_kms"]):
                prof = next(p for p in profiles if p["model"] == model and p["upsilon"] == U and p["sigma_sys_kms"] == sigma)
                nodes = [prof["best_node"]]
                if U == 0.5 and sigma == 5.0:
                    nodes = sorted({max(0, prof["best_node"] - 1), prof["best_node"], min(24, prof["best_node"] + 1)})
                for node in nodes:
                    key = (model, U, sigma, node)
                    if key in seen:
                        continue
                    pars = {
                        "U": np.full(91, float(U)), "sigma": float(sigma),
                        "eps": float(eps[node]), "model": model,
                        "_H0": H0_SI, "_c0": 10.0, "_c_slope": -0.1,
                    }
                    total = 0.0
                    max_gap = 0.0
                    for gi, galaxy in enumerate(galaxies):
                        calls += 1
                        value = float(integrate(
                            gsimple(galaxy), pars, gi, "segmented_quantile", 384, floors=("B0",),
                        )["logL"]["B0"])
                        total += value
                        max_gap = max(max_gap, abs(value - float(logL[im, iu, iss, node, gi])))
                    row = {
                        "model": model, "upsilon": U, "sigma": sigma, "node": node,
                        "fine384_logL": total,
                        "fine192_logL": float(logL[im, iu, iss, node].sum()),
                        "max_abs_galaxy_gap": max_gap,
                        "sum_gap": total - float(logL[im, iu, iss, node].sum()),
                    }
                    done.append(row)
                    PARTIAL.write_text(json.dumps(done, indent=2) + "\n")
                    print("fine384", model, U, sigma, node, f"sum_gap={row['sum_gap']:.6e}", f"max={max_gap:.6e}", flush=True)
    lin = next(p for p in profiles if p["model"] == "ptq" and p["upsilon"] == 0.5 and p["sigma_sys_kms"] == 5.0)
    nu = next(p for p in profiles if p["model"] == "ptq-nu" and p["upsilon"] == 0.5 and p["sigma_sys_kms"] == 5.0)
    outer = []
    n_dom = 0
    for name in study["galaxies"]:
        galaxy = loaded[name]
        vbar2 = vbar_squared_kms2(0.5, galaxy.v_disk, galaxy.v_bulge, galaxy.v_gas)
        term = linear_term_kms2(lin["best_epsilon"], galaxy.r_kpc)
        ratio = float(term[-1] / vbar2[-1]) if vbar2[-1] != 0 else float("nan")
        if np.isfinite(ratio) and ratio > 1:
            n_dom += 1
        if name in study["representative_galaxies"]:
            outer.append({
                "galaxy": name,
                "r_last_kpc": float(galaxy.r_kpc[-1]),
                "linear_over_vbar2_at_last": ratio,
                "n_radii_linear_exceeds_abs_vbar2": int(np.sum(term > np.abs(vbar2))),
                "n_radii": int(len(galaxy.r_kpc)),
                "n_negative_gas": int(np.sum(galaxy.v_gas < 0)),
            })
    outer.append({
        "n_galaxies_linear_term_exceeds_vbar2_at_last_radius": n_dom,
        "n_galaxies": 91,
        "median_last_ratio": float(np.nanmedian([
            linear_term_kms2(lin["best_epsilon"], loaded[n].r_kpc)[-1]
            / vbar_squared_kms2(0.5, loaded[n].v_disk, loaded[n].v_bulge, loaded[n].v_gas)[-1]
            for n in study["galaxies"]
        ])),
        "upsilon": 0.5,
        "sigma_sys_kms": 5.0,
        "linear_epsilon": lin["best_epsilon"],
        "ptq_nu_epsilon": nu["best_epsilon"],
        "note": "Nominal catalog templates. Not an extra likelihood evaluation.",
    })
    (TEST / "logs/fine384_calls.json").write_text(json.dumps({
        "fine384_calls_this_process": calls,
        "wall_seconds": time.perf_counter() - t0,
        "n_rows": len(done),
    }, indent=2) + "\n")
    (TEST / "logs/outer_radius.json").write_text(json.dumps(outer, indent=2) + "\n")
    print("FINE384_DONE", calls, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
