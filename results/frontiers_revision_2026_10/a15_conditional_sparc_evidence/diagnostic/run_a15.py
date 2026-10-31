"""Conditional two-family fine192 grid. No optimizer."""
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

from ptquat.data import load_tidy_sparc
from ptquat.marginal_likelihood import PRIMARY_B0, galaxy_log_marginal

T0 = time.perf_counter()
CAP = 20500
CALLS = 0


def load_galaxies():
    loaded = load_tidy_sparc(str(A10 / "logs/s_domain13_filtered_tidy.csv"))
    galaxies = [loaded[name] for name in sorted(loaded)]
    if len(galaxies) != 91 or sum(len(g.v_obs) for g in galaxies) != 1769:
        raise SystemExit("sample identity failed")
    return galaxies


def main():
    global CALLS
    study = json.loads((TEST / "PREDECLARED_STUDY.json").read_text())
    galaxies = load_galaxies()
    names = [g.name for g in galaxies]
    if names != study["galaxies"]:
        raise SystemExit("galaxy list differs from predeclaration")
    models = study["families"]
    ups = study["nuisance"]["upsilon"]
    sigs = study["nuisance"]["sigma_sys_kms"]
    cache_path = TEST / "PER_GALAXY_LOGL_CACHE.npz"
    if cache_path.exists():
        store = np.load(cache_path)
        logl = store["logL"]
        done = store["done"]
        CALLS = int(store["calls"])
    else:
        logl = np.full((2, 2, 2, 25, 91), np.nan)
        done = np.zeros((2, 2, 2, 25), dtype=bool)
        CALLS = 0
    errors = []
    for im, model in enumerate(models):
        eps = np.asarray(study["grids"][model]["epsilon"], float)
        if eps.size != 25:
            raise SystemExit("grid length")
        for iu, upsilon in enumerate(ups):
            U = np.full(91, float(upsilon))
            for iss, sigma in enumerate(sigs):
                for node, epsilon in enumerate(eps):
                    if done[im, iu, iss, node]:
                        continue
                    if time.perf_counter() - T0 > 3 * 3600:
                        raise SystemExit("INCOMPLETE_BUDGET wall")
                    pars = {"U": U, "sigma": float(sigma), "eps": float(epsilon), "model": model}
                    for gi, galaxy in enumerate(galaxies):
                        if CALLS >= CAP:
                            raise SystemExit("INCOMPLETE_BUDGET count")
                        CALLS += 1
                        try:
                            record = galaxy_log_marginal(
                                galaxy, pars, gi, model,
                                likelihood_version=PRIMARY_B0, order=192,
                            )
                            logl[im, iu, iss, node, gi] = float(record["logL"]["B0"])
                        except Exception as exc:
                            logl[im, iu, iss, node, gi] = np.nan
                            errors.append({
                                "model": model, "upsilon": upsilon, "sigma": sigma,
                                "node": node, "galaxy": galaxy.name,
                                "error": type(exc).__name__ + ": " + str(exc),
                            })
                    done[im, iu, iss, node] = True
                    np.savez_compressed(
                        cache_path, logL=logl, done=done, calls=np.array(CALLS),
                        models=np.array(models), upsilon=np.array(ups), sigma=np.array(sigs),
                    )
                    finite = int(np.isfinite(logl[im, iu, iss, node]).sum())
                    print(
                        f"{model} U={upsilon} s={sigma} node={node} finite={finite}/91 "
                        f"calls={CALLS} wall={time.perf_counter()-T0:.1f}",
                        flush=True,
                    )
    (TEST / "logs/grid_errors.json").write_text(json.dumps(errors, indent=2) + "\n")
    (TEST / "logs/grid_complete.json").write_text(json.dumps({
        "calls": CALLS,
        "wall_seconds": time.perf_counter() - T0,
        "all_nodes_done": bool(done.all()),
        "n_nan": int(np.isnan(logl).sum()),
        "n_errors": len(errors),
    }, indent=2) + "\n")
    print("GRID_DONE", CALLS, time.perf_counter() - T0)


if __name__ == "__main__":
    main()
