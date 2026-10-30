"""Replay the last accepted checkpoint. Does not call an optimizer."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_a7 import (
    IMPL, NEW, IDENTITY_LIMIT, REPLAY_OBJECTIVE_FLOOR, THETA_CONTENT_SHA,
    Budget, Store, kkt, native_bounds, theta_hash, unpack,
)

def main():
    store = Store(NEW / "state/scientific")
    theta, meta, manifest = store.recover()
    if meta["kind"] != "ACCEPTED_ITERATION" or meta["accepted_index"] != 1:
        raise SystemExit("STOP expected the one committed accepted step")
    if theta_hash(theta) != meta["theta_sha256"]:
        raise SystemExit("STOP theta")
    bounds = np.asarray(native_bounds(), float)
    if np.any(theta < bounds[:, 0]) or np.any(theta > bounds[:, 1]):
        raise SystemExit("STOP bounds")
    tidy = pd.read_csv(IMPL / "dataset/sparc_tidy.csv")
    mask = pd.read_csv(IMPL / "results/frontiers_revision_2026_10/mle_pilot_a4_domain_feasibility/S_DOMAIN13_MASK.csv")
    filtered = NEW / "logs/s_domain13_filtered_tidy.csv"
    weights = {row["galaxy"]: float(row["fullsample_equivalent"]) for row in json.loads(
        (IMPL / "results/frontiers_revision_2026_10/mle_pilot_a/GALAXY_WORK_WEIGHTS.json").read_text()
    )["baryon"]["192"]}
    # evaluate() closes over galaxies from run_a7.main. Reimplement the call here.
    from ptquat.data import load_tidy_sparc
    from signed_jacobian import galaxy_jacobian_row
    loaded = load_tidy_sparc(str(filtered))
    galaxies = [loaded[name] for name in sorted(loaded)]
    if len(galaxies) != 91 or int((mask.included == 1).sum()) != 1769:
        raise SystemExit("STOP sample")
    budget = Budget(weights)
    started = time.perf_counter()
    pars = unpack(theta)
    rows = []
    for gi, galaxy in enumerate(galaxies):
        budget.charge_galaxy(galaxy.name)
        rows.append(galaxy_jacobian_row(galaxy, pars, gi, "ptq-nu"))
    seconds = time.perf_counter() - started
    gap = max(row["abs_gap"] for row in rows)
    objective = -sum(row["official_logL"] for row in rows)
    gradient = np.zeros(93)
    gradient[0] = -sum(row["dlogL"]["epsilon"] for row in rows)
    for gi, row in enumerate(rows):
        gradient[1 + gi] = -row["dlogL"]["upsilon"]
    gradient[92] = -sum(row["dlogL"]["ln_sigma_sys"] for row in rows)
    diag = kkt(theta, gradient)
    recorded = float(meta["objective_after"])
    tolerance = REPLAY_OBJECTIVE_FLOOR
    payload = {
        "optimizer_invoked_by_this_script": False,
        "recovered_accepted_index": 1,
        "recovered_theta_sha256": theta_hash(theta),
        "origin_theta_sha256": THETA_CONTENT_SHA,
        "galaxies_completed": len(rows),
        "quadrature": "fine192",
        "max_per_galaxy_path_gap": gap,
        "identity_pass": gap <= IDENTITY_LIMIT,
        "recorded_objective": recorded,
        "replay_objective": objective,
        "objective_abs_gap": abs(objective - recorded),
        "replay_tolerance": tolerance,
        "objective_within_tolerance": abs(objective - recorded) <= tolerance,
        "outside_bounds": diag["outside_bounds"],
        "gradient_all_finite": bool(np.isfinite(gradient).all()),
        "flags": sorted({flag for row in rows for flag in row["flags"]}),
        "domains": sorted({row["domain"] for row in rows}),
        "projected_gradient_inf_norm": diag["projected_gradient_inf_norm"],
        "full_gradient_inf_norm": diag["full_gradient_inf_norm"],
        "replay_seconds": seconds,
        "replay_galaxy_calls": budget.calls,
        "replay_equivalents": budget.eq,
        "manifest": manifest,
    }
    (NEW / "logs/endpoint_replay.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in (
        "max_per_galaxy_path_gap", "objective_abs_gap", "identity_pass",
        "objective_within_tolerance", "projected_gradient_inf_norm", "replay_seconds"
    )}, indent=2))


if __name__ == "__main__":
    main()
