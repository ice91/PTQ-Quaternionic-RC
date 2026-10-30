"""Bounded fixed-point check of the isolated B0 derivative contract. No optimizer."""
import hashlib, json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/mle_pilot_a3_deriv")
sys.path.insert(0, str(ROOT / "diagnostic"))
CAND = Path("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/mle_pilot_a/candidate/src")
sys.path.insert(0, str(CAND))
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from b0_analytic_contract import evaluate_fixed_b0
from ptquat.data import load_tidy_sparc
from ptquat.marginal_likelihood import PRIMARY_B0
import ptquat.marginal_likelihood as ml

EX = Path("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/mle_pilot_a/execution_v1")
IMPL = Path("/home/rocky/PTQ-Quaternionic-RC")
MAX_CALLS = 300
MAX_EQ = 2.0
ARCHIVED = {
    "ptq": {
        "objective": 7149.884126318863,
        "epsilon": 82.52506788569912,
        "ln_sigma_sys": 7.837096471983812,
        "fd_epsilon_1e-4": 76.82584514576808,
    },
    "ptq-nu": {
        "objective": 6657.818831613563,
        "epsilon": 0.9494002718727226,
        "ln_sigma_sys": -0.44859569759110923,
        "fd_epsilon_1e-4": 0.9492829796648327,
    },
}


def main():
    if Path(ml.__file__).resolve() != (CAND / "ptquat" / "marginal_likelihood.py").resolve():
        raise SystemExit("import authority")
    if PRIMARY_B0 != "REV26-P1E-UNIFORM-EXPLICIT-B0-PHYSICAL-v1":
        raise SystemExit(PRIMARY_B0)
    weights = json.loads((IMPL / "results/frontiers_revision_2026_10/mle_pilot_a/GALAXY_WORK_WEIGHTS.json").read_text())
    gd = load_tidy_sparc(str(IMPL / "dataset/sparc_tidy.csv"))
    gs = [gd[k] for k in sorted(gd)]
    if len(gs) != 91 or sum(len(g.v_obs) for g in gs) != 1782:
        raise SystemExit("sample")
    calls = 0
    eq = 0.0
    t0 = time.monotonic()

    def charge(model, gi, order):
        nonlocal calls, eq
        unit = weights[model][str(order)][gi]["fullsample_equivalent"]
        if calls + 1 > MAX_CALLS or eq + unit > MAX_EQ + 1e-12:
            raise RuntimeError("budget")
        calls += 1
        eq += unit

    out = {"task": "REV26-P1G-A3-DERIV", "models": {}}
    for model in ("ptq", "ptq-nu"):
        theta = np.load(EX / f"{model.replace('-', '_')}_last_accepted_theta.npy")
        result = evaluate_fixed_b0(theta, model, gs, order=192, charge=charge)
        arch = ARCHIVED[model]
        grads = result["analytic_partial_gradients"]
        out["models"][model] = {
            "theta_sha256": hashlib.sha256(np.asarray(theta, dtype="<f8").tobytes()).hexdigest(),
            "objective": result["objective"],
            "archived_objective": arch["objective"],
            "objective_minus_archived": result["objective"] - arch["objective"],
            "max_abs_galaxy_logL_gap": result["max_abs_galaxy_logL_gap"],
            "identity_pass": result["identity_pass"],
            "epsilon_native_dF": grads["epsilon"]["native_dF"],
            "epsilon_minus_archived_analytic": grads["epsilon"]["native_dF"] - arch["epsilon"],
            "epsilon_minus_frozen_fd": grads["epsilon"]["native_dF"] - arch["fd_epsilon_1e-4"],
            "ln_sigma_native_dF": grads["ln_sigma_sys"]["native_dF"],
            "ln_sigma_minus_archived_analytic": grads["ln_sigma_sys"]["native_dF"] - arch["ln_sigma_sys"],
            "domain": result["validation_metadata"]["domain"],
            "domain_flags": result["validation_metadata"]["domain_flags"],
            "weights_depend_on_theta": False,
            "activated_as_optimizer_jacobian": False,
        }
        print(model, "gap", result["max_abs_galaxy_logL_gap"], "eps", grads["epsilon"]["native_dF"], "sig", grads["ln_sigma_sys"]["native_dF"], flush=True)
    out["budget"] = {"calls": calls, "equivalents": eq, "wall_seconds": time.monotonic() - t0, "caps": {"calls": MAX_CALLS, "equivalents": MAX_EQ}}
    (ROOT / "FIXED_POINT_VALIDATION.json").write_text(json.dumps(out, indent=2) + "\n")
    print("DONE", calls, eq, flush=True)


if __name__ == "__main__":
    main()
