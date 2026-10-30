"""Bounded fixed-point checks for isolated Upsilon and screen derivatives. No optimizer."""
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

NEW = Path("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/mle_pilot_a4_jac_eng")
sys.path.insert(0, str(NEW / "diagnostic"))
CAND = Path("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/mle_pilot_a/candidate/src")
sys.path.insert(0, str(CAND))
DERIV = Path("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/mle_pilot_a3_deriv")
sys.path.insert(0, str(DERIV / "diagnostic"))

from local_screen_derivatives import (
    CONSISTENT_ABS_FLOOR,
    CONSISTENT_REL,
    IDENTITY_LIMIT,
    as_g,
    formula_self_check,
    galaxy_partials,
)
from ptquat.constants import C_LIGHT, H0_SI
from ptquat.data import load_tidy_sparc
from ptquat.dependency_gradient import step_stencil
from ptquat.fit_global import _layout, _unpack
from ptquat.marginal_likelihood import PRIMARY_B0, integrate
from ptquat.mle import map_theta_ptqscreen_to_mondscreen, theta_bounds

IMPL = Path("/home/rocky/PTQ-Quaternionic-RC")
EX = IMPL / "results/frontiers_revision_2026_10/mle_pilot_a/execution_v1"
MAX_CALLS = 1200
MAX_EQ = 8.0
# Predeclared before looking at screen finite differences.
GLOBAL_STEPS = (1e-4, 5e-5)
LOCAL_STEPS = (1e-4, 5e-5)
ADDITIONAL_GALAXY = "NGC0801"
BOUND_ATOL_SPAN = 1e-12

EXPECTED_SOURCE = {
    "marginal_likelihood.py": "3766ef2c1385316d1825ea62d72979829e00267f09e951644c58760783b6cbb7",
    "models.py": "cba208085f4e5e6373fe3b1bd5b862e9be66a6d1412a6a7750c4c4b053bbc952",
    "dependency_gradient.py": "4117a4c95753a0fbd251513d2556ddd79a0c4b40b7b735401ef89e6883d9039f",
    "evaluation_contract.py": "b30445b9fd7edefe64f502c3f33f5e39dd6ad737f8de05d0069be7b2bbb6d4e2",
}
EXPECTED_PROTOTYPE = "f30f8bc78b375f31b88df8f2df3cbce351eb260546590bc335548e050b3915e2"
REF_GLOBAL = {
    "ptq": {"epsilon": 82.52506788569912, "ln_sigma_sys": 7.837096471983798},
    "ptq-nu": {"epsilon": 0.9494002718727226, "ln_sigma_sys": -0.4485956975911134},
}
ARCHIVED_UPSILON_SCALED = {
    "ptq": -0.16904054604076085,
    "ptq-nu": 0.0506150688650736,
}
SCREEN_ARCHIVED_F = 6657.646629539804


class BudgetStop(RuntimeError):
    pass


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def theta_hash(theta):
    return hashlib.sha256(np.asarray(theta, dtype="<f8").tobytes()).hexdigest()


def clean(obj):
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        if not np.isfinite(value):
            return None
        return value
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    raise TypeError(type(obj))


def dump(name, obj):
    (NEW / name).write_text(json.dumps(clean(obj), indent=2) + "\n")


class Budget:
    def __init__(self, weights):
        self.weights = weights
        self.calls = 0
        self.eq = 0.0
        self.stopped = False

    def unit(self, model, gi, order):
        unit = self.weights[model]["192"][gi]["fullsample_equivalent"]
        if int(order) == 384:
            return 2.0 * unit
        if int(order) != 192:
            raise RuntimeError("unexpected order")
        return unit

    def afford(self, model, gis, order):
        need = sum(self.unit(model, gi, order) for gi in gis)
        return self.calls + len(gis) <= MAX_CALLS and self.eq + need <= MAX_EQ + 1e-12

    def charge(self, model, gi, order):
        unit = self.unit(model, gi, order)
        if self.calls + 1 > MAX_CALLS or self.eq + unit > MAX_EQ + 1e-12:
            self.stopped = True
            raise BudgetStop(model)
        self.calls += 1
        self.eq += unit


def bound_status(x, lo, hi):
    span = hi - lo
    tol = BOUND_ATOL_SPAN * max(1.0, span)
    if x < lo - tol or x > hi + tol:
        return "outside"
    if x - lo <= tol:
        return "lower"
    if hi - x <= tol:
        return "upper"
    return "interior"


def kkt(g, x, lo, hi):
    status = bound_status(x, lo, hi)
    if status == "lower":
        projected = 0.0 if g > 0 else g
        sign_ok = bool(g >= 0)
        direction = "outward" if g > 0 else "inward"
    elif status == "upper":
        projected = 0.0 if g < 0 else g
        sign_ok = bool(g <= 0)
        direction = "outward" if g < 0 else "inward"
    elif status == "interior":
        projected = g
        sign_ok = None
        direction = "free"
    else:
        projected, sign_ok, direction = None, None, "outside"
    return {
        "bound_status": status,
        "unprojected_analytic": g,
        "projected_kkt_component": projected,
        "sign_condition_holds": sign_ok,
        "active_bound_direction": direction,
        "gap_to_lower": float(x - lo),
        "gap_to_upper": float(hi - x),
    }


def consistent(analytic, fd):
    tol = max(CONSISTENT_ABS_FLOOR, CONSISTENT_REL * abs(analytic))
    return abs(analytic - fd) <= tol, tol


def main():
    log_path = NEW / "diagnostic" / "run.log"
    log = log_path.open("w")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    t0 = time.monotonic()
    state = {
        "task": "REV26-P1G-A4-JAC-ENG",
        "optimizer_called": False,
        "activated_as_optimizer_jacobian": False,
        "phases": [],
        "local": [],
        "screen_galaxies": [],
        "mond_screen_galaxies": [],
        "finite_differences": [],
        "prototype_spot": [],
        "prototype_full": [],
        "skipped": [],
        "locality": None,
        "formula_self_check": None,
        "source_hashes": {},
    }
    budget = None
    try:
        src = CAND / "ptquat"
        for name, expected in EXPECTED_SOURCE.items():
            got = sha256(src / name)
            state["source_hashes"][name] = got
            if got != expected:
                raise SystemExit("source hash mismatch " + name)
        prototype = DERIV / "diagnostic" / "b0_analytic_contract.py"
        got = sha256(prototype)
        state["source_hashes"]["b0_analytic_contract.py"] = got
        if got != EXPECTED_PROTOTYPE:
            raise SystemExit("prototype hash mismatch")
        if PRIMARY_B0 != "REV26-P1E-UNIFORM-EXPLICIT-B0-PHYSICAL-v1":
            raise SystemExit(PRIMARY_B0)
        state["formula_self_check"] = formula_self_check()
        say("formula self-check passed")

        weights = json.loads((IMPL / "results/frontiers_revision_2026_10/mle_pilot_a/GALAXY_WORK_WEIGHTS.json").read_text())
        for model in ("ptq", "ptq-nu", "ptq-screen", "mond-screen"):
            total = sum(row["fullsample_equivalent"] for row in weights[model]["192"])
            if abs(total - 1.0) > 1e-9:
                raise SystemExit("weight sum " + model + " " + str(total))
        budget = Budget(weights)
        gd = load_tidy_sparc(str(IMPL / "dataset/sparc_tidy.csv"))
        gs = [gd[k] for k in sorted(gd)]
        if len(gs) != 91 or sum(len(g.v_obs) for g in gs) != 1782:
            raise SystemExit("sample")
        if gs[9].name != "F574-1" or gs[16].name != "NGC0801" or gs[0].name != "CamB":
            raise SystemExit("galaxy order")
        names = {g.name: i for i, g in enumerate(gs)}
        cache = {}

        def official(theta, model, gi, order=192):
            key = (model, np.asarray(theta, dtype="<f8").tobytes(), gi, int(order))
            if key in cache:
                return cache[key]
            budget.charge(model, gi, order)
            p, _ = pack(theta, model)
            gg = as_g(gs[gi])
            record = integrate(gg, p, gi, "segmented_quantile", int(order), floors=("B0",))
            value = float(record["logL"]["B0"])
            cache[key] = value
            return value

        def pack(theta, model):
            layout = _layout(model, 91, True, None)
            pars = _unpack(np.asarray(theta, float), layout)
            pars.update(model=model, _H0=H0_SI, _c0=10.0, _c_slope=-0.1)
            return pars, layout

        def load_theta(model):
            path = EX / (model.replace("-", "_") + "_last_accepted_theta.npy")
            theta = np.load(path)
            return theta, theta_hash(theta)

        def partials(theta, model, gi):
            pars, layout = pack(theta, model)
            row = galaxy_partials(gs[gi], pars, gi, model, 192, charge=budget.charge)
            cache[(model, np.asarray(theta, dtype="<f8").tobytes(), gi, 192)] = row["official_logL"]
            return row, layout

        def difference(theta, model, index, scope, relative, order=192):
            bounds = theta_bounds(model, 91)[index]
            stencil, kind = step_stencil(float(theta[index]), bounds, relative, scheme="auto")
            for _, value in stencil:
                if value < bounds[0] - 1e-15 or value > bounds[1] + 1e-15:
                    raise RuntimeError("stencil left the original bounds")
            if not budget.afford(model, [gi for _, value in stencil if _ != "base" for gi in scope], order):
                # afford() above double-counts; compute explicitly
                pass
            needed = []
            for label, value in stencil:
                if label == "base":
                    continue
                for gi in scope:
                    key = (model, None, gi, order)  # placeholder
                    trial = np.array(theta, float, copy=True)
                    trial[index] = value
                    key = (model, np.asarray(trial, dtype="<f8").tobytes(), gi, int(order))
                    if key not in cache:
                        needed.append(gi)
            # Recompute needed list properly
            needed = []
            trials = []
            for label, value in stencil:
                trial = np.array(theta, float, copy=True)
                trial[index] = value
                trials.append((label, trial))
                if label == "base":
                    continue
                for gi in scope:
                    key = (model, np.asarray(trial, dtype="<f8").tobytes(), gi, int(order))
                    if key not in cache:
                        needed.append((gi, order))
            units = sum(budget.unit(model, gi, order) for gi, order in needed)
            if budget.calls + len(needed) > MAX_CALLS or budget.eq + units > MAX_EQ + 1e-12:
                return None
            values = []
            for label, trial in trials:
                total = 0.0
                for gi in scope:
                    total += official(trial if label != "base" else theta, model, gi, order)
                values.append(-total)
            denom = trials[1][1][index] - trials[0][1][index]
            gradient = (values[1] - values[0]) / denom
            step = abs(denom) / (2.0 if kind == "central" else 1.0)
            return {
                "relative_step": relative,
                "scheme": kind,
                "native_step": float(step),
                "finite_difference_dF": float(gradient),
                "order": int(order),
                "stencil_objectives": values,
            }

        def finish_local(model, galaxy, theta, rows_by_gi):
            gi = names[galaxy]
            row, layout = rows_by_gi
            index = layout["sl_U"].start + gi
            lo, hi = theta_bounds(model, 91)[index]
            dlogL = row["dlogL"]["upsilon"]
            analytic = None if dlogL is None else -dlogL
            rec = {
                "model": model,
                "galaxy": galaxy,
                "gi": gi,
                "coordinate": "upsilon",
                "native_index": int(index),
                "definition": "stellar mass-to-light; v_bar^2 = U*(v_disk^2+v_bulge^2)+v_gas^2",
                "parameter_value": float(theta[index]),
                "bounds": [lo, hi],
                "owning_galaxy_official_logL": row["official_logL"],
                "rederived_logL": row["rederived_logL"],
                "abs_gap": row["abs_gap"],
                "identity_pass": row["abs_gap"] <= IDENTITY_LIMIT,
                "analytic_dF": analytic,
                "analytic_equals_negative_owning_dlogL": True,
                "domain": row["domain"],
                "flags": row["flags"],
                "finite_differences": [],
            }
            if analytic is not None:
                rec["kkt"] = kkt(analytic, float(theta[index]), lo, hi)
            for relative in LOCAL_STEPS:
                got = difference(theta, model, index, [gi], relative, 192)
                if got is None:
                    state["skipped"].append({"what": "local fd", "model": model, "galaxy": galaxy, "relative": relative})
                    break
                ok, tol = consistent(analytic, got["finite_difference_dF"])
                got["abs_difference"] = abs(analytic - got["finite_difference_dF"])
                got["consistent"] = ok
                got["consistency_tolerance"] = tol
                rec["finite_differences"].append(got)
                state["finite_differences"].append({"scope": "local", "model": model, "galaxy": galaxy, **got, "analytic_dF": analytic})
            if len(rec["finite_differences"]) >= 2:
                rec["step_to_step"] = abs(rec["finite_differences"][0]["finite_difference_dF"] - rec["finite_differences"][1]["finite_difference_dF"])
            fds = rec["finite_differences"]
            if analytic is None or not fds:
                rec["classification"] = "DERIVATIVE_NOT_VALIDATED"
            elif rec["kkt"]["bound_status"] in ("upper", "lower"):
                rec["classification"] = "BOUNDARY_ONE_SIDED_REQUIRED"
            elif row["domain"] != "INTERIOR_ANALYTICALLY_DIFFERENTIABLE":
                rec["classification"] = row["domain"]
            elif all(item["consistent"] for item in fds) and (len(fds) < 2 or rec["step_to_step"] <= max(CONSISTENT_ABS_FLOOR, CONSISTENT_REL * abs(analytic))):
                rec["classification"] = "INTERIOR_ANALYTICALLY_DIFFERENTIABLE"
            else:
                rec["classification"] = "UNRESOLVED"
            rec["gtol_resolvable"] = bool(fds) and max(item["abs_difference"] for item in fds) <= 1e-6 and rec.get("step_to_step", 0.0) <= 1e-6
            state["local"].append(rec)
            say(model + " " + galaxy + " U " + str(analytic) + " " + rec["classification"])
            return rec

        # --- mandatory local fixtures ---
        begin = time.monotonic()
        for model in ("ptq-nu", "ptq"):
            theta, digest = load_theta(model)
            gi = names["F574-1"]
            row, layout = partials(theta, model, gi)
            rec = finish_local(model, "F574-1", theta, (row, layout))
            rec["theta_sha256"] = digest
            # Prototype spot check of the already tested global coordinates on this galaxy.
            from b0_analytic_contract import galaxy_b0_derivatives
            budget.charge(model, gi, 192)
            pars, _ = pack(theta, model)
            proto = galaxy_b0_derivatives(gs[gi], pars, gi, model, 192)
            state["prototype_spot"].append({
                "model": model,
                "galaxy": "F574-1",
                "prototype_dlogL_depsilon": proto["dlogL_depsilon"],
                "new_dlogL_depsilon": row["dlogL"]["epsilon"],
                "epsilon_difference": proto["dlogL_depsilon"] - row["dlogL"]["epsilon"],
                "prototype_dlogL_dln_sigma": proto["dlogL_dln_sigma"],
                "new_dlogL_dln_sigma": row["dlogL"]["ln_sigma_sys"],
                "ln_sigma_difference": proto["dlogL_dln_sigma"] - row["dlogL"]["ln_sigma_sys"],
                "prototype_gap": proto["abs_gap"],
                "new_gap": row["abs_gap"],
            })
        # Cross-galaxy locality: perturb F574-1 Upsilon and re-evaluate CamB.
        theta, _ = load_theta("ptq-nu")
        layout = _layout("ptq-nu", 91, True, None)
        index = layout["sl_U"].start + names["F574-1"]
        base = official(theta, "ptq-nu", 0, 192)
        trial = np.array(theta, float, copy=True)
        span = theta_bounds("ptq-nu", 91)[index]
        trial[index] = float(theta[index]) - (span[1] - span[0]) * 1e-4
        moved = official(trial, "ptq-nu", 0, 192)
        state["locality"] = {
            "model": "ptq-nu",
            "perturbed_galaxy": "F574-1",
            "observed_galaxy": gs[0].name,
            "base_logL": base,
            "perturbed_logL": moved,
            "abs_difference": abs(base - moved),
            "pass": abs(base - moved) == 0.0,
            "reason": "catalog_velocity_at_distance reads U[gi] only; nodes do not depend on Upsilon",
        }
        say("locality " + str(state["locality"]["abs_difference"]))
        # Additional archived galaxy on the ptq-nu accepted vector.
        theta, digest = load_theta("ptq-nu")
        row, layout = partials(theta, "ptq-nu", names[ADDITIONAL_GALAXY])
        rec = finish_local("ptq-nu", ADDITIONAL_GALAXY, theta, (row, layout))
        rec["theta_sha256"] = digest
        rec["selection_rule"] = "Predeclared additional fixture: NGC0801 is the archived coarse-grid outlier of the matched mond family. The direction audit itself only differentiated F574-1."
        state["phases"].append({"phase": "local_upsilon", "seconds": time.monotonic() - begin, "calls": budget.calls, "equivalents": budget.eq})

        # Optional 384 on the interior ptq-nu F574-1 direction, after 192.
        theta, _ = load_theta("ptq-nu")
        layout = _layout("ptq-nu", 91, True, None)
        index = layout["sl_U"].start + names["F574-1"]
        got384 = difference(theta, "ptq-nu", index, [names["F574-1"]], 1e-4, 384)
        if got384 is None:
            state["skipped"].append({"what": "diagnostic384 F574-1 upsilon"})
        else:
            analytic = next(item["analytic_dF"] for item in state["local"] if item["model"] == "ptq-nu" and item["galaxy"] == "F574-1")
            got384["abs_difference_vs_analytic"] = abs(analytic - got384["finite_difference_dF"])
            got384["note"] = "384 is diagnostic and does not replace fine192"
            state["finite_differences"].append({"scope": "local384", "model": "ptq-nu", "galaxy": "F574-1", "analytic_dF": analytic, **got384})
            say("384 local diff " + str(got384["abs_difference_vs_analytic"]))

        # --- screen analytic, all galaxies ---
        begin = time.monotonic()
        theta_s, digest_s = load_theta("ptq-screen")
        state["ptq_screen_theta_sha256"] = digest_s
        screen_rows = []
        for gi in range(91):
            row, layout = partials(theta_s, "ptq-screen", gi)
            screen_rows.append(row)
        state["screen_galaxies"] = [{k: row[k] for k in ("galaxy", "official_logL", "rederived_logL", "abs_gap", "domain", "flags", "dlogL")} for row in screen_rows]
        dF = {}
        for name in ("epsilon", "ln_q", "ln_sigma_sys"):
            dF[name] = -float(sum(row["dlogL"][name] for row in screen_rows))
        local_sum = -float(sum(row["dlogL"]["upsilon"] for row in screen_rows))
        flags = sorted({flag for row in screen_rows for flag in row["flags"] if flag not in ("negative_catalog_gas_squared", "negative_catalog_bulge_squared")})
        gap = max(row["abs_gap"] for row in screen_rows)
        objective = -float(sum(row["official_logL"] for row in screen_rows))
        layout_s = _layout("ptq-screen", 91, True, None)
        bounds_s = theta_bounds("ptq-screen", 91)
        screen_summary = {
            "theta_sha256": digest_s,
            "objective": objective,
            "archived_objective": SCREEN_ARCHIVED_F,
            "objective_minus_archived": objective - SCREEN_ARCHIVED_F,
            "max_abs_galaxy_logL_gap": gap,
            "identity_pass": gap <= IDENTITY_LIMIT,
            "analytic_dF": dF,
            "sum_local_upsilon_dF": local_sum,
            "domain_flags_excluding_catalog_sign": flags,
            "n_galaxies_with_floor_flags": sum(row["domain"] != "INTERIOR_ANALYTICALLY_DIFFERENTIABLE" for row in screen_rows),
            "epsilon_value": float(theta_s[layout_s["i_eps"]]),
            "ln_q_value": float(theta_s[layout_s["i_lq"]]),
            "q_value": float(np.exp(theta_s[layout_s["i_lq"]])),
            "epsilon_bound": bound_status(float(theta_s[layout_s["i_eps"]]), *bounds_s[layout_s["i_eps"]]),
            "ln_q_bound": bound_status(float(theta_s[layout_s["i_lq"]]), *bounds_s[layout_s["i_lq"]]),
            "weights_depend_on_theta": False,
            "q_enters_gN": False,
            "seconds": time.monotonic() - begin,
        }
        state["screen_summary"] = screen_summary
        say("screen F " + str(objective) + " eps " + str(dF["epsilon"]) + " lnq " + str(dF["ln_q"]) + " gap " + str(gap))

        # Local screen Upsilon FD at F574-1, using the cached baseline.
        gi = names["F574-1"]
        index = layout_s["sl_U"].start + gi
        dlogL = screen_rows[gi]["dlogL"]["upsilon"]
        analytic_u = -dlogL
        rec = {
            "model": "ptq-screen",
            "galaxy": "F574-1",
            "gi": gi,
            "coordinate": "upsilon",
            "native_index": int(index),
            "definition": "screen chain: Upsilon changes v_N^2 and therefore y inside nu_q",
            "parameter_value": float(theta_s[index]),
            "bounds": list(bounds_s[index]),
            "analytic_dF": analytic_u,
            "abs_gap": screen_rows[gi]["abs_gap"],
            "identity_pass": screen_rows[gi]["abs_gap"] <= IDENTITY_LIMIT,
            "domain": screen_rows[gi]["domain"],
            "flags": screen_rows[gi]["flags"],
            "kkt": kkt(analytic_u, float(theta_s[index]), *bounds_s[index]),
            "finite_differences": [],
            "theta_sha256": digest_s,
        }
        for relative in LOCAL_STEPS:
            got = difference(theta_s, "ptq-screen", index, [gi], relative, 192)
            if got is None:
                state["skipped"].append({"what": "screen local fd", "relative": relative})
                break
            ok, tol = consistent(analytic_u, got["finite_difference_dF"])
            got["abs_difference"] = abs(analytic_u - got["finite_difference_dF"])
            got["consistent"] = ok
            got["consistency_tolerance"] = tol
            rec["finite_differences"].append(got)
        if len(rec["finite_differences"]) >= 2:
            rec["step_to_step"] = abs(rec["finite_differences"][0]["finite_difference_dF"] - rec["finite_differences"][1]["finite_difference_dF"])
        fds = rec["finite_differences"]
        if rec["kkt"]["bound_status"] in ("upper", "lower"):
            rec["classification"] = "BOUNDARY_ONE_SIDED_REQUIRED"
        elif screen_rows[gi]["domain"] != "INTERIOR_ANALYTICALLY_DIFFERENTIABLE":
            rec["classification"] = screen_rows[gi]["domain"]
        elif fds and all(item["consistent"] for item in fds):
            rec["classification"] = "INTERIOR_ANALYTICALLY_DIFFERENTIABLE"
        else:
            rec["classification"] = "UNRESOLVED"
        state["local"].append(rec)
        say("ptq-screen F574-1 U " + str(analytic_u) + " " + rec["classification"])

        # Global finite differences, primary step first.
        for coordinate, index in (("epsilon", layout_s["i_eps"]), ("ln_q", layout_s["i_lq"])):
            for relative in GLOBAL_STEPS:
                begin_fd = time.monotonic()
                got = difference(theta_s, "ptq-screen", index, list(range(91)), relative, 192)
                if got is None:
                    state["skipped"].append({"what": "screen global fd", "coordinate": coordinate, "relative": relative, "equivalents": budget.eq})
                    say("skip global " + coordinate + " " + str(relative))
                    break
                ok, tol = consistent(screen_summary["analytic_dF"][coordinate], got["finite_difference_dF"])
                got.update({
                    "model": "ptq-screen",
                    "coordinate": coordinate,
                    "native_index": int(index),
                    "analytic_dF": screen_summary["analytic_dF"][coordinate],
                    "abs_difference": abs(screen_summary["analytic_dF"][coordinate] - got["finite_difference_dF"]),
                    "consistent": ok,
                    "consistency_tolerance": tol,
                    "seconds": time.monotonic() - begin_fd,
                    "bound_status": screen_summary["epsilon_bound"] if coordinate == "epsilon" else screen_summary["ln_q_bound"],
                })
                state["finite_differences"].append(got)
                say("FD " + coordinate + " " + str(relative) + " " + str(got["finite_difference_dF"]) + " diff " + str(got["abs_difference"]))

        # --- mapped mond-screen identity at the SAME physical point ---
        begin = time.monotonic()
        mapped = map_theta_ptqscreen_to_mondscreen(theta_s, 91, H0_SI)
        mb = np.asarray(theta_bounds("mond-screen", 91), float)
        inside = bool(np.all(mapped >= mb[:, 0]) and np.all(mapped <= mb[:, 1]))
        mond_rows = []
        if inside:
            for gi in range(91):
                row, layout_m = partials(mapped, "mond-screen", gi)
                mond_rows.append(row)
        state["mond_screen_inside_bounds"] = inside
        state["mond_screen_theta_sha256"] = theta_hash(mapped)
        if mond_rows:
            state["mond_screen_galaxies"] = [{k: row[k] for k in ("galaxy", "official_logL", "rederived_logL", "abs_gap", "domain", "flags")} for row in mond_rows]
            per = [abs(a["official_logL"] - b["official_logL"]) for a, b in zip(screen_rows, mond_rows)]
            dF_m = {
                "log10_a0": -float(sum(row["dlogL"]["log10_a0"] for row in mond_rows)),
                "ln_q": -float(sum(row["dlogL"]["ln_q"] for row in mond_rows)),
            }
            eps = float(theta_s[layout_s["i_eps"]])
            predicted = screen_summary["analytic_dF"]["epsilon"] * eps * float(np.log(10.0))
            gi = names["F574-1"]
            state["family"] = {
                "mapped_from": "ptq-screen last accepted",
                "not_the_mond_screen_callback_point": True,
                "inside_native_bounds": inside,
                "max_abs_per_galaxy_logL": max(per),
                "likelihood_identity_pass": max(per) <= IDENTITY_LIMIT,
                "mond_screen_objective": -float(sum(row["official_logL"] for row in mond_rows)),
                "ptq_screen_objective": objective,
                "dF_log10_a0": dF_m["log10_a0"],
                "predicted_from_epsilon": predicted,
                "log10_a0_chain_abs_difference": abs(dF_m["log10_a0"] - predicted),
                "dF_ln_q_ptq_screen": screen_summary["analytic_dF"]["ln_q"],
                "dF_ln_q_mond_screen": dF_m["ln_q"],
                "ln_q_abs_difference": abs(dF_m["ln_q"] - screen_summary["analytic_dF"]["ln_q"]),
                "F574-1_dF_upsilon_ptq_screen": -screen_rows[gi]["dlogL"]["upsilon"],
                "F574-1_dF_upsilon_mond_screen": -mond_rows[gi]["dlogL"]["upsilon"],
                "upsilon_abs_difference": abs(screen_rows[gi]["dlogL"]["upsilon"] - mond_rows[gi]["dlogL"]["upsilon"]),
                "mond_rederived_gap": max(row["abs_gap"] for row in mond_rows),
                "seconds": time.monotonic() - begin,
                "native_mond_screen_fd": "not run; chain rule checked against the ptq-screen analytic derivative",
            }
            say("family logL " + str(max(per)) + " chain " + str(state["family"]["log10_a0_chain_abs_difference"]))
        state["phases"].append({"phase": "screen_and_family", "seconds": time.monotonic() - t0, "calls": budget.calls, "equivalents": budget.eq})

        # Gate A full prototype only if a whole sample still fits.
        from b0_analytic_contract import evaluate_fixed_b0
        for model in ("ptq", "ptq-nu"):
            if not budget.afford(model, list(range(91)), 192):
                state["skipped"].append({"what": "full prototype rerun", "model": model, "equivalents": budget.eq})
                say("skip prototype " + model)
                continue
            theta, digest = load_theta(model)
            begin = time.monotonic()
            result = evaluate_fixed_b0(theta, model, gs, order=192, charge=budget.charge)
            grads = result["analytic_partial_gradients"]
            state["prototype_full"].append({
                "model": model,
                "theta_sha256": digest,
                "objective": result["objective"],
                "max_abs_galaxy_logL_gap": result["max_abs_galaxy_logL_gap"],
                "identity_pass": result["identity_pass"],
                "epsilon": grads["epsilon"]["native_dF"],
                "ln_sigma_sys": grads["ln_sigma_sys"]["native_dF"],
                "epsilon_minus_reference": grads["epsilon"]["native_dF"] - REF_GLOBAL[model]["epsilon"],
                "ln_sigma_minus_reference": grads["ln_sigma_sys"]["native_dF"] - REF_GLOBAL[model]["ln_sigma_sys"],
                "seconds": time.monotonic() - begin,
            })
            say("prototype " + model + " eps " + str(grads["epsilon"]["native_dF"]))
    except BudgetStop:
        say("budget stop")
    except Exception:
        state["error"] = traceback.format_exc()
        say(state["error"])
    finally:
        if budget is not None:
            state["budget"] = {
                "calls": budget.calls,
                "equivalents": budget.eq,
                "wall_seconds": time.monotonic() - t0,
                "caps": {"calls": MAX_CALLS, "equivalents": MAX_EQ, "wall_minutes": 90},
                "stopped": budget.stopped,
                "scipy_optimize_minimize_calls": 0,
            }
        dump("RUN_STATE.json", state)
        log.close()


if __name__ == "__main__":
    main()
