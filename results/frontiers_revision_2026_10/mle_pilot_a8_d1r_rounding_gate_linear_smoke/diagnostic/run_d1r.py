"""Linear PTQ smoke with a bounded roundoff exception for provisional trials only."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize

NEW = Path("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/mle_pilot_a8_d1r_rounding_gate_linear_smoke")
IMPL = Path("/home/rocky/PTQ-Quaternionic-RC")
D1 = Path("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/mle_pilot_a8_d1_linear_ptq_forensic")
sys.path.insert(0, str(NEW / "diagnostic"))
sys.path.insert(0, str(NEW / "candidate/src"))

from ptquat.data import load_tidy_sparc
from signed_jacobian import SCIENTIFIC_TARGET, VARIANCE_TOKEN, galaxy_jacobian_row

ABSOLUTE_LIMIT = 1e-9
RELATIVE_LIMIT = 1e-12
ULP_FACTOR = 8.0
REPLAY_TOLERANCE = 1e-8
START_F = 7101.671260839833
START_PROJECTED = 28.81029102042328
START_SHA = "96ac8469398b416aa0740522da74c844c12ea8e95364d69f3920f2df460fd6c4"
D1_TRIAL_SHA = "dc65035bc1f195edc72a71c9afa162fbc42a6cd6cec322454ada7f16310b5223"
D1_TRIAL_GAP = 2.7939677238464355e-09
NGC2403_LOGL = -5486154.322253605
JACOBIAN_SHA = "d23cb42195870aab92745d1735e54366435c8ae49f3d214f858164364d439585"
MODELS_SHA = "7214664b9a4a3e182f99748a9c4c7da91a98b930399a832589203395f3b89b1b"
MASK_SHA = "df0086b1e67f96403ff09f5c5bfb7f5818593d141d14234d752cb433a9511e6e"
TIDY_SHA = "3ab1c3c2b110c1e48680ebae77c1ceefaa4f778f81119dcf04881b630b274cbb"
EPS_WIDE = (1.0e-6, 3.999)
UPSILON = (0.05, 1.5)
SIGMA = (1e-3, 200.0)
OPTIONS = {"maxiter": 3, "maxls": 20, "maxcor": 10, "ftol": 1e-12, "gtol": 1e-6}
CLIP_FLAGS = {"y_floor", "a0_floor", "radius_floor", "gc_floor", "concentration_floor"}
FATAL_FLAGS = {
    "nonpositive_velocity", "velocity_formula_mismatch", "nu_q_formula_mismatch",
    "q_power_nonfinite", "nonpositive_vN2_off_floor", "nonpositive_halo_speed",
    "negative_stellar_component", "nonpositive_interpolation_argument",
}
SCIENTIFIC_CALLS = 0


class BudgetLimit(RuntimeError):
    pass


class IdentityStop(RuntimeError):
    pass


class SchemaError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def theta_hash(theta) -> str:
    return sha256_bytes(np.ascontiguousarray(theta, dtype="<f8").tobytes())


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def dumps(payload) -> bytes:
    return (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()


def atomic_json(path: Path, payload) -> str:
    data = dumps(payload)
    atomic_bytes(path, data)
    return sha256_bytes(data)


def atomic_npy(path: Path, theta) -> str:
    array = np.ascontiguousarray(theta, dtype="<f8")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, array)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return theta_hash(np.load(path))


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(row, allow_nan=False) + "\n").encode()
    with path.open("ab") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def positive_ulp(logl) -> float | None:
    try:
        magnitude = abs(float(logl))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(magnitude):
        return None
    ulp = math.ulp(magnitude)
    if (not math.isfinite(ulp)) or ulp <= 0.0:
        return None
    return float(ulp)


def relative_difference(absolute: float, left: float, right: float) -> float | None:
    if not all(math.isfinite(value) for value in (absolute, left, right)):
        return None
    return absolute / max(1.0, abs(left), abs(right))


def classify_galaxy(logl_a5, logl_a6, *, strict_only: bool) -> dict:
    if not (isinstance(logl_a5, (int, float)) and isinstance(logl_a6, (int, float))):
        return {"class": "FAIL", "reason": "nonfinite", "absolute": None, "relative": None, "ulp": None}
    if isinstance(logl_a5, bool) or isinstance(logl_a6, bool):
        return {"class": "FAIL", "reason": "nonfinite", "absolute": None, "relative": None, "ulp": None}
    if not math.isfinite(float(logl_a5)) or not math.isfinite(float(logl_a6)):
        return {"class": "FAIL", "reason": "nonfinite", "absolute": None, "relative": None, "ulp": None}
    absolute = abs(float(logl_a5) - float(logl_a6))
    relative = relative_difference(absolute, float(logl_a5), float(logl_a6))
    ulp = positive_ulp(logl_a5)
    if absolute <= ABSOLUTE_LIMIT:
        return {"class": "STRICT_PASS", "reason": "absolute", "absolute": absolute, "relative": relative, "ulp": ulp}
    if strict_only:
        return {"class": "FAIL", "reason": "accepted_absolute", "absolute": absolute, "relative": relative, "ulp": ulp}
    failed = []
    if relative is None or relative > RELATIVE_LIMIT:
        failed.append("relative")
    if ulp is None or absolute > ULP_FACTOR * ulp:
        failed.append("ulp")
    if failed:
        return {"class": "FAIL", "reason": "+".join(failed), "absolute": absolute, "relative": relative, "ulp": ulp}
    scaled = absolute / ulp
    if not math.isfinite(scaled):
        return {"class": "FAIL", "reason": "ulp", "absolute": absolute, "relative": relative, "ulp": ulp}
    return {
        "class": "TRIAL_ROUNDOFF_ADMISSIBLE",
        "reason": "bounded_roundoff",
        "absolute": absolute,
        "relative": relative,
        "ulp": ulp,
        "ulp_scaled": scaled,
    }


def classify_trial(rows: list[dict], *, strict_only: bool, expected_count: int = 91) -> dict:
    if len(rows) != expected_count or any(not row.get("complete", False) for row in rows):
        return {"class": "INCOMPLETE", "reason": "missing_row", "rows": len(rows)}
    classes = []
    for row in rows:
        if row.get("nonfinite_gradient") or row.get("domain_fail") or row.get("unexpected_branch"):
            return {"class": "FAIL", "reason": "domain_or_branch", "galaxy": row.get("galaxy")}
        verdict = classify_galaxy(row["logL_A5"], row["logL_A6"], strict_only=strict_only)
        classes.append(verdict)
        if verdict["class"] == "FAIL":
            return {"class": "FAIL", "reason": verdict["reason"], "galaxy": row.get("galaxy"), "detail": verdict}
    if any(item["class"] == "TRIAL_ROUNDOFF_ADMISSIBLE" for item in classes):
        return {
            "class": "TRIAL_ROUNDOFF_ADMISSIBLE",
            "roundoff_galaxies": sum(item["class"] == "TRIAL_ROUNDOFF_ADMISSIBLE" for item in classes),
            "max_absolute": max(item["absolute"] for item in classes),
            "max_relative": max(item["relative"] for item in classes),
            "max_ulp_scaled": max((item.get("ulp_scaled") or 0.0) for item in classes),
        }
    return {
        "class": "TRIAL_STRICT_PASS" if not strict_only else "ACCEPTED_STRICT_PASS",
        "roundoff_galaxies": 0,
        "max_absolute": max(item["absolute"] for item in classes),
        "max_relative": max(item["relative"] for item in classes),
        "max_ulp_scaled": max(((item["absolute"] / item["ulp"]) if item["ulp"] else 0.0) for item in classes),
    }


def native_bounds():
    bounds = [EPS_WIDE]
    bounds.extend([UPSILON] * 91)
    bounds.append((math.log(SIGMA[0]), math.log(SIGMA[1])))
    if len(bounds) != 93:
        raise SchemaError("dimension")
    return bounds


def unpack(theta):
    theta = np.asarray(theta, float)
    if theta.shape != (93,):
        raise SchemaError("dimension")
    return {"eps": float(theta[0]), "U": np.asarray(theta[1:92], float), "sigma": float(math.exp(theta[92]))}


def kkt(theta, gradient):
    x = np.asarray(theta, float)
    g = np.asarray(gradient, float)
    bounds = np.asarray(native_bounds(), float)
    lower = x <= bounds[:, 0]
    upper = x >= bounds[:, 1]
    projected = g.copy()
    projected[lower & (g > 0.0)] = 0.0
    projected[upper & (g < 0.0)] = 0.0
    return {
        "full_gradient_norm": float(np.max(np.abs(g))),
        "projected_gradient_norm": float(np.max(np.abs(projected))),
        "active_bounds": {
            "lower": np.flatnonzero(lower).astype(int).tolist(),
            "upper": np.flatnonzero(upper).astype(int).tolist(),
        },
        "outside_bounds": np.flatnonzero((x < bounds[:, 0]) | (x > bounds[:, 1])).astype(int).tolist(),
        "projected_gradient": projected,
    }


class Store:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def commit(self, theta, record: dict) -> dict:
        if record.get("kind") not in ("INITIAL", "ACCEPTED"):
            raise SchemaError("kind")
        if record["kind"] == "ACCEPTED" and record.get("identity_class") != "ACCEPTED_STRICT_PASS":
            raise SchemaError("accepted identity")
        digest = theta_hash(theta)
        if digest != record["theta_sha256"] or theta.shape != (93,):
            raise SchemaError("theta")
        index = int(record["iteration_index"])
        atomic_npy(self.root / f"accepted_{index}_theta.npy", theta)
        stored = {key: value for key, value in record.items() if key != "theta"}
        meta_sha = atomic_json(self.root / f"accepted_{index}_meta.json", stored)
        manifest = {
            "latest_kind": record["kind"],
            "iteration_index": index,
            "theta_file": f"accepted_{index}_theta.npy",
            "meta_file": f"accepted_{index}_meta.json",
            "theta_sha256": digest,
            "meta_sha256": meta_sha,
            "objective_after": record["objective_after"],
            "model": "ptq",
            "scientific_target": SCIENTIFIC_TARGET,
            "mask_sha256": MASK_SHA,
            "models_sha256": MODELS_SHA,
            "jacobian_sha256": JACOBIAN_SHA,
        }
        atomic_json(self.root / "ACCEPTED_MANIFEST.json", manifest)
        return manifest

    def recover(self):
        manifest = json.loads((self.root / "ACCEPTED_MANIFEST.json").read_text())
        if sha256_file(self.root / manifest["meta_file"]) != manifest["meta_sha256"]:
            raise SchemaError("mixed meta")
        theta = np.load(self.root / manifest["theta_file"])
        if theta_hash(theta) != manifest["theta_sha256"]:
            raise SchemaError("mixed theta")
        record = json.loads((self.root / manifest["meta_file"]).read_text())
        if record["iteration_index"] != manifest["iteration_index"] or record["objective_after"] != manifest["objective_after"]:
            raise SchemaError("manifest mismatch")
        if record["theta_sha256"] == D1_TRIAL_SHA:
            raise SchemaError("D1 trial cannot be a restart vector")
        return theta, record, manifest


def regression_tests() -> dict:
    root = NEW / "state/mock_regression"
    if root.exists():
        for child in sorted(root.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
    root.mkdir(parents=True, exist_ok=True)
    results = {}
    results["A_strict_pass"] = classify_galaxy(-40.0, -40.0 + 1e-12, strict_only=False)["class"] == "STRICT_PASS"
    ngc = classify_galaxy(NGC2403_LOGL, NGC2403_LOGL + D1_TRIAL_GAP, strict_only=False)
    results["B_ngc2403_roundoff"] = (
        ngc["class"] == "TRIAL_ROUNDOFF_ADMISSIBLE"
        and ngc["absolute"] > ABSOLUTE_LIMIT
        and ngc["ulp_scaled"] <= ULP_FACTOR
    )
    large_ulp = classify_galaxy(NGC2403_LOGL, NGC2403_LOGL + 9.0 * positive_ulp(NGC2403_LOGL), strict_only=False)
    results["C_above_8_ulp_fail"] = large_ulp["class"] == "FAIL" and "ulp" in large_ulp["reason"]
    relative_case = classify_galaxy(-100.0, -100.0 + 1e-8, strict_only=False)
    results["D_relative_fail"] = relative_case["class"] == "FAIL" and "relative" in relative_case["reason"]
    formula = classify_galaxy(-50.0, -49.0, strict_only=False)
    results["E_formula_mismatch_fail"] = formula["class"] == "FAIL"
    results["F_nonfinite_fail"] = classify_galaxy(float("nan"), -1.0, strict_only=False)["class"] == "FAIL"
    incomplete = classify_trial([{"complete": True, "logL_A5": -1.0, "logL_A6": -1.0, "galaxy": "only"}] * 90, strict_only=False)
    results["G_missing_row_incomplete"] = incomplete["class"] == "INCOMPLETE"
    provisional = [{"complete": True, "galaxy": "NGC2403", "logL_A5": NGC2403_LOGL, "logL_A6": NGC2403_LOGL + D1_TRIAL_GAP}]
    provisional += [{"complete": True, "galaxy": f"g{i}", "logL_A5": -20.0, "logL_A6": -20.0} for i in range(90)]
    trial_ok = classify_trial(provisional, strict_only=False)["class"] == "TRIAL_ROUNDOFF_ADMISSIBLE"
    accepted_fail = classify_trial(provisional, strict_only=True)["class"] == "FAIL"
    store = Store(root / "accepted")
    base = np.linspace(0.2, 0.4, 93)
    manifest = store.commit(base, {
        "kind": "INITIAL", "iteration_index": 0, "theta_sha256": theta_hash(base),
        "objective_before": 3.0, "objective_after": 3.0, "identity_class": "INITIAL",
        "projected_gradient_norm": 1.0, "full_gradient_norm": 1.0,
        "active_bounds": {"lower": [], "upper": []}, "evaluation_count": 0, "elapsed_seconds": 0.0,
    })
    refused = False
    try:
        store.commit(base, {
            "kind": "ACCEPTED", "iteration_index": 1, "theta_sha256": theta_hash(base),
            "objective_before": 3.0, "objective_after": 2.0, "identity_class": "TRIAL_ROUNDOFF_ADMISSIBLE",
            "projected_gradient_norm": 1.0, "full_gradient_norm": 1.0,
            "active_bounds": {"lower": [], "upper": []}, "evaluation_count": 1, "elapsed_seconds": 1.0,
        })
    except SchemaError:
        refused = True
    _, record, recovered = store.recover()
    results["H_roundoff_not_committed"] = trial_ok and accepted_fail and refused and recovered["iteration_index"] == 0
    good = base.copy(); good[10] = 0.25
    store.commit(good, {
        "kind": "ACCEPTED", "iteration_index": 1, "theta_sha256": theta_hash(good),
        "objective_before": 3.0, "objective_after": 2.5, "identity_class": "ACCEPTED_STRICT_PASS",
        "projected_gradient_norm": 0.5, "full_gradient_norm": 0.5,
        "active_bounds": {"lower": [], "upper": []}, "evaluation_count": 2, "elapsed_seconds": 2.0,
    })
    _, committed, manifest = store.recover()
    results["I_strict_commit"] = committed["identity_class"] == "ACCEPTED_STRICT_PASS" and manifest["iteration_index"] == 1
    atomic_json(root / "accepted" / "rejected_trial.json", {"kind": "REJECTED_TRIAL", "objective_after": -1e9})
    results["J_rejected_trial_ignored"] = store.recover()[2]["iteration_index"] == 1 and store.recover()[1]["objective_after"] == 2.5
    atomic_npy(root / "accepted" / "accepted_2_theta.npy", base)
    atomic_json(root / "accepted" / "accepted_2_meta.json", {"kind": "ACCEPTED", "iteration_index": 2, "theta_sha256": theta_hash(base), "objective_after": 1.0})
    results["K_crash_recovery"] = store.recover()[2]["iteration_index"] == 1
    edge = {
        "negative_logl_ulp": positive_ulp(NGC2403_LOGL),
        "zero_ulp": positive_ulp(0.0),
        "subnormal_ulp": positive_ulp(5e-324),
    }
    atomic_json(root / "edge.json", edge)
    results["L_ulp_json_edges"] = (
        edge["negative_logl_ulp"] > 0 and edge["zero_ulp"] > 0 and edge["subnormal_ulp"] > 0
        and math.isfinite(edge["negative_logl_ulp"]) and "Infinity" not in (root / "edge.json").read_text()
    )
    evidence = root / "evidence_before_exception.json"
    atomic_json(evidence, {"status": "PROVISIONAL_TRIAL", "theta_sha256": "ab" * 32, "retained": True})
    raised = False
    try:
        if evidence.exists():
            raise IdentityStop("after evidence")
    except IdentityStop:
        raised = True
    results["M_evidence_before_exception"] = raised and json.loads(evidence.read_text())["retained"] is True
    lock = root / "OPTIMIZER_INVOCATION.json"
    def try_invoke():
        if lock.exists():
            return False
        atomic_json(lock, {"started": True})
        return True
    first = try_invoke()
    second = try_invoke()
    results["N_second_invocation_refused"] = first is True and second is False
    inside = np.linspace(0.2, 0.5, 93)
    outside = inside.copy(); outside[0] = 10.0
    bounds = np.asarray(native_bounds(), float)
    results["O_dimension_and_bounds"] = (
        len(native_bounds()) == 93 and inside.shape == (93,)
        and np.all(inside >= bounds[:, 0]) and np.all(inside <= bounds[:, 1])
        and np.any(outside > bounds[:, 1]) and D1_TRIAL_SHA != START_SHA
    )
    results["scientific_likelihood_calls"] = SCIENTIFIC_CALLS
    keys = [key for key in results if key not in ("pass", "scientific_likelihood_calls")]
    results["pass"] = SCIENTIFIC_CALLS == 0 and all(bool(results[key]) for key in keys)
    return results


def verify_sources():
    frozen = {
        IMPL / "dataset/sparc_tidy.csv": TIDY_SHA,
        IMPL / "results/frontiers_revision_2026_10/mle_pilot_a4_domain_feasibility/S_DOMAIN13_MASK.csv": MASK_SHA,
        IMPL / "results/frontiers_revision_2026_10/mle_pilot_a5_signed_b0_integration/candidate/src/ptquat/models.py": MODELS_SHA,
        IMPL / "results/frontiers_revision_2026_10/mle_pilot_a6_signed_jac/diagnostic/signed_jacobian.py": JACOBIAN_SHA,
        NEW / "diagnostic/signed_jacobian.py": JACOBIAN_SHA,
        NEW / "candidate/src/ptquat/models.py": MODELS_SHA,
    }
    for path, digest in frozen.items():
        if sha256_file(path) != digest:
            raise SystemExit("STOP hash " + str(path))
    trial = np.load(D1 / "TRIAL_THETA.npy")
    if theta_hash(trial) != D1_TRIAL_SHA:
        raise SystemExit("STOP D1 trial hash")


def load_sample():
    tidy = pd.read_csv(IMPL / "dataset/sparc_tidy.csv")
    raw = pd.read_csv(IMPL / "dataset/raw/vizier_table2.csv")
    mask = pd.read_csv(IMPL / "results/frontiers_revision_2026_10/mle_pilot_a4_domain_feasibility/S_DOMAIN13_MASK.csv")
    joined = tidy.merge(raw, left_on=["galaxy", "r_kpc"], right_on=["Name", "Rad"], how="left", validate="one_to_one")
    included = mask.included.to_numpy(int) == 1
    if int(included.sum()) != 1769 or not np.array_equal(mask.source_recno.to_numpy(), joined.recno.to_numpy()):
        raise SystemExit("STOP mask")
    filtered = NEW / "logs/s_domain13_filtered_tidy.csv"
    tidy.loc[included].to_csv(filtered, index=False)
    loaded = load_tidy_sparc(str(filtered))
    galaxies = [loaded[name] for name in sorted(loaded)]
    if len(galaxies) != 91 or sum(len(g.v_obs) for g in galaxies) != 1769:
        raise SystemExit("STOP sample")
    weights = {
        row["galaxy"]: float(row["fullsample_equivalent"])
        for row in json.loads((IMPL / "results/frontiers_revision_2026_10/mle_pilot_a/GALAXY_WORK_WEIGHTS.json").read_text())["baryon"]["192"]
    }
    archived = np.asarray(json.loads(
        (IMPL / "results/frontiers_revision_2026_10/mle_pilot_a6_signed_jac/FIVE_FAMILY_GRADIENT_VECTORS.json").read_text()
    )["vectors"]["ptq"]["native_dF"], float)
    return galaxies, weights, archived


class Budget:
    def __init__(self, weights):
        self.weights = weights
        self.calls = 0
        self.eq = 0.0
        self.requests = 0
        self.t0 = time.perf_counter()

    def charge(self, name):
        weight = float(self.weights[name])
        if self.calls + 2 > 3200 or self.eq + 2.0 * weight > 32.0 or time.perf_counter() - self.t0 > 45 * 60:
            raise BudgetLimit("budget")
        self.calls += 2
        self.eq += 2.0 * weight

    def begin_request(self):
        if self.requests >= 12:
            raise BudgetLimit("request cap")
        if self.eq + 4.0 > 32.0 or self.calls + 364 > 3200 or time.perf_counter() - self.t0 > 45 * 60:
            raise BudgetLimit("reserve replay")
        self.requests += 1

    def allow_independent(self):
        if self.eq + 4.0 > 32.0 or self.calls + 364 > 3200 or time.perf_counter() - self.t0 > 45 * 60:
            raise BudgetLimit("reserve replay")


def main():
    global SCIENTIFIC_CALLS
    verify_sources()
    if scipy.__version__ != "1.17.1":
        raise SystemExit("STOP scipy " + scipy.__version__)
    tests = regression_tests()
    (NEW / "logs").mkdir(parents=True, exist_ok=True)
    atomic_json(NEW / "logs/policy_regression.json", tests)
    print("REGRESSION", tests["pass"], flush=True)
    if not tests["pass"]:
        raise SystemExit("ENGINEERING_REGRESSION_FAIL")
    if os.environ.get("D1R_REGRESSION_ONLY") == "1":
        print("REGRESSION_ONLY", flush=True)
        return
    galaxies, weights, archived_gradient = load_sample()
    theta0 = np.load(IMPL / "results/frontiers_revision_2026_10/mle_pilot_a/execution_v1/ptq_last_accepted_theta.npy")
    if theta_hash(theta0) != START_SHA or theta_hash(theta0) == D1_TRIAL_SHA or theta0.shape != (93,):
        raise SystemExit("STOP start vector")
    bounds = native_bounds()
    array_bounds = np.asarray(bounds, float)
    if np.any(theta0 < array_bounds[:, 0]) or np.any(theta0 > array_bounds[:, 1]):
        raise SystemExit("STOP bounds")
    budget = Budget(weights)
    ledger = NEW / "logs/galaxy_rows.jsonl"
    requests = []
    accepted = []
    cache = {}

    def evaluate(theta, digest):
        global SCIENTIFIC_CALLS
        pars = unpack(theta)
        rows = []
        records = []
        for gi, galaxy in enumerate(galaxies):
            budget.charge(galaxy.name)
            SCIENTIFIC_CALLS += 1
            raw = galaxy_jacobian_row(galaxy, pars, gi, "ptq")
            flags = list(raw["flags"])
            unexpected = any(flag in CLIP_FLAGS or flag in FATAL_FLAGS for flag in flags)
            domain_fail = raw["domain"] != "INTERIOR_ANALYTICALLY_DIFFERENTIABLE"
            gradient_bad = any(raw["dlogL"].get(key) is None for key in ("epsilon", "upsilon", "ln_sigma_sys"))
            verdict = classify_galaxy(raw["official_logL"], raw["rederived_logL"], strict_only=False)
            record = {
                "complete": True,
                "galaxy": galaxy.name,
                "gi": gi,
                "n_retained": int(len(galaxy.v_obs)),
                "theta_sha256": digest,
                "logL_A5": float(raw["official_logL"]),
                "logL_A6": float(raw["rederived_logL"]),
                "absolute_difference": verdict["absolute"],
                "relative_difference": verdict["relative"],
                "ulp": verdict["ulp"],
                "ulp_scaled": (
                    None if verdict["ulp"] in (None, 0.0) or verdict["absolute"] is None
                    or not math.isfinite(verdict["absolute"] / verdict["ulp"])
                    else verdict["absolute"] / verdict["ulp"]
                ),
                "galaxy_class": verdict["class"],
                "flags": flags,
                "domain": raw["domain"],
                "unexpected_branch": unexpected,
                "domain_fail": domain_fail,
                "nonfinite_gradient": gradient_bad or not all(math.isfinite(float(raw["dlogL"][key])) for key in ("epsilon", "upsilon", "ln_sigma_sys")),
            }
            append_jsonl(ledger, record)
            records.append(record)
            rows.append(raw)
        gradient = np.full(93, np.nan)
        if any(record["nonfinite_gradient"] for record in records):
            return {"records": records, "gradient": gradient, "F": None, "rows": rows}
        gradient[0] = -sum(float(row["dlogL"]["epsilon"]) for row in rows)
        for gi, row in enumerate(rows):
            gradient[1 + gi] = -float(row["dlogL"]["upsilon"])
        gradient[92] = -sum(float(row["dlogL"]["ln_sigma_sys"]) for row in rows)
        value = -sum(float(row["official_logL"]) for row in rows)
        return {"records": records, "gradient": gradient, "F": float(value), "rows": rows}

    def summarize(result, digest, theta, kind):
        records = result["records"]
        trial = classify_trial(records, strict_only=False)
        finite_gradient = bool(np.isfinite(result["gradient"]).all()) and result["F"] is not None and math.isfinite(result["F"])
        diag = None if not finite_gradient else kkt(theta, result["gradient"])
        absolutes = [item["absolute_difference"] for item in records if item["absolute_difference"] is not None]
        relatives = [item["relative_difference"] for item in records if item["relative_difference"] is not None]
        scaled = [item["ulp_scaled"] for item in records if item["ulp_scaled"] is not None]
        row = {
            "kind": kind,
            "identity_class": trial["class"],
            "theta_sha256": digest,
            "epsilon": float(theta[0]),
            "ln_sigma": float(theta[92]),
            "F_A5": result["F"],
            "F_A6": None if any(not math.isfinite(item["logL_A6"]) for item in records) else -sum(item["logL_A6"] for item in records),
            "max_absolute": None if not absolutes else max(absolutes),
            "max_relative": None if not relatives else max(relatives),
            "max_ulp_scaled": None if not scaled else max(scaled),
            "roundoff_galaxies": sum(item["galaxy_class"] == "TRIAL_ROUNDOFF_ADMISSIBLE" for item in records),
            "active_bounds": None if diag is None else diag["active_bounds"],
            "outside_bounds": None if diag is None else diag["outside_bounds"],
            "projected_gradient_norm": None if diag is None else diag["projected_gradient_norm"],
            "full_gradient_norm": None if diag is None else diag["full_gradient_norm"],
            "finite_gradient": finite_gradient,
            "flags": sorted({flag for item in records for flag in item["flags"]}),
            "domains": sorted({item["domain"] for item in records}),
            "calls": budget.calls,
            "equivalents": budget.eq,
        }
        if row["F_A5"] is not None and not math.isfinite(row["F_A5"]):
            row["F_A5"] = None
        return row, trial, diag

    theta0_digest = theta_hash(theta0)
    start = evaluate(theta0, theta0_digest)
    start_summary, start_trial, start_kkt = summarize(start, theta0_digest, theta0, "INITIAL")
    gradient_gap = float(np.max(np.abs(start["gradient"] - archived_gradient)))
    preflight_ok = (
        start_trial["class"] == "TRIAL_STRICT_PASS"
        and abs(start["F"] - START_F) <= REPLAY_TOLERANCE
        and abs(start_kkt["projected_gradient_norm"] - START_PROJECTED) <= 1e-8
        and gradient_gap <= 1e-8
        and not start_kkt["outside_bounds"]
    )
    print(
        f"PREFLIGHT F {start['F']:.12f} proj {start_kkt['projected_gradient_norm']:.14f} "
        f"gap {start_summary['max_absolute']:.3e} ok {preflight_ok}",
        flush=True,
    )
    if not preflight_ok:
        atomic_json(NEW / "logs/run_state.json", {"preflight": start_summary, "gradient_gap": gradient_gap})
        raise SystemExit("STOP preflight")
    cache[theta0_digest] = start
    store = Store(NEW / "state/scientific")
    store.commit(theta0, {
        "kind": "INITIAL",
        "iteration_index": 0,
        "theta_sha256": theta0_digest,
        "objective_before": start["F"],
        "objective_after": start["F"],
        "identity_class": "INITIAL",
        "projected_gradient_norm": start_kkt["projected_gradient_norm"],
        "full_gradient_norm": start_kkt["full_gradient_norm"],
        "active_bounds": start_kkt["active_bounds"],
        "evaluation_count": 0,
        "elapsed_seconds": 0.0,
        "gradient": start["gradient"].tolist(),
        "projected_gradient": start_kkt["projected_gradient"].tolist(),
    })
    requests.append(dict(start_summary, request_index=0, step_l2=0.0, step_inf=0.0))
    failure = None
    first_trial = None
    scipy_result = None

    def fun(theta):
        nonlocal first_trial
        theta = np.asarray(theta, float).copy()
        digest = theta_hash(theta)
        budget.begin_request()
        reference = theta0 if not accepted else np.asarray(accepted[-1]["theta"], float)
        displacement = theta - reference
        if digest == theta0_digest and digest in cache:
            result = cache[digest]
        elif digest in cache:
            result = cache[digest]
        else:
            if np.any(theta < array_bounds[:, 0]) or np.any(theta > array_bounds[:, 1]) or not np.isfinite(theta).all():
                append_jsonl(NEW / "logs/requests.jsonl", {"status": "TRIAL_IDENTITY_FAIL", "theta_sha256": digest, "reason": "bounds"})
                raise IdentityStop("trial outside original bounds")
            result = evaluate(theta, digest)
            cache[digest] = result
        summary, trial, diag = summarize(result, digest, theta, "PROVISIONAL_TRIAL")
        admissible = (
            trial["class"] in ("TRIAL_STRICT_PASS", "TRIAL_ROUNDOFF_ADMISSIBLE")
            and summary["finite_gradient"]
            and diag is not None
            and not diag["outside_bounds"]
        )
        if digest == theta0_digest and trial["class"] == "TRIAL_STRICT_PASS":
            summary["kind"] = "INITIAL"
        elif trial["class"] == "TRIAL_STRICT_PASS" and admissible:
            summary["kind"] = "TRIAL_STRICT_PASS"
        elif trial["class"] == "TRIAL_ROUNDOFF_ADMISSIBLE" and admissible:
            summary["kind"] = "TRIAL_ROUNDOFF_ADMISSIBLE"
        else:
            summary["kind"] = "TRIAL_IDENTITY_FAIL"
        summary.update({
            "request_index": budget.requests,
            "step_l2": float(np.linalg.norm(displacement)),
            "step_inf": float(np.max(np.abs(displacement))),
            "matches_d1_trial": digest == D1_TRIAL_SHA,
        })
        append_jsonl(NEW / "logs/requests.jsonl", {key: value for key, value in summary.items() if key != "active_bounds"})
        requests.append(summary)
        if digest != theta0_digest and first_trial is None:
            first_trial = summary
            print(
                f"FIRST_TRIAL {digest} class {summary['kind']} F {summary['F_A5']} "
                f"eps {summary['epsilon']} max_abs {summary['max_absolute']}",
                flush=True,
            )
        if not admissible:
            raise IdentityStop("trial identity " + trial["class"] + " " + str(trial.get("reason")))
        print(f"REQUEST {summary['request_index']} {summary['kind']} F {result['F']:.8f}", flush=True)
        return result["F"], result["gradient"].copy()

    def callback(xk):
        if hasattr(xk, "x"):
            xk = xk.x
        theta = np.asarray(xk, float).copy()
        digest = theta_hash(theta)
        append_jsonl(NEW / "logs/requests.jsonl", {
            "kind": "SOLVER_ACCEPTED_PENDING_VALIDATION",
            "theta_sha256": digest,
        })
        budget.allow_independent()
        result = evaluate(theta, digest)
        cache[digest + ":accepted"] = result
        strict = classify_trial(result["records"], strict_only=True)
        if strict["class"] != "ACCEPTED_STRICT_PASS" or not np.isfinite(result["gradient"]).all():
            raise IdentityStop("ACCEPTED_IDENTITY_FAIL")
        diag = kkt(theta, result["gradient"])
        if diag["outside_bounds"]:
            raise IdentityStop("accepted outside bounds")
        previous_theta = theta0 if not accepted else np.asarray(accepted[-1]["theta"], float)
        previous_f = start["F"] if not accepted else accepted[-1]["objective_after"]
        if result["F"] > previous_f + REPLAY_TOLERANCE:
            raise IdentityStop("accepted objective increased")
        record = {
            "kind": "ACCEPTED",
            "iteration_index": len(accepted) + 1,
            "theta_sha256": digest,
            "objective_before": previous_f,
            "objective_after": result["F"],
            "delta_f": result["F"] - previous_f,
            "identity_class": "ACCEPTED_STRICT_PASS",
            "max_absolute": strict["max_absolute"],
            "projected_gradient_norm": diag["projected_gradient_norm"],
            "full_gradient_norm": diag["full_gradient_norm"],
            "active_bounds": diag["active_bounds"],
            "evaluation_count": budget.requests,
            "elapsed_seconds": time.perf_counter() - budget.t0,
            "step_l2": float(np.linalg.norm(theta - previous_theta)),
            "step_inf": float(np.max(np.abs(theta - previous_theta))),
            "gradient": result["gradient"].tolist(),
            "projected_gradient": diag["projected_gradient"].tolist(),
            "epsilon": float(theta[0]),
            "ln_sigma": float(theta[92]),
        }
        store.commit(theta, record)
        record["theta"] = theta.tolist()
        accepted.append(record)
        print(f"ACCEPTED {record['iteration_index']} F {record['objective_after']:.8f} dF {record['delta_f']:.8f}", flush=True)
        if len(accepted) > 3:
            raise BudgetLimit("accepted iteration cap")

    invocation = NEW / "state/OPTIMIZER_INVOCATION.json"
    if invocation.exists():
        raise SystemExit("STOP optimizer already invoked")
    atomic_json(invocation, {"model": "ptq", "started": True, "scientific_invocations": 1, "start_sha256": START_SHA})
    try:
        scipy_result = minimize(
            fun, theta0, method="L-BFGS-B", jac=True, bounds=bounds, callback=callback, options=dict(OPTIONS),
        )
    except (IdentityStop, BudgetLimit, SchemaError) as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
        print("STOP", failure, flush=True)
    theta_end, meta_end, manifest = store.recover()
    try:
        replay = evaluate(theta_end, theta_hash(theta_end))
    except BudgetLimit as exc:
        failure = failure or {"type": "BudgetLimit", "message": "replay " + str(exc)}
        atomic_json(NEW / "logs/run_state.json", {"failure": failure, "accepted": len(accepted)})
        raise SystemExit("BUDGET_LIMIT before endpoint replay")
    replay_strict = classify_trial(replay["records"], strict_only=True)
    replay_kkt = kkt(theta_end, replay["gradient"])
    endpoint = {
        "pass": (
            replay_strict["class"] == "ACCEPTED_STRICT_PASS"
            and abs(replay["F"] - meta_end["objective_after"]) <= REPLAY_TOLERANCE
            and not replay_kkt["outside_bounds"]
            and bool(np.isfinite(replay["gradient"]).all())
            and replay["records"] and all(row["domain"] == "INTERIOR_ANALYTICALLY_DIFFERENTIABLE" for row in replay["records"])
        ),
        "recovered_kind": meta_end["kind"],
        "recovered_iteration_index": meta_end["iteration_index"],
        "recovered_theta_sha256": theta_hash(theta_end),
        "recorded_objective": meta_end["objective_after"],
        "replay_objective": replay["F"],
        "objective_abs_gap": abs(replay["F"] - meta_end["objective_after"]),
        "max_per_galaxy_gap": replay_strict.get("max_absolute"),
        "projected_gradient_norm": replay_kkt["projected_gradient_norm"],
        "full_gradient_norm": replay_kkt["full_gradient_norm"],
        "outside_bounds": replay_kkt["outside_bounds"],
        "galaxies": len(replay["records"]),
        "quadrature": "fine192",
    }
    accepted_public = []
    for row in accepted:
        accepted_public.append({key: value for key, value in row.items() if key not in ("theta", "gradient", "projected_gradient")})
    payload = {
        "regression": tests,
        "preflight": start_summary,
        "gradient_gap": gradient_gap,
        "first_trial": first_trial,
        "requests": requests,
        "accepted": accepted_public,
        "failure": failure,
        "scipy": None if scipy_result is None else {
            "success": bool(scipy_result.success),
            "status": int(scipy_result.status),
            "message": str(scipy_result.message),
            "nit": int(scipy_result.nit),
            "nfev": int(scipy_result.nfev),
            "njev": int(scipy_result.njev),
        },
        "endpoint": endpoint,
        "manifest": manifest,
        "budget": {
            "a5_a6_marginal_pairs": budget.calls // 2,
            "counted_a5_plus_a6": budget.calls,
            "pair_equivalents": budget.eq,
            "optimizer_requests": budget.requests,
            "wall_seconds": time.perf_counter() - budget.t0,
            "scientific_likelihood_calls": SCIENTIFIC_CALLS,
            "minimize_invocations": 1,
        },
        "scipy_version": scipy.__version__,
        "factr_from_ftol": OPTIONS["ftol"] / float(np.finfo(float).eps),
    }
    atomic_json(NEW / "logs/run_state.json", payload)
    print("D1R_DONE", json.dumps({
        "accepted": len(accepted),
        "first_class": None if first_trial is None else first_trial["identity_class"],
        "first_match": None if first_trial is None else first_trial["matches_d1_trial"],
        "endpoint": endpoint["pass"],
        "failure": failure,
        "scipy": payload["scipy"],
    }), flush=True)


if __name__ == "__main__":
    main()
