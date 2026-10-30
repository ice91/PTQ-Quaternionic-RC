"""Repaired ptq-nu smoke. Mock callback tests run before the single optimizer call."""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize

NEW = Path("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/mle_pilot_a7_r1_ptqnu_smoke_repair")
IMPL = Path("/home/rocky/PTQ-Quaternionic-RC")
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
sys.path.insert(0, str(NEW / "diagnostic"))
sys.path.insert(0, str(NEW / "candidate/src"))

from ptquat.constants import C_LIGHT, H0_SI
from ptquat.data import load_tidy_sparc
from signed_jacobian import SCIENTIFIC_TARGET, VARIANCE_TOKEN, galaxy_jacobian_row

MODEL = "ptq-nu"
IDENTITY_LIMIT = 1e-9
REPLAY_TOLERANCE = 1e-8
EXPECTED_INITIAL_F = 6614.933826799582
THETA_CONTENT_SHA = "7abded50306ec11a32380f1ae85c555faef1d2d9318f39c673b691ae974473b2"
A7_ACCEPTED_STEP1_SHA = "7d96e270aac86144b23df21f2a4e61590d46bfa3f2a1a67f353048ab5cfe8eda"
A7_REQUEST5_SHA = "ed6114309504d12618d4596dceb0ca7c251fb2d61d149f4814fc3f9cf6960b09"
JACOBIAN_SHA = "d23cb42195870aab92745d1735e54366435c8ae49f3d214f858164364d439585"
MODELS_SHA = "7214664b9a4a3e182f99748a9c4c7da91a98b930399a832589203395f3b89b1b"
MASK_SHA = "df0086b1e67f96403ff09f5c5bfb7f5818593d141d14234d752cb433a9511e6e"
TIDY_SHA = "3ab1c3c2b110c1e48680ebae77c1ceefaa4f778f81119dcf04881b630b274cbb"
A0_BOUNDS = (5.0e-11, 2.0e-10)
UPSILON_BOUNDS = (0.05, 1.5)
SIGMA_BOUNDS = (1e-3, 200.0)
MAX_ACCEPTED = 3
MAX_OPTIMIZER_REQUESTS = 24
MAX_GALAXY_CALLS = 3000
MAX_EQUIVALENTS = 32.0
MAX_WALL = 35 * 60
FROZEN_OPTIONS = {"maxiter": 3, "maxls": 20, "maxcor": 10, "ftol": 1e-12, "gtol": 1e-6}
CLIP_FLAGS = {"y_floor", "a0_floor", "radius_floor", "gc_floor", "concentration_floor"}
REQUIRED_FIELDS = (
    "iteration_index",
    "theta_sha256",
    "objective_before",
    "objective_after",
    "projected_gradient_norm",
    "full_gradient_norm",
    "active_bounds",
    "evaluation_count",
    "elapsed_seconds",
)


class BudgetLimit(RuntimeError):
    pass


class SourceStop(RuntimeError):
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


def atomic_json(path: Path, payload) -> str:
    data = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()
    atomic_bytes(path, data)
    return sha256_bytes(data)


def atomic_npy(path: Path, theta) -> str:
    array = np.ascontiguousarray(theta, dtype="<f8")
    tmp = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("wb") as handle:
        np.save(handle, array)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    if theta_hash(np.load(path)) != theta_hash(array):
        raise SourceStop("atomic theta hash mismatch")
    return theta_hash(array)


def native_bounds():
    eps_lo = A0_BOUNDS[0] / (C_LIGHT * H0_SI)
    eps_hi = A0_BOUNDS[1] / (C_LIGHT * H0_SI)
    bounds = [(eps_lo, eps_hi)]
    bounds.extend([UPSILON_BOUNDS] * 91)
    bounds.append((math.log(SIGMA_BOUNDS[0]), math.log(SIGMA_BOUNDS[1])))
    if len(bounds) != 93:
        raise SourceStop("dimension")
    return bounds


def unpack(theta):
    theta = np.asarray(theta, float)
    return {"eps": float(theta[0]), "U": np.asarray(theta[1:92], float), "sigma": float(math.exp(theta[92]))}


def kkt(theta, gradient):
    x = np.asarray(theta, float)
    g = np.asarray(gradient, float)
    bounds = np.asarray(native_bounds(), float)
    lower = x <= bounds[:, 0]
    upper = x >= bounds[:, 1]
    outside = (x < bounds[:, 0]) | (x > bounds[:, 1])
    projected = g.copy()
    projected[lower & (g > 0)] = 0.0
    projected[upper & (g < 0)] = 0.0
    return {
        "full_gradient_norm": float(np.max(np.abs(g))),
        "projected_gradient_norm": float(np.max(np.abs(projected))),
        "active_bounds": {
            "lower": np.flatnonzero(lower).astype(int).tolist(),
            "upper": np.flatnonzero(upper).astype(int).tolist(),
        },
        "outside_bounds": np.flatnonzero(outside).astype(int).tolist(),
        "projected_gradient": projected.tolist(),
    }


def validate_schema(record: dict) -> None:
    missing = [name for name in REQUIRED_FIELDS if name not in record]
    if missing:
        raise SchemaError("missing " + ",".join(missing))
    if record.get("kind") not in ("INITIAL", "ACCEPTED"):
        raise SchemaError("record kind is not an accepted-state kind")
    for name in ("objective_before", "objective_after", "projected_gradient_norm", "full_gradient_norm", "elapsed_seconds"):
        value = record[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise SchemaError("nonfinite " + name)
    if not isinstance(record["iteration_index"], int) or isinstance(record["iteration_index"], bool):
        raise SchemaError("iteration_index")
    if not isinstance(record["evaluation_count"], int) or isinstance(record["evaluation_count"], bool):
        raise SchemaError("evaluation_count")
    if not isinstance(record["theta_sha256"], str) or len(record["theta_sha256"]) != 64:
        raise SchemaError("theta_sha256")
    bounds = record["active_bounds"]
    if not isinstance(bounds, dict) or "lower" not in bounds or "upper" not in bounds:
        raise SchemaError("active_bounds")


class Store:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def commit(self, theta, record: dict) -> dict:
        validate_schema(record)
        theta = np.asarray(theta, float)
        digest = theta_hash(theta)
        if digest != record["theta_sha256"]:
            raise SchemaError("theta does not match theta_sha256")
        index = record["iteration_index"]
        theta_name = f"accepted_{index}_theta.npy"
        meta_name = f"accepted_{index}_meta.json"
        atomic_npy(self.root / theta_name, theta)
        meta_sha = atomic_json(self.root / meta_name, record)
        manifest = {
            "latest_kind": record["kind"],
            "iteration_index": index,
            "theta_file": theta_name,
            "meta_file": meta_name,
            "theta_sha256": digest,
            "meta_sha256": meta_sha,
            "objective_after": record["objective_after"],
            "model": MODEL,
            "scientific_target": SCIENTIFIC_TARGET,
            "mask_sha256": MASK_SHA,
            "models_sha256": MODELS_SHA,
            "jacobian_sha256": JACOBIAN_SHA,
        }
        atomic_json(self.root / "ACCEPTED_MANIFEST.json", manifest)
        return manifest

    def recover(self):
        manifest_path = self.root / "ACCEPTED_MANIFEST.json"
        if not manifest_path.exists():
            raise SourceStop("no committed manifest")
        manifest = json.loads(manifest_path.read_text())
        meta_path = self.root / manifest["meta_file"]
        theta_path = self.root / manifest["theta_file"]
        if sha256_file(meta_path) != manifest["meta_sha256"]:
            raise SourceStop("mixed meta version")
        theta = np.load(theta_path)
        if theta_hash(theta) != manifest["theta_sha256"]:
            raise SourceStop("mixed theta version")
        record = json.loads(meta_path.read_text())
        validate_schema(record)
        if record["theta_sha256"] != manifest["theta_sha256"]:
            raise SourceStop("meta does not name this theta")
        if record["iteration_index"] != manifest["iteration_index"]:
            raise SourceStop("index mismatch")
        if record["objective_after"] != manifest["objective_after"]:
            raise SourceStop("objective mismatch")
        return theta, record, manifest


def _record(index, theta, before, after, projected, full, bounds, evaluations, elapsed, kind):
    return {
        "kind": kind,
        "iteration_index": index,
        "theta_sha256": theta_hash(theta),
        "objective_before": before,
        "objective_after": after,
        "projected_gradient_norm": projected,
        "full_gradient_norm": full,
        "active_bounds": bounds,
        "evaluation_count": evaluations,
        "elapsed_seconds": elapsed,
        "model": MODEL,
        "scientific_target": SCIENTIFIC_TARGET,
        "mask_sha256": MASK_SHA,
        "models_sha256": MODELS_SHA,
        "jacobian_sha256": JACOBIAN_SHA,
    }


def regression_tests() -> dict:
    root = NEW / "state/regression_fixture"
    if root.exists():
        for child in root.iterdir():
            if child.is_file():
                child.unlink()
    store = Store(root)
    base = np.linspace(0.2, 0.3, 93)
    results = {}

    def current():
        return store.recover()

    initial = _record(0, base, 10.0, 10.0, 1.0, 2.0, {"lower": [], "upper": []}, 0, 0.0, "INITIAL")
    store.commit(base, initial)
    theta, record, manifest = current()
    results["A_first_callback"] = record["iteration_index"] == 0 and record["objective_after"] == 10.0 and theta_hash(theta) == theta_hash(base)

    step1 = base.copy(); step1[0] = 0.21
    rec1 = _record(1, step1, 10.0, 9.5, 0.8, 1.5, {"lower": [1], "upper": []}, 2, 1.0, "ACCEPTED")
    store.commit(step1, rec1)
    theta, record, manifest = current()
    before_manifest = sha256_file(root / "ACCEPTED_MANIFEST.json")
    results["B_second_callback"] = record["iteration_index"] == 1 and record["objective_before"] == 10.0 and record["objective_after"] == 9.5

    step2 = step1.copy(); step2[1] = 0.22
    rec2 = _record(2, step2, 9.5, 9.0, 0.4, 1.1, {"lower": [1], "upper": [4]}, 4, 2.0, "ACCEPTED")
    store.commit(step2, rec2)
    theta, record, _ = current()
    results["C_third_callback"] = record["iteration_index"] == 2 and record["objective_after"] == 9.0

    atomic_json(root / "rejected_trial.json", {"kind": "REJECTED_TRIAL", "objective_after": -1000.0, "theta_sha256": "0" * 64})
    theta, record, _ = current()
    results["D_rejected_trial_ignored"] = record["iteration_index"] == 2 and record["objective_after"] == 9.0

    atomic_npy(root / "newer_uncommitted_theta.npy", base)
    theta, record, _ = current()
    results["E_uncommitted_trial_ignored"] = theta_hash(theta) == theta_hash(step2)

    broken = dict(rec2)
    del broken["objective_after"]
    broken["iteration_index"] = 3
    failed_closed = False
    try:
        store.commit(step2, broken)
    except SchemaError:
        failed_closed = True
    theta, record, _ = current()
    results["F_missing_objective_after"] = failed_closed and record["iteration_index"] == 2 and record["objective_after"] == 9.0

    atomic_npy(root / "accepted_2_theta.npy", base)
    refused = False
    try:
        store.recover()
    except SourceStop:
        refused = True
    store.commit(step2, rec2)
    results["G_hash_mismatch_rejected"] = refused and theta_hash(current()[0]) == theta_hash(step2)

    staged = step2.copy(); staged[2] = 0.33
    staged_record = _record(3, staged, 9.0, 8.5, 0.2, 0.9, {"lower": [], "upper": []}, 6, 3.0, "ACCEPTED")
    validate_schema(staged_record)
    atomic_npy(root / "accepted_3_theta.npy", staged)
    atomic_json(root / "accepted_3_meta.json", staged_record)
    theta, record, _ = current()
    results["H_crash_before_manifest"] = record["iteration_index"] == 2 and theta_hash(theta) == theta_hash(step2)

    store.commit(staged, staged_record)
    _, first, _ = current()
    nxt = staged.copy(); nxt[3] = 0.34
    rec_next = _record(4, nxt, first["objective_after"], 8.0, 0.1, 0.5, {"lower": [], "upper": []}, 8, 4.0, "ACCEPTED")
    store.commit(nxt, rec_next)
    _, second, _ = current()
    results["I_objective_continuity"] = second["objective_before"] == first["objective_after"] == 8.5 and second["objective_after"] == 8.0

    trial = {"kind": "TRIAL", "iteration_index": 9, "objective_after": -1e9, "theta_sha256": theta_hash(base)}
    atomic_json(root / "trial_record.json", trial)
    _, recovered, manifest = current()
    results["J_trial_is_not_accepted"] = recovered["kind"] == "ACCEPTED" and manifest["latest_kind"] == "ACCEPTED" and recovered["iteration_index"] == 4
    results["pass"] = all(results[key] for key in list(results) if key != "pass")
    results["scientific_likelihood_calls"] = 0
    return results


class Budget:
    def __init__(self, weights):
        self.weights = weights
        self.calls = 0
        self.eq = 0.0
        self.optimizer_requests = 0
        self.t0 = time.perf_counter()

    def charge_galaxy(self, name):
        weight = float(self.weights[name])
        if self.calls + 1 > MAX_GALAXY_CALLS or self.eq + weight > MAX_EQUIVALENTS or time.perf_counter() - self.t0 > MAX_WALL:
            raise BudgetLimit("budget")
        self.calls += 1
        self.eq += weight

    def begin_optimizer_request(self):
        if self.optimizer_requests >= MAX_OPTIMIZER_REQUESTS:
            raise BudgetLimit("optimizer request cap")
        if self.calls + 182 > MAX_GALAXY_CALLS or self.eq + 2.0 > MAX_EQUIVALENTS or time.perf_counter() - self.t0 > MAX_WALL:
            raise BudgetLimit("reserve terminal replay")
        self.optimizer_requests += 1


def main():
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
    tests = regression_tests()
    (NEW / "logs/callback_schema_tests.json").write_text(json.dumps(tests, indent=2) + "\n")
    print("REGRESSION", tests["pass"], flush=True)
    if not tests["pass"]:
        raise SystemExit("ENGINEERING_REGRESSION_FAIL")

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
    weights = {row["galaxy"]: float(row["fullsample_equivalent"]) for row in json.loads(
        (IMPL / "results/frontiers_revision_2026_10/mle_pilot_a/GALAXY_WORK_WEIGHTS.json").read_text()
    )["baryon"]["192"]}
    budget = Budget(weights)
    theta0 = np.load(IMPL / "results/frontiers_revision_2026_10/mle_pilot_a/execution_v1/ptq_nu_last_accepted_theta.npy")
    digest0 = theta_hash(theta0)
    if digest0 != THETA_CONTENT_SHA or digest0 in (A7_ACCEPTED_STEP1_SHA, A7_REQUEST5_SHA) or theta0.shape != (93,):
        raise SystemExit("STOP start vector")
    bounds = native_bounds()
    array_bounds = np.asarray(bounds, float)
    if not np.isfinite(theta0).all() or np.any(theta0 < array_bounds[:, 0]) or np.any(theta0 > array_bounds[:, 1]):
        raise SystemExit("STOP bounds")

    def evaluate(theta):
        started = time.perf_counter()
        pars = unpack(theta)
        rows = []
        for gi, galaxy in enumerate(galaxies):
            budget.charge_galaxy(galaxy.name)
            rows.append(galaxy_jacobian_row(galaxy, pars, gi, MODEL))
        gap = max(row["abs_gap"] for row in rows)
        flags = sorted({flag for row in rows for flag in row["flags"]})
        domains = sorted({row["domain"] for row in rows})
        if gap > IDENTITY_LIMIT:
            raise SourceStop("objective identity " + str(gap))
        if any(row["dlogL"]["epsilon"] is None for row in rows):
            raise SourceStop("nonfinite derivative")
        if any(flag in CLIP_FLAGS for flag in flags) or domains != ["INTERIOR_ANALYTICALLY_DIFFERENTIABLE"]:
            raise SourceStop("unexpected floor " + ",".join(flags))
        gradient = np.zeros(93)
        gradient[0] = -sum(row["dlogL"]["epsilon"] for row in rows)
        for gi, row in enumerate(rows):
            gradient[1 + gi] = -row["dlogL"]["upsilon"]
        gradient[92] = -sum(row["dlogL"]["ln_sigma_sys"] for row in rows)
        value = -sum(row["official_logL"] for row in rows)
        if not np.isfinite(gradient).all() or not math.isfinite(value):
            raise SourceStop("nonfinite objective")
        return {
            "F": float(value),
            "gradient": gradient,
            "max_identity_gap": float(gap),
            "flags": flags,
            "domains": domains,
            "seconds": time.perf_counter() - started,
            "galaxy_logL": [row["official_logL"] for row in rows],
        }

    initial = evaluate(theta0)
    if abs(initial["F"] - EXPECTED_INITIAL_F) > REPLAY_TOLERANCE:
        raise SystemExit("STOP initial F " + str(initial["F"]))
    archived = np.asarray(json.loads(
        (IMPL / "results/frontiers_revision_2026_10/mle_pilot_a6_signed_jac/FIVE_FAMILY_GRADIENT_VECTORS.json").read_text()
    )["vectors"]["ptq-nu"]["native_dF"], float)
    grad_gap = float(np.max(np.abs(archived - initial["gradient"])))
    if grad_gap > 1e-8:
        raise SystemExit("STOP gradient " + str(grad_gap))
    initial_kkt = kkt(theta0, initial["gradient"])
    if initial_kkt["outside_bounds"]:
        raise SystemExit("STOP outside")
    store = Store(NEW / "state/scientific")
    store.commit(theta0, _record(
        0, theta0, initial["F"], initial["F"], initial_kkt["projected_gradient_norm"],
        initial_kkt["full_gradient_norm"], initial_kkt["active_bounds"], 0, 0.0, "INITIAL",
    ))
    requests = []
    accepted = []
    cache = {}
    warning_log = []

    def fun(theta):
        theta = np.asarray(theta, float).copy()
        digest = theta_hash(theta)
        budget.begin_optimizer_request()
        if digest in cache:
            cached = cache[digest]
            requests.append({"request_index": budget.optimizer_requests, "theta_sha256": digest, "kind": "TRIAL", "status": "CACHE_HIT", "F": cached["F"]})
            return cached["F"], cached["gradient"].copy()
        if (not np.isfinite(theta).all()) or np.any(theta < array_bounds[:, 0]) or np.any(theta > array_bounds[:, 1]):
            raise SourceStop("trial outside original bounds")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = evaluate(theta)
        for item in caught:
            warning_log.append({"request": budget.optimizer_requests, "category": item.category.__name__, "message": str(item.message)})
        cache[digest] = result
        diag = kkt(theta, result["gradient"])
        row = {
            "request_index": budget.optimizer_requests,
            "theta_sha256": digest,
            "kind": "TRIAL",
            "status": "COMPLETE_NOT_ACCEPTED",
            "F": result["F"],
            "full_gradient_norm": diag["full_gradient_norm"],
            "projected_gradient_norm": diag["projected_gradient_norm"],
            "max_identity_gap": result["max_identity_gap"],
            "flags": result["flags"],
            "seconds": result["seconds"],
            "galaxy_calls": budget.calls,
            "equivalents": budget.eq,
            "elapsed_seconds": time.perf_counter() - budget.t0,
        }
        requests.append(row)
        with (NEW / "logs/requests.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(f"REQUEST {row['request_index']} F {result['F']:.8f}", flush=True)
        return result["F"], result["gradient"].copy()

    def callback(xk):
        if hasattr(xk, "x"):
            xk = xk.x
        theta = np.asarray(xk, float).copy()
        digest = theta_hash(theta)
        if digest not in cache:
            raise SourceStop("callback theta was not an exact completed evaluation")
        result = cache[digest]
        diag = kkt(theta, result["gradient"])
        if diag["outside_bounds"]:
            raise SourceStop("accepted theta outside bounds")
        previous_f = initial["F"] if not accepted else accepted[-1]["objective_after"]
        if result["F"] > previous_f + REPLAY_TOLERANCE:
            raise SourceStop("accepted objective increased")
        record = _record(
            len(accepted) + 1, theta, previous_f, result["F"], diag["projected_gradient_norm"],
            diag["full_gradient_norm"], diag["active_bounds"], budget.optimizer_requests,
            time.perf_counter() - budget.t0, "ACCEPTED",
        )
        record.update({
            "delta_f": result["F"] - previous_f,
            "step_l2": float(np.linalg.norm(theta - (theta0 if not accepted else np.asarray(accepted[-1]["theta"], float)))),
            "step_inf": float(np.max(np.abs(theta - (theta0 if not accepted else np.asarray(accepted[-1]["theta"], float))))),
            "gradient": result["gradient"].tolist(),
            "projected_gradient": diag["projected_gradient"],
            "galaxy_calls": budget.calls,
            "equivalents": budget.eq,
        })
        validate_schema(record)
        store.commit(theta, {k: v for k, v in record.items() if k != "theta"})
        record["theta"] = theta.tolist()
        accepted.append(record)
        for item in requests:
            if item["theta_sha256"] == digest:
                item["kind"] = "ACCEPTED"
                item["status"] = "ACCEPTED"
        print(f"ACCEPTED {record['iteration_index']} F {record['objective_after']:.8f} dF {record['delta_f']:.8f}", flush=True)
        if len(accepted) > MAX_ACCEPTED:
            raise BudgetLimit("accepted iteration cap")

    invocation = NEW / "state/OPTIMIZER_INVOCATION.json"
    if invocation.exists():
        raise SystemExit("STOP optimizer already invoked")
    atomic_json(invocation, {"model": MODEL, "started": True, "scientific_invocations": 1, "start_sha256": digest0})
    scipy_result = None
    failure = None
    try:
        scipy_result = minimize(
            fun, theta0, method="L-BFGS-B", jac=True, bounds=bounds, callback=callback, options=dict(FROZEN_OPTIONS),
        )
    except (BudgetLimit, SourceStop, SchemaError) as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
    theta_end, meta_end, manifest = store.recover()
    if accepted and theta_hash(theta_end) != accepted[-1]["theta_sha256"]:
        raise SystemExit("STOP recovery mismatch")
    if not accepted and theta_hash(theta_end) != digest0:
        raise SystemExit("STOP initial recovery mismatch")
    replay_eval = evaluate(theta_end)
    replay_kkt = kkt(theta_end, replay_eval["gradient"])
    recorded_f = meta_end["objective_after"]
    endpoint = {
        "accepted_steps": len(accepted),
        "recovered_kind": meta_end["kind"],
        "recovered_iteration_index": meta_end["iteration_index"],
        "recovered_theta_sha256": theta_hash(theta_end),
        "galaxies": 91,
        "max_per_galaxy_path_gap": replay_eval["max_identity_gap"],
        "recorded_objective": recorded_f,
        "replay_objective": replay_eval["F"],
        "objective_abs_gap": abs(replay_eval["F"] - recorded_f),
        "objective_within_tolerance": abs(replay_eval["F"] - recorded_f) <= REPLAY_TOLERANCE,
        "identity_pass": replay_eval["max_identity_gap"] <= IDENTITY_LIMIT,
        "outside_bounds": replay_kkt["outside_bounds"],
        "gradient_finite": bool(np.isfinite(replay_eval["gradient"]).all()),
        "flags": replay_eval["flags"],
        "domains": replay_eval["domains"],
        "projected_gradient_norm": replay_kkt["projected_gradient_norm"],
        "full_gradient_norm": replay_kkt["full_gradient_norm"],
        "active_bounds": replay_kkt["active_bounds"],
        "quadrature": "fine192",
    }
    for item in requests:
        if item["kind"] == "TRIAL" and item["theta_sha256"] == digest0:
            item["kind"] = "INITIAL"
    payload = {
        "regression": tests,
        "scipy_version": scipy.__version__,
        "options": FROZEN_OPTIONS,
        "factr_from_ftol": FROZEN_OPTIONS["ftol"] / float(np.finfo(float).eps),
        "initial_F": initial["F"],
        "initial_abs_gap_from_a7": abs(initial["F"] - EXPECTED_INITIAL_F),
        "archived_gradient_max_abs_gap": grad_gap,
        "initial_kkt": {k: v for k, v in initial_kkt.items() if k != "projected_gradient"},
        "requests": requests,
        "accepted": [{k: v for k, v in row.items() if k not in ("theta", "gradient", "projected_gradient")} for row in accepted],
        "warning_log": warning_log,
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
        "budget": {
            "calls": budget.calls,
            "equivalents": budget.eq,
            "optimizer_requests": budget.optimizer_requests,
            "wall_seconds": time.perf_counter() - budget.t0,
        },
        "manifest": manifest,
    }
    (NEW / "logs/run_state.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "accepted": len(accepted),
        "F0": initial["F"],
        "steps": [{k: row[k] for k in ("iteration_index", "objective_before", "objective_after", "delta_f", "projected_gradient_norm")} for row in accepted],
        "endpoint_gap": endpoint["objective_abs_gap"],
        "identity": endpoint["max_per_galaxy_path_gap"],
        "failure": failure,
        "scipy": payload["scipy"],
        "budget": payload["budget"],
        "initial_projected": initial_kkt["projected_gradient_norm"],
        "final_projected": replay_kkt["projected_gradient_norm"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
