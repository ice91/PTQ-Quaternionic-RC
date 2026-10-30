"""One isolated ptq-nu L-BFGS-B smoke test. At most three accepted iterations."""
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

NEW = Path("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/mle_pilot_a7_ptqnu_smoke")
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

# Predeclared before any optimizer result is known.
MODEL = "ptq-nu"
IDENTITY_LIMIT = 1e-9
REPLAY_OBJECTIVE_FLOOR = 1e-8
THETA_CONTENT_SHA = "7abded50306ec11a32380f1ae85c555faef1d2d9318f39c673b691ae974473b2"
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


class BudgetLimit(RuntimeError):
    pass


class SourceStop(RuntimeError):
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
    raw = array.tobytes()
    # NumPy's .npy format, written atomically, then content hash checked on read.
    tmp = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("wb") as handle:
        np.save(handle, array)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    loaded = np.load(path)
    if theta_hash(loaded) != sha256_bytes(raw):
        raise SourceStop("atomic theta hash mismatch")
    return sha256_bytes(raw)


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
    span = bounds[:, 1] - bounds[:, 0]
    return {
        "full_gradient_inf_norm": float(np.max(np.abs(g))),
        "projected_gradient_inf_norm": float(np.max(np.abs(projected))),
        "scaled_projected_gradient_inf_norm": float(np.max(np.abs(projected * span))),
        "lower_contacts": np.flatnonzero(lower).astype(int).tolist(),
        "upper_contacts": np.flatnonzero(upper).astype(int).tolist(),
        "outside_bounds": np.flatnonzero(outside).astype(int).tolist(),
        "projected_gradient": projected.tolist(),
        "scaled_projected_gradient": (projected * span).tolist(),
    }


class Store:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def commit(self, kind: str, accepted_index: int, theta, meta: dict) -> dict:
        theta = np.asarray(theta, float)
        digest = atomic_npy(self.root / f"accepted_{accepted_index}_theta.npy", theta)
        body = dict(meta)
        body.update({
            "kind": kind,
            "accepted_index": accepted_index,
            "theta_sha256": digest,
            "model": MODEL,
            "scientific_target": SCIENTIFIC_TARGET,
            "variance_token": VARIANCE_TOKEN,
            "models_sha256": MODELS_SHA,
            "jacobian_sha256": JACOBIAN_SHA,
            "mask_sha256": MASK_SHA,
        })
        meta_name = f"accepted_{accepted_index}_meta.json"
        meta_sha = atomic_json(self.root / meta_name, body)
        manifest = {
            "latest_kind": kind,
            "latest_accepted_index": accepted_index,
            "theta_file": f"accepted_{accepted_index}_theta.npy",
            "meta_file": meta_name,
            "theta_sha256": digest,
            "meta_sha256": meta_sha,
            "model": MODEL,
            "scientific_target": SCIENTIFIC_TARGET,
            "origin_theta_content_sha256": THETA_CONTENT_SHA,
        }
        atomic_json(self.root / "ACCEPTED_MANIFEST.json", manifest)
        return manifest

    def recover(self):
        manifest_path = self.root / "ACCEPTED_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text())
        if sha256_file(manifest_path) != sha256_bytes(manifest_path.read_bytes()):
            raise SourceStop("manifest reread mismatch")
        meta_path = self.root / manifest["meta_file"]
        theta_path = self.root / manifest["theta_file"]
        if sha256_file(meta_path) != manifest["meta_sha256"]:
            raise SourceStop("mixed meta version")
        theta = np.load(theta_path)
        if theta_hash(theta) != manifest["theta_sha256"]:
            raise SourceStop("mixed theta version")
        meta = json.loads(meta_path.read_text())
        if meta["theta_sha256"] != manifest["theta_sha256"]:
            raise SourceStop("meta does not name this theta")
        if meta["accepted_index"] != manifest["latest_accepted_index"]:
            raise SourceStop("index mismatch")
        return theta, meta, manifest


def recovery_tests() -> dict:
    root = NEW / "state/recovery_fixture"
    if root.exists():
        for child in root.iterdir():
            child.unlink()
    store = Store(root)
    theta0 = np.linspace(0.1, 0.2, 93)
    store.commit("INITIAL_STATE", 0, theta0, {"note": "fixture"})
    atomic_json(root / "trial_not_accepted.json", {"kind": "TRIAL_EVALUATION", "theta": [1.0]})
    recovered, meta, _ = store.recover()
    first = theta_hash(recovered) == theta_hash(theta0) and meta["kind"] == "INITIAL_STATE"
    theta1 = theta0.copy()
    theta1[0] = 0.11
    store.commit("ACCEPTED_ITERATION", 1, theta1, {"note": "accepted fixture"})
    atomic_json(root / "later_trial.json", {"kind": "REJECTED_TRIAL", "theta": [9.0]})
    recovered, meta, _ = store.recover()
    second = theta_hash(recovered) == theta_hash(theta1) and meta["accepted_index"] == 1
    # Corrupt the committed theta after the manifest. Recovery must refuse it.
    atomic_npy(root / "accepted_1_theta.npy", theta0)
    refused = False
    try:
        store.recover()
    except SourceStop:
        refused = True
    store.commit("ACCEPTED_ITERATION", 1, theta1, {"note": "restored fixture"})
    recovered, _, _ = store.recover()
    restored = theta_hash(recovered) == theta_hash(theta1)
    return {
        "ignores_newer_trial_and_returns_initial": first,
        "returns_latest_accepted_not_later_trial": second,
        "refuses_mixed_theta_and_manifest": refused,
        "restores_consistent_pair": restored,
        "scientific_likelihood_calls": 0,
        "pass": bool(first and second and refused and restored),
    }


class Budget:
    def __init__(self, weights):
        self.weights = weights
        self.calls = 0
        self.eq = 0.0
        self.optimizer_requests = 0
        self.t0 = time.perf_counter()

    def charge_galaxy(self, name):
        weight = float(self.weights[name])
        if self.calls + 1 > MAX_GALAXY_CALLS or self.eq + weight > MAX_EQUIVALENTS:
            raise BudgetLimit("galaxy budget")
        if time.perf_counter() - self.t0 > MAX_WALL:
            raise BudgetLimit("wall budget")
        self.calls += 1
        self.eq += weight

    def begin_optimizer_request(self):
        if self.optimizer_requests >= MAX_OPTIMIZER_REQUESTS:
            raise BudgetLimit("optimizer request cap")
        if self.calls + 91 > MAX_GALAXY_CALLS or self.eq + 1.0 > MAX_EQUIVALENTS:
            raise BudgetLimit("reserve terminal replay")
        if time.perf_counter() - self.t0 > MAX_WALL:
            raise BudgetLimit("wall budget")
        self.optimizer_requests += 1


def main():
    frozen = {
        IMPL / "dataset/sparc_tidy.csv": TIDY_SHA,
        IMPL / "results/frontiers_revision_2026_10/mle_pilot_a4_domain_feasibility/S_DOMAIN13_MASK.csv": MASK_SHA,
        IMPL / "results/frontiers_revision_2026_10/mle_pilot_a5_signed_b0_integration/candidate/src/ptquat/models.py": MODELS_SHA,
        NEW / "candidate/src/ptquat/models.py": MODELS_SHA,
        NEW / "diagnostic/signed_jacobian.py": JACOBIAN_SHA,
        IMPL / "results/frontiers_revision_2026_10/mle_pilot_a6_signed_jac/diagnostic/signed_jacobian.py": JACOBIAN_SHA,
    }
    for path, digest in frozen.items():
        if sha256_file(path) != digest:
            raise SystemExit("STOP hash " + str(path))
    recovery = recovery_tests()
    if not recovery["pass"]:
        raise SystemExit("STOP recovery fixture")
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
    if theta_hash(theta0) != THETA_CONTENT_SHA or theta0.shape != (93,):
        raise SystemExit("STOP theta")
    bounds = native_bounds()
    array_bounds = np.asarray(bounds, float)
    if not np.isfinite(theta0).all() or np.any(theta0 < array_bounds[:, 0]) or np.any(theta0 > array_bounds[:, 1]):
        raise SystemExit("STOP bounds")

    def evaluate(theta, phase):
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
        if any(flag in CLIP_FLAGS for flag in flags) or any(domain != "INTERIOR_ANALYTICALLY_DIFFERENTIABLE" for domain in domains):
            raise SourceStop("unexpected floor " + ",".join(flags))
        gradient = np.zeros(93)
        gradient[0] = -sum(row["dlogL"]["epsilon"] for row in rows)
        for gi, row in enumerate(rows):
            gradient[1 + gi] = -row["dlogL"]["upsilon"]
        gradient[92] = -sum(row["dlogL"]["ln_sigma_sys"] for row in rows)
        if not np.isfinite(gradient).all():
            raise SourceStop("nonfinite gradient")
        objective = -sum(row["official_logL"] for row in rows)
        if not math.isfinite(objective):
            raise SourceStop("nonfinite objective")
        return {
            "phase": phase,
            "theta_sha256": theta_hash(theta),
            "objective": float(objective),
            "gradient": gradient,
            "max_identity_gap": float(gap),
            "flags": flags,
            "domains": domains,
            "seconds": time.perf_counter() - started,
            "galaxy_logL": [row["official_logL"] for row in rows],
            "rederived_logL": [row["rederived_logL"] for row in rows],
        }

    pre_t = time.perf_counter()
    initial = evaluate(theta0, "preflight")
    preflight_seconds = time.perf_counter() - pre_t
    archived = json.loads((IMPL / "results/frontiers_revision_2026_10/mle_pilot_a6_signed_jac/FIVE_FAMILY_GRADIENT_VECTORS.json").read_text())
    archived_grad = np.asarray(archived["vectors"]["ptq-nu"]["native_dF"], float)
    grad_gap = float(np.max(np.abs(archived_grad - initial["gradient"])))
    if grad_gap > 1e-8:
        raise SystemExit("STOP archived gradient mismatch " + str(grad_gap))
    initial_kkt = kkt(theta0, initial["gradient"])
    if initial_kkt["outside_bounds"]:
        raise SystemExit("STOP outside")
    replay_tolerance = max(REPLAY_OBJECTIVE_FLOOR, 91.0 * initial["max_identity_gap"])
    store = Store(NEW / "state/scientific")
    checkpoint_t = time.perf_counter()
    store.commit("INITIAL_STATE", 0, theta0, {
        "objective": initial["objective"],
        "full_gradient_inf_norm": initial_kkt["full_gradient_inf_norm"],
        "projected_gradient_inf_norm": initial_kkt["projected_gradient_inf_norm"],
        "lower_contacts": initial_kkt["lower_contacts"],
        "upper_contacts": initial_kkt["upper_contacts"],
    })
    checkpoint_seconds = time.perf_counter() - checkpoint_t
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
            requests.append({"request_index": len(requests) + 1, "theta_sha256": digest, "status": "CACHE_HIT",
                             "objective": cached["objective"], "kind": "TRIAL_EVALUATION"})
            return cached["objective"], cached["gradient"].copy()
        array_b = np.asarray(bounds, float)
        if (not np.isfinite(theta).all()) or np.any(theta < array_b[:, 0]) or np.any(theta > array_b[:, 1]):
            raise SourceStop("trial outside original bounds")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = evaluate(theta, "optimizer")
        for item in caught:
            warning_log.append({"request": len(requests) + 1, "category": item.category.__name__, "message": str(item.message)})
        cache[digest] = result
        diag = kkt(theta, result["gradient"])
        requests.append({
            "request_index": len(requests) + 1,
            "theta_sha256": digest,
            "status": "COMPLETE_NOT_YET_ACCEPTED",
            "kind": "TRIAL_EVALUATION",
            "objective": result["objective"],
            "full_gradient_inf_norm": diag["full_gradient_inf_norm"],
            "projected_gradient_inf_norm": diag["projected_gradient_inf_norm"],
            "max_identity_gap": result["max_identity_gap"],
            "flags": result["flags"],
            "seconds": result["seconds"],
            "calls": budget.calls,
            "equivalents": budget.eq,
            "wall_seconds": time.perf_counter() - budget.t0,
        })
        with (NEW / "logs/requests.jsonl").open("a") as handle:
            handle.write(json.dumps(requests[-1]) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(f"REQUEST {len(requests)} F {result['objective']:.8f} eq {budget.eq:.3f}", flush=True)
        return result["objective"], result["gradient"].copy()

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
        previous_theta = theta0 if not accepted else np.asarray(accepted[-1]["theta"], float)
        previous_f = initial["objective"] if not accepted else accepted[-1]["objective_after"]
        if result["objective"] > previous_f + replay_tolerance:
            raise SourceStop("accepted objective increased beyond replay tolerance")
        record = {
            "kind": "ACCEPTED_ITERATION",
            "accepted_index": len(accepted) + 1,
            "theta": theta.tolist(),
            "theta_sha256": digest,
            "objective_before": previous_f,
            "objective_after": result["objective"],
            "delta_f": result["objective"] - previous_f,
            "step_l2": float(np.linalg.norm(theta - previous_theta)),
            "step_inf": float(np.max(np.abs(theta - previous_theta))),
            "full_gradient_inf_norm": diag["full_gradient_inf_norm"],
            "projected_gradient_inf_norm": diag["projected_gradient_inf_norm"],
            "lower_contacts": diag["lower_contacts"],
            "upper_contacts": diag["upper_contacts"],
            "requests_so_far": len(requests),
            "calls": budget.calls,
            "equivalents": budget.eq,
            "wall_seconds": time.perf_counter() - budget.t0,
        }
        accepted.append(record)
        store.commit("ACCEPTED_ITERATION", record["accepted_index"], theta, {k: v for k, v in record.items() if k != "theta"})
        for item in requests:
            if item["theta_sha256"] == digest:
                item["kind"] = "ACCEPTED_ITERATION"
                item["status"] = "ACCEPTED"
        print(f"ACCEPTED {record['accepted_index']} F {record['objective_after']:.8f} dF {record['delta_f']:.8f}", flush=True)
        if len(accepted) > MAX_ACCEPTED:
            raise BudgetLimit("accepted iteration cap")

    invocation = NEW / "state/OPTIMIZER_INVOCATION.json"
    if invocation.exists():
        raise SystemExit("STOP optimizer already invoked")
    atomic_json(invocation, {"model": MODEL, "maxiter": 3, "started": False})
    scipy_result = None
    failure = None
    opt_t = time.perf_counter()
    atomic_json(invocation, {"model": MODEL, "maxiter": 3, "started": True, "scientific_invocations": 1})
    try:
        scipy_result = minimize(
            fun, theta0, method="L-BFGS-B", jac=True, bounds=bounds, callback=callback,
            options=dict(FROZEN_OPTIONS),
        )
    except (BudgetLimit, SourceStop) as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
    optimizer_seconds = time.perf_counter() - opt_t
    theta_end, meta_end, manifest = store.recover()
    if accepted and theta_hash(theta_end) != accepted[-1]["theta_sha256"]:
        raise SystemExit("STOP recovery does not match last callback")
    if not accepted and theta_hash(theta_end) != THETA_CONTENT_SHA:
        raise SystemExit("STOP empty run did not recover the initial state")
    replay = evaluate(theta_end, "endpoint_replay")
    replay_kkt = kkt(theta_end, replay["gradient"])
    recorded_f = initial["objective"] if not accepted else accepted[-1]["objective_after"]
    endpoint = {
        "accepted_steps": len(accepted),
        "recovered_kind": meta_end["kind"],
        "recovered_theta_sha256": theta_hash(theta_end),
        "galaxies": 91,
        "max_per_galaxy_path_gap": replay["max_identity_gap"],
        "recorded_objective": recorded_f,
        "replay_objective": replay["objective"],
        "objective_abs_gap": abs(replay["objective"] - recorded_f),
        "replay_tolerance": replay_tolerance,
        "objective_within_tolerance": abs(replay["objective"] - recorded_f) <= replay_tolerance,
        "identity_pass": replay["max_identity_gap"] <= IDENTITY_LIMIT,
        "outside_bounds": replay_kkt["outside_bounds"],
        "gradient_finite": True,
        "flags": replay["flags"],
        "domains": replay["domains"],
        "projected_gradient_inf_norm": replay_kkt["projected_gradient_inf_norm"],
        "full_gradient_inf_norm": replay_kkt["full_gradient_inf_norm"],
        "quadrature": "fine192",
    }
    for item in requests:
        if item["kind"] == "TRIAL_EVALUATION" and item["theta_sha256"] == THETA_CONTENT_SHA:
            item["kind"] = "INITIAL_EVALUATION"
    rejected = [item for item in requests if item["kind"] == "TRIAL_EVALUATION"]
    payload = {
        "scipy_version": scipy.__version__,
        "options": FROZEN_OPTIONS,
        "factr_from_ftol": FROZEN_OPTIONS["ftol"] / float(np.finfo(float).eps),
        "user_parameter_scaling": None,
        "initial_objective": initial["objective"],
        "initial_kkt": {k: v for k, v in initial_kkt.items() if k not in ("projected_gradient", "scaled_projected_gradient")},
        "initial_gradient": initial["gradient"].tolist(),
        "archived_gradient_max_abs_gap": grad_gap,
        "preflight_seconds": preflight_seconds,
        "checkpoint_seconds_initial": checkpoint_seconds,
        "optimizer_seconds": optimizer_seconds,
        "replay_tolerance": replay_tolerance,
        "requests": requests,
        "accepted": [{k: v for k, v in row.items() if k != "theta"} for row in accepted],
        "rejected_trial_count": len(rejected),
        "warning_log": warning_log,
        "failure": failure,
        "scipy": None if scipy_result is None else {
            "success": bool(scipy_result.success),
            "status": int(scipy_result.status),
            "message": str(scipy_result.message),
            "nit": int(scipy_result.nit),
            "nfev": int(scipy_result.nfev),
            "njev": int(scipy_result.njev),
            "fun": float(scipy_result.fun),
        },
        "endpoint": endpoint,
        "recovery_tests": recovery,
        "budget": {"calls": budget.calls, "equivalents": budget.eq, "optimizer_requests": budget.optimizer_requests,
                   "wall_seconds": time.perf_counter() - budget.t0},
        "manifest": manifest,
    }
    (NEW / "logs/run_state.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "accepted": len(accepted),
        "F0": initial["objective"],
        "steps": [{k: row[k] for k in ("accepted_index", "objective_before", "objective_after", "delta_f", "projected_gradient_inf_norm")} for row in accepted],
        "endpoint_gap": endpoint["objective_abs_gap"],
        "identity": endpoint["max_per_galaxy_path_gap"],
        "failure": failure,
        "scipy": payload["scipy"],
        "budget": payload["budget"],
        "rejected": len(rejected),
        "initial_projected": initial_kkt["projected_gradient_inf_norm"],
        "final_projected": replay_kkt["projected_gradient_inf_norm"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
