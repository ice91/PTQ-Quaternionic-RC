"""One final ptq-screen smoke after an actual-type Budget regression."""
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
from scipy.special import logsumexp

NEW = Path("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/mle_pilot_a8_d2r_r1_strategic_closeout")
IMPL = Path("/home/rocky/PTQ-Quaternionic-RC")
sys.path.insert(0, str(NEW / "diagnostic"))
sys.path.insert(0, str(NEW / "candidate/src"))

from ptquat.constants import C_LIGHT, H0_SI, KM, KPC
from ptquat.data import load_tidy_sparc
from ptquat.marginal_likelihood import integrate, nuisance_nodes, velocity
from ptquat.models import model_v_ptq_screen
from signed_jacobian import SCIENTIFIC_TARGET, VARIANCE_TOKEN, as_g, galaxy_jacobian_row

ABSOLUTE_LIMIT = 1e-9
RELATIVE_LIMIT = 1e-12
ULP_FACTOR = 8.0
REPLAY_TOLERANCE = 1e-8
START_F = 6614.725282751667
START_PROJECTED = 32.3078606499194
START_SHA = "b0a24278778944f59b3ae440ba3ae5a63d9dec584c851f93a64fa3e693ff3ab4"
D2_TRIAL_SHA = "49ddefe591521a171457bd110bda4b5baf8c1ca55a8a8554a6d15518aef68693"
C_KMS = C_LIGHT / KM
A0_BOUNDS = (5.0e-11, 2.0e-10)
UPSILON = (0.05, 1.5)
Q_BOUNDS = (0.25, 8.0)
SIGMA = (1e-3, 200.0)
OPTIONS = {"maxiter": 3, "maxls": 20, "maxcor": 10, "ftol": 1e-12, "gtol": 1e-6}
CLIP_FLAGS = {"y_floor", "a0_floor", "radius_floor", "gc_floor", "concentration_floor"}
FORMULA_FLAGS = {"nu_q_formula_mismatch", "velocity_formula_mismatch", "q_power_nonfinite"}
JACOBIAN_SHA = "d23cb42195870aab92745d1735e54366435c8ae49f3d214f858164364d439585"
MODELS_SHA = "7214664b9a4a3e182f99748a9c4c7da91a98b930399a832589203395f3b89b1b"
MASK_SHA = "df0086b1e67f96403ff09f5c5bfb7f5818593d141d14234d752cb433a9511e6e"
TIDY_SHA = "3ab1c3c2b110c1e48680ebae77c1ceefaa4f778f81119dcf04881b630b274cbb"
MARGINAL_CALLS = 0
MINIMIZE_CALLS = 0


class IdentityStop(RuntimeError):
    pass


class PhysicalReviewStop(RuntimeError):
    pass


class SchemaError(RuntimeError):
    pass


class BudgetLimit(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def theta_hash(theta) -> str:
    return sha256_bytes(np.ascontiguousarray(theta, dtype="<f8").tobytes())


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def dumps(payload) -> bytes:
    return (json.dumps(clean(payload), indent=2, allow_nan=False) + "\n").encode()


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
    data = (json.dumps(clean(row), allow_nan=False) + "\n").encode()
    with path.open("ab") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def positive_ulp(logl):
    try:
        number = float(logl)
    except (TypeError, ValueError):
        return None
    if isinstance(logl, bool) or not math.isfinite(number):
        return None
    ulp = math.ulp(abs(number))
    if (not math.isfinite(ulp)) or ulp <= 0.0:
        return None
    return float(ulp)


def relative_difference(absolute, left, right):
    if not all(math.isfinite(float(value)) for value in (absolute, left, right)):
        return None
    return float(absolute) / max(1.0, abs(float(left)), abs(float(right)))


def classify_galaxy(logl_a5, logl_a6, *, strict_only: bool) -> dict:
    if isinstance(logl_a5, bool) or isinstance(logl_a6, bool):
        return {"class": "FAIL", "reason": "nonfinite", "absolute": None, "relative": None, "ulp": None}
    try:
        left = float(logl_a5)
        right = float(logl_a6)
    except (TypeError, ValueError):
        return {"class": "FAIL", "reason": "nonfinite", "absolute": None, "relative": None, "ulp": None}
    if not math.isfinite(left) or not math.isfinite(right):
        return {"class": "FAIL", "reason": "nonfinite", "absolute": None, "relative": None, "ulp": None}
    absolute = abs(left - right)
    relative = relative_difference(absolute, left, right)
    ulp = positive_ulp(left)
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


def classify_table(rows, *, strict_only: bool, expected_count: int = 91) -> dict:
    if len(rows) != expected_count or any(not row.get("complete", False) for row in rows):
        return {"class": "INCOMPLETE", "reason": "missing_row"}
    classes = []
    for row in rows:
        if row.get("nonfinite_gradient") or row.get("domain_fail") or row.get("unexpected_branch") or row.get("formula_fail") or row.get("normalization_fail"):
            return {"class": "FAIL", "reason": "domain_formula_or_branch", "galaxy": row.get("galaxy")}
        verdict = classify_galaxy(row["logL_A5"], row["logL_A6"], strict_only=strict_only)
        classes.append(verdict)
        if verdict["class"] == "FAIL":
            return {"class": "FAIL", "reason": verdict["reason"], "galaxy": row.get("galaxy"), "detail": verdict}
    if any(item["class"] == "TRIAL_ROUNDOFF_ADMISSIBLE" for item in classes):
        label = "FAIL" if strict_only else "TRIAL_ROUNDOFF_ADMISSIBLE"
        if strict_only:
            return {"class": "FAIL", "reason": "accepted_absolute"}
        return {
            "class": label,
            "roundoff_galaxies": sum(item["class"] == "TRIAL_ROUNDOFF_ADMISSIBLE" for item in classes),
            "max_absolute": max(item["absolute"] for item in classes),
            "max_ulp_scaled": max(item.get("ulp_scaled") or 0.0 for item in classes),
        }
    return {
        "class": "ACCEPTED_STRICT_PASS" if strict_only else "TRIAL_STRICT_PASS",
        "roundoff_galaxies": 0,
        "max_absolute": max(item["absolute"] for item in classes),
        "max_ulp_scaled": max(((item["absolute"] / item["ulp"]) if item["ulp"] else 0.0) for item in classes),
    }


def epsilon_bounds():
    return (A0_BOUNDS[0] / (C_LIGHT * H0_SI), A0_BOUNDS[1] / (C_LIGHT * H0_SI))


def native_bounds():
    low, high = epsilon_bounds()
    bounds = [(low, high), (math.log(Q_BOUNDS[0]), math.log(Q_BOUNDS[1]))]
    bounds.extend([UPSILON] * 91)
    bounds.append((math.log(SIGMA[0]), math.log(SIGMA[1])))
    if len(bounds) != 94:
        raise SchemaError("dimension")
    return bounds


def unpack(theta):
    theta = np.asarray(theta, float)
    if theta.shape != (94,) or not np.isfinite(theta).all():
        raise SchemaError("dimension")
    return {
        "eps": float(theta[0]),
        "lq": float(theta[1]),
        "U": np.asarray(theta[2:93], float),
        "sigma": float(math.exp(theta[93])),
    }


def bound_census(theta):
    x = np.asarray(theta, float)
    bounds = np.asarray(native_bounds(), float)
    lower = x <= bounds[:, 0]
    upper = x >= bounds[:, 1]
    upsilon = x[2:93]
    return {
        "lower": np.flatnonzero(lower).astype(int).tolist(),
        "upper": np.flatnonzero(upper).astype(int).tolist(),
        "outside": np.flatnonzero((x < bounds[:, 0]) | (x > bounds[:, 1])).astype(int).tolist(),
        "epsilon_at_lower": bool(lower[0]),
        "epsilon_at_upper": bool(upper[0]),
        "ln_q_at_lower": bool(lower[1]),
        "ln_q_at_upper": bool(upper[1]),
        "ln_sigma_at_lower": bool(lower[93]),
        "ln_sigma_at_upper": bool(upper[93]),
        "upsilon_lower": int(np.sum(upsilon <= UPSILON[0])),
        "upsilon_upper": int(np.sum(upsilon >= UPSILON[1])),
        "upsilon_interior": int(np.sum((upsilon > UPSILON[0]) & (upsilon < UPSILON[1]))),
    }


def projected_kkt(theta, gradient):
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
        "active_bounds": bound_census(theta),
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
        if record["kind"] == "ACCEPTED" and record.get("physical_regime") == "PHYSICAL_REGIME_REVIEW_REQUIRED":
            raise SchemaError("physical review is not a canonical commit")
        digest = theta_hash(theta)
        if digest != record["theta_sha256"] or np.asarray(theta).shape != (94,):
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
            "model": "ptq-screen",
            "scientific_target": SCIENTIFIC_TARGET,
            "mask_sha256": MASK_SHA,
            "models_sha256": MODELS_SHA,
            "jacobian_sha256": JACOBIAN_SHA,
            "canonical": True,
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
        if record["theta_sha256"] == D2_TRIAL_SHA:
            raise SchemaError("D2 trial cannot be a restart vector")
        return theta, record, manifest


def regression_tests() -> dict:
    root = NEW / "state/mock_regression"
    if root.exists():
        for child in sorted(root.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
    root.mkdir(parents=True, exist_ok=True)
    results = {}
    results["strict_provisional_pass"] = classify_galaxy(-40.0, -40.0 + 1e-12, strict_only=False)["class"] == "STRICT_PASS"
    one = classify_galaxy(-226811322.2915751, -226811322.2915751 - 2.9802322387695312e-08, strict_only=False)
    results["screen_1ulp_admissible"] = one["class"] == "TRIAL_ROUNDOFF_ADMISSIBLE" and abs(one["ulp_scaled"] - 1.0) < 1e-9
    two = classify_galaxy(-43139560.486632235, -43139560.486632235 - 1.4901161193847656e-08, strict_only=False)
    results["screen_2ulp_admissible"] = two["class"] == "TRIAL_ROUNDOFF_ADMISSIBLE" and abs(two["ulp_scaled"] - 2.0) < 1e-8
    three_ulp = 3.0 * positive_ulp(-226811322.2915751)
    three = classify_galaxy(-226811322.2915751, -226811322.2915751 - three_ulp, strict_only=False)
    results["screen_3ulp_admissible"] = three["class"] == "TRIAL_ROUNDOFF_ADMISSIBLE" and three["ulp_scaled"] <= ULP_FACTOR
    nine = classify_galaxy(-226811322.2915751, -226811322.2915751 - 9.0 * positive_ulp(-226811322.2915751), strict_only=False)
    results["above_8ulp_rejected"] = nine["class"] == "FAIL" and "ulp" in nine["reason"]
    results["formula_mismatch_rejected"] = classify_galaxy(-50.0, -49.0, strict_only=False)["class"] == "FAIL"
    results["nonfinite_likelihood_rejected"] = classify_galaxy(float("nan"), -1.0, strict_only=False)["class"] == "FAIL"
    bad_grad = classify_table([
        {"complete": True, "galaxy": "g", "logL_A5": -1.0, "logL_A6": -1.0, "nonfinite_gradient": True}
    ] * 91, strict_only=False)
    results["nonfinite_gradient_rejected"] = bad_grad["class"] == "FAIL"
    incomplete = classify_table([{"complete": True, "logL_A5": -1.0, "logL_A6": -1.0, "galaxy": "g"}] * 90, strict_only=False)
    results["incomplete_table_rejected"] = incomplete["class"] == "INCOMPLETE"
    dimension_rejected = False
    try:
        unpack(np.ones(93))
    except SchemaError:
        dimension_rejected = True
    bounds = np.asarray(native_bounds(), float)
    outside = np.full(94, 0.2)
    outside[0] = 10.0
    results["invalid_vector_rejected"] = dimension_rejected and bool(np.any(outside > bounds[:, 1])) and len(native_bounds()) == 94
    floor_row = [{"complete": True, "galaxy": "g", "logL_A5": -1.0, "logL_A6": -1.0, "unexpected_branch": True}] * 91
    results["unexpected_floor_rejected"] = classify_table(floor_row, strict_only=False)["class"] == "FAIL"
    roundoff_rows = [{"complete": True, "galaxy": "IC2574", "logL_A5": -226811322.2915751, "logL_A6": -226811322.2915751 - 2.9802322387695312e-08}]
    roundoff_rows += [{"complete": True, "galaxy": f"g{i}", "logL_A5": -20.0, "logL_A6": -20.0} for i in range(90)]
    trial_ok = classify_table(roundoff_rows, strict_only=False)["class"] == "TRIAL_ROUNDOFF_ADMISSIBLE"
    accepted_blocked = classify_table(roundoff_rows, strict_only=True)["class"] == "FAIL"
    store = Store(root / "accepted")
    base = np.full(94, 0.2)
    base[0] = 0.12
    base[1] = 0.0
    manifest = store.commit(base, {
        "kind": "INITIAL", "iteration_index": 0, "theta_sha256": theta_hash(base),
        "objective_before": 4.0, "objective_after": 4.0, "identity_class": "INITIAL",
        "physical_regime": "NO_FLAG_OBSERVED", "projected_gradient_norm": 1.0, "full_gradient_norm": 1.0,
        "active_bounds": {"lower": [], "upper": []}, "evaluation_count": 0, "elapsed_seconds": 0.0,
    })
    refused = False
    try:
        store.commit(base, {
            "kind": "ACCEPTED", "iteration_index": 1, "theta_sha256": theta_hash(base),
            "objective_before": 4.0, "objective_after": 3.0, "identity_class": "TRIAL_ROUNDOFF_ADMISSIBLE",
            "physical_regime": "NO_FLAG_OBSERVED", "projected_gradient_norm": 1.0, "full_gradient_norm": 1.0,
            "active_bounds": {"lower": [], "upper": []}, "evaluation_count": 1, "elapsed_seconds": 1.0,
        })
    except SchemaError:
        refused = True
    results["provisional_roundoff_not_committed"] = trial_ok and accepted_blocked and refused and store.recover()[2]["iteration_index"] == 0
    good = base.copy()
    good[10] = 0.25
    store.commit(good, {
        "kind": "ACCEPTED", "iteration_index": 1, "theta_sha256": theta_hash(good),
        "objective_before": 4.0, "objective_after": 3.5, "identity_class": "ACCEPTED_STRICT_PASS",
        "physical_regime": "NO_FLAG_OBSERVED", "projected_gradient_norm": 0.8, "full_gradient_norm": 0.8,
        "active_bounds": {"lower": [], "upper": []}, "evaluation_count": 2, "elapsed_seconds": 2.0,
    })
    results["accepted_strict_pass_committed"] = store.recover()[1]["identity_class"] == "ACCEPTED_STRICT_PASS"
    strict_fail_refused = False
    try:
        store.commit(good, {
            "kind": "ACCEPTED", "iteration_index": 2, "theta_sha256": theta_hash(good),
            "objective_before": 3.5, "objective_after": 3.4, "identity_class": "FAIL",
            "physical_regime": "NO_FLAG_OBSERVED", "projected_gradient_norm": 0.7, "full_gradient_norm": 0.7,
            "active_bounds": {"lower": [], "upper": []}, "evaluation_count": 3, "elapsed_seconds": 3.0,
        })
    except SchemaError:
        strict_fail_refused = True
    results["accepted_strict_fail_not_committed"] = strict_fail_refused and store.recover()[2]["iteration_index"] == 1
    warning = {"regime": "WEAK_FIELD_REGIME_WARNING", "v_max_kms": C_KMS + 10.0, "identity_class": "TRIAL_ROUNDOFF_ADMISSIBLE"}
    atomic_json(root / "provisional_warning.json", warning)
    results["physical_warning_independent"] = json.loads((root / "provisional_warning.json").read_text())["regime"] == "WEAK_FIELD_REGIME_WARNING"
    superluminal = base.copy()
    superluminal[4] = 0.22
    atomic_npy(root / "noncanonical_superluminal.npy", superluminal)
    atomic_json(root / "noncanonical_superluminal.json", {
        "status": "NONCANONICAL_DIAGNOSTIC",
        "theta_sha256": theta_hash(superluminal),
        "physical_regime": "PHYSICAL_REGIME_REVIEW_REQUIRED",
        "canonical": False,
    })
    results["superluminal_provisional_not_canonical"] = (
        store.recover()[2]["theta_sha256"] != theta_hash(superluminal)
        and json.loads((root / "noncanonical_superluminal.json").read_text())["canonical"] is False
    )
    second = base.copy(); second[12] = 0.3
    third = base.copy(); third[12] = 0.31
    store.commit(second, {
        "kind": "ACCEPTED", "iteration_index": 2, "theta_sha256": theta_hash(second),
        "objective_before": 3.5, "objective_after": 3.2, "identity_class": "ACCEPTED_STRICT_PASS",
        "physical_regime": "NO_FLAG_OBSERVED", "projected_gradient_norm": 0.4, "full_gradient_norm": 0.4,
        "active_bounds": {"lower": [], "upper": []}, "evaluation_count": 4, "elapsed_seconds": 4.0,
    })
    store.commit(third, {
        "kind": "ACCEPTED", "iteration_index": 3, "theta_sha256": theta_hash(third),
        "objective_before": 3.2, "objective_after": 3.0, "identity_class": "ACCEPTED_STRICT_PASS",
        "physical_regime": "NO_FLAG_OBSERVED", "projected_gradient_norm": 0.2, "full_gradient_norm": 0.2,
        "active_bounds": {"lower": [], "upper": []}, "evaluation_count": 5, "elapsed_seconds": 5.0,
    })
    _, record3, manifest3 = store.recover()
    results["second_and_third_callback_committed"] = manifest3["iteration_index"] == 3 and record3["objective_after"] == 3.0
    atomic_json(root / "accepted" / "rejected_trial.json", {"kind": "REJECTED_TRIAL", "objective_after": 1e12})
    results["rejected_trial_isolated"] = store.recover()[2]["iteration_index"] == 3 and store.recover()[1]["objective_after"] == 3.0
    atomic_npy(root / "accepted" / "accepted_4_theta.npy", base)
    atomic_json(root / "accepted" / "accepted_4_meta.json", {
        "kind": "ACCEPTED", "iteration_index": 4, "theta_sha256": theta_hash(base), "objective_after": 1.0,
    })
    recovered_theta, recovered_record, recovered_manifest = store.recover()
    results["crash_recovery"] = recovered_manifest["iteration_index"] == 3 and theta_hash(recovered_theta) == theta_hash(third)
    results["theta_manifest_hash_consistent"] = (
        sha256_file(root / "accepted" / recovered_manifest["meta_file"]) == recovered_manifest["meta_sha256"]
        and theta_hash(recovered_theta) == recovered_manifest["theta_sha256"]
    )
    lock = root / "OPTIMIZER_INVOCATION.json"

    def try_invoke():
        if lock.exists():
            return False
        atomic_json(lock, {"started": True, "scientific": False})
        return True

    results["invocation_lock"] = try_invoke() is True and try_invoke() is False
    edge = {"negative_ulp": positive_ulp(-226811322.2915751), "zero_ulp": positive_ulp(0.0), "subnormal_ulp": positive_ulp(5e-324)}
    atomic_json(root / "edge.json", edge)
    rejected_nan = False
    try:
        json.dumps({"bad": float("nan")}, allow_nan=False)
    except ValueError:
        rejected_nan = True
    results["ulp_json_edges"] = (
        edge["negative_ulp"] > 0 and edge["zero_ulp"] > 0 and edge["subnormal_ulp"] > 0
        and rejected_nan and "Infinity" not in (root / "edge.json").read_text()
    )
    probe = Budget({"CamB": 0.0123, "D564-8": 0.0100})
    subscript_fails = False
    try:
        probe["weights"]
    except TypeError:
        subscript_fails = True
    ordered = list(probe.weights)
    results["budget_attribute_access"] = (
        subscript_fails
        and not isinstance(probe, dict)
        and probe.weights["CamB"] == 0.0123
        and ordered == ["CamB", "D564-8"]
        and abs(sum(probe.weights.values()) - 0.0223) < 1e-15
    )
    origin = np.full(94, 0.2)
    origin[0] = 0.12
    side = Store(root / "failure_store")
    side.commit(origin, {
        "kind": "INITIAL", "iteration_index": 0, "theta_sha256": theta_hash(origin),
        "objective_before": 4.0, "objective_after": 4.0, "identity_class": "INITIAL",
        "physical_regime": "NO_FLAG_OBSERVED", "projected_gradient_norm": 1.0, "full_gradient_norm": 1.0,
        "active_bounds": {"lower": [], "upper": []}, "evaluation_count": 0, "elapsed_seconds": 0.0,
    })
    failed_trial = origin.copy()
    failed_trial[5] = 0.21
    failed_digest = persist_uncommitted(root / "uncommitted", failed_trial, 2)
    simulated = False
    try:
        raise RuntimeError("simulated evaluation failure")
    except RuntimeError:
        simulated = True
    kept_theta, kept_record, kept_manifest = side.recover()
    results["failure_preserves_trial_and_checkpoint"] = (
        simulated
        and kept_manifest["iteration_index"] == 0
        and kept_record["kind"] == "INITIAL"
        and theta_hash(kept_theta) == theta_hash(origin)
        and theta_hash(np.load(root / "uncommitted" / "request_2_theta.npy")) == failed_digest
        and json.loads((root / "uncommitted" / "request_2.json").read_text())["status"] == "UNCOMMITTED_TRIAL"
    )
    results["accepted_state_strict_identity"] = (
        classify_galaxy(-10.0, -10.0, strict_only=True)["class"] == "STRICT_PASS"
        and classify_galaxy(-10.0, -10.0 - 2.0e-9, strict_only=True)["class"] == "FAIL"
    )
    restart_count = {"n": 0}

    def one_shot():
        if restart_count["n"]:
            return "restarted"
        restart_count["n"] += 1
        raise RuntimeError("single failure")

    restarted = False
    try:
        one_shot()
    except RuntimeError:
        restarted = False
    results["no_automatic_restart"] = restart_count["n"] == 1 and restarted is False
    results["scientific_likelihood_calls"] = MARGINAL_CALLS
    results["scipy_minimize_calls"] = MINIMIZE_CALLS
    keys = [key for key in results if key not in ("pass", "scientific_likelihood_calls", "scipy_minimize_calls")]
    results["pass"] = MARGINAL_CALLS == 0 and MINIMIZE_CALLS == 0 and all(bool(results[key]) for key in keys)
    del manifest
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
    d2 = np.load("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/mle_pilot_a8_d2_screen_forensic/TRIAL_THETA.npy")
    if theta_hash(d2) != D2_TRIAL_SHA:
        raise SystemExit("STOP D2 trial hash")


def load_sample():
    tidy = pd.read_csv(IMPL / "dataset/sparc_tidy.csv")
    raw = pd.read_csv(IMPL / "dataset/raw/vizier_table2.csv")
    mask = pd.read_csv(IMPL / "results/frontiers_revision_2026_10/mle_pilot_a4_domain_feasibility/S_DOMAIN13_MASK.csv")
    joined = tidy.merge(raw, left_on=["galaxy", "r_kpc"], right_on=["Name", "Rad"], how="left", validate="one_to_one")
    included = mask.included.to_numpy(int) == 1
    if int(included.sum()) != 1769 or not np.array_equal(mask.source_recno.to_numpy(), joined.recno.to_numpy()):
        raise SystemExit("STOP mask")
    out = NEW / "logs/s_domain13_filtered_tidy.csv"
    tidy.loc[included].to_csv(out, index=False)
    loaded = load_tidy_sparc(str(out))
    galaxies = [loaded[name] for name in sorted(loaded)]
    if len(galaxies) != 91 or sum(len(g.v_obs) for g in galaxies) != 1769:
        raise SystemExit("STOP sample")
    weights = {
        row["galaxy"]: float(row["fullsample_equivalent"])
        for row in json.loads((IMPL / "results/frontiers_revision_2026_10/mle_pilot_a/GALAXY_WORK_WEIGHTS.json").read_text())["baryon"]["192"]
    }
    archived = np.asarray(json.loads(
        (IMPL / "results/frontiers_revision_2026_10/mle_pilot_a6_signed_jac/FIVE_FAMILY_GRADIENT_VECTORS.json").read_text()
    )["vectors"]["ptq-screen"]["native_dF"], float)
    return galaxies, weights, archived


def finite_span(values):
    array = np.asarray(values, float).ravel()
    finite = array[np.isfinite(array)]
    return None if finite.size == 0 else float(np.min(finite)), None if finite.size == 0 else float(np.max(finite))


def score_mass(galaxy, pars, gi, vd, wd, wi, incs):
    gg = as_g(galaxy)
    sigma = float(pars["sigma"])
    b = np.sin(incs) / np.sin(gg.inc)
    var = np.broadcast_to(gg.e[None, :] ** 2 + sigma ** 2, (len(incs), len(gg.y)))
    inv = 1.0 / var
    yi = gg.y[None, :] * inv
    y2 = (gg.y[None, :] ** 2 * inv).sum(axis=1)
    norm = np.log(2.0 * np.pi * var).sum(axis=1)
    scores = []
    masks = []
    for start in range(0, len(vd), 128):
        end = min(start + 128, len(vd))
        v = vd[start:end]
        cross = v @ yi.T
        vv = (v * v) @ inv.T
        quad = y2[None, :] - 2.0 * b[None, :] * cross + (b[None, :] ** 2) * vv
        ll = -0.5 * (quad + norm[None, :])
        score = ll + np.log(wd[start:end, None]) + np.log(wi)[None, :]
        cell_super = np.any(v >= C_KMS, axis=1)
        masks.append(np.repeat(cell_super, score.shape[1]))
        scores.append(score.ravel())
    flat = np.concatenate(scores)
    mask = np.concatenate(masks)
    total = float(logsumexp(flat))
    if not np.any(mask):
        return {"logsumexp_total": total, "superluminal_score_mass": 0.0}
    part = float(logsumexp(flat[mask]))
    mass = math.exp(part - total) if math.isfinite(part - total) else None
    return {"logsumexp_total": total, "superluminal_logsumexp": part, "superluminal_score_mass": mass}


def speed_audit(galaxy, pars, gi, budget):
    gg = as_g(galaxy)
    upsilon = float(pars["U"][gi])
    q = float(math.exp(pars["lq"]))
    nominal = model_v_ptq_screen(
        upsilon, float(pars["eps"]), q, galaxy.r_kpc, galaxy.v_disk, galaxy.v_bulge, galaxy.v_gas, H0_si=H0_SI,
    )
    pp = dict(pars, model="ptq-screen", _H0=H0_SI, _c0=10.0, _c_slope=-0.1)
    ds, wd, incs, wi, _, _ = nuisance_nodes(gg, "segmented_quantile", 192)
    vd = velocity(gg, pp, gi, ds[:, None])
    nominal_super = np.asarray(nominal, float) >= C_KMS
    quad_super = np.asarray(vd, float) >= C_KMS
    distance_super = np.any(quad_super, axis=1)
    weight = float(np.sum(wd[distance_super]) * np.sum(wi))
    nmin, nmax = finite_span(nominal)
    qmin, qmax = finite_span(vd)
    audit = {
        "nominal_min_kms": nmin,
        "nominal_max_kms": nmax,
        "nominal_superluminal_count": int(np.sum(nominal_super)),
        "quadrature_min_kms": qmin,
        "quadrature_max_kms": qmax,
        "quadrature_superluminal_count": int(np.sum(quad_super)),
        "quadrature_superluminal_weight": weight,
        "weight_sum": float(np.sum(wd) * np.sum(wi)),
        "c_kms": C_KMS,
        "score": None,
    }
    if audit["nominal_superluminal_count"] or audit["quadrature_superluminal_count"]:
        weight_cost = float(budget.weights[galaxy.name])
        if budget.marginals + 1 > 3600 or budget.eq + weight_cost > 36.0:
            raise BudgetLimit("score diagnostic")
        budget.marginals += 1
        budget.eq += weight_cost
        budget.score_diagnostics += 1
        audit["score"] = score_mass(galaxy, pars, gi, vd, wd, incs, wi)
    audit["superluminal"] = bool(audit["nominal_superluminal_count"] or audit["quadrature_superluminal_count"])
    return audit


class Budget:
    def __init__(self, weights):
        self.weights = weights
        self.marginals = 0
        self.eq = 0.0
        self.requests = 0
        self.score_diagnostics = 0
        self.attempted_galaxies = 0
        self.completed_galaxies = 0
        self.t0 = time.perf_counter()

    def charge_pair(self, name):
        weight = 2.0 * float(self.weights[name])
        if self.marginals + 2 > 3600 or self.eq + weight > 36.0 or time.perf_counter() - self.t0 > 45 * 60:
            raise BudgetLimit("budget")
        self.marginals += 2
        self.eq += weight

    def begin_request(self):
        if self.requests >= 12:
            raise BudgetLimit("request cap")
        if self.eq + 6.0 > 36.0 or self.marginals + 600 > 3600 or time.perf_counter() - self.t0 > 45 * 60:
            raise BudgetLimit("reserve replay")
        self.requests += 1


def persist_uncommitted(folder: Path, theta, request_index: int, status: str = "UNCOMMITTED_TRIAL") -> str:
    digest = theta_hash(theta)
    atomic_npy(folder / f"request_{request_index}_theta.npy", theta)
    atomic_json(folder / f"request_{request_index}.json", {
        "status": status,
        "theta_sha256": digest,
        "request_index": request_index,
        "canonical": False,
        "used_as_optimizer_start": False,
    })
    return digest


def main():
    global MARGINAL_CALLS, MINIMIZE_CALLS
    verify_sources()
    if scipy.__version__ != "1.17.1":
        raise SystemExit("STOP scipy " + scipy.__version__)
    tests = regression_tests()
    atomic_json(NEW / "logs/mock_regression.json", tests)
    print("REGRESSION", tests["pass"], flush=True)
    if not tests["pass"]:
        raise SystemExit("ENGINEERING_REGRESSION_FAIL")
    if os.environ.get("D2R_REGRESSION_ONLY") == "1":
        print("REGRESSION_ONLY", flush=True)
        return
    galaxies, weights, archived_gradient = load_sample()
    theta0 = np.load(IMPL / "results/frontiers_revision_2026_10/mle_pilot_a/execution_v1/ptq_screen_last_accepted_theta.npy")
    if theta_hash(theta0) != START_SHA or theta_hash(theta0) == D2_TRIAL_SHA or theta0.shape != (94,):
        raise SystemExit("STOP start vector")
    array_bounds = np.asarray(native_bounds(), float)
    if np.any(theta0 < array_bounds[:, 0]) or np.any(theta0 > array_bounds[:, 1]):
        raise SystemExit("STOP bounds")
    if archived_gradient.shape != (94,):
        raise SystemExit("STOP archived gradient")
    budget = Budget(weights)
    ledger = NEW / "logs/galaxy_rows.jsonl"
    requests = []
    accepted = []
    noncanonical = []

    def evaluate(theta, digest, phase):
        global MARGINAL_CALLS
        pars = unpack(theta)
        rows = []
        records = []
        for gi, galaxy in enumerate(galaxies):
            budget.attempted_galaxies += 1
            budget.charge_pair(galaxy.name)
            MARGINAL_CALLS += 1
            raw = galaxy_jacobian_row(galaxy, pars, gi, "ptq-screen")
            flags = list(raw["flags"])
            audit = speed_audit(galaxy, pars, gi, budget)
            a5 = float(raw["official_logL"])
            a6 = float(raw["rederived_logL"])
            verdict = classify_galaxy(a5, a6, strict_only=False)
            scaled = None
            if verdict["ulp"] not in (None, 0.0) and verdict["absolute"] is not None:
                scaled = verdict["absolute"] / verdict["ulp"]
                if not math.isfinite(scaled):
                    scaled = None
            gradient_keys = ("epsilon", "ln_q", "upsilon", "ln_sigma_sys")
            gradient_bad = any(raw["dlogL"].get(key) is None or not math.isfinite(float(raw["dlogL"][key])) for key in gradient_keys)
            record = {
                "complete": True,
                "phase": phase,
                "galaxy": galaxy.name,
                "gi": gi,
                "n_retained": int(len(galaxy.v_obs)),
                "theta_sha256": digest,
                "logL_A5": a5,
                "logL_A6": a6,
                "absolute_difference": verdict["absolute"],
                "relative_difference": verdict["relative"],
                "ulp": verdict["ulp"],
                "ulp_scaled": scaled,
                "galaxy_class": verdict["class"],
                "flags": flags,
                "domain": raw["domain"],
                "variance_token": raw["variance_token"],
                "unexpected_branch": any(flag in CLIP_FLAGS for flag in flags),
                "formula_fail": any(flag in FORMULA_FLAGS for flag in flags),
                "normalization_fail": raw["variance_token"] != VARIANCE_TOKEN,
                "domain_fail": raw["domain"] != "INTERIOR_ANALYTICALLY_DIFFERENTIABLE",
                "nonfinite_gradient": gradient_bad,
                "physical": audit,
            }
            append_jsonl(ledger, record)
            records.append(record)
            rows.append(raw)
            budget.completed_galaxies += 1
        gradient = np.full(94, np.nan)
        if any(record["nonfinite_gradient"] for record in records):
            return {"records": records, "gradient": gradient, "F": None, "pars": pars}
        gradient[0] = -sum(float(row["dlogL"]["epsilon"]) for row in rows)
        gradient[1] = -sum(float(row["dlogL"]["ln_q"]) for row in rows)
        for gi, row in enumerate(rows):
            gradient[2 + gi] = -float(row["dlogL"]["upsilon"])
        gradient[93] = -sum(float(row["dlogL"]["ln_sigma_sys"]) for row in rows)
        value = -sum(float(row["official_logL"]) for row in rows)
        return {"records": records, "gradient": gradient, "F": float(value), "pars": pars}

    def regime_of(records):
        superluminal = [row for row in records if row["physical"]["superluminal"]]
        nominal = [row["galaxy"] for row in superluminal if row["physical"]["nominal_superluminal_count"]]
        quadrature = [row["galaxy"] for row in superluminal if row["physical"]["quadrature_superluminal_count"]]
        labels = ["MATHEMATICAL_DOMAIN_VALID", "FLOATING_POINT_EVALUATION_VALID"]
        if superluminal:
            labels.append("WEAK_FIELD_REGIME_WARNING")
        return {
            "labels": labels,
            "superluminal_galaxies": [row["galaxy"] for row in superluminal],
            "nominal_superluminal_galaxies": nominal,
            "quadrature_superluminal_galaxies": quadrature,
            "max_nominal_kms": max(row["physical"]["nominal_max_kms"] for row in records),
            "max_quadrature_kms": max(row["physical"]["quadrature_max_kms"] for row in records),
            "max_superluminal_weight": max((row["physical"]["quadrature_superluminal_weight"] for row in records), default=0.0),
            "review_required": bool(nominal or quadrature),
        }

    def summarize(result, digest, theta):
        records = result["records"]
        trial = classify_table(records, strict_only=False)
        finite_gradient = bool(np.isfinite(result["gradient"]).all()) and result["F"] is not None and math.isfinite(result["F"])
        diag = None if not finite_gradient else projected_kkt(theta, result["gradient"])
        absolutes = [row["absolute_difference"] for row in records if row["absolute_difference"] is not None]
        scaled = [row["ulp_scaled"] for row in records if row["ulp_scaled"] is not None]
        regime = regime_of(records)
        return {
            "theta_sha256": digest,
            "epsilon": float(theta[0]),
            "ln_q": float(theta[1]),
            "q": float(math.exp(theta[1])),
            "ln_sigma": float(theta[93]),
            "identity_class": trial["class"],
            "F_A5": result["F"],
            "F_A6": None if any(not math.isfinite(row["logL_A6"]) for row in records) else -sum(row["logL_A6"] for row in records),
            "max_absolute": None if not absolutes else max(absolutes),
            "max_ulp_scaled": None if not scaled else max(scaled),
            "roundoff_galaxies": sum(row["galaxy_class"] == "TRIAL_ROUNDOFF_ADMISSIBLE" for row in records),
            "roundoff_names": [row["galaxy"] for row in records if row["galaxy_class"] == "TRIAL_ROUNDOFF_ADMISSIBLE"],
            "active_bounds": None if diag is None else diag["active_bounds"],
            "outside_bounds": None if diag is None else diag["active_bounds"]["outside"],
            "projected_gradient_norm": None if diag is None else diag["projected_gradient_norm"],
            "full_gradient_norm": None if diag is None else diag["full_gradient_norm"],
            "finite_gradient": finite_gradient,
            "physical": regime,
            "calls": budget.marginals,
            "equivalents": budget.eq,
        }, trial, diag

    names = [galaxy.name for galaxy in galaxies]
    weight_sum = float(sum(float(weights[name]) for name in names))
    camb = next(galaxy for galaxy in galaxies if galaxy.name == "CamB")
    camb_index = names.index("CamB")
    subscript_fails = False
    try:
        budget["weights"]
    except TypeError:
        subscript_fails = True
    budget.marginals += 1
    budget.eq += float(budget.weights[camb.name])
    budget.attempted_galaxies += 1
    MARGINAL_CALLS += 1
    camb_pars = unpack(theta0)
    camb_nominal = model_v_ptq_screen(
        float(camb_pars["U"][camb_index]), float(camb_pars["eps"]), float(math.exp(camb_pars["lq"])),
        camb.r_kpc, camb.v_disk, camb.v_bulge, camb.v_gas, H0_si=H0_SI,
    )
    camb_audit = speed_audit(camb, camb_pars, camb_index, budget)
    budget.completed_galaxies += 1
    d2_theta = np.load(IMPL / "results/frontiers_revision_2026_10/mle_pilot_a8_d2_screen_forensic/TRIAL_THETA.npy")
    d2_digest = persist_uncommitted(NEW / "state/fixed_point", d2_theta, 0, "UNCOMMITTED_DIAGNOSTIC")
    print("FIXED_POINT_BEGIN", d2_digest, flush=True)
    fixed_point = {"pass": False}
    try:
        if d2_digest != D2_TRIAL_SHA or d2_digest == theta_hash(theta0):
            raise SchemaError("D2 diagnostic hash")
        d2_result = evaluate(d2_theta, d2_digest, "FIXED_POINT_DIAGNOSTIC")
        d2_summary, d2_trial, _d2_kkt = summarize(d2_result, d2_digest, d2_theta)
        raw_objective = -sum(float(row["logL_A5"]) for row in d2_result["records"])
        super_rows = [row for row in d2_result["records"] if row["physical"]["superluminal"]]
        positive_weight = [
            {
                "galaxy": row["galaxy"],
                "quadrature_superluminal_weight": row["physical"]["quadrature_superluminal_weight"],
                "nominal_max_kms": row["physical"]["nominal_max_kms"],
                "quadrature_max_kms": row["physical"]["quadrature_max_kms"],
            }
            for row in super_rows
            if float(row["physical"]["quadrature_superluminal_weight"]) > 0.0
        ]
        fixed_point = {
            "status": "FIXED_POINT_DIAGNOSTIC_ONLY",
            "used_as_optimizer_start": False,
            "theta_sha256": d2_digest,
            "identity_class": d2_trial["class"],
            "F_A5": d2_result["F"],
            "objective_equals_negative_sum_logL": d2_result["F"] == raw_objective,
            "max_absolute": d2_summary["max_absolute"],
            "max_ulp_scaled": d2_summary["max_ulp_scaled"],
            "roundoff_galaxies": d2_summary["roundoff_galaxies"],
            "epsilon": d2_summary["epsilon"],
            "ln_q": d2_summary["ln_q"],
            "q": d2_summary["q"],
            "ln_sigma": d2_summary["ln_sigma"],
            "max_nominal_kms": d2_summary["physical"]["max_nominal_kms"],
            "max_quadrature_kms": d2_summary["physical"]["max_quadrature_kms"],
            "superluminal_galaxy_count": len(super_rows),
            "positive_weight_superluminal_count": len(positive_weight),
            "velocity_clipped_at_c": False,
            "likelihood_penalty_added": False,
        }
        serialized = json.dumps(clean(fixed_point), allow_nan=False)
        fixed_point["pass"] = bool(
            subscript_fails
            and abs(weight_sum - 1.0) <= 1e-9
            and all(name in budget.weights for name in names)
            and camb_audit["superluminal"] is False
            and float(camb_audit["nominal_max_kms"]) < C_KMS
            and len(d2_result["records"]) == 91
            and d2_trial["class"] in ("TRIAL_STRICT_PASS", "TRIAL_ROUNDOFF_ADMISSIBLE")
            and d2_summary["finite_gradient"]
            and fixed_point["objective_equals_negative_sum_logL"]
            and float(d2_summary["physical"]["max_quadrature_kms"]) > 1.0e6
            and len(positive_weight) > 0
            and "Infinity" not in serialized
            and "NaN" not in serialized
        )
    except (Exception, SchemaError) as exc:
        fixed_point = {"pass": False, "error": type(exc).__name__, "message": str(exc), "used_as_optimizer_start": False}
    atomic_json(NEW / "logs/actual_type_regression.json", {
        "weight_sum": weight_sum,
        "weight_order_matches_galaxies": names == sorted(names) and all(name in budget.weights for name in names),
        "budget_type": type(budget).__name__,
        "subscript_raises_typeerror": subscript_fails,
        "camb": {
            "nominal_max_kms": camb_audit["nominal_max_kms"],
            "quadrature_max_kms": camb_audit["quadrature_max_kms"],
            "superluminal": camb_audit["superluminal"],
            "score": camb_audit["score"],
        },
        "fixed_point": fixed_point,
        "pass": bool(fixed_point.get("pass")),
    })
    print("ACTUAL_TYPE", fixed_point.get("pass"), fixed_point.get("identity_class"), fixed_point.get("error"), flush=True)
    if not fixed_point.get("pass"):
        raise SystemExit("ENGINEERING_PREFLIGHT_BLOCKED")

    digest0 = theta_hash(theta0)
    start = evaluate(theta0, digest0, "INITIAL")
    start_summary, start_trial, start_kkt = summarize(start, digest0, theta0)
    gradient_gap = float(np.max(np.abs(start["gradient"] - archived_gradient)))
    preflight_ok = (
        start_trial["class"] == "TRIAL_STRICT_PASS"
        and abs(start["F"] - START_F) <= REPLAY_TOLERANCE
        and abs(start_kkt["projected_gradient_norm"] - START_PROJECTED) <= 1e-8
        and gradient_gap <= 1e-8
        and not start_kkt["active_bounds"]["outside"]
        and all(row["domain"] == "INTERIOR_ANALYTICALLY_DIFFERENTIABLE" for row in start["records"])
        and not any(row["unexpected_branch"] or row["formula_fail"] or row["normalization_fail"] for row in start["records"])
    )
    print(
        f"PREFLIGHT F {start['F']:.12f} proj {start_kkt['projected_gradient_norm']:.14f} "
        f"gap {start_summary['max_absolute']:.3e} vmax {start_summary['physical']['max_quadrature_kms']:.6g} ok {preflight_ok}",
        flush=True,
    )
    if not preflight_ok:
        atomic_json(NEW / "logs/run_state.json", {"preflight": start_summary, "gradient_gap": gradient_gap})
        raise SystemExit("STOP preflight")
    cache = {digest0: start}
    store = Store(NEW / "state/scientific")
    store.commit(theta0, {
        "kind": "INITIAL",
        "iteration_index": 0,
        "theta_sha256": digest0,
        "objective_before": start["F"],
        "objective_after": start["F"],
        "identity_class": "INITIAL",
        "physical_regime": "INITIAL_REFERENCE",
        "projected_gradient_norm": start_kkt["projected_gradient_norm"],
        "full_gradient_norm": start_kkt["full_gradient_norm"],
        "active_bounds": start_kkt["active_bounds"],
        "evaluation_count": 0,
        "elapsed_seconds": 0.0,
        "gradient": start["gradient"].tolist(),
        "projected_gradient": start_kkt["projected_gradient"].tolist(),
        "physical_audit": start_summary["physical"],
    })
    requests.append(dict(start_summary, request_index=0, kind="INITIAL", step_l2=0.0, step_inf=0.0))
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
        if digest not in cache:
            if digest != digest0:
                persist_uncommitted(NEW / "state/uncommitted", theta, budget.requests)
            if np.any(theta < array_bounds[:, 0]) or np.any(theta > array_bounds[:, 1]) or not np.isfinite(theta).all():
                append_jsonl(NEW / "logs/requests.jsonl", {"status": "TRIAL_IDENTITY_FAIL", "theta_sha256": digest, "reason": "bounds"})
                raise IdentityStop("trial outside original bounds")
            cache[digest] = evaluate(theta, digest, "PROVISIONAL_TRIAL")
        result = cache[digest]
        summary, trial, diag = summarize(result, digest, theta)
        admissible = (
            trial["class"] in ("TRIAL_STRICT_PASS", "TRIAL_ROUNDOFF_ADMISSIBLE")
            and summary["finite_gradient"]
            and diag is not None
            and not diag["active_bounds"]["outside"]
        )
        if digest == digest0 and trial["class"] == "TRIAL_STRICT_PASS":
            summary["kind"] = "INITIAL"
        elif admissible and trial["class"] == "TRIAL_STRICT_PASS":
            summary["kind"] = "TRIAL_STRICT_PASS"
        elif admissible and trial["class"] == "TRIAL_ROUNDOFF_ADMISSIBLE":
            summary["kind"] = "TRIAL_ROUNDOFF_ADMISSIBLE"
        else:
            summary["kind"] = "TRIAL_IDENTITY_FAIL"
        summary.update({
            "request_index": budget.requests,
            "step_l2": float(np.linalg.norm(displacement)),
            "step_inf": float(np.max(np.abs(displacement))),
            "matches_d2_trial": digest == D2_TRIAL_SHA,
        })
        append_jsonl(NEW / "logs/requests.jsonl", {key: value for key, value in summary.items() if key != "active_bounds"})
        requests.append(summary)
        if digest != digest0 and first_trial is None:
            first_trial = summary
            print(
                f"FIRST_TRIAL {digest} {summary['kind']} F {summary['F_A5']} eps {summary['epsilon']} "
                f"q {summary['q']} max_abs {summary['max_absolute']} roundoff {summary['roundoff_galaxies']}",
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
        append_jsonl(NEW / "logs/requests.jsonl", {"kind": "SOLVER_ACCEPTED_PENDING_VALIDATION", "theta_sha256": digest})
        if budget.eq + 6.0 > 36.0 or budget.marginals + 600 > 3600:
            raise BudgetLimit("reserve replay")
        result = evaluate(theta, digest, "ACCEPTED_CANDIDATE")
        strict = classify_table(result["records"], strict_only=True)
        if strict["class"] != "ACCEPTED_STRICT_PASS" or not np.isfinite(result["gradient"]).all():
            raise IdentityStop("ACCEPTED_IDENTITY_FAIL")
        diag = projected_kkt(theta, result["gradient"])
        if diag["active_bounds"]["outside"]:
            raise IdentityStop("accepted outside bounds")
        regime = regime_of(result["records"])
        previous_f = start["F"] if not accepted else accepted[-1]["objective_after"]
        if result["F"] > previous_f + REPLAY_TOLERANCE:
            raise IdentityStop("accepted objective increased")
        if regime["review_required"]:
            atomic_npy(NEW / "state/noncanonical" / f"candidate_{len(accepted)}_theta.npy", theta)
            atomic_json(NEW / "state/noncanonical" / f"candidate_{len(accepted)}.json", {
                "status": "NONCANONICAL_DIAGNOSTIC",
                "physical_regime": "PHYSICAL_REGIME_REVIEW_REQUIRED",
                "theta_sha256": digest,
                "objective": result["F"],
                "identity_class": "ACCEPTED_STRICT_PASS",
                "canonical": False,
                "physical": regime,
                "note": "Solver accepted this theta. It is not a canonical checkpoint.",
            })
            noncanonical.append({"theta_sha256": digest, "objective": result["F"], "physical": regime})
            raise PhysicalReviewStop("PHYSICAL_REGIME_REVIEW_REQUIRED")
        previous_theta = theta0 if not accepted else np.asarray(accepted[-1]["theta"], float)
        record = {
            "kind": "ACCEPTED",
            "iteration_index": len(accepted) + 1,
            "theta_sha256": digest,
            "objective_before": previous_f,
            "objective_after": result["F"],
            "delta_f": result["F"] - previous_f,
            "identity_class": "ACCEPTED_STRICT_PASS",
            "max_absolute": strict["max_absolute"],
            "physical_regime": "NO_FLAG_OBSERVED",
            "physical_audit": regime,
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
            "ln_q": float(theta[1]),
            "q": float(math.exp(theta[1])),
            "ln_sigma": float(theta[93]),
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
    atomic_json(invocation, {"model": "ptq-screen", "started": True, "scientific_invocations": 1, "start_sha256": START_SHA})
    try:
        MINIMIZE_CALLS += 1
        scipy_result = minimize(fun, theta0, method="L-BFGS-B", jac=True, bounds=native_bounds(), callback=callback, options=dict(OPTIONS))
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc), "restarted": False}
        print("STOP", failure, flush=True)
    theta_end, meta_end, manifest = store.recover()
    try:
        replay = evaluate(theta_end, theta_hash(theta_end), "ENDPOINT_REPLAY")
    except BudgetLimit as exc:
        replay = None
        endpoint_budget_stop = str(exc)
    else:
        endpoint_budget_stop = None
    if replay is None:
        endpoint = {
            "pass": False,
            "status": "BUDGET_LIMIT",
            "message": endpoint_budget_stop,
            "recovered_kind": meta_end["kind"],
            "recovered_iteration_index": meta_end["iteration_index"],
            "recovered_theta_sha256": theta_hash(theta_end),
            "recorded_objective": meta_end["objective_after"],
            "quadrature": "fine192",
        }
    else:
        replay_strict = classify_table(replay["records"], strict_only=True)
        replay_kkt = projected_kkt(theta_end, replay["gradient"])
        replay_regime = regime_of(replay["records"])
        endpoint = {
            "pass": (
                replay_strict["class"] == "ACCEPTED_STRICT_PASS"
                and abs(replay["F"] - meta_end["objective_after"]) <= REPLAY_TOLERANCE
                and not replay_kkt["active_bounds"]["outside"]
                and bool(np.isfinite(replay["gradient"]).all())
                and all(row["domain"] == "INTERIOR_ANALYTICALLY_DIFFERENTIABLE" for row in replay["records"])
            ),
            "status": "COMPLETED",
            "recovered_kind": meta_end["kind"],
            "recovered_iteration_index": meta_end["iteration_index"],
            "recovered_theta_sha256": theta_hash(theta_end),
            "recorded_objective": meta_end["objective_after"],
            "replay_objective": replay["F"],
            "objective_abs_gap": abs(replay["F"] - meta_end["objective_after"]),
            "max_per_galaxy_gap": replay_strict.get("max_absolute"),
            "projected_gradient_norm": replay_kkt["projected_gradient_norm"],
            "full_gradient_norm": replay_kkt["full_gradient_norm"],
            "outside_bounds": replay_kkt["active_bounds"]["outside"],
            "physical": replay_regime,
            "galaxies": len(replay["records"]),
            "quadrature": "fine192",
        }
    accepted_public = []
    for row in accepted:
        accepted_public.append({key: value for key, value in row.items() if key not in ("theta", "gradient", "projected_gradient")})
    payload = {
        "regression": {key: value for key, value in tests.items() if key != "pass"},
        "regression_pass": tests["pass"],
        "fixed_point": fixed_point,
        "preflight": start_summary,
        "gradient_gap": gradient_gap,
        "first_trial": first_trial,
        "requests": requests,
        "accepted": accepted_public,
        "noncanonical": noncanonical,
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
            "marginals": budget.marginals,
            "pair_equivalents": budget.eq,
            "optimizer_requests": budget.requests,
            "score_diagnostics": budget.score_diagnostics,
            "wall_seconds": time.perf_counter() - budget.t0,
            "scientific_likelihood_calls": MARGINAL_CALLS,
            "minimize_invocations": MINIMIZE_CALLS,
            "attempted_galaxies": budget.attempted_galaxies,
            "completed_galaxies": budget.completed_galaxies,
        },
        "scipy_version": scipy.__version__,
        "factr_from_ftol": OPTIONS["ftol"] / float(np.finfo(float).eps),
    }
    atomic_json(NEW / "logs/run_state.json", payload)
    print("D2R_DONE", json.dumps({
        "accepted": len(accepted),
        "first_class": None if first_trial is None else first_trial["kind"],
        "first_match": None if first_trial is None else first_trial["matches_d2_trial"],
        "endpoint": endpoint["pass"],
        "failure": failure,
        "scipy": payload["scipy"],
        "noncanonical": len(noncanonical),
    }), flush=True)


if __name__ == "__main__":
    main()
