"""A8 five-family pilot. Four new L-BFGS-B smokes. ptq-nu is not optimized."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import minimize

NEW = Path("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/mle_pilot_a8_five_family_pilot")
IMPL = Path("/home/rocky/PTQ-Quaternionic-RC")
A7R1 = Path("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/mle_pilot_a7_r1_ptqnu_smoke_repair")
sys.path.insert(0, str(NEW / "diagnostic"))
sys.path.insert(0, str(NEW / "candidate/src"))

from ptquat.constants import C_LIGHT, H0_SI
from ptquat.data import load_tidy_sparc
from ptquat.models import c_powerlaw
from signed_jacobian import SCIENTIFIC_TARGET, VARIANCE_TOKEN, galaxy_jacobian_row

IDENTITY_LIMIT = 1e-9
REPLAY_TOLERANCE = 1e-8
JACOBIAN_SHA = "d23cb42195870aab92745d1735e54366435c8ae49f3d214f858164364d439585"
MODELS_SHA = "7214664b9a4a3e182f99748a9c4c7da91a98b930399a832589203395f3b89b1b"
MASK_SHA = "df0086b1e67f96403ff09f5c5bfb7f5818593d141d14234d752cb433a9511e6e"
TIDY_SHA = "3ab1c3c2b110c1e48680ebae77c1ceefaa4f778f81119dcf04881b630b274cbb"
PTQNU_START_SHA = "7abded50306ec11a32380f1ae85c555faef1d2d9318f39c673b691ae974473b2"
PTQNU_ACCEPTED3_SHA = "0977c53ffb201756b9a81d9df0a97a2fb5426a89df5d0cb72f48e6695fdfd7d3"
A0_BOUNDS = (5.0e-11, 2.0e-10)
EPS_WIDE_BOUNDS = (1.0e-6, 3.999)
UPSILON_BOUNDS = (0.05, 1.5)
Q_BOUNDS = (0.25, 8.0)
SIGMA_BOUNDS = (1e-3, 200.0)
LOGM_BOUNDS = (9.0, 13.0)
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
FAMILY_SPECS = (
    ("baryon", "baryon_last_accepted_theta.npy", "1e9c848c43e456d6eb233b90ff4796edaf2a30fb7cd9daf274fe40e95f680714", 92),
    ("ptq", "ptq_last_accepted_theta.npy", "96ac8469398b416aa0740522da74c844c12ea8e95364d69f3920f2df460fd6c4", 93),
    ("ptq-screen", "ptq_screen_last_accepted_theta.npy", "b0a24278778944f59b3ae440ba3ae5a63d9dec584c851f93a64fa3e693ff3ab4", 94),
    ("nfw1p", "nfw1p_initial_theta.npy", "5c4ad3c07e46f3fa3df746c3cb5a1eeb42fc8577fbd5ec316dcd1c12ae617d7e", 183),
)
FORBIDDEN_MODELS = {"ptq-nu", "mond", "mond-screen"}
NFW_FIXTURES = (
    ("UGC01281", "upsilon", "sl_U", 0.05, -0.8430147826831202),
    ("UGC08837", "log10_M200", "sl_lM", 9.0, -1.3048003757702087),
)
SCIENTIFIC_CALLS = 0


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


def layout(model, g=91):
    if model in FORBIDDEN_MODELS:
        raise SourceStop("model is not authorized for this pilot: " + model)
    slot = 0
    spec = {"model": model}
    if model == "ptq":
        spec["i_eps"] = slot
        slot += 1
        spec["sl_U"] = slice(slot, slot + g)
        slot += g
    elif model == "ptq-screen":
        spec["i_eps"] = slot
        slot += 1
        spec["i_lq"] = slot
        slot += 1
        spec["sl_U"] = slice(slot, slot + g)
        slot += g
    elif model == "baryon":
        spec["sl_U"] = slice(slot, slot + g)
        slot += g
    elif model == "nfw1p":
        spec["sl_U"] = slice(slot, slot + g)
        slot += g
        spec["sl_lM"] = slice(slot, slot + g)
        slot += g
    else:
        raise SourceStop("unsupported model " + model)
    spec["i_lnsig"] = slot
    slot += 1
    spec["k"] = slot
    return spec


def native_bounds(model):
    spec = layout(model)
    use_matched = model == "ptq-screen"
    if use_matched:
        eps_lo = A0_BOUNDS[0] / (C_LIGHT * H0_SI)
        eps_hi = A0_BOUNDS[1] / (C_LIGHT * H0_SI)
    else:
        eps_lo, eps_hi = EPS_WIDE_BOUNDS
    bounds = [None] * spec["k"]
    if "i_eps" in spec:
        bounds[spec["i_eps"]] = (eps_lo, eps_hi)
    if "i_lq" in spec:
        bounds[spec["i_lq"]] = (math.log(Q_BOUNDS[0]), math.log(Q_BOUNDS[1]))
    for index in range(spec["sl_U"].start, spec["sl_U"].stop):
        bounds[index] = UPSILON_BOUNDS
    if "sl_lM" in spec:
        for index in range(spec["sl_lM"].start, spec["sl_lM"].stop):
            bounds[index] = LOGM_BOUNDS
    bounds[spec["i_lnsig"]] = (math.log(SIGMA_BOUNDS[0]), math.log(SIGMA_BOUNDS[1]))
    if any(item is None for item in bounds) or len(bounds) != spec["k"]:
        raise SourceStop("incomplete bounds " + model)
    return bounds


def unpack(theta, model):
    spec = layout(model)
    theta = np.asarray(theta, float)
    if theta.shape != (spec["k"],):
        raise SourceStop("dimension " + model)
    out = {"U": np.asarray(theta[spec["sl_U"]], float), "sigma": float(math.exp(theta[spec["i_lnsig"]]))}
    if "i_eps" in spec:
        out["eps"] = float(theta[spec["i_eps"]])
    if "i_lq" in spec:
        out["lq"] = float(theta[spec["i_lq"]])
    if "sl_lM" in spec:
        out["lM"] = np.asarray(theta[spec["sl_lM"]], float)
    return out


def assemble_gradient(rows, model):
    spec = layout(model)
    local_keys = ["upsilon"] + (["log10_M200"] if "sl_lM" in spec else [])
    global_keys = ["ln_sigma_sys"]
    if "i_eps" in spec:
        global_keys.append("epsilon")
    if "i_lq" in spec:
        global_keys.append("ln_q")
    gradient = np.full(spec["k"], np.nan)
    for gi, row in enumerate(rows):
        dlogl = row["dlogL"]
        for key in local_keys:
            value = None if dlogl is None else dlogl.get(key, None)
            if value is None or isinstance(value, bool) or not math.isfinite(float(value)):
                raise SourceStop(f"missing gradient coordinate {model} {key} galaxy {gi}")
        gradient[spec["sl_U"].start + gi] = -float(dlogl["upsilon"])
        if "sl_lM" in spec:
            gradient[spec["sl_lM"].start + gi] = -float(dlogl["log10_M200"])
    for key in global_keys:
        total = 0.0
        for gi, row in enumerate(rows):
            value = row["dlogL"].get(key, None)
            if value is None or isinstance(value, bool) or not math.isfinite(float(value)):
                raise SourceStop(f"missing gradient coordinate {model} {key} galaxy {gi}")
            total += float(value)
        slot = {"ln_sigma_sys": "i_lnsig", "epsilon": "i_eps", "ln_q": "i_lq"}[key]
        gradient[spec[slot]] = -total
    if gradient.shape != (spec["k"],) or not np.isfinite(gradient).all():
        raise SourceStop("unsupported gradient coordinate " + model)
    return gradient


def kkt(theta, gradient, model):
    x = np.asarray(theta, float)
    g = np.asarray(gradient, float)
    if x.shape != g.shape:
        raise SourceStop("gradient dimension")
    bounds = np.asarray(native_bounds(model), float)
    lower = x <= bounds[:, 0]
    upper = x >= bounds[:, 1]
    outside = (x < bounds[:, 0]) | (x > bounds[:, 1])
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
        "outside_bounds": np.flatnonzero(outside).astype(int).tolist(),
        "projected_gradient": projected,
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
    def __init__(self, root: Path, model: str):
        self.root = root
        self.model = model
        self.root.mkdir(parents=True, exist_ok=True)

    def commit(self, theta, record: dict) -> dict:
        validate_schema(record)
        if record.get("model") != self.model:
            raise SchemaError("model mismatch")
        theta = np.asarray(theta, float)
        if theta.shape != (layout(self.model)["k"],):
            raise SchemaError("theta dimension")
        digest = theta_hash(theta)
        if digest != record["theta_sha256"]:
            raise SchemaError("theta does not match theta_sha256")
        index = record["iteration_index"]
        theta_name = f"accepted_{index}_theta.npy"
        meta_name = f"accepted_{index}_meta.json"
        atomic_npy(self.root / theta_name, theta)
        stored = {key: value for key, value in record.items() if key != "theta"}
        meta_sha = atomic_json(self.root / meta_name, stored)
        manifest = {
            "latest_kind": record["kind"],
            "iteration_index": index,
            "theta_file": theta_name,
            "meta_file": meta_name,
            "theta_sha256": digest,
            "meta_sha256": meta_sha,
            "objective_after": record["objective_after"],
            "model": self.model,
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
        if manifest.get("model") != self.model:
            raise SourceStop("manifest model mismatch")
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
        if record.get("model") != self.model:
            raise SourceStop("recovered model mismatch")
        return theta, record, manifest


def _record(model, index, theta, before, after, projected, full, bounds, evaluations, elapsed, kind):
    return {
        "kind": kind,
        "iteration_index": int(index),
        "theta_sha256": theta_hash(theta),
        "objective_before": float(before),
        "objective_after": float(after),
        "projected_gradient_norm": float(projected),
        "full_gradient_norm": float(full),
        "active_bounds": bounds,
        "evaluation_count": int(evaluations),
        "elapsed_seconds": float(elapsed),
        "model": model,
        "dimension": int(layout(model)["k"]),
        "scientific_target": SCIENTIFIC_TARGET,
        "variance_token": VARIANCE_TOKEN,
        "mask_sha256": MASK_SHA,
        "models_sha256": MODELS_SHA,
        "jacobian_sha256": JACOBIAN_SHA,
    }


def regression_tests() -> dict:
    root = NEW / "state/regression_fixture"
    if root.exists():
        for child in sorted(root.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
    results = {}
    expected = {"baryon": 92, "ptq": 93, "ptq-screen": 94, "nfw1p": 183}
    for model, dimension in expected.items():
        if layout(model)["k"] != dimension or len(native_bounds(model)) != dimension:
            raise SchemaError("dimension hardcoded incorrectly")
    for forbidden in FORBIDDEN_MODELS:
        refused = False
        try:
            layout(forbidden)
        except SourceStop:
            refused = True
        results[f"refuses_{forbidden}"] = refused

    store = Store(root, "baryon")
    base = np.linspace(0.2, 1.2, 92)
    initial = _record("baryon", 0, base, 10.0, 10.0, 1.0, 2.0, {"lower": [], "upper": []}, 0, 0.0, "INITIAL")
    store.commit(base, initial)
    theta, record, _ = store.recover()
    results["A_first_callback"] = record["iteration_index"] == 0 and record["objective_after"] == 10.0 and theta.shape == (92,)

    step1 = base.copy()
    step1[0] = 0.21
    rec1 = _record("baryon", 1, step1, 10.0, 9.5, 0.8, 1.5, {"lower": [1], "upper": []}, 2, 1.0, "ACCEPTED")
    store.commit(step1, rec1)
    theta, record, _ = store.recover()
    results["B_second_callback"] = record["iteration_index"] == 1 and record["objective_before"] == 10.0 and record["objective_after"] == 9.5

    step2 = step1.copy()
    step2[1] = 0.22
    rec2 = _record("baryon", 2, step2, 9.5, 9.0, 0.4, 1.1, {"lower": [1], "upper": [4]}, 4, 2.0, "ACCEPTED")
    store.commit(step2, rec2)
    _, record, _ = store.recover()
    results["C_third_callback"] = record["iteration_index"] == 2 and record["objective_after"] == 9.0

    atomic_json(root / "rejected_trial.json", {"kind": "REJECTED_TRIAL", "objective_after": -1000.0})
    _, record, _ = store.recover()
    results["D_rejected_trial_ignored"] = record["iteration_index"] == 2 and record["objective_after"] == 9.0

    atomic_npy(root / "newer_uncommitted_theta.npy", base)
    theta, record, _ = store.recover()
    results["E_uncommitted_trial_ignored"] = theta_hash(theta) == theta_hash(step2)

    broken = dict(rec2)
    del broken["objective_after"]
    broken["iteration_index"] = 3
    failed_closed = False
    try:
        store.commit(step2, broken)
    except SchemaError:
        failed_closed = True
    _, record, _ = store.recover()
    results["F_missing_objective_after"] = failed_closed and record["iteration_index"] == 2 and record["objective_after"] == 9.0

    atomic_npy(root / "accepted_2_theta.npy", base)
    refused = False
    try:
        store.recover()
    except SourceStop:
        refused = True
    store.commit(step2, rec2)
    results["G_hash_mismatch_rejected"] = refused and theta_hash(store.recover()[0]) == theta_hash(step2)

    staged = step2.copy()
    staged[2] = 0.33
    staged_record = _record("baryon", 3, staged, 9.0, 8.5, 0.2, 0.9, {"lower": [], "upper": []}, 6, 3.0, "ACCEPTED")
    validate_schema(staged_record)
    atomic_npy(root / "accepted_3_theta.npy", staged)
    atomic_json(root / "accepted_3_meta.json", staged_record)
    theta, record, _ = store.recover()
    results["H_crash_before_manifest"] = record["iteration_index"] == 2 and theta_hash(theta) == theta_hash(step2)

    store.commit(staged, staged_record)
    _, first, _ = store.recover()
    nxt = staged.copy()
    nxt[3] = 0.34
    rec_next = _record("baryon", 4, nxt, first["objective_after"], 8.0, 0.1, 0.5, {"lower": [], "upper": []}, 8, 4.0, "ACCEPTED")
    store.commit(nxt, rec_next)
    _, second, _ = store.recover()
    results["I_objective_continuity"] = second["objective_before"] == first["objective_after"] == 8.5 and second["objective_after"] == 8.0

    trial = {"kind": "TRIAL", "iteration_index": 9, "objective_after": -1e9, "theta_sha256": theta_hash(base)}
    atomic_json(root / "trial_record.json", trial)
    _, recovered, manifest = store.recover()
    trial_commit_rejected = False
    try:
        store.commit(base, _record("baryon", 5, base, 8.0, 7.0, 0.1, 0.2, {"lower": [], "upper": []}, 9, 5.0, "TRIAL"))
    except SchemaError:
        trial_commit_rejected = True
    results["J_trial_is_not_accepted"] = (
        recovered["kind"] == "ACCEPTED" and manifest["latest_kind"] == "ACCEPTED"
        and recovered["iteration_index"] == 4 and trial_commit_rejected
    )
    results["schema_validation"] = results["F_missing_objective_after"] and results["J_trial_is_not_accepted"]

    dimension_ok = True
    for model, dimension in expected.items():
        sample = np.linspace(0.1, 0.2, dimension)
        local = Store(root / f"dim_{dimension}", model)
        local.commit(sample, _record(model, 0, sample, 1.0, 1.0, 0.5, 0.5, {"lower": [], "upper": []}, 0, 0.0, "INITIAL"))
        theta, record, manifest = local.recover()
        dimension_ok = dimension_ok and theta.shape == (dimension,) and record["dimension"] == dimension and manifest["model"] == model
    results["dimension_roundtrip_92_93_94_183"] = dimension_ok

    missing = False
    incomplete = [{"dlogL": {"upsilon": 1.0, "ln_sigma_sys": 0.1}} for _ in range(91)]
    try:
        assemble_gradient(incomplete, "nfw1p")
    except SourceStop:
        missing = True
    complete = [{"dlogL": {"upsilon": 0.2, "log10_M200": -0.3, "ln_sigma_sys": 0.01}} for _ in range(91)]
    assembled = assemble_gradient(complete, "nfw1p")
    results["missing_coordinate_stops_without_zero_fill"] = missing and abs(float(assembled[91]) - 0.3) < 1e-15
    results["scientific_likelihood_calls"] = SCIENTIFIC_CALLS
    keys = [key for key in results if key not in ("pass", "scientific_likelihood_calls")]
    results["pass"] = SCIENTIFIC_CALLS == 0 and all(bool(results[key]) for key in keys)
    return results


class Budget:
    def __init__(self, weights):
        self.weights = weights
        self.calls = 0
        self.eq = 0.0
        self.t0 = time.perf_counter()
        self.models = {
            model: {"calls": 0, "eq": 0.0, "requests": 0}
            for model, _, _, _ in FAMILY_SPECS
        }

    def charge(self, model, name):
        weight = float(self.weights[name])
        slot = self.models[model]
        if self.calls + 1 > 6400 or self.eq + weight > 64.0 or slot["eq"] + weight > 16.0:
            raise BudgetLimit("budget")
        if time.perf_counter() - self.t0 > 90 * 60:
            raise BudgetLimit("wall")
        self.calls += 1
        self.eq += weight
        slot["calls"] += 1
        slot["eq"] += weight

    def begin_optimizer_request(self, model):
        slot = self.models[model]
        if slot["requests"] >= 12:
            raise BudgetLimit(model + " request cap")
        if slot["eq"] + 2.0 > 16.0 or self.eq + 2.0 > 64.0 or self.calls + 182 > 6400:
            raise BudgetLimit(model + " reserve terminal replay")
        if time.perf_counter() - self.t0 > 90 * 60:
            raise BudgetLimit("wall")
        slot["requests"] += 1

    def snapshot(self, model):
        slot = dict(self.models[model])
        slot["wall_seconds"] = time.perf_counter() - self.t0
        slot["total_calls"] = self.calls
        slot["total_equivalents"] = self.eq
        return slot


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


def verify_a7r1():
    mismatches = []
    for line in (A7R1 / "SHA256SUMS").read_text().splitlines():
        if not line.strip():
            continue
        digest, path = line.split("  ", 1)
        if sha256_file(path) != digest:
            mismatches.append(path)
    manifest = json.loads((A7R1 / "state/scientific/ACCEPTED_MANIFEST.json").read_text())
    theta = np.load(A7R1 / "state/scientific" / manifest["theta_file"])
    meta = json.loads((A7R1 / "state/scientific" / manifest["meta_file"]).read_text())
    consistent = (
        not mismatches
        and manifest["iteration_index"] == 3
        and manifest["objective_after"] == 6614.908093993494
        and theta_hash(theta) == PTQNU_ACCEPTED3_SHA
        and meta["objective_after"] == 6614.908093993494
        and meta["kind"] == "ACCEPTED"
        and sha256_file(A7R1 / "state/scientific" / manifest["meta_file"]) == manifest["meta_sha256"]
    )
    start = np.load(IMPL / "results/frontiers_revision_2026_10/mle_pilot_a/execution_v1/ptq_nu_last_accepted_theta.npy")
    return {
        "integrity": "PASS" if consistent and theta_hash(start) == PTQNU_START_SHA else "FAIL",
        "sha256sums_mismatches": mismatches,
        "process_status_error_was_not_file_corruption": True,
        "manifest_iteration_index": manifest["iteration_index"],
        "objective_after": manifest["objective_after"],
        "accepted_iteration_3_sha256": theta_hash(theta),
        "original_start_sha256": theta_hash(start),
        "a7r1_audit_status": json.loads((A7R1 / "AUDIT_STATUS.json").read_text()),
    }


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
    archived = json.loads((IMPL / "results/frontiers_revision_2026_10/mle_pilot_a6_signed_jac/FIVE_FAMILY_GRADIENT_VECTORS.json").read_text())["vectors"]
    return galaxies, weights, archived


def nfw_relation_ok():
    pivot = c_powerlaw(1.0e12, c0=10.0, beta=-0.1, Mpiv=1.0e12)
    tenth = c_powerlaw(1.0e11, c0=10.0, beta=-0.1, Mpiv=1.0e12)
    expected = 10.0 * (0.1 ** -0.1)
    return pivot == 10.0 and abs(tenth - expected) == 0.0


def fixture_rows(galaxies, theta, gradient, model):
    if model != "nfw1p":
        return []
    spec = layout(model)
    names = [galaxy.name for galaxy in galaxies]
    bounds = native_bounds(model)
    rows = []
    for galaxy, coordinate, slot, bound_value, reference in NFW_FIXTURES:
        gi = names.index(galaxy)
        index = spec[slot].start + gi
        component = float(gradient[index])
        value = float(theta[index])
        lo, hi = bounds[index]
        projected = component
        if value <= lo and component > 0.0:
            projected = 0.0
        if value >= hi and component < 0.0:
            projected = 0.0
        rows.append({
            "galaxy": galaxy,
            "coordinate": coordinate,
            "index": int(index),
            "value": value,
            "bound_value": bound_value,
            "on_recorded_bound": abs(value - bound_value) <= 1e-12,
            "unprojected": component,
            "projected": projected,
            "reference_unprojected": reference,
            "reference_abs_gap": abs(component - reference),
            "nonzero_projected_on_bound": (value <= lo or value >= hi) and projected != 0.0,
        })
    return rows


def classify_verdict(preflight_pass, accepted, failure, endpoint, scipy_result):
    if failure and failure.get("type") == "BudgetLimit":
        return "BUDGET_LIMIT"
    if not preflight_pass:
        return "BLOCKED"
    if failure and failure.get("type") == "SchemaError":
        return "FAIL"
    replay_ok = bool(endpoint and endpoint.get("pass"))
    decreases = bool(accepted) and all(row["delta_f"] <= REPLAY_TOLERANCE for row in accepted)
    improved = bool(accepted) and all(row["delta_f"] < 0.0 for row in accepted)
    if failure and failure.get("type") == "SourceStop":
        if accepted and decreases and replay_ok:
            return "CONDITIONAL"
        return "FAIL"
    if not replay_ok or not decreases:
        return "FAIL"
    if len(accepted) >= 2 and improved:
        return "PASS_WITH_WARN"
    if len(accepted) == 1 and improved and scipy_result is not None:
        return "CONDITIONAL"
    if len(accepted) == 0:
        return "FAIL"
    return "CONDITIONAL"


def main():
    verify_sources()
    evidence = verify_a7r1()
    if evidence["integrity"] != "PASS":
        raise SystemExit("STOP A7-R1 evidence integrity")
    if abs(c_powerlaw(1.0e12) - 10.0) != 0.0 or not nfw_relation_ok():
        raise SystemExit("STOP c(M200) relation")
    tests = regression_tests()
    (NEW / "logs").mkdir(parents=True, exist_ok=True)
    (NEW / "logs/callback_schema_tests.json").write_text(json.dumps(tests, indent=2) + "\n")
    print("REGRESSION", tests["pass"], "likelihood_calls", tests["scientific_likelihood_calls"], flush=True)
    if not tests["pass"]:
        raise SystemExit("ENGINEERING_REGRESSION_FAIL")
    if os.environ.get("A8_REGRESSION_ONLY") == "1":
        print("REGRESSION_ONLY", flush=True)
        return

    galaxies, weights, archived_vectors = load_sample()
    budget = Budget(weights)
    families = []
    invocations = 0
    for model, filename, digest, dimension in FAMILY_SPECS:
        if model in FORBIDDEN_MODELS:
            raise SystemExit("STOP forbidden model")
        family_root = NEW / "families" / model
        log_path = family_root / "logs/requests.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        state_dir = family_root / "state"
        record = {
            "model": model,
            "authorized_filename": filename,
            "expected_sha256": digest,
            "expected_dimension": dimension,
        }
        failure = None
        accepted = []
        requests = []
        warning_log = []
        scipy_result = None
        endpoint = None
        preflight = {"pass": False}
        store = Store(state_dir, model)
        try:
            theta0 = np.load(IMPL / "results/frontiers_revision_2026_10/mle_pilot_a/execution_v1" / filename)
            got = theta_hash(theta0)
            if got != digest or theta0.shape != (dimension,) or layout(model)["k"] != dimension:
                raise SourceStop("start vector")
            bounds = native_bounds(model)
            array_bounds = np.asarray(bounds, float)
            if not np.isfinite(theta0).all() or np.any(theta0 < array_bounds[:, 0]) or np.any(theta0 > array_bounds[:, 1]):
                raise SourceStop("start outside bounds")
            if model == "nfw1p" and not nfw_relation_ok():
                raise SourceStop("c(M200) relation changed")

            def evaluate(theta, phase_model=model):
                global SCIENTIFIC_CALLS
                started = time.perf_counter()
                pars = unpack(theta, phase_model)
                rows = []
                for gi, galaxy in enumerate(galaxies):
                    budget.charge(phase_model, galaxy.name)
                    SCIENTIFIC_CALLS += 1
                    rows.append(galaxy_jacobian_row(galaxy, pars, gi, phase_model))
                gap = max(row["abs_gap"] for row in rows)
                if gap > IDENTITY_LIMIT:
                    raise SourceStop("objective identity " + str(gap))
                flags = sorted({flag for row in rows for flag in row["flags"]})
                domains = sorted({row["domain"] for row in rows})
                if any(flag in CLIP_FLAGS for flag in flags) or domains != ["INTERIOR_ANALYTICALLY_DIFFERENTIABLE"]:
                    raise SourceStop("unexpected domain " + ",".join(flags + domains))
                gradient = assemble_gradient(rows, phase_model)
                value = -sum(row["official_logL"] for row in rows)
                if not math.isfinite(value):
                    raise SourceStop("nonfinite objective")
                return {
                    "F": float(value),
                    "gradient": gradient,
                    "max_identity_gap": float(gap),
                    "flags": flags,
                    "domains": domains,
                    "seconds": time.perf_counter() - started,
                    "n_galaxies": len(rows),
                }

            initial = evaluate(theta0)
            archived = archived_vectors[model]
            archived_gradient = np.asarray(archived["native_dF"], float)
            gradient_gap = float(np.max(np.abs(archived_gradient - initial["gradient"])))
            objective_gap = abs(initial["F"] - float(archived["objective"]))
            initial_kkt = kkt(theta0, initial["gradient"], model)
            preflight = {
                "pass": gradient_gap <= 1e-8 and objective_gap <= REPLAY_TOLERANCE and not initial_kkt["outside_bounds"],
                "theta_sha256": got,
                "dimension": dimension,
                "initial_F": initial["F"],
                "a6_objective": float(archived["objective"]),
                "objective_abs_gap": objective_gap,
                "archived_gradient_max_abs_gap": gradient_gap,
                "max_identity_gap": initial["max_identity_gap"],
                "projected_gradient_norm": initial_kkt["projected_gradient_norm"],
                "full_gradient_norm": initial_kkt["full_gradient_norm"],
                "active_bounds": initial_kkt["active_bounds"],
                "outside_bounds": initial_kkt["outside_bounds"],
                "flags": initial["flags"],
                "domains": initial["domains"],
                "gradient_finite": True,
                "c_m200_unchanged": nfw_relation_ok(),
                "zero_filled_coordinates": False,
                "fixtures": fixture_rows(galaxies, theta0, initial["gradient"], model),
            }
            print(
                f"PREFLIGHT {model} F {initial['F']:.8f} gap {objective_gap:.3e} grad {gradient_gap:.3e} "
                f"identity {initial['max_identity_gap']:.3e} proj {initial_kkt['projected_gradient_norm']:.8f}",
                flush=True,
            )
            if not preflight["pass"]:
                raise SourceStop("preflight")
            if model == "nfw1p" and any(
                (not item["on_recorded_bound"]) or item["reference_abs_gap"] > 1e-8 or (not item["nonzero_projected_on_bound"])
                for item in preflight["fixtures"]
            ):
                preflight["pass"] = False
                raise SourceStop("nfw archived bound fixture mismatch")
            store.commit(theta0, _record(
                model, 0, theta0, initial["F"], initial["F"], initial_kkt["projected_gradient_norm"],
                initial_kkt["full_gradient_norm"], initial_kkt["active_bounds"], 0, 0.0, "INITIAL",
            ))
            cache = {}
            warning_log = []

            def fun(theta):
                theta = np.asarray(theta, float).copy()
                digest_now = theta_hash(theta)
                budget.begin_optimizer_request(model)
                if digest_now in cache:
                    cached = cache[digest_now]
                    requests.append({
                        "request_index": budget.models[model]["requests"],
                        "theta_sha256": digest_now,
                        "kind": "TRIAL",
                        "status": "CACHE_HIT",
                        "F": cached["F"],
                    })
                    return cached["F"], cached["gradient"].copy()
                if (not np.isfinite(theta).all()) or np.any(theta < array_bounds[:, 0]) or np.any(theta > array_bounds[:, 1]):
                    raise SourceStop("trial outside original bounds")
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    result = evaluate(theta)
                for item in caught:
                    warning_log.append({"request": budget.models[model]["requests"], "category": item.category.__name__, "message": str(item.message)})
                cache[digest_now] = result
                diag = kkt(theta, result["gradient"], model)
                row = {
                    "request_index": budget.models[model]["requests"],
                    "theta_sha256": digest_now,
                    "kind": "TRIAL",
                    "status": "COMPLETE_NOT_ACCEPTED",
                    "F": result["F"],
                    "full_gradient_norm": diag["full_gradient_norm"],
                    "projected_gradient_norm": diag["projected_gradient_norm"],
                    "max_identity_gap": result["max_identity_gap"],
                    "flags": result["flags"],
                    "seconds": result["seconds"],
                    "galaxy_calls": budget.models[model]["calls"],
                    "equivalents": budget.models[model]["eq"],
                    "elapsed_seconds": time.perf_counter() - budget.t0,
                }
                requests.append(row)
                with log_path.open("a") as handle:
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                print(f"REQUEST {model} {row['request_index']} F {result['F']:.8f}", flush=True)
                return result["F"], result["gradient"].copy()

            def callback(xk):
                if hasattr(xk, "x"):
                    xk = xk.x
                theta = np.asarray(xk, float).copy()
                digest_now = theta_hash(theta)
                if digest_now not in cache:
                    raise SourceStop("callback theta was not an exact completed evaluation")
                result = cache[digest_now]
                diag = kkt(theta, result["gradient"], model)
                if diag["outside_bounds"]:
                    raise SourceStop("accepted theta outside bounds")
                previous = theta0 if not accepted else np.asarray(accepted[-1]["theta"], float)
                previous_f = initial["F"] if not accepted else accepted[-1]["objective_after"]
                if result["F"] > previous_f + REPLAY_TOLERANCE:
                    raise SourceStop("accepted objective increased")
                built = _record(
                    model, len(accepted) + 1, theta, previous_f, result["F"], diag["projected_gradient_norm"],
                    diag["full_gradient_norm"], diag["active_bounds"], budget.models[model]["requests"],
                    time.perf_counter() - budget.t0, "ACCEPTED",
                )
                built.update({
                    "delta_f": result["F"] - previous_f,
                    "step_l2": float(np.linalg.norm(theta - previous)),
                    "step_inf": float(np.max(np.abs(theta - previous))),
                    "gradient": result["gradient"].tolist(),
                    "projected_gradient": diag["projected_gradient"].tolist(),
                    "fixtures": fixture_rows(galaxies, theta, result["gradient"], model),
                    "max_identity_gap": result["max_identity_gap"],
                    "flags": result["flags"],
                    "domains": result["domains"],
                    "c_m200_unchanged": nfw_relation_ok(),
                    "zero_filled_coordinates": False,
                })
                validate_schema(built)
                store.commit(theta, built)
                built["theta"] = theta.tolist()
                accepted.append(built)
                for item in requests:
                    if item["theta_sha256"] == digest_now and item["kind"] == "TRIAL":
                        item["kind"] = "ACCEPTED"
                        item["status"] = "ACCEPTED"
                print(
                    f"ACCEPTED {model} {built['iteration_index']} F {built['objective_after']:.8f} dF {built['delta_f']:.8f}",
                    flush=True,
                )
                if len(accepted) > 3:
                    raise BudgetLimit("accepted iteration cap")

            invocation = state_dir / "OPTIMIZER_INVOCATION.json"
            if invocation.exists():
                raise SourceStop("optimizer already invoked")
            invocations += 1
            if invocations > 4:
                raise SourceStop("optimizer invocation cap")
            atomic_json(invocation, {
                "model": model,
                "started": True,
                "scientific_invocations_for_model": 1,
                "start_sha256": got,
                "options": FROZEN_OPTIONS,
            })
            scipy_result = minimize(
                fun, theta0, method="L-BFGS-B", jac=True, bounds=bounds,
                callback=callback, options=dict(FROZEN_OPTIONS),
            )
        except (BudgetLimit, SourceStop, SchemaError) as exc:
            failure = {"type": type(exc).__name__, "message": str(exc)}
            print(f"STOP {model} {failure['type']} {failure['message']}", flush=True)

        manifest = None
        if (state_dir / "ACCEPTED_MANIFEST.json").exists():
            try:
                theta_end, meta_end, manifest = store.recover()
                if failure and failure.get("type") == "BudgetLimit" and time.perf_counter() - budget.t0 > 90 * 60:
                    endpoint = {"pass": False, "reason": "wall exhausted before replay"}
                else:
                    replay = evaluate(theta_end)
                    replay_kkt = kkt(theta_end, replay["gradient"], model)
                    endpoint = {
                        "pass": (
                            replay["n_galaxies"] == 91
                            and abs(replay["F"] - meta_end["objective_after"]) <= REPLAY_TOLERANCE
                            and replay["max_identity_gap"] <= IDENTITY_LIMIT
                            and not replay_kkt["outside_bounds"]
                            and bool(np.isfinite(replay["gradient"]).all())
                            and replay["domains"] == ["INTERIOR_ANALYTICALLY_DIFFERENTIABLE"]
                        ),
                        "recovered_kind": meta_end["kind"],
                        "recovered_iteration_index": meta_end["iteration_index"],
                        "recovered_theta_sha256": theta_hash(theta_end),
                        "galaxies": replay["n_galaxies"],
                        "quadrature": "fine192",
                        "recorded_objective": meta_end["objective_after"],
                        "replay_objective": replay["F"],
                        "objective_abs_gap": abs(replay["F"] - meta_end["objective_after"]),
                        "max_per_galaxy_path_gap": replay["max_identity_gap"],
                        "outside_bounds": replay_kkt["outside_bounds"],
                        "gradient_finite": bool(np.isfinite(replay["gradient"]).all()),
                        "flags": replay["flags"],
                        "domains": replay["domains"],
                        "projected_gradient_norm": replay_kkt["projected_gradient_norm"],
                        "full_gradient_norm": replay_kkt["full_gradient_norm"],
                        "active_bounds": replay_kkt["active_bounds"],
                        "fixtures": fixture_rows(galaxies, theta_end, replay["gradient"], model),
                    }
            except (BudgetLimit, SourceStop, SchemaError) as exc:
                endpoint = {"pass": False, "reason": type(exc).__name__ + ": " + str(exc)}
                if failure is None:
                    failure = {"type": type(exc).__name__, "message": str(exc)}
        elif preflight.get("initial_F") is not None and failure and failure.get("message") == "preflight":
            endpoint = {
                "pass": False,
                "reason": "preflight failed; optimizer not started; start evaluation was not committed",
                "recorded_objective": preflight.get("initial_F"),
                "max_per_galaxy_path_gap": preflight.get("max_identity_gap"),
            }
        verdict = classify_verdict(bool(preflight.get("pass")), accepted, failure, endpoint, scipy_result)
        if failure and failure.get("type") == "SchemaError":
            verdict = "FAIL"
        public_accepted = []
        for row in accepted:
            public_accepted.append({key: value for key, value in row.items() if key not in ("theta", "gradient", "projected_gradient")})
        for item in requests:
            if item["kind"] == "TRIAL" and item["theta_sha256"] == record.get("expected_sha256") and item["status"] != "ACCEPTED":
                item["kind"] = "INITIAL"
        family = {
            "model": model,
            "verdict": verdict,
            "preflight": preflight,
            "accepted": public_accepted,
            "requests": requests,
            "failure": failure,
            "warnings": warning_log,
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
            "budget": budget.snapshot(model),
            "minimize_invocations": 0 if scipy_result is None and not (state_dir / "OPTIMIZER_INVOCATION.json").exists() else 1,
        }
        if not (state_dir / "OPTIMIZER_INVOCATION.json").exists():
            family["minimize_invocations"] = 0
        families.append(family)
        (family_root / "family_result.json").write_text(json.dumps(family, indent=2) + "\n")
        print(json.dumps({
            "model": model,
            "verdict": verdict,
            "accepted": len(accepted),
            "F0": preflight.get("initial_F"),
            "steps": [
                {key: row[key] for key in ("iteration_index", "objective_before", "objective_after", "delta_f", "projected_gradient_norm")}
                for row in public_accepted
            ],
            "endpoint": None if endpoint is None else {
                "pass": endpoint.get("pass"),
                "gap": endpoint.get("objective_abs_gap"),
                "projected": endpoint.get("projected_gradient_norm"),
            },
            "failure": failure,
            "scipy": family["scipy"],
            "budget": family["budget"],
        }, indent=2), flush=True)
        if failure and failure.get("type") == "SchemaError":
            break

    payload = {
        "a7r1_evidence": evidence,
        "regression": tests,
        "scipy_version": scipy.__version__,
        "options": FROZEN_OPTIONS,
        "factr_from_ftol": FROZEN_OPTIONS["ftol"] / float(np.finfo(float).eps),
        "families": families,
        "scientific_likelihood_calls": SCIENTIFIC_CALLS,
        "budget": {
            "calls": budget.calls,
            "equivalents": budget.eq,
            "wall_seconds": time.perf_counter() - budget.t0,
            "minimize_invocations": invocations,
        },
        "ptq_nu_reused": {
            "optimized_again": False,
            "initial_F": 6614.933826799582,
            "accepted_iteration_3_F": 6614.908093993494,
            "projected_start": 8.12382175678397,
            "projected_endpoint": 4.495983185189471,
            "accepted_iterations": 3,
            "endpoint_replay": "PASS",
            "verdict": "PASS_WITH_WARN",
        },
    }
    (NEW / "logs/run_state.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("A8_DONE", json.dumps({"calls": budget.calls, "eq": budget.eq, "wall": payload["budget"]["wall_seconds"], "invocations": invocations}), flush=True)


if __name__ == "__main__":
    main()
