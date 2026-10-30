"""Capture the first PTQ-screen line-search trial and stop before any accepted step."""
from __future__ import annotations

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

NEW = Path("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/mle_pilot_a8_d2_screen_forensic")
IMPL = Path("/home/rocky/PTQ-Quaternionic-RC")
sys.path.insert(0, str(NEW / "diagnostic"))
sys.path.insert(0, str(NEW / "candidate/src"))

from ptquat.constants import C_LIGHT, H0_SI, KPC, KM
from ptquat.data import load_tidy_sparc
from ptquat.marginal_likelihood import integrate, nuisance_nodes, velocity
from ptquat.models import model_v_ptq_screen, nu_q, vbar_squared_kms2
from signed_jacobian import SCIENTIFIC_TARGET, as_g, galaxy_jacobian_row

IDENTITY_LIMIT = 1e-9
RELATIVE_LIMIT = 1e-12
ULP_FACTOR = 8.0
START_F = 6614.725282751667
START_PROJECTED = 32.30786064991939
START_SHA = "b0a24278778944f59b3ae440ba3ae5a63d9dec584c851f93a64fa3e693ff3ab4"
LINEAR_START_SHA = "96ac8469398b416aa0740522da74c844c12ea8e95364d69f3920f2df460fd6c4"
LINEAR_TRIAL_SHA = "dc65035bc1f195edc72a71c9afa162fbc42a6cd6cec322454ada7f16310b5223"
HISTORICAL_SCREEN_GAP = 1.1920928955078125e-07
HISTORICAL_LINEAR_GAP = 2.7939677238464355e-09
JACOBIAN_SHA = "d23cb42195870aab92745d1735e54366435c8ae49f3d214f858164364d439585"
MODELS_SHA = "7214664b9a4a3e182f99748a9c4c7da91a98b930399a832589203395f3b89b1b"
MASK_SHA = "df0086b1e67f96403ff09f5c5bfb7f5818593d141d14234d752cb433a9511e6e"
TIDY_SHA = "3ab1c3c2b110c1e48680ebae77c1ceefaa4f778f81119dcf04881b630b274cbb"
A0_BOUNDS = (5.0e-11, 2.0e-10)
UPSILON = (0.05, 1.5)
Q_BOUNDS = (0.25, 8.0)
SIGMA = (1e-3, 200.0)
OPTIONS = {"maxiter": 3, "maxls": 20, "maxcor": 10, "ftol": 1e-12, "gtol": 1e-6}
CLIP_FLAGS = {"y_floor", "a0_floor", "radius_floor", "gc_floor", "concentration_floor"}
MARGINAL_CALLS = 0
MINIMIZE_CALLS = 0


class DiagnosticStop(RuntimeError):
    pass


class SchemaError(RuntimeError):
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
    if not math.isfinite(number):
        return None
    ulp = math.ulp(abs(number))
    if (not math.isfinite(ulp)) or ulp <= 0.0:
        return None
    return float(ulp)


def relative_difference(absolute, left, right):
    if not all(math.isfinite(float(value)) for value in (absolute, left, right)):
        return None
    return float(absolute) / max(1.0, abs(float(left)), abs(float(right)))


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
    if theta.shape != (94,):
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


def projected_norm(theta, gradient):
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
    }


class InitialStore:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def commit_initial(self, theta, objective: float) -> dict:
        digest = theta_hash(theta)
        record = {
            "kind": "INITIAL",
            "iteration_index": 0,
            "theta_sha256": digest,
            "objective_before": float(objective),
            "objective_after": float(objective),
            "status": "INITIAL_REFERENCE",
            "model": "ptq-screen",
        }
        atomic_npy(self.root / "accepted_0_theta.npy", theta)
        meta_sha = atomic_json(self.root / "accepted_0_meta.json", record)
        manifest = {
            "latest_kind": "INITIAL",
            "iteration_index": 0,
            "theta_file": "accepted_0_theta.npy",
            "meta_file": "accepted_0_meta.json",
            "theta_sha256": digest,
            "meta_sha256": meta_sha,
            "objective_after": float(objective),
            "model": "ptq-screen",
            "status": "INITIAL_REFERENCE",
        }
        atomic_json(self.root / "ACCEPTED_MANIFEST.json", manifest)
        return manifest

    def reject_accepted_commit(self, theta) -> None:
        del theta
        raise SchemaError("diagnostic trial cannot be committed as an accepted iteration")

    def recover(self):
        manifest = json.loads((self.root / "ACCEPTED_MANIFEST.json").read_text())
        if manifest.get("status") == "UNCOMMITTED_DIAGNOSTIC" or manifest.get("latest_kind") != "INITIAL":
            raise SchemaError("refusing diagnostic trial as a restart state")
        if manifest["iteration_index"] != 0:
            raise SchemaError("accepted iteration is not authorized")
        theta = np.load(self.root / manifest["theta_file"])
        if theta_hash(theta) != manifest["theta_sha256"]:
            raise SchemaError("theta hash mismatch")
        return theta, manifest


def regression_tests() -> dict:
    root = NEW / "state/mock_regression"
    if root.exists():
        for child in sorted(root.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
    root.mkdir(parents=True, exist_ok=True)
    results = {}
    store = InitialStore(root / "accepted")
    start = np.linspace(0.2, 0.8, 94)
    start[0] = 0.12
    start[1] = 0.0
    manifest = store.commit_initial(start, 4.0)
    append_jsonl(root / "rows.jsonl", {
        "kind": "INITIAL",
        "theta_sha256": theta_hash(start),
        "objective": 4.0,
        "complete": True,
    })
    results["1_initial_recorded"] = json.loads((root / "rows.jsonl").read_text().splitlines()[0])["kind"] == "INITIAL"

    trial = start.copy()
    trial[0] = 0.15
    digest = theta_hash(trial)
    persisted = {"done": False}

    def validate():
        if not persisted["done"]:
            raise RuntimeError("validator ran before persistence")
        raise RuntimeError("identity failure")

    atomic_npy(root / "TRIAL_THETA.npy", trial)
    atomic_json(root / "TRIAL_CAPTURE_MANIFEST.json", {
        "status": "UNCOMMITTED_DIAGNOSTIC",
        "theta_sha256": digest,
        "accepted_iteration": False,
    })
    persisted["done"] = True
    try:
        validate()
    except RuntimeError as exc:
        results["2_trial_persisted_before_validation"] = "identity" in str(exc) and (root / "TRIAL_THETA.npy").exists()
    manifest_trial = json.loads((root / "TRIAL_CAPTURE_MANIFEST.json").read_text())
    results["3_uncommitted_status"] = manifest_trial["status"] == "UNCOMMITTED_DIAGNOSTIC"
    append_jsonl(root / "rows.jsonl", {
        "status": "COMPLETE",
        "theta_sha256": digest,
        "galaxy": "synthetic",
        "abs_diff": 1.1920928955078125e-07,
        "complete": True,
    })
    text = (root / "rows.jsonl").read_text()
    results["4_identity_failure_retains_rows"] = "1.1920928955078125e-07" in text
    results["5_rows_carry_theta_hash"] = all("theta_sha256" in json.loads(line) for line in text.splitlines())
    append_jsonl(root / "rows.jsonl", {"status": "INCOMPLETE", "theta_sha256": digest, "complete": False, "galaxy": "partial"})
    results["6_incomplete_marked"] = json.loads((root / "rows.jsonl").read_text().splitlines()[-1])["status"] == "INCOMPLETE"
    ulp = positive_ulp(-5486154.322253605)
    results["7_negative_logl_positive_ulp"] = ulp is not None and ulp > 0.0 and math.isfinite(ulp)
    rejected_nonfinite = False
    try:
        json.dumps({"bad": float("nan")}, allow_nan=False)
    except ValueError:
        rejected_nonfinite = True
    atomic_json(root / "ulp_edge.json", {"ulp": ulp, "zero_ulp": positive_ulp(0.0)})
    results["8_json_rejects_nonfinite"] = rejected_nonfinite and "Infinity" not in (root / "ulp_edge.json").read_text()
    refused = False
    try:
        store.reject_accepted_commit(trial)
    except SchemaError:
        refused = True
    recovered_theta, recovered = store.recover()
    results["9_callback_cannot_commit"] = refused and recovered["iteration_index"] == 0
    results["10_trial_cannot_replace_start"] = theta_hash(recovered_theta) == theta_hash(start) and theta_hash(trial) != theta_hash(start)
    lock = root / "OPTIMIZER_INVOCATION.json"

    def try_invoke():
        if lock.exists():
            return False
        atomic_json(lock, {"started": True, "scientific": False})
        return True

    results["11_second_invocation_blocked"] = try_invoke() is True and try_invoke() is False
    staged = start.copy()
    staged[3] = 0.4
    atomic_npy(root / "accepted" / "accepted_1_theta.npy", staged)
    atomic_json(root / "accepted" / "accepted_1_meta.json", {
        "kind": "ACCEPTED",
        "iteration_index": 1,
        "theta_sha256": theta_hash(staged),
    })
    crashed_theta, crashed = store.recover()
    results["12_crash_recovers_initial"] = crashed["iteration_index"] == 0 and theta_hash(crashed_theta) == theta_hash(start)
    inside = start.copy()
    bounds = np.asarray(native_bounds(), float)
    outside = inside.copy()
    outside[0] = 10.0
    dimension_rejected = False
    try:
        unpack(np.ones(93))
    except SchemaError:
        dimension_rejected = True
    results["13_dimension_and_bounds"] = (
        len(native_bounds()) == 94
        and dimension_rejected
        and np.all((inside >= bounds[:, 0]) & (inside <= bounds[:, 1]))
        and np.any(outside > bounds[:, 1])
        and START_SHA not in (LINEAR_START_SHA, LINEAR_TRIAL_SHA)
    )
    calls = {"n": 0, "callback": 0}

    def synthetic_step(vector):
        calls["n"] += 1
        if calls["n"] == 1:
            append_jsonl(root / "synthetic_rows.jsonl", {"kind": "INITIAL", "theta_sha256": theta_hash(vector), "complete": True})
            return
        atomic_npy(root / "synthetic_trial.npy", vector)
        atomic_json(root / "synthetic_manifest.json", {"status": "UNCOMMITTED_DIAGNOSTIC", "theta_sha256": theta_hash(vector)})
        append_jsonl(root / "synthetic_rows.jsonl", {"kind": "UNCOMMITTED_DIAGNOSTIC", "theta_sha256": theta_hash(vector), "complete": True})
        raise DiagnosticStop("stop before accepted iteration")

    synthetic_step(start)
    stopped = False
    try:
        synthetic_step(trial)
    except DiagnosticStop:
        stopped = True
    results["14_synthetic_capture_stops_before_accept"] = (
        stopped and calls["callback"] == 0 and calls["n"] == 2 and (root / "synthetic_trial.npy").exists()
        and json.loads((root / "synthetic_manifest.json").read_text())["status"] == "UNCOMMITTED_DIAGNOSTIC"
    )
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


def load_sample():
    tidy = pd.read_csv(IMPL / "dataset/sparc_tidy.csv")
    raw = pd.read_csv(IMPL / "dataset/raw/vizier_table2.csv")
    mask = pd.read_csv(IMPL / "results/frontiers_revision_2026_10/mle_pilot_a4_domain_feasibility/S_DOMAIN13_MASK.csv")
    joined = tidy.merge(raw, left_on=["galaxy", "r_kpc"], right_on=["Name", "Rad"], how="left", validate="one_to_one")
    included = mask.included.to_numpy(int) == 1
    if int(included.sum()) != 1769 or not np.array_equal(mask.source_recno.to_numpy(), joined.recno.to_numpy()):
        raise SystemExit("STOP mask")
    filtered = tidy.loc[included]
    out = NEW / "logs/s_domain13_filtered_tidy.csv"
    filtered.to_csv(out, index=False)
    loaded = load_tidy_sparc(str(out))
    galaxies = [loaded[name] for name in sorted(loaded)]
    if len(galaxies) != 91 or sum(len(g.v_obs) for g in galaxies) != 1769:
        raise SystemExit("STOP sample")
    kept = mask.loc[included]
    recnos = {name: group.source_recno.astype(int).tolist() for name, group in kept.groupby("galaxy")}
    weights = {
        row["galaxy"]: float(row["fullsample_equivalent"])
        for row in json.loads((IMPL / "results/frontiers_revision_2026_10/mle_pilot_a/GALAXY_WORK_WEIGHTS.json").read_text())["baryon"]["192"]
    }
    archived = np.asarray(json.loads(
        (IMPL / "results/frontiers_revision_2026_10/mle_pilot_a6_signed_jac/FIVE_FAMILY_GRADIENT_VECTORS.json").read_text()
    )["vectors"]["ptq-screen"]["native_dF"], float)
    return galaxies, recnos, weights, archived


def finite_range(values):
    array = np.asarray(values, float)
    finite = array[np.isfinite(array)]
    return {
        "min": None if finite.size == 0 else float(np.min(finite)),
        "max": None if finite.size == 0 else float(np.max(finite)),
        "n_nonfinite": int(np.size(array) - finite.size),
        "n": int(array.size),
    }


def catalog_screen_scales(galaxy, upsilon, epsilon, q):
    radius = np.asarray(galaxy.r_kpc, float)
    baryon = vbar_squared_kms2(upsilon, galaxy.v_disk, galaxy.v_bulge, galaxy.v_gas)
    a0 = max(float(epsilon) * C_LIGHT * H0_SI, 1e-30)
    r_m = radius * KPC
    g_n = (np.asarray(baryon, float) * (KM ** 2)) / np.maximum(r_m, 1e-12)
    y_raw = g_n / a0
    y = np.maximum(y_raw, 1e-12)
    with np.errstate(over="ignore", invalid="ignore"):
        power = np.power(y, -float(q))
        exp_form = np.exp((-float(q)) * np.log(y))
    nu_power = 0.5 + np.sqrt(0.25 + np.where(np.isfinite(power), power, np.nan))
    speed = model_v_ptq_screen(
        upsilon, epsilon, q, radius, galaxy.v_disk, galaxy.v_bulge, galaxy.v_gas, H0_si=H0_SI,
    )
    return {
        "a0_si": float(epsilon * C_LIGHT * H0_SI),
        "a0_floor_active": bool(epsilon * C_LIGHT * H0_SI <= 1e-30),
        "y_raw": finite_range(y_raw),
        "y_floor_count": int(np.sum(y_raw < 1e-12)),
        "screened_power": finite_range(power),
        "screened_exp_form": finite_range(exp_form),
        "power_minus_exp_max_abs": None if not (np.isfinite(power).any() and np.isfinite(exp_form).any()) else float(np.nanmax(np.abs(power - exp_form))),
        "nu": finite_range(nu_power),
        "speed_kms": finite_range(speed),
        "signed_baryon_kms2": finite_range(baryon),
    }


def summation_probe(galaxy, pars, gi):
    gg = as_g(galaxy)
    pp = dict(pars, model="ptq-screen", _H0=H0_SI, _c0=10.0, _c_slope=-0.1)
    official = integrate(gg, pp, gi, "segmented_quantile", 192, floors=("B0",))
    ds, wd, incs, wi, _, _ = nuisance_nodes(gg, "segmented_quantile", 192)
    vd = velocity(gg, pp, gi, ds[:, None])
    q = float(math.exp(pars["lq"]))
    radius, disk, bulge, gas = (
        gg.r * (ds / gg.D)[:, None],
        gg.disk * np.sqrt(ds / gg.D)[:, None],
        gg.bulge * np.sqrt(ds / gg.D)[:, None],
        gg.gas * np.sqrt(ds / gg.D)[:, None],
    )
    # Quadrature-node screen arguments. Diagnostic only; the stored likelihood is unchanged.
    upsilon = float(pars["U"][gi])
    a0 = max(float(pars["eps"]) * C_LIGHT * H0_SI, 1e-30)
    vn2 = vbar_squared_kms2(upsilon, disk, bulge, gas)
    y_raw = (vn2 * (KM ** 2)) / np.maximum(radius * KPC, 1e-12) / a0
    y = np.maximum(y_raw, 1e-12)
    with np.errstate(over="ignore", invalid="ignore"):
        power = np.power(y, -q)
        exp_form = np.exp((-q) * np.log(y))
        power_long = np.power(y.astype(np.longdouble), np.longdouble(-q))
    nu_power = nu_q(y, q)
    nu_exp = 0.5 + np.sqrt(0.25 + exp_form)
    nu_long = 0.5 + np.sqrt(np.longdouble(0.25) + power_long)
    b = np.sin(incs) / np.sin(gg.inc)
    var = np.broadcast_to(gg.e[None, :] ** 2 + pp["sigma"] ** 2, (len(incs), len(gg.y)))
    inv = 1.0 / var
    yi = gg.y[None, :] * inv
    y2 = (gg.y[None, :] ** 2 * inv).sum(axis=1)
    norm = np.log(2.0 * np.pi * var).sum(axis=1)
    scores = []
    blocked = -math.inf
    residual_min = math.inf
    residual_max = -math.inf
    for start in range(0, len(ds), 128):
        end = min(start + 128, len(ds))
        v = vd[start:end]
        cross = v @ yi.T
        vv = (v * v) @ inv.T
        quad = y2[None, :] - 2.0 * b[None, :] * cross + (b[None, :] ** 2) * vv
        residual_min = min(residual_min, float(np.min(quad)))
        residual_max = max(residual_max, float(np.max(quad)))
        ll = -0.5 * (quad + norm[None, :])
        score = ll + np.log(wd[start:end, None]) + np.log(wi)[None, :]
        scores.append(score.ravel())
        blocked = float(np.logaddexp(blocked, float(logsumexp(score))))
    flat_scores = np.concatenate(scores)
    flat = float(logsumexp(flat_scores))
    center = float(np.max(flat_scores))
    compensated = center + math.log(math.fsum(np.exp(flat_scores - center)))
    var1 = gg.e ** 2 + pp["sigma"] ** 2
    inv1 = 1.0 / var1
    norm1 = float(np.sum(np.log(2.0 * math.pi * var1)))
    y2_scalar = float(np.sum((gg.y ** 2) * inv1))
    a6_scores = []
    for start in range(0, len(ds), 128):
        end = min(start + 128, len(ds))
        v = vd[start:end]
        cross = v @ (gg.y * inv1)
        vv = (v * v) @ inv1
        quad = y2_scalar - 2.0 * b[None, :] * cross[:, None] + (b ** 2)[None, :] * vv[:, None]
        ll = -0.5 * (quad + norm1)
        a6_scores.append((ll + np.log(wd[start:end])[:, None] + np.log(wi)[None, :]).ravel())
    a6_flat = float(logsumexp(np.concatenate(a6_scores)))
    both = np.isfinite(power) & np.isfinite(exp_form)
    return {
        "distance_nodes": int(len(ds)),
        "inclination_nodes": int(len(incs)),
        "weight_sum": float(wd.sum() * wi.sum()),
        "official_integrate_logL": float(official["logL"]["B0"]),
        "official_max_quadratic_roundoff_bound": float(official["max_quadratic_roundoff_bound"]),
        "blocked_logaddexp": blocked,
        "flat_logsumexp": flat,
        "compensated_fsum": compensated,
        "a6_style_flat_logsumexp": a6_flat,
        "blocked_minus_flat": blocked - flat,
        "compensated_minus_flat": compensated - flat,
        "a6_style_minus_blocked": a6_flat - blocked,
        "score_min": float(np.min(flat_scores)),
        "score_max": float(np.max(flat_scores)),
        "quadratic_residual_min": residual_min,
        "quadratic_residual_max": residual_max,
        "quadrature_speed": finite_range(vd),
        "y_raw": finite_range(y_raw),
        "y_floor_fraction": float(np.mean(y_raw < 1e-12)),
        "screened_power": finite_range(power),
        "screened_exp_form": finite_range(exp_form),
        "power_minus_exp_max_abs": None if not np.any(both) else float(np.max(np.abs(power[both] - exp_form[both]))),
        "nu_power_minus_exp_max_abs": float(np.nanmax(np.abs(nu_power - nu_exp))),
        "nu_power_minus_longdouble_max_abs": float(np.max(np.abs(nu_power.astype(np.longdouble) - nu_long))),
        "quadrature_speed_finite": bool(np.isfinite(vd).all()),
    }


def main():
    global MARGINAL_CALLS, MINIMIZE_CALLS
    verify_sources()
    if scipy.__version__ != "1.17.1":
        raise SystemExit("STOP scipy " + scipy.__version__)
    tests = regression_tests()
    (NEW / "logs").mkdir(parents=True, exist_ok=True)
    atomic_json(NEW / "logs/mock_regression.json", tests)
    print("REGRESSION", tests["pass"], flush=True)
    if not tests["pass"]:
        raise SystemExit("ENGINEERING_REGRESSION_FAIL")
    if os.environ.get("D2_REGRESSION_ONLY") == "1":
        print("REGRESSION_ONLY", flush=True)
        return
    galaxies, recnos, weights, archived_gradient = load_sample()
    theta0 = np.load(IMPL / "results/frontiers_revision_2026_10/mle_pilot_a/execution_v1/ptq_screen_last_accepted_theta.npy")
    if theta_hash(theta0) != START_SHA or theta0.shape != (94,) or theta_hash(theta0) in (LINEAR_START_SHA, LINEAR_TRIAL_SHA):
        raise SystemExit("STOP start vector")
    bounds = native_bounds()
    array_bounds = np.asarray(bounds, float)
    if np.any(theta0 < array_bounds[:, 0]) or np.any(theta0 > array_bounds[:, 1]) or not np.isfinite(theta0).all():
        raise SystemExit("STOP bounds")
    if archived_gradient.shape != (94,):
        raise SystemExit("STOP archived gradient")
    budget = {"a5": 0, "a6": 0, "eq": 0.0, "fun": 0, "t0": time.perf_counter()}
    ledger = NEW / "logs/galaxy_rows.jsonl"

    def charge(name):
        weight = float(weights[name])
        if budget["a5"] + budget["a6"] + 2 > 1200 or budget["eq"] + 2.0 * weight > 10.0:
            raise DiagnosticStop("budget")
        if time.perf_counter() - budget["t0"] > 35 * 60:
            raise DiagnosticStop("wall")
        budget["a5"] += 1
        budget["a6"] += 1
        budget["eq"] += 2.0 * weight

    def one_galaxy(galaxy, pars, gi, digest, phase):
        global MARGINAL_CALLS
        try:
            charge(galaxy.name)
            MARGINAL_CALLS += 1
            row = galaxy_jacobian_row(galaxy, pars, gi, "ptq-screen")
            a5 = float(row["official_logL"])
            a6 = float(row["rederived_logL"])
            absolute = abs(a5 - a6)
            ulp = positive_ulp(a5)
            scaled = None if ulp is None else absolute / ulp
            if scaled is not None and not math.isfinite(scaled):
                scaled = None
            membership = recnos[galaxy.name]
            q = float(math.exp(pars["lq"]))
            record = {
                "status": "COMPLETE",
                "complete": True,
                "phase": phase,
                "galaxy": galaxy.name,
                "gi": gi,
                "n_retained": int(len(galaxy.v_obs)),
                "source_recnos": membership,
                "recno_sha256": sha256_bytes(np.asarray(membership, dtype="<i8").tobytes()),
                "recno_min": int(min(membership)),
                "recno_max": int(max(membership)),
                "theta_sha256": digest,
                "logL_A5": a5,
                "logL_A6": a6,
                "signed_difference_A5_minus_A6": a5 - a6,
                "absolute_difference": absolute,
                "relative_difference": relative_difference(absolute, a5, a6),
                "ulp": ulp,
                "ulp_scaled_difference": scaled,
                "finite_A5": bool(math.isfinite(a5)),
                "finite_A6": bool(math.isfinite(a6)),
                "finite_gradient": all(math.isfinite(float(row["dlogL"][key])) for key in ("epsilon", "ln_q", "upsilon", "ln_sigma_sys") if row["dlogL"].get(key) is not None),
                "flags": list(row["flags"]),
                "domain": row["domain"],
                "scales": catalog_screen_scales(galaxy, float(pars["U"][gi]), float(pars["eps"]), q),
            }
            append_jsonl(ledger, record)
            return row, record
        except Exception as exc:
            append_jsonl(ledger, {
                "status": "INCOMPLETE",
                "complete": False,
                "phase": phase,
                "galaxy": galaxy.name,
                "theta_sha256": digest,
                "error": type(exc).__name__ + ": " + str(exc),
            })
            raise

    def evaluate(theta, digest, phase):
        pars = unpack(theta)
        rows = []
        records = []
        for gi, galaxy in enumerate(galaxies):
            row, record = one_galaxy(galaxy, pars, gi, digest, phase)
            rows.append(row)
            records.append(record)
        gradient = np.full(94, np.nan)
        if any(row["dlogL"].get(key) is None for row in rows for key in ("epsilon", "ln_q", "upsilon", "ln_sigma_sys")):
            return {"F": None, "gradient": gradient, "records": records, "pars": pars}
        gradient[0] = -sum(float(row["dlogL"]["epsilon"]) for row in rows)
        gradient[1] = -sum(float(row["dlogL"]["ln_q"]) for row in rows)
        for gi, row in enumerate(rows):
            gradient[2 + gi] = -float(row["dlogL"]["upsilon"])
        gradient[93] = -sum(float(row["dlogL"]["ln_sigma_sys"]) for row in rows)
        value = -sum(float(row["official_logL"]) for row in rows)
        return {"F": float(value), "gradient": gradient, "records": records, "pars": pars}

    digest0 = theta_hash(theta0)
    start = evaluate(theta0, digest0, "INITIAL")
    start_gap = max(record["absolute_difference"] for record in start["records"])
    gradient_gap = float(np.max(np.abs(start["gradient"] - archived_gradient)))
    start_kkt = projected_norm(theta0, start["gradient"])
    preflight_ok = (
        start["F"] is not None
        and abs(start["F"] - START_F) <= 1e-8
        and start_gap <= IDENTITY_LIMIT
        and gradient_gap <= 1e-8
        and abs(start_kkt["projected_gradient_norm"] - START_PROJECTED) <= 1e-6
        and not start_kkt["active_bounds"]["outside"]
        and np.isfinite(start["gradient"]).all()
        and all(record["domain"] == "INTERIOR_ANALYTICALLY_DIFFERENTIABLE" for record in start["records"])
        and all(record["finite_A5"] and record["finite_A6"] for record in start["records"])
    )
    print(
        f"PREFLIGHT F {start['F']} gap {start_gap:.3e} proj {start_kkt['projected_gradient_norm']:.14f} ok {preflight_ok}",
        flush=True,
    )
    if not preflight_ok:
        atomic_json(NEW / "logs/run_state.json", {"preflight_ok": False, "gap": start_gap, "gradient_gap": gradient_gap})
        raise SystemExit("STOP preflight")

    cache = {"start": start, "trial": None, "callback": 0}
    store = InitialStore(NEW / "state/initial_reference")

    def fun(theta):
        theta = np.asarray(theta, float).copy()
        digest = theta_hash(theta)
        budget["fun"] += 1
        if budget["fun"] > 2:
            raise DiagnosticStop("second line-search trial is not authorized")
        if budget["fun"] == 1:
            if digest != START_SHA:
                raise DiagnosticStop("first optimizer evaluation is not the archived start")
            print("INITIAL reused preflight", flush=True)
            return cache["start"]["F"], cache["start"]["gradient"].copy()
        atomic_npy(NEW / "TRIAL_THETA.npy", theta)
        displacement = theta - theta0
        census = bound_census(theta)
        q = float(math.exp(theta[1]))
        atomic_json(NEW / "TRIAL_CAPTURE_MANIFEST.json", {
            "status": "UNCOMMITTED_DIAGNOSTIC",
            "accepted_iteration": False,
            "request_index": 2,
            "theta_sha256": digest,
            "theta_file": "TRIAL_THETA.npy",
            "model": "ptq-screen",
            "native_dimension": 94,
            "scientific_target": SCIENTIFIC_TARGET,
            "epsilon": float(theta[0]),
            "ln_q": float(theta[1]),
            "q": q,
            "ln_sigma": float(theta[93]),
            "step_l2": float(np.linalg.norm(displacement)),
            "step_inf": float(np.max(np.abs(displacement))),
            "delta_epsilon": float(displacement[0]),
            "delta_ln_q": float(displacement[1]),
            "delta_ln_sigma": float(displacement[93]),
            "bound_census": census,
            "note": "Not an accepted optimizer state and not a restart vector.",
        })
        result = evaluate(theta, digest, "TRIAL")
        cache["trial"] = result
        cache["trial_sha"] = digest
        print(f"TRIAL captured {digest}", flush=True)
        raise DiagnosticStop("first trial captured; accepted callback is not authorized")

    def callback(_xk):
        cache["callback"] += 1
        raise DiagnosticStop("accepted callback is not authorized")

    invocation = NEW / "state/OPTIMIZER_INVOCATION.json"
    if invocation.exists():
        raise SystemExit("STOP optimizer already invoked")
    atomic_json(invocation, {
        "model": "ptq-screen",
        "started": True,
        "scientific_invocations": 1,
        "start_sha256": START_SHA,
        "accepted_iterations_authorized": 0,
    })
    failure = None
    try:
        MINIMIZE_CALLS += 1
        minimize(fun, theta0, method="L-BFGS-B", jac=True, bounds=bounds, callback=callback, options=dict(OPTIONS))
    except DiagnosticStop as exc:
        failure = {"type": "DiagnosticStop", "message": str(exc)}
    print("STOP", failure, cache["callback"], flush=True)
    if cache["trial"] is None or cache["callback"] != 0:
        atomic_json(NEW / "logs/run_state.json", {"failure": failure, "callback": cache["callback"], "budget": budget})
        raise SystemExit("INCOMPLETE capture")
    store.commit_initial(theta0, cache["start"]["F"])
    recovered, manifest = store.recover()
    if theta_hash(recovered) != START_SHA or manifest["iteration_index"] != 0:
        raise SystemExit("STOP accepted state changed")
    if theta_hash(np.load(NEW / "TRIAL_THETA.npy")) != cache["trial_sha"]:
        raise SystemExit("STOP trial file mismatch")
    records = cache["trial"]["records"]
    if len(records) != 91 or any(not record["complete"] for record in records):
        raise SystemExit("INCOMPLETE galaxy rows")
    ranked = sorted(records, key=lambda record: record["absolute_difference"], reverse=True)
    pars = unpack(np.load(NEW / "TRIAL_THETA.npy"))
    probes = []
    for record in ranked[:5]:
        galaxy = galaxies[record["gi"]]
        charge(galaxy.name)
        MARGINAL_CALLS += 1
        probe = summation_probe(galaxy, pars, record["gi"])
        probe["galaxy"] = record["galaxy"]
        probe["stored_A5"] = record["logL_A5"]
        probe["stored_A6"] = record["logL_A6"]
        probe["replica_minus_A5"] = probe["blocked_logaddexp"] - record["logL_A5"]
        probe["a6_style_minus_A6"] = probe["a6_style_flat_logsumexp"] - record["logL_A6"]
        probe["official_integrate_minus_A5"] = probe["official_integrate_logL"] - record["logL_A5"]
        probes.append(probe)
        print(
            f"PROBE {record['galaxy']} abs {record['absolute_difference']:.6e} "
            f"a6-blocked {probe['a6_style_minus_blocked']:.6e} yfloor {probe['y_floor_fraction']}",
            flush=True,
        )
    signed = [record["signed_difference_A5_minus_A6"] for record in records]
    payload = {
        "failure": failure,
        "callback_count": cache["callback"],
        "scipy_version": scipy.__version__,
        "options": OPTIONS,
        "factr_from_ftol": OPTIONS["ftol"] / float(np.finfo(float).eps),
        "start_F": cache["start"]["F"],
        "start_gap": start_gap,
        "start_projected": start_kkt["projected_gradient_norm"],
        "start_full": start_kkt["full_gradient_norm"],
        "start_active_bounds": start_kkt["active_bounds"],
        "gradient_gap": gradient_gap,
        "start_parameters": {
            "epsilon": float(theta0[0]),
            "ln_q": float(theta0[1]),
            "q": float(math.exp(theta0[1])),
            "ln_sigma": float(theta0[93]),
        },
        "trial_sha256": cache["trial_sha"],
        "trial_F_A5": cache["trial"]["F"],
        "trial_F_A6": -sum(record["logL_A6"] for record in records),
        "signed_sum": float(np.sum(signed)),
        "accepted_iteration_index": manifest["iteration_index"],
        "records": records,
        "ranked_names": [record["galaxy"] for record in ranked[:5]],
        "probes": probes,
        "historical_gap_ratio_screen_over_linear": HISTORICAL_SCREEN_GAP / HISTORICAL_LINEAR_GAP,
        "budget": {
            "a5_marginals": budget["a5"],
            "a6_marginals": budget["a6"],
            "pair_equivalents": budget["eq"],
            "optimizer_fun_calls": budget["fun"],
            "minimize_invocations": MINIMIZE_CALLS,
            "wall_seconds": time.perf_counter() - budget["t0"],
            "scientific_likelihood_calls": MARGINAL_CALLS,
        },
        "manifest": manifest,
    }
    atomic_json(NEW / "logs/run_state.json", payload)
    print("D2_DONE", json.dumps({
        "trial": cache["trial_sha"],
        "max_abs": ranked[0]["absolute_difference"],
        "galaxy": ranked[0]["galaxy"],
        "ulp": ranked[0]["ulp_scaled_difference"],
        "fun": budget["fun"],
        "accepted": manifest["iteration_index"],
        "callback": cache["callback"],
    }), flush=True)


if __name__ == "__main__":
    main()
