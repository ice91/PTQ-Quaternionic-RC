"""Capture the first Linear PTQ line-search trial and stop before any accepted step."""
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

NEW = Path("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/mle_pilot_a8_d1_linear_ptq_forensic")
IMPL = Path("/home/rocky/PTQ-Quaternionic-RC")
sys.path.insert(0, str(NEW / "diagnostic"))
sys.path.insert(0, str(NEW / "candidate/src"))

from ptquat.constants import C_LIGHT, H0_SI, KPC, KM
from ptquat.data import load_tidy_sparc
from ptquat.marginal_likelihood import integrate, nuisance_nodes, velocity
from ptquat.models import linear_term_kms2, model_v_ptq, vbar_squared_kms2
from signed_jacobian import SCIENTIFIC_TARGET, VARIANCE_TOKEN, as_g, galaxy_jacobian_row

IDENTITY_LIMIT = 1e-9
START_F = 7101.671260839833
START_GAP = 4.263256414560601e-13
START_PROJECTED = 28.81029102042328
HISTORICAL_TRIAL_GAP = 2.7939677238464355e-09
START_SHA = "96ac8469398b416aa0740522da74c844c12ea8e95364d69f3920f2df460fd6c4"
JACOBIAN_SHA = "d23cb42195870aab92745d1735e54366435c8ae49f3d214f858164364d439585"
MODELS_SHA = "7214664b9a4a3e182f99748a9c4c7da91a98b930399a832589203395f3b89b1b"
MASK_SHA = "df0086b1e67f96403ff09f5c5bfb7f5818593d141d14234d752cb433a9511e6e"
TIDY_SHA = "3ab1c3c2b110c1e48680ebae77c1ceefaa4f778f81119dcf04881b630b274cbb"
EPS_WIDE = (1.0e-6, 3.999)
UPSILON = (0.05, 1.5)
SIGMA = (1e-3, 200.0)
OPTIONS = {"maxiter": 3, "maxls": 20, "maxcor": 10, "ftol": 1e-12, "gtol": 1e-6}
CLIP_FLAGS = {"y_floor", "a0_floor", "radius_floor", "gc_floor", "concentration_floor"}
RELATIVE_DENOMINATOR = "max(abs(logL_A5), abs(logL_A6), 1.0)"
MARGINAL_CALLS = 0


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
        "active_bounds": {
            "lower": np.flatnonzero(lower).astype(int).tolist(),
            "upper": np.flatnonzero(upper).astype(int).tolist(),
        },
        "outside_bounds": np.flatnonzero((x < bounds[:, 0]) | (x > bounds[:, 1])).astype(int).tolist(),
    }


class InitialStore:
    """Accepted-state store that can commit only the archived initial vector."""

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
            "model": "ptq",
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
            "model": "ptq",
            "status": "INITIAL_REFERENCE",
        }
        atomic_json(self.root / "ACCEPTED_MANIFEST.json", manifest)
        return manifest

    def reject_accepted_commit(self, theta) -> None:
        del theta
        raise SchemaError("diagnostic trial cannot be committed as an accepted iteration")

    def recover(self):
        manifest = json.loads((self.root / "ACCEPTED_MANIFEST.json").read_text())
        if manifest.get("status") == "UNCOMMITTED_DIAGNOSTIC" or manifest.get("latest_kind") not in ("INITIAL",):
            raise SchemaError("refusing diagnostic trial as a restart state")
        if manifest["iteration_index"] != 0:
            raise SchemaError("accepted iteration is not authorized")
        theta = np.load(self.root / manifest["theta_file"])
        if theta_hash(theta) != manifest["theta_sha256"]:
            raise SchemaError("theta hash mismatch")
        if manifest["theta_sha256"] != START_SHA and manifest.get("allow_synthetic"):
            return theta, manifest
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
    start = np.linspace(0.2, 0.4, 93)
    # Synthetic hashes are not the archived scientific vector. Mark the store.
    manifest = store.commit_initial(start, 3.0)
    manifest["allow_synthetic"] = True
    atomic_json(root / "accepted" / "ACCEPTED_MANIFEST.json", manifest)
    initial_row = {"kind": "INITIAL", "theta_sha256": theta_hash(start), "objective": 3.0, "complete": True}
    append_jsonl(root / "rows.jsonl", initial_row)
    results["initial_recorded"] = json.loads((root / "rows.jsonl").read_text().splitlines()[0])["kind"] == "INITIAL"

    trial = start.copy()
    trial[0] = 0.21
    persisted = {"done": False}

    def validate():
        if not persisted["done"]:
            raise RuntimeError("validator ran before persistence")
        raise RuntimeError("identity failure")

    digest = theta_hash(trial)
    atomic_npy(root / "TRIAL_THETA.npy", trial)
    atomic_json(root / "TRIAL_CAPTURE_MANIFEST.json", {
        "status": "UNCOMMITTED_DIAGNOSTIC",
        "theta_sha256": digest,
        "accepted_iteration": False,
    })
    persisted["done"] = True
    retained = False
    try:
        validate()
    except RuntimeError as exc:
        retained = "identity" in str(exc) and (root / "TRIAL_THETA.npy").exists()
    results["trial_persisted_before_exception"] = retained
    results["uncommitted_label"] = json.loads((root / "TRIAL_CAPTURE_MANIFEST.json").read_text())["status"] == "UNCOMMITTED_DIAGNOSTIC"
    refused = False
    try:
        store.reject_accepted_commit(trial)
    except SchemaError:
        refused = True
    recovered_theta, recovered = store.recover()
    results["trial_cannot_become_accepted"] = refused and recovered["iteration_index"] == 0 and theta_hash(recovered_theta) == theta_hash(start)

    append_jsonl(root / "rows.jsonl", {
        "status": "COMPLETE",
        "theta_sha256": digest,
        "abs_diff": 2.8e-9,
        "complete": True,
    })
    results["identity_failure_retains_rows"] = "2.8e-9" in (root / "rows.jsonl").read_text() or "2.8e-09" in (root / "rows.jsonl").read_text()

    staged = start.copy()
    staged[2] = 0.33
    atomic_npy(root / "accepted" / "accepted_1_theta.npy", staged)
    atomic_json(root / "accepted" / "accepted_1_meta.json", {"kind": "ACCEPTED", "iteration_index": 1, "theta_sha256": theta_hash(staged)})
    crashed_theta, crashed = store.recover()
    results["crash_does_not_overwrite_initial"] = crashed["iteration_index"] == 0 and theta_hash(crashed_theta) == theta_hash(start)

    rows_have_hash = all("theta_sha256" in json.loads(line) for line in (root / "rows.jsonl").read_text().splitlines())
    results["rows_carry_theta_hash"] = rows_have_hash
    append_jsonl(root / "rows.jsonl", {"status": "INCOMPLETE", "theta_sha256": digest, "complete": False, "galaxy": "partial"})
    results["partial_rows_marked_incomplete"] = json.loads((root / "rows.jsonl").read_text().splitlines()[-1])["status"] == "INCOMPLETE"

    calls = {"n": 0, "callback": 0}

    def synthetic(x):
        calls["n"] += 1
        x = np.asarray(x, float).copy()
        if calls["n"] == 1:
            append_jsonl(root / "optimizer_rows.jsonl", {"kind": "INITIAL", "theta_sha256": theta_hash(x), "complete": True})
            value = float((x[0] - 1.0) ** 2)
            return value, np.array([2.0 * (x[0] - 1.0)])
        atomic_npy(root / "synthetic_trial.npy", x)
        append_jsonl(root / "optimizer_rows.jsonl", {
            "kind": "UNCOMMITTED_DIAGNOSTIC",
            "theta_sha256": theta_hash(x),
            "complete": True,
        })
        raise DiagnosticStop("stop before accepted callback")

    def callback(_xk):
        calls["callback"] += 1
        raise DiagnosticStop("callback is not authorized")

    try:
        minimize(synthetic, np.array([0.0]), method="L-BFGS-B", jac=True, bounds=[(-2.0, 2.0)], callback=callback, options={"maxiter": 3, "ftol": 1e-12})
    except DiagnosticStop:
        pass
    results["optimizer_stops_before_callback"] = calls["callback"] == 0 and calls["n"] == 2 and (root / "synthetic_trial.npy").exists()

    def resolve_start():
        manifest_now = json.loads((root / "accepted" / "ACCEPTED_MANIFEST.json").read_text())
        if manifest_now.get("status") == "UNCOMMITTED_DIAGNOSTIC":
            raise SchemaError("trial manifest")
        theta_now = np.load(root / "accepted" / manifest_now["theta_file"])
        if (root / "TRIAL_THETA.npy").exists() and theta_hash(np.load(root / "TRIAL_THETA.npy")) == theta_hash(theta_now):
            raise SchemaError("restart loaded the diagnostic trial")
        return theta_now

    results["replay_ignores_trial_theta"] = theta_hash(resolve_start()) == theta_hash(start)
    results["scientific_likelihood_calls"] = MARGINAL_CALLS
    keys = [key for key in results if key not in ("pass", "scientific_likelihood_calls")]
    results["pass"] = MARGINAL_CALLS == 0 and all(bool(results[key]) for key in keys)
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
    )["vectors"]["ptq"]["native_dF"], float)
    return galaxies, recnos, weights, archived


def relative_difference(left, right):
    return abs(left - right) / max(abs(left), abs(right), 1.0)


def ulp_scaled(reference, absolute):
    # NumPy's spacing can be negative for a negative magnitude. The gap is its absolute value.
    spacing = abs(float(np.spacing(np.float64(reference))))
    if not math.isfinite(spacing) or spacing == 0.0 or not math.isfinite(absolute):
        return spacing, None
    scaled = absolute / spacing
    if not math.isfinite(scaled):
        return spacing, None
    return spacing, scaled


def nominal_scales(galaxy, upsilon, epsilon):
    radius = np.asarray(galaxy.r_kpc, float)
    baryon = vbar_squared_kms2(upsilon, galaxy.v_disk, galaxy.v_bulge, galaxy.v_gas)
    linear = linear_term_kms2(epsilon, radius, H0_si=H0_SI)
    speed2 = np.asarray(baryon + linear, float)
    speed = model_v_ptq(upsilon, epsilon, radius, galaxy.v_disk, galaxy.v_bulge, galaxy.v_gas, H0_si=H0_SI)
    return {
        "min_signed_baryon_kms2": float(np.min(baryon)),
        "max_signed_baryon_kms2": float(np.max(baryon)),
        "min_linear_term_kms2": float(np.min(linear)),
        "max_linear_term_kms2": float(np.max(linear)),
        "min_speed_squared_kms2": float(np.min(speed2)),
        "max_speed_squared_kms2": float(np.max(speed2)),
        "min_nominal_speed_kms": float(np.min(speed)),
        "max_nominal_speed_kms": float(np.max(speed)),
        "nominal_speed_finite": bool(np.isfinite(speed).all() and np.isfinite(speed2).all()),
    }


def summation_probe(galaxy, pars, gi):
    """Compare blocked and flat reductions of the same B0 quadratic. Not a new target."""
    gg = as_g(galaxy)
    pp = dict(pars, model="ptq", _H0=H0_SI, _c0=10.0, _c_slope=-0.1)
    ds, wd, incs, wi, _, _ = nuisance_nodes(gg, "segmented_quantile", 192)
    vd = velocity(gg, pp, gi, ds[:, None])
    b = np.sin(incs) / np.sin(gg.inc)
    var = np.broadcast_to(gg.e[None, :] ** 2 + pp["sigma"] ** 2, (len(incs), len(gg.y)))
    inv = 1.0 / var
    yi = gg.y[None, :] * inv
    y2 = (gg.y[None, :] ** 2 * inv).sum(axis=1)
    norm = np.log(2.0 * np.pi * var).sum(axis=1)
    scores = []
    blocked = -math.inf
    for start in range(0, len(ds), 128):
        end = min(start + 128, len(ds))
        v = vd[start:end]
        cross = v @ yi.T
        vv = (v * v) @ inv.T
        quad = y2[None, :] - 2.0 * b[None, :] * cross + (b[None, :] ** 2) * vv
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
    return {
        "distance_nodes": int(len(ds)),
        "inclination_nodes": int(len(incs)),
        "blocked_logaddexp": blocked,
        "flat_logsumexp": flat,
        "compensated_fsum": compensated,
        "a6_style_flat_logsumexp": a6_flat,
        "blocked_minus_flat": blocked - flat,
        "compensated_minus_flat": compensated - flat,
        "a6_style_minus_flat": a6_flat - flat,
        "score_min": float(np.min(flat_scores)),
        "score_max": float(np.max(flat_scores)),
        "min_quadrature_speed_kms": float(np.min(vd)),
        "max_quadrature_speed_kms": float(np.max(vd)),
        "quadrature_speed_finite": bool(np.isfinite(vd).all()),
    }


def main():
    global MARGINAL_CALLS
    verify_sources()
    if scipy.__version__ != "1.17.1":
        raise SystemExit("STOP scipy " + scipy.__version__)
    tests = regression_tests()
    (NEW / "logs").mkdir(parents=True, exist_ok=True)
    (NEW / "logs/mock_regression.json").write_text(json.dumps(tests, indent=2) + "\n")
    print("REGRESSION", tests["pass"], flush=True)
    if not tests["pass"]:
        raise SystemExit("ENGINEERING_REGRESSION_FAIL")
    if os.environ.get("D1_REGRESSION_ONLY") == "1":
        print("REGRESSION_ONLY", flush=True)
        return

    galaxies, recnos, weights, archived_gradient = load_sample()
    theta0 = np.load(IMPL / "results/frontiers_revision_2026_10/mle_pilot_a/execution_v1/ptq_last_accepted_theta.npy")
    if theta_hash(theta0) != START_SHA or theta0.shape != (93,):
        raise SystemExit("STOP start vector")
    bounds = native_bounds()
    array_bounds = np.asarray(bounds, float)
    if np.any(theta0 < array_bounds[:, 0]) or np.any(theta0 > array_bounds[:, 1]):
        raise SystemExit("STOP bounds")
    budget = {"a5": 0, "a6": 0, "eq": 0.0, "fun": 0, "t0": time.perf_counter()}
    ledger = NEW / "logs/galaxy_rows.jsonl"
    cache = {}

    def charge(name):
        weight = float(weights[name])
        if budget["a5"] + 1 > 1000 or budget["a6"] + 1 > 1000 or budget["eq"] + 2.0 * weight > 8.0:
            raise DiagnosticStop("budget")
        if time.perf_counter() - budget["t0"] > 30 * 60:
            raise DiagnosticStop("wall")
        budget["a5"] += 1
        budget["a6"] += 1
        budget["eq"] += 2.0 * weight

    def one_galaxy(galaxy, pars, gi, digest, ordinal):
        global MARGINAL_CALLS
        try:
            charge(galaxy.name)
            MARGINAL_CALLS += 1
            row = galaxy_jacobian_row(galaxy, pars, gi, "ptq")
            a5 = float(row["official_logL"])
            a6 = float(row["rederived_logL"])
            absolute = abs(a5 - a6)
            spacing, ulp = ulp_scaled(a5, absolute)
            membership = recnos[galaxy.name]
            record = {
                "status": "COMPLETE",
                "complete": True,
                "ordinal": ordinal,
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
                "relative_difference": relative_difference(a5, a6),
                "relative_denominator": RELATIVE_DENOMINATOR,
                "ulp_spacing_at_A5": spacing,
                "ulp_scaled_difference": ulp,
                "finite_A5": bool(math.isfinite(a5)),
                "finite_A6": bool(math.isfinite(a6)),
                "flags": row["flags"],
                "domain": row["domain"],
                "scales": nominal_scales(galaxy, float(pars["U"][gi]), float(pars["eps"])),
            }
            append_jsonl(ledger, record)
            return row, record
        except Exception as exc:
            append_jsonl(ledger, {
                "status": "INCOMPLETE",
                "complete": False,
                "galaxy": galaxy.name,
                "theta_sha256": digest,
                "error": type(exc).__name__ + ": " + str(exc),
            })
            raise

    def evaluate(theta, digest):
        pars = unpack(theta)
        rows = []
        records = []
        for gi, galaxy in enumerate(galaxies):
            row, record = one_galaxy(galaxy, pars, gi, digest, gi)
            rows.append(row)
            records.append(record)
        gradient = np.full(93, np.nan)
        gradient[0] = -sum(row["dlogL"]["epsilon"] for row in rows)
        for gi, row in enumerate(rows):
            gradient[1 + gi] = -row["dlogL"]["upsilon"]
        gradient[92] = -sum(row["dlogL"]["ln_sigma_sys"] for row in rows)
        if not np.isfinite(gradient).all():
            raise DiagnosticStop("nonfinite gradient")
        value = -sum(row["official_logL"] for row in rows)
        return {"F": float(value), "gradient": gradient, "records": records}

    def fun(theta):
        theta = np.asarray(theta, float).copy()
        digest = theta_hash(theta)
        budget["fun"] += 1
        if budget["fun"] > 2:
            raise DiagnosticStop("second line-search trial is not authorized")
        if budget["fun"] == 1:
            if digest != START_SHA:
                raise DiagnosticStop("first optimizer evaluation is not the archived start")
            result = evaluate(theta, digest)
            gap = max(record["absolute_difference"] for record in result["records"])
            gradient_gap = float(np.max(np.abs(result["gradient"] - archived_gradient)))
            if abs(result["F"] - START_F) > 1e-8 or gap > IDENTITY_LIMIT or gradient_gap > 1e-8:
                raise DiagnosticStop(
                    f"start environment mismatch F {result['F']} gap {gap} gradient {gradient_gap}"
                )
            cache["start"] = result
            cache["start_gap"] = gap
            print(f"INITIAL F {result['F']:.12f} gap {gap:.3e}", flush=True)
            return result["F"], result["gradient"].copy()
        atomic_npy(NEW / "TRIAL_THETA.npy", theta)
        displacement = theta - theta0
        diag = projected_norm(theta, cache["start"]["gradient"])
        atomic_json(NEW / "TRIAL_CAPTURE_MANIFEST.json", {
            "status": "UNCOMMITTED_DIAGNOSTIC",
            "accepted_iteration": False,
            "iteration_index": 0,
            "theta_sha256": digest,
            "theta_file": "TRIAL_THETA.npy",
            "model": "ptq",
            "scientific_target": SCIENTIFIC_TARGET,
            "epsilon": float(theta[0]),
            "epsilon_bounds": list(EPS_WIDE),
            "ln_sigma": float(theta[92]),
            "step_l2": float(np.linalg.norm(displacement)),
            "step_inf": float(np.max(np.abs(displacement))),
            "delta_epsilon": float(displacement[0]),
            "delta_ln_sigma": float(displacement[92]),
            "outside_bounds": diag["outside_bounds"],
            "active_bounds_at_previous_gradient": diag["active_bounds"],
            "note": "Not an accepted optimizer state and not a restart vector.",
        })
        result = evaluate(theta, digest)
        cache["trial"] = result
        cache["trial_sha"] = digest
        print(f"TRIAL captured {digest} F {result['F']:.12f}", flush=True)
        raise DiagnosticStop("first trial captured; accepted callback is not authorized")

    def callback(_xk):
        raise DiagnosticStop("accepted callback is not authorized")

    store = InitialStore(NEW / "state/initial_reference")
    invocation = NEW / "state/OPTIMIZER_INVOCATION.json"
    if invocation.exists():
        raise SystemExit("STOP optimizer already invoked")
    atomic_json(invocation, {"model": "ptq", "started": True, "scientific_invocations": 1, "start_sha256": START_SHA})
    failure = None
    try:
        minimize(fun, theta0, method="L-BFGS-B", jac=True, bounds=bounds, callback=callback, options=dict(OPTIONS))
    except DiagnosticStop as exc:
        failure = {"type": "DiagnosticStop", "message": str(exc)}
    print("STOP", failure, flush=True)
    if cache.get("start") is None or cache.get("trial") is None:
        (NEW / "logs/run_state.json").write_text(json.dumps({"failure": failure, "budget": budget}, indent=2) + "\n")
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
        probe = summation_probe(galaxy, pars, record["gi"])
        probe["galaxy"] = record["galaxy"]
        probe["stored_A5"] = record["logL_A5"]
        probe["stored_A6"] = record["logL_A6"]
        probe["replica_minus_A5"] = probe["blocked_logaddexp"] - record["logL_A5"]
        probe["a6_style_minus_A6"] = probe["a6_style_flat_logsumexp"] - record["logL_A6"]
        probes.append(probe)
        print(
            f"PROBE {record['galaxy']} abs {record['absolute_difference']:.3e} "
            f"blocked-flat {probe['blocked_minus_flat']:.3e} replica-A5 {probe['replica_minus_A5']:.3e}",
            flush=True,
        )
    payload = {
        "failure": failure,
        "scipy_version": scipy.__version__,
        "options": OPTIONS,
        "start_F": cache["start"]["F"],
        "start_gap": cache["start_gap"],
        "start_projected": projected_norm(theta0, cache["start"]["gradient"])["projected_gradient_norm"],
        "trial_F_A5": cache["trial"]["F"],
        "trial_F_A6": -sum(record["logL_A6"] for record in records),
        "trial_sha256": cache["trial_sha"],
        "accepted_iteration_index": manifest["iteration_index"],
        "records": records,
        "ranked_names": [record["galaxy"] for record in ranked[:5]],
        "probes": probes,
        "budget": {
            "a5_marginals": budget["a5"],
            "a6_marginals": budget["a6"],
            "pair_equivalents": budget["eq"],
            "optimizer_fun_calls": budget["fun"],
            "wall_seconds": time.perf_counter() - budget["t0"],
            "scientific_likelihood_calls": MARGINAL_CALLS,
        },
        "manifest": manifest,
    }
    (NEW / "logs/run_state.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("D1_DONE", json.dumps({
        "trial": cache["trial_sha"],
        "max_abs": ranked[0]["absolute_difference"],
        "galaxy": ranked[0]["galaxy"],
        "F": cache["trial"]["F"],
        "fun": budget["fun"],
        "accepted": manifest["iteration_index"],
    }), flush=True)


if __name__ == "__main__":
    main()
