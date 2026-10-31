"""Four-family signed-B0 science pilot. One L-BFGS-B call per family."""
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

NEW = Path("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/mle_pilot_a10_four_family_science_pilot")
IMPL = Path("/home/rocky/PTQ-Quaternionic-RC")
sys.path.insert(0, str(NEW / "diagnostic"))
sys.path.insert(0, str(NEW / "candidate/src"))

from ptquat.constants import C_LIGHT, H0_SI, KM
from ptquat.data import load_tidy_sparc
from ptquat.marginal_likelihood import integrate, nuisance_nodes, velocity
from ptquat.models import model_v_baryon, model_v_nfw1p, model_v_ptq, model_v_ptq_nu
from signed_jacobian import SCIENTIFIC_TARGET, VARIANCE_TOKEN, as_g, galaxy_jacobian_row

ABSOLUTE = 1e-9
RELATIVE = 1e-12
ULP_FACTOR = 8.0
REPLAY_TOL = 1e-8
C_KMS = C_LIGHT / KM
A0 = (5.0e-11, 2.0e-10)
EPS_WIDE = (1.0e-6, 3.999)
UPSILON = (0.05, 1.5)
SIGMA = (1e-3, 200.0)
LOGM = (9.0, 13.0)
OPTIONS = {"maxiter": 8, "maxls": 20, "maxcor": 10, "ftol": 1e-12, "gtol": 1e-6}
CLIP = {"y_floor", "a0_floor", "radius_floor", "gc_floor", "concentration_floor"}
FORMULA = {"nu_q_formula_mismatch", "velocity_formula_mismatch", "q_power_nonfinite"}
MODELS_SHA = "7214664b9a4a3e182f99748a9c4c7da91a98b930399a832589203395f3b89b1b"
JACOBIAN_SHA = "d23cb42195870aab92745d1735e54366435c8ae49f3d214f858164364d439585"
MASK_SHA = "df0086b1e67f96403ff09f5c5bfb7f5818593d141d14234d752cb433a9511e6e"
TIDY_SHA = "3ab1c3c2b110c1e48680ebae77c1ceefaa4f778f81119dcf04881b630b274cbb"
EXPERIMENT_TARGET = "SIGNED_B0_S_DOMAIN13_91_1769"
ARCHIVED = {
    "baryon": "1e9c848c43e456d6eb233b90ff4796edaf2a30fb7cd9daf274fe40e95f680714",
    "ptq-nu": "7abded50306ec11a32380f1ae85c555faef1d2d9318f39c673b691ae974473b2",
    "ptq": "96ac8469398b416aa0740522da74c844c12ea8e95364d69f3920f2df460fd6c4",
    "nfw1p": "5c4ad3c07e46f3fa3df746c3cb5a1eeb42fc8577fbd5ec316dcd1c12ae617d7e",
}
FAMILIES = [
    {
        "model": "baryon",
        "dim": 92,
        "k_count": 92,
        "archived": IMPL / "results/frontiers_revision_2026_10/mle_pilot_a/execution_v1/baryon_last_accepted_theta.npy",
        "checkpoint_dir": IMPL / "results/frontiers_revision_2026_10/mle_pilot_a8_five_family_pilot/families/baryon/state",
        "historical_replay": IMPL / "results/frontiers_revision_2026_10/mle_pilot_a8_five_family_pilot/families/baryon/family_result.json",
        "provisional": False,
    },
    {
        "model": "ptq-nu",
        "dim": 93,
        "k_count": 93,
        "archived": IMPL / "results/frontiers_revision_2026_10/mle_pilot_a/execution_v1/ptq_nu_last_accepted_theta.npy",
        "checkpoint_dir": IMPL / "results/frontiers_revision_2026_10/mle_pilot_a7_r1_ptqnu_smoke_repair/state/scientific",
        "historical_replay": IMPL / "results/frontiers_revision_2026_10/mle_pilot_a7_r1_ptqnu_smoke_repair/ENDPOINT_REPLAY.json",
        "provisional": False,
    },
    {
        "model": "ptq",
        "dim": 93,
        "k_count": 93,
        "archived": IMPL / "results/frontiers_revision_2026_10/mle_pilot_a/execution_v1/ptq_last_accepted_theta.npy",
        "checkpoint_dir": IMPL / "results/frontiers_revision_2026_10/mle_pilot_a8_d1r_rounding_gate_linear_smoke/state/scientific",
        "historical_replay": IMPL / "results/frontiers_revision_2026_10/mle_pilot_a8_d1r_rounding_gate_linear_smoke/ENDPOINT_REPLAY.json",
        "provisional": True,
    },
    {
        "model": "nfw1p",
        "dim": 183,
        "k_count": 183,
        "archived": IMPL / "results/frontiers_revision_2026_10/mle_pilot_a/execution_v1/nfw1p_initial_theta.npy",
        "checkpoint_dir": IMPL / "results/frontiers_revision_2026_10/mle_pilot_a8_five_family_pilot/families/nfw1p/state",
        "historical_replay": IMPL / "results/frontiers_revision_2026_10/mle_pilot_a8_five_family_pilot/families/nfw1p/family_result.json",
        "provisional": False,
    },
]


class StopRun(RuntimeError):
    pass


class BudgetLimit(StopRun):
    pass


class IdentityStop(StopRun):
    pass


class PhysicalReviewStop(StopRun):
    pass


class StagnationStop(StopRun):
    pass


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def theta_hash(theta) -> str:
    return hashlib.sha256(np.ascontiguousarray(theta, dtype="<f8").tobytes()).hexdigest()


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


def atomic_json(path: Path, payload) -> str:
    data = (json.dumps(clean(payload), indent=2, allow_nan=False) + "\n").encode()
    atomic_bytes(path, data)
    return hashlib.sha256(data).hexdigest()


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


def positive_ulp(logl):
    number = float(logl)
    if not math.isfinite(number):
        return None
    ulp = math.ulp(abs(number))
    return float(ulp) if math.isfinite(ulp) and ulp > 0.0 else None


def classify_pair(left, right, provisional: bool) -> dict:
    if not math.isfinite(left) or not math.isfinite(right):
        return {"class": "FAIL", "absolute": None, "relative": None, "ulp": None}
    absolute = abs(float(left) - float(right))
    relative = absolute / max(1.0, abs(float(left)), abs(float(right)))
    ulp = positive_ulp(left)
    if absolute <= ABSOLUTE:
        return {"class": "STRICT", "absolute": absolute, "relative": relative, "ulp": ulp}
    if not provisional:
        return {"class": "FAIL", "absolute": absolute, "relative": relative, "ulp": ulp}
    if ulp is None or absolute > ULP_FACTOR * ulp or relative > RELATIVE:
        return {"class": "FAIL", "absolute": absolute, "relative": relative, "ulp": ulp}
    return {"class": "ROUNDOFF", "absolute": absolute, "relative": relative, "ulp": ulp, "ulp_scaled": absolute / ulp}


def layout(model: str) -> dict:
    slot = 0
    spec = {"model": model}
    if model in ("ptq", "ptq-nu"):
        spec["i_eps"] = slot
        slot += 1
    spec["sl_U"] = slice(slot, slot + 91)
    slot += 91
    if model == "nfw1p":
        spec["sl_lM"] = slice(slot, slot + 91)
        slot += 91
    spec["i_lnsig"] = slot
    slot += 1
    spec["k"] = slot
    return spec


def native_bounds(model: str):
    spec = layout(model)
    bounds = [None] * spec["k"]
    if "i_eps" in spec:
        if model == "ptq-nu":
            bounds[spec["i_eps"]] = (A0[0] / (C_LIGHT * H0_SI), A0[1] / (C_LIGHT * H0_SI))
        else:
            bounds[spec["i_eps"]] = EPS_WIDE
    for index in range(spec["sl_U"].start, spec["sl_U"].stop):
        bounds[index] = UPSILON
    if "sl_lM" in spec:
        for index in range(spec["sl_lM"].start, spec["sl_lM"].stop):
            bounds[index] = LOGM
    bounds[spec["i_lnsig"]] = (math.log(SIGMA[0]), math.log(SIGMA[1]))
    return bounds


def unpack(theta, model: str) -> dict:
    spec = layout(model)
    theta = np.asarray(theta, float)
    out = {"U": np.asarray(theta[spec["sl_U"]], float), "sigma": float(math.exp(theta[spec["i_lnsig"]]))}
    if "i_eps" in spec:
        out["eps"] = float(theta[spec["i_eps"]])
    if "sl_lM" in spec:
        out["lM"] = np.asarray(theta[spec["sl_lM"]], float)
    return out


def assemble(rows, model: str) -> np.ndarray:
    spec = layout(model)
    gradient = np.full(spec["k"], np.nan)
    gradient[spec["i_lnsig"]] = -sum(float(row["dlogL"]["ln_sigma_sys"]) for row in rows)
    if "i_eps" in spec:
        gradient[spec["i_eps"]] = -sum(float(row["dlogL"]["epsilon"]) for row in rows)
    for gi, row in enumerate(rows):
        gradient[spec["sl_U"].start + gi] = -float(row["dlogL"]["upsilon"])
        if "sl_lM" in spec:
            gradient[spec["sl_lM"].start + gi] = -float(row["dlogL"]["log10_M200"])
    return gradient


def projected(theta, gradient, model: str) -> dict:
    x = np.asarray(theta, float)
    g = np.asarray(gradient, float)
    bounds = np.asarray(native_bounds(model), float)
    lower = x <= bounds[:, 0]
    upper = x >= bounds[:, 1]
    outside = np.flatnonzero((x < bounds[:, 0]) | (x > bounds[:, 1]))
    proj = g.copy()
    proj[lower & (g > 0.0)] = 0.0
    proj[upper & (g < 0.0)] = 0.0
    return {
        "full_gradient_norm": float(np.max(np.abs(g))),
        "projected_gradient_norm": float(np.max(np.abs(proj))),
        "active_bounds": {
            "lower": np.flatnonzero(lower).astype(int).tolist(),
            "upper": np.flatnonzero(upper).astype(int).tolist(),
            "outside": outside.astype(int).tolist(),
        },
        "max_projected_index": int(np.argmax(np.abs(proj))),
        "max_full_index": int(np.argmax(np.abs(g))),
    }


def nominal_speeds(galaxy, pars, gi, model: str) -> np.ndarray:
    if model == "baryon":
        return model_v_baryon(float(pars["U"][gi]), galaxy.r_kpc, galaxy.v_disk, galaxy.v_bulge, galaxy.v_gas)
    if model == "ptq":
        return model_v_ptq(float(pars["U"][gi]), float(pars["eps"]), galaxy.r_kpc, galaxy.v_disk, galaxy.v_bulge, galaxy.v_gas, H0_si=H0_SI)
    if model == "ptq-nu":
        return model_v_ptq_nu(float(pars["U"][gi]), float(pars["eps"]), galaxy.r_kpc, galaxy.v_disk, galaxy.v_bulge, galaxy.v_gas, H0_si=H0_SI)
    return model_v_nfw1p(
        float(pars["U"][gi]), float(pars["lM"][gi]), galaxy.r_kpc, galaxy.v_disk, galaxy.v_bulge, galaxy.v_gas,
        H0_si=H0_SI, c0=10.0, c_slope=-0.1,
    )


def speed_audit(galaxy, pars, gi, model: str) -> dict:
    nominal = np.asarray(nominal_speeds(galaxy, pars, gi, model), float)
    gg = as_g(galaxy)
    pp = dict(pars, model=model, _H0=H0_SI, _c0=10.0, _c_slope=-0.1)
    ds, wd, incs, wi, _, _ = nuisance_nodes(gg, "segmented_quantile", 192)
    vd = np.asarray(velocity(gg, pp, gi, ds[:, None]), float)
    nominal_super = nominal >= C_KMS
    quad_super = vd >= C_KMS
    distance_super = np.any(quad_super, axis=1)
    weight = float(np.sum(wd[distance_super]) * np.sum(wi))
    return {
        "nominal_max_kms": float(np.nanmax(nominal)),
        "quadrature_max_kms": float(np.nanmax(vd)),
        "nominal_superluminal_count": int(np.sum(nominal_super)),
        "quadrature_superluminal_count": int(np.sum(quad_super)),
        "quadrature_superluminal_weight": weight,
        "superluminal": bool(np.any(nominal_super) or np.any(quad_super)),
    }


def verify_sources() -> None:
    frozen = {
        IMPL / "dataset/sparc_tidy.csv": TIDY_SHA,
        IMPL / "results/frontiers_revision_2026_10/mle_pilot_a4_domain_feasibility/S_DOMAIN13_MASK.csv": MASK_SHA,
        NEW / "candidate/src/ptquat/models.py": MODELS_SHA,
        NEW / "diagnostic/signed_jacobian.py": JACOBIAN_SHA,
    }
    for path, digest in frozen.items():
        if sha256_file(path) != digest:
            raise SystemExit("STOP hash " + str(path))
    if scipy.__version__ != "1.17.1":
        raise SystemExit("STOP scipy")
    if VARIANCE_TOKEN != "REV26-P1E-UNIFORM-EXPLICIT-B0-PHYSICAL-v1":
        raise SystemExit("STOP variance token")
    if SCIENTIFIC_TARGET != "REV26-P1E-UNIFORM-EXPLICIT-B0-SIGNED-SPARC-CANDIDATE-v2":
        raise SystemExit("STOP code target token")


def load_sample():
    tidy = pd.read_csv(IMPL / "dataset/sparc_tidy.csv")
    raw = pd.read_csv(IMPL / "dataset/raw/vizier_table2.csv")
    mask = pd.read_csv(IMPL / "results/frontiers_revision_2026_10/mle_pilot_a4_domain_feasibility/S_DOMAIN13_MASK.csv")
    joined = tidy.merge(raw, left_on=["galaxy", "r_kpc"], right_on=["Name", "Rad"], how="left", validate="one_to_one")
    included = mask.included.to_numpy(int) == 1
    if int(included.sum()) != 1769 or not np.array_equal(mask.source_recno.to_numpy(), joined.recno.to_numpy()):
        raise SystemExit("STOP mask")
    out = NEW / "logs/s_domain13_filtered_tidy.csv"
    if not out.exists():
        tidy.loc[included].to_csv(out, index=False)
    loaded = load_tidy_sparc(str(out))
    galaxies = [loaded[name] for name in sorted(loaded)]
    if len(galaxies) != 91 or sum(len(g.v_obs) for g in galaxies) != 1769:
        raise SystemExit("STOP sample")
    weights = {
        row["galaxy"]: float(row["fullsample_equivalent"])
        for row in json.loads((IMPL / "results/frontiers_revision_2026_10/mle_pilot_a/GALAXY_WORK_WEIGHTS.json").read_text())["baryon"]["192"]
    }
    return galaxies, weights


def historical_replay_ok(path: Path, theta_sha: str) -> bool:
    payload = json.loads(path.read_text())
    endpoint = payload.get("endpoint", payload)
    passed = endpoint.get("pass")
    if passed is None:
        passed = bool(endpoint.get("identity_pass")) and bool(endpoint.get("objective_within_tolerance"))
    return bool(passed) and endpoint.get("recovered_theta_sha256") == theta_sha and float(endpoint.get("objective_abs_gap", 1.0)) <= REPLAY_TOL


def load_warm(spec: dict) -> dict:
    root = spec["checkpoint_dir"]
    manifest = json.loads((root / "ACCEPTED_MANIFEST.json").read_text())
    meta = json.loads((root / manifest["meta_file"]).read_text())
    theta = np.load(root / manifest["theta_file"])
    archived = np.load(spec["archived"])
    problems = []
    if theta_hash(theta) != manifest["theta_sha256"] or theta_hash(theta) != meta["theta_sha256"]:
        problems.append("theta hash")
    if sha256_file(root / manifest["meta_file"]) != manifest["meta_sha256"]:
        problems.append("meta hash")
    if manifest["model"] != spec["model"] or meta["kind"] != "ACCEPTED":
        problems.append("kind")
    if manifest.get("mask_sha256") != MASK_SHA or manifest.get("models_sha256") != MODELS_SHA or manifest.get("jacobian_sha256") != JACOBIAN_SHA:
        problems.append("source hash")
    if manifest.get("scientific_target") != SCIENTIFIC_TARGET:
        problems.append("target token")
    if theta.shape != (spec["dim"],) or theta_hash(archived) != ARCHIVED[spec["model"]] or theta_hash(theta) == theta_hash(archived):
        problems.append("lineage")
    bounds = np.asarray(native_bounds(spec["model"]), float)
    if np.any(theta < bounds[:, 0]) or np.any(theta > bounds[:, 1]) or not np.isfinite(theta).all():
        problems.append("bounds")
    if not historical_replay_ok(spec["historical_replay"], theta_hash(theta)):
        problems.append("historical replay")
    return {
        "ok": not problems,
        "problems": problems,
        "theta": theta,
        "theta_sha256": theta_hash(theta),
        "archived_sha256": theta_hash(archived),
        "objective_after": meta["objective_after"],
        "iteration_index": meta["iteration_index"],
        "projected_gradient_norm": meta.get("projected_gradient_norm"),
        "manifest": manifest,
    }


class Budget:
    def __init__(self, weights, global_state):
        self.weights = weights
        self.global_state = global_state
        self.marginals = 0
        self.eq = 0.0
        self.requests = 0
        self.t0 = time.perf_counter()

    def charge(self, name: str, count: int) -> None:
        weight = float(count) * float(self.weights[name])
        self._room(count, weight)
        self.marginals += count
        self.eq += weight
        self.global_state["marginals"] += count
        self.global_state["eq"] += weight

    def _room(self, count: int, weight: float) -> None:
        if self.marginals + count > 4500 or self.eq + weight > 45.0:
            raise BudgetLimit("family budget")
        if self.global_state["marginals"] + count > 18000 or self.global_state["eq"] + weight > 160.0:
            raise BudgetLimit("global budget")
        if time.perf_counter() - self.t0 > 40 * 60 or time.perf_counter() - self.global_state["t0"] > 150 * 60:
            raise BudgetLimit("wall clock")

    def begin_request(self) -> None:
        if self.requests >= 24:
            raise BudgetLimit("request cap")
        self._room(400, 6.0)
        self.requests += 1


def main() -> None:
    verify_sources()
    galaxies, weights = load_sample()
    global_state = {"eq": 0.0, "marginals": 0, "t0": time.perf_counter(), "minimize": 0}
    results = []
    for spec in FAMILIES:
        if global_state["eq"] > 160.0 or time.perf_counter() - global_state["t0"] > 150 * 60 or global_state["minimize"] >= 4:
            results.append({"model": spec["model"], "status": "NOT_RUN_GLOBAL_BUDGET"})
            continue
        results.append(run_family(spec, galaxies, weights, global_state))
        atomic_json(NEW / "logs" / f"{spec['model']}_result.json", results[-1])
        print("FAMILY_DONE", spec["model"], results[-1].get("status"), flush=True)
    atomic_json(NEW / "logs/run_state.json", {"families": results, "global": {
        "eq": global_state["eq"], "marginals": global_state["marginals"],
        "minimize": global_state["minimize"], "wall_seconds": time.perf_counter() - global_state["t0"],
    }})
    print("A10_DONE", json.dumps({row["model"]: row.get("status") for row in results}), flush=True)


def run_family(spec, galaxies, weights, global_state) -> dict:
    model = spec["model"]
    warm = load_warm(spec)
    record = {
        "model": model,
        "provisional_trials": spec["provisional"],
        "archived_sha256": warm["archived_sha256"],
        "warm_sha256": warm["theta_sha256"],
        "historical_objective": warm["objective_after"],
        "historical_iterations": warm["iteration_index"],
        "historical_projected": warm["projected_gradient_norm"],
        "checkpoint_problems": warm["problems"],
        "code_scientific_target": SCIENTIFIC_TARGET,
        "variance_token": VARIANCE_TOKEN,
        "experiment_target": EXPERIMENT_TARGET,
        "k_count": spec["k_count"],
    }
    if not warm["ok"]:
        record["status"] = "BLOCKED_WARM_START"
        return record
    budget = Budget(weights, global_state)
    root = NEW / "state" / model
    theta0 = warm["theta"]
    try:
        start = evaluate(galaxies, theta0, model, False, budget, "PREFLIGHT")
    except BudgetLimit as exc:
        record["status"] = "BUDGET_LIMIT"
        record["failure"] = str(exc)
        return record
    start_kkt = projected(theta0, start["gradient"], model)
    identity = table_class(start["records"], False)
    preflight_ok = (
        identity["class"] == "STRICT"
        and start["finite"]
        and not start_kkt["active_bounds"]["outside"]
        and abs(start["F"] - warm["objective_after"]) <= REPLAY_TOL
        and not start["bad_domain"]
        and not start["physical"]["review_required"]
    )
    record["preflight"] = summarize(start, theta0, model, start_kkt)
    record["preflight_ok"] = preflight_ok
    print(f"PREFLIGHT {model} F {start['F']:.8f} proj {start_kkt['projected_gradient_norm']:.6g} ok {preflight_ok}", flush=True)
    if not preflight_ok:
        record["status"] = "BLOCKED_PREFLIGHT"
        return record
    store = Store(root)
    store.commit(theta0, base_record(model, theta0, start, start_kkt, 0, start["F"], start["F"], 0.0))
    accepted = []
    failure = None
    scipy_result = None
    lock = root / "OPTIMIZER_INVOCATION.json"
    if lock.exists():
        record["status"] = "BLOCKED_EXISTING_INVOCATION"
        return record
    atomic_json(lock, {"model": model, "started": True, "warm_sha256": theta_hash(theta0)})

    def fun(theta):
        theta = np.asarray(theta, float).copy()
        digest = theta_hash(theta)
        budget.begin_request()
        atomic_npy(root / "uncommitted" / f"request_{budget.requests}_theta.npy", theta)
        atomic_json(root / "uncommitted" / f"request_{budget.requests}.json", {
            "status": "UNCOMMITTED_TRIAL", "theta_sha256": digest, "request_index": budget.requests, "canonical": False,
        })
        bounds = np.asarray(native_bounds(model), float)
        if np.any(theta < bounds[:, 0]) or np.any(theta > bounds[:, 1]) or not np.isfinite(theta).all():
            raise IdentityStop("bounds")
        result = evaluate(galaxies, theta, model, spec["provisional"], budget, "TRIAL")
        kind = table_class(result["records"], False)
        if kind["class"] not in ("STRICT", "ROUNDOFF") or not result["finite"] or result["bad_domain"]:
            raise IdentityStop("trial " + kind["class"])
        diag = projected(theta, result["gradient"], model)
        if diag["active_bounds"]["outside"]:
            raise IdentityStop("outside")
        print(f"REQUEST {model} {budget.requests} {kind['class']} F {result['F']:.8f}", flush=True)
        return result["F"], result["gradient"].copy()

    def callback(xk):
        if hasattr(xk, "x"):
            xk = xk.x
        theta = np.asarray(xk, float).copy()
        result = evaluate(galaxies, theta, model, False, budget, "ACCEPTED_CANDIDATE")
        kind = table_class(result["records"], False)
        if kind["class"] != "STRICT" or not result["finite"] or result["bad_domain"]:
            raise IdentityStop("ACCEPTED_IDENTITY_FAIL")
        diag = projected(theta, result["gradient"], model)
        if diag["active_bounds"]["outside"]:
            raise IdentityStop("accepted bounds")
        previous_f = start["F"] if not accepted else accepted[-1]["objective_after"]
        if result["F"] > previous_f + REPLAY_TOL:
            raise IdentityStop("objective increased")
        if result["physical"]["review_required"]:
            atomic_npy(root / "noncanonical" / f"candidate_{len(accepted)}_theta.npy", theta)
            atomic_json(root / "noncanonical" / f"candidate_{len(accepted)}.json", {
                "status": "NONCANONICAL", "theta_sha256": theta_hash(theta), "objective": result["F"], "physical": result["physical"],
            })
            raise PhysicalReviewStop("PHYSICAL_REGIME_REVIEW_REQUIRED")
        delta = result["F"] - previous_f
        row = base_record(model, theta, result, diag, len(accepted) + 1, previous_f, result["F"], delta)
        store.commit(theta, row)
        accepted.append(row)
        print(f"ACCEPTED {model} {row['iteration_index']} F {row['objective_after']:.8f} dF {delta:.8g} proj {diag['projected_gradient_norm']:.6g}", flush=True)
        if len(accepted) >= 2:
            last = accepted[-2:]
            if all(abs(item["delta_f"]) < 1e-4 for item in last) and diag["projected_gradient_norm"] > 1e-3:
                raise StagnationStop("STAGNATED_NONSTATIONARY")
        if len(accepted) > 8:
            raise BudgetLimit("accepted cap")

    try:
        global_state["minimize"] += 1
        scipy_result = minimize(fun, theta0, method="L-BFGS-B", jac=True, bounds=native_bounds(model), callback=callback, options=dict(OPTIONS))
    except StopRun as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
        print("STOP", model, failure, flush=True)
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc), "restarted": False}
        print("STOP", model, failure, flush=True)
    theta_end, meta_end = store.recover()
    replay = None
    refine = None
    try:
        replay_eval = evaluate(galaxies, theta_end, model, False, budget, "ENDPOINT_REPLAY")
        replay_kind = table_class(replay_eval["records"], False)
        replay_kkt = projected(theta_end, replay_eval["gradient"], model)
        replay = {
            "pass": replay_kind["class"] == "STRICT" and abs(replay_eval["F"] - meta_end["objective_after"]) <= REPLAY_TOL and not replay_kkt["active_bounds"]["outside"] and replay_eval["finite"] and not replay_eval["bad_domain"],
            "F": replay_eval["F"],
            "objective_abs_gap": abs(replay_eval["F"] - meta_end["objective_after"]),
            "max_per_galaxy_gap": replay_kind["max_absolute"],
            "projected_gradient_norm": replay_kkt["projected_gradient_norm"],
            "full_gradient_norm": replay_kkt["full_gradient_norm"],
            "physical": replay_eval["physical"],
            "theta_sha256": theta_hash(theta_end),
        }
        refine = quadrature_refine(galaxies, unpack(theta_end, model), model, replay_eval, budget)
    except BudgetLimit as exc:
        if replay is None:
            replay = {"pass": False, "status": "BUDGET_LIMIT", "message": str(exc)}
        else:
            refine = {"status": "BUDGET_LIMIT", "message": str(exc)}
    record.update(classify_family(start, start_kkt, accepted, replay, refine, failure, scipy_result, budget, meta_end))
    return record


def quadrature_refine(galaxies, pars, model, replay_eval, budget) -> dict:
    rows = []
    for gi, galaxy in enumerate(galaxies):
        budget.charge(galaxy.name, 1)
        gg = as_g(galaxy)
        pp = dict(pars, model=model, _H0=H0_SI, _c0=10.0, _c_slope=-0.1)
        refined = integrate(gg, pp, gi, "segmented_quantile", 384, floors=("B0",))
        fine = float(replay_eval["records"][gi]["logL_A5"])
        high = float(refined["logL"]["B0"])
        rows.append({"galaxy": galaxy.name, "fine192": fine, "refined384": high, "absolute": abs(high - fine)})
    fine_f = -sum(row["fine192"] for row in rows)
    refined_f = -sum(row["refined384"] for row in rows)
    return {
        "node_count": 384,
        "method": "segmented_quantile",
        "fine192_F": fine_f,
        "refined_F": refined_f,
        "absolute_difference": abs(refined_f - fine_f),
        "max_per_galaxy": max(row["absolute"] for row in rows),
        "relative": abs(refined_f - fine_f) / max(1.0, abs(fine_f)),
        "replaces_fine192": False,
    }


def table_class(records, strict_unused: bool) -> dict:
    if len(records) != 91:
        return {"class": "INCOMPLETE", "max_absolute": None}
    classes = []
    for row in records:
        if row["bad"]:
            return {"class": "FAIL", "max_absolute": row["absolute"], "galaxy": row["galaxy"]}
        classes.append(row["pair_class"])
        if row["pair_class"] == "FAIL":
            return {"class": "FAIL", "max_absolute": row["absolute"], "galaxy": row["galaxy"]}
    if any(item == "ROUNDOFF" for item in classes):
        return {"class": "ROUNDOFF", "max_absolute": max(row["absolute"] for row in records)}
    return {"class": "STRICT", "max_absolute": max(row["absolute"] for row in records)}


def evaluate(galaxies, theta, model, provisional, budget, phase) -> dict:
    pars = unpack(theta, model)
    records = []
    rows = []
    physical_rows = []
    for gi, galaxy in enumerate(galaxies):
        budget.charge(galaxy.name, 2)
        raw = galaxy_jacobian_row(galaxy, pars, gi, model)
        pair = classify_pair(float(raw["official_logL"]), float(raw["rederived_logL"]), provisional and phase == "TRIAL")
        flags = list(raw["flags"])
        bad = raw["domain"] != "INTERIOR_ANALYTICALLY_DIFFERENTIABLE" or any(flag in CLIP or flag in FORMULA for flag in flags) or raw["variance_token"] != VARIANCE_TOKEN
        gradient_bad = any(raw["dlogL"].get(key) is None or not math.isfinite(float(raw["dlogL"][key])) for key in raw["dlogL"])
        records.append({
            "galaxy": galaxy.name, "logL_A5": float(raw["official_logL"]), "logL_A6": float(raw["rederived_logL"]),
            "absolute": pair["absolute"], "pair_class": pair["class"], "bad": bad or gradient_bad, "domain": raw["domain"],
        })
        rows.append(raw)
        if phase in ("PREFLIGHT", "ACCEPTED_CANDIDATE", "ENDPOINT_REPLAY"):
            audit = speed_audit(galaxy, pars, gi, model)
            audit["galaxy"] = galaxy.name
            physical_rows.append(audit)
    finite = all(not row["bad"] for row in records) and all(math.isfinite(float(row["official_logL"])) for row in rows)
    gradient = assemble(rows, model) if finite else np.full(layout(model)["k"], np.nan)
    value = -sum(float(row["official_logL"]) for row in rows)
    physical = regime(physical_rows)
    return {"records": records, "gradient": gradient, "F": float(value), "finite": bool(finite and np.isfinite(gradient).all() and math.isfinite(value)), "bad_domain": any(row["bad"] for row in records), "physical": physical, "pars": pars}


def regime(rows) -> dict:
    if not rows:
        return {"review_required": False, "max_nominal_kms": None, "max_quadrature_kms": None, "superluminal_galaxies": [], "max_superluminal_weight": 0.0}
    flagged = [row for row in rows if row["superluminal"]]
    return {
        "review_required": bool(flagged),
        "max_nominal_kms": max(row["nominal_max_kms"] for row in rows),
        "max_quadrature_kms": max(row["quadrature_max_kms"] for row in rows),
        "superluminal_galaxies": [row.get("galaxy", "") for row in flagged],
        "max_superluminal_weight": max((row["quadrature_superluminal_weight"] for row in rows), default=0.0),
        "galaxies_audited": len(rows),
    }


def summarize(result, theta, model, diag) -> dict:
    return {
        "F": result["F"],
        "projected_gradient_norm": diag["projected_gradient_norm"],
        "full_gradient_norm": diag["full_gradient_norm"],
        "max_projected_index": diag["max_projected_index"],
        "max_full_index": diag["max_full_index"],
        "active_lower": len(diag["active_bounds"]["lower"]),
        "active_upper": len(diag["active_bounds"]["upper"]),
        "max_abs_gap": max(row["absolute"] for row in result["records"]),
        "physical": result["physical"],
        "theta_sha256": theta_hash(theta),
    }


def base_record(model, theta, result, diag, index, before, after, delta) -> dict:
    kind = "INITIAL" if index == 0 else "ACCEPTED"
    return {
        "kind": kind,
        "iteration_index": index,
        "theta_sha256": theta_hash(theta),
        "objective_before": before,
        "objective_after": after,
        "delta_f": delta,
        "identity_class": "ACCEPTED_STRICT_PASS" if index else "INITIAL",
        "projected_gradient_norm": diag["projected_gradient_norm"],
        "full_gradient_norm": diag["full_gradient_norm"],
        "max_projected_index": diag["max_projected_index"],
        "active_bounds": diag["active_bounds"],
        "physical_regime": "PHYSICAL_REGIME_REVIEW_REQUIRED" if result["physical"]["review_required"] else "NO_FLAG",
        "physical": result["physical"],
        "model": model,
    }


def classify_family(start, start_kkt, accepted, replay, refine, failure, scipy_result, budget, meta_end) -> dict:
    terminal_f = meta_end["objective_after"]
    terminal_proj = meta_end["projected_gradient_norm"]
    decreased = any(row["delta_f"] < 0.0 for row in accepted)
    engineering = "PASS" if decreased else "NO_ACCEPTED_DECREASE"
    identity_ok = bool(replay and replay.get("pass"))
    refine_ok = bool(refine and "absolute_difference" in refine)
    physical_ok = not bool((replay or {}).get("physical", {}).get("review_required"))
    improvement = start["F"] - terminal_f
    precision = "NOT_RUN"
    if refine_ok:
        gap = float(refine["absolute_difference"])
        precision = "PRECISION_UNRESOLVED" if gap >= max(1e-3, 0.5 * abs(improvement)) and abs(improvement) > 0.0 else "RECORDED"
        if abs(improvement) == 0.0:
            precision = "RECORDED"
    approx = bool(terminal_proj <= 1e-4 and identity_ok and refine_ok and physical_ok and precision != "PRECISION_UNRESOLVED")
    gtol = bool(terminal_proj <= 1e-6 and identity_ok and refine_ok and precision != "PRECISION_UNRESOLVED" and physical_ok)
    stagnated = bool(failure and failure["type"] == "StagnationStop")
    if failure and failure["type"] == "BudgetLimit":
        status = "BUDGET_LIMIT"
    elif failure and failure["type"] == "PhysicalReviewStop":
        status = "PHYSICAL_REVIEW_STOP"
    elif stagnated:
        status = "STAGNATED_NONSTATIONARY"
    elif failure and failure["type"] == "IdentityStop":
        status = "IDENTITY_STOP"
    elif failure:
        status = "FAIL"
    elif gtol:
        status = "ORIGINAL_GTOL_REACHED"
    elif approx:
        status = "SINGLE_START_APPROX_STATIONARY"
    elif decreased:
        status = "ENGINEERING_PROGRESS"
    else:
        status = "NO_PROGRESS"
    return {
        "status": status,
        "engineering_progress": engineering if decreased else "NO",
        "single_start_approx_stationary": approx,
        "original_gtol_reached": gtol,
        "stagnated": stagnated,
        "precision": precision,
        "initial_F": start["F"],
        "terminal_F": terminal_f,
        "objective_improvement": improvement,
        "accepted_iterations": len(accepted),
        "accepted": [{key: value for key, value in row.items() if key != "active_bounds"} | {"active_lower": len(row["active_bounds"]["lower"]), "active_upper": len(row["active_bounds"]["upper"])} for row in accepted],
        "initial_projected": start_kkt["projected_gradient_norm"],
        "terminal_projected": terminal_proj,
        "initial_full": start_kkt["full_gradient_norm"],
        "terminal_full": meta_end["full_gradient_norm"],
        "initial_max_index": start_kkt["max_projected_index"],
        "terminal_max_index": meta_end.get("max_projected_index"),
        "replay": replay,
        "quadrature": refine,
        "failure": failure,
        "scipy": None if scipy_result is None else {
            "success": bool(scipy_result.success), "status": int(scipy_result.status),
            "message": str(scipy_result.message), "nit": int(scipy_result.nit), "nfev": int(scipy_result.nfev), "njev": int(scipy_result.njev),
        },
        "budget": {"marginals": budget.marginals, "eq": budget.eq, "requests": budget.requests, "wall_seconds": time.perf_counter() - budget.t0},
        "terminal_theta_sha256": meta_end["theta_sha256"],
    }


class Store:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def commit(self, theta, record: dict) -> None:
        if record["kind"] == "ACCEPTED" and record["identity_class"] != "ACCEPTED_STRICT_PASS":
            raise IdentityStop("commit identity")
        if record["physical_regime"] == "PHYSICAL_REGIME_REVIEW_REQUIRED":
            raise PhysicalReviewStop("commit physical")
        digest = theta_hash(theta)
        if digest != record["theta_sha256"]:
            raise IdentityStop("commit hash")
        index = int(record["iteration_index"])
        atomic_npy(self.root / f"accepted_{index}_theta.npy", theta)
        meta_sha = atomic_json(self.root / f"accepted_{index}_meta.json", record)
        atomic_json(self.root / "ACCEPTED_MANIFEST.json", {
            "iteration_index": index,
            "theta_sha256": digest,
            "meta_sha256": meta_sha,
            "theta_file": f"accepted_{index}_theta.npy",
            "meta_file": f"accepted_{index}_meta.json",
            "objective_after": record["objective_after"],
            "model": record["model"],
            "experiment_target": EXPERIMENT_TARGET,
            "code_scientific_target": SCIENTIFIC_TARGET,
            "variance_token": VARIANCE_TOKEN,
            "mask_sha256": MASK_SHA,
            "models_sha256": MODELS_SHA,
            "jacobian_sha256": JACOBIAN_SHA,
        })

    def recover(self):
        manifest = json.loads((self.root / "ACCEPTED_MANIFEST.json").read_text())
        meta = json.loads((self.root / manifest["meta_file"]).read_text())
        theta = np.load(self.root / manifest["theta_file"])
        if theta_hash(theta) != manifest["theta_sha256"] or sha256_file(self.root / manifest["meta_file"]) != manifest["meta_sha256"]:
            raise IdentityStop("recover hash")
        return theta, meta


if __name__ == "__main__":
    main()
