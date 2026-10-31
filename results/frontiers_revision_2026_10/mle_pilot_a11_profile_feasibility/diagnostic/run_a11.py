"""A11 separability audit and bounded local-profile feasibility.

Selection is written to disk before any new likelihood evaluation.
No full-sample optimizer is called.
"""
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

NEW = Path("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/mle_pilot_a11_profile_feasibility")
A10 = Path("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/mle_pilot_a10_four_family_science_pilot")
IMPL = Path("/home/rocky/PTQ-Quaternionic-RC")
sys.path.insert(0, str(A10 / "diagnostic"))
sys.path.insert(0, str(A10 / "candidate/src"))

from ptquat.constants import C_LIGHT, H0_SI, KM
from ptquat.data import load_tidy_sparc
from ptquat.marginal_likelihood import integrate, nuisance_nodes, velocity
from ptquat.models import model_v_baryon, model_v_nfw1p, model_v_ptq, model_v_ptq_nu
from signed_jacobian import VARIANCE_TOKEN, as_g, galaxy_jacobian_row

MODELS_SHA = "7214664b9a4a3e182f99748a9c4c7da91a98b930399a832589203395f3b89b1b"
JACOBIAN_SHA = "d23cb42195870aab92745d1735e54366435c8ae49f3d214f858164364d439585"
MASK_SHA = "df0086b1e67f96403ff09f5c5bfb7f5818593d141d14234d752cb433a9511e6e"
TIDY_SHA = "3ab1c3c2b110c1e48680ebae77c1ceefaa4f778f81119dcf04881b630b274cbb"
RUN_A10_SHA = "5b318db3b06ab71d69fd7560377f8133f46198ace4cf97633cc7d7507081cabe"
TERMINAL = {
    "baryon": "792135832c515052090f2cdd9eefa9962f6d06fe46b4e2420b9de6542b369158",
    "ptq-nu": "4b90929db7808b914d6e32d88b5b24bb30672c1fd052fee07b1a921b32b8fc54",
    "ptq": "d1537c99a968b2c8c1f6749c2e9ecfce450d6f2d8a6e3e29dd2146e46cff9677",
    "nfw1p": "3df4048e243ec0a83b14adfa2414627a9cadadde7ee9517f01e63f2178be1da3",
}
WARM_BARYON = "93b72fe3630b1d4ee53c086063f961255cad49778ff3f08415c832e4827eccdf"
UPS = (0.05, 1.5)
LOGM = (9.0, 13.0)
SIGMA = (1e-3, 200.0)
A0 = (5.0e-11, 2.0e-10)
EPS_WIDE = (1.0e-6, 3.999)
EPS_COS = 1.4919136877222163
C_KMS = C_LIGHT / KM
DOMAIN_EXCLUSION_COUNTS = {
    "UGC01281": 7, "UGC07577": 2, "UGC08286": 1, "UGC08837": 2, "UGCA444": 1,
}
BUDGET_CAP = 1200
WALL_CAP = 20 * 60


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def theta_hash(theta) -> str:
    return hashlib.sha256(np.ascontiguousarray(theta, dtype="<f8").tobytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
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


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(clean(payload), indent=2, allow_nan=False) + "\n").encode()
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def verify_sources() -> None:
    frozen = {
        IMPL / "dataset/sparc_tidy.csv": TIDY_SHA,
        IMPL / "results/frontiers_revision_2026_10/mle_pilot_a4_domain_feasibility/S_DOMAIN13_MASK.csv": MASK_SHA,
        A10 / "candidate/src/ptquat/models.py": MODELS_SHA,
        A10 / "diagnostic/signed_jacobian.py": JACOBIAN_SHA,
        A10 / "diagnostic/run_a10.py": RUN_A10_SHA,
    }
    for path, digest in frozen.items():
        if sha256_file(path) != digest:
            raise SystemExit("STOP hash " + str(path))
    if VARIANCE_TOKEN != "REV26-P1E-UNIFORM-EXPLICIT-B0-PHYSICAL-v1":
        raise SystemExit("STOP variance token")


def load_galaxies():
    path = A10 / "logs/s_domain13_filtered_tidy.csv"
    loaded = load_tidy_sparc(str(path))
    galaxies = [loaded[name] for name in sorted(loaded)]
    if len(galaxies) != 91 or sum(len(g.v_obs) for g in galaxies) != 1769:
        raise SystemExit("STOP sample")
    return galaxies


def load_terminal(model: str):
    manifest = json.loads((A10 / "state" / model / "ACCEPTED_MANIFEST.json").read_text())
    theta = np.load(A10 / "state" / model / manifest["theta_file"])
    if theta_hash(theta) != TERMINAL[model] or theta_hash(theta) != manifest["theta_sha256"]:
        raise SystemExit("STOP terminal " + model)
    return theta


def upsilon_block(model: str):
    if model == "baryon":
        return slice(0, 91), None, 91
    if model in ("ptq", "ptq-nu"):
        return slice(1, 92), 0, 92
    return slice(0, 91), None, 182


def unpack(theta, model: str) -> dict:
    theta = np.asarray(theta, float)
    if model == "baryon":
        return {"U": theta[:91].copy(), "sigma": float(math.exp(theta[91]))}
    if model in ("ptq", "ptq-nu"):
        return {"U": theta[1:92].copy(), "eps": float(theta[0]), "sigma": float(math.exp(theta[92]))}
    return {"U": theta[:91].copy(), "lM": theta[91:182].copy(), "sigma": float(math.exp(theta[182]))}


def select(galaxies) -> dict:
    names = [g.name for g in galaxies]
    baryon = load_terminal("baryon")
    ptqnu = load_terminal("ptq-nu")
    linear = load_terminal("ptq")
    chosen = []

    def add(name, role, detail):
        if name in [row["galaxy"] for row in chosen]:
            return False
        if name not in names:
            raise SystemExit("STOP missing " + name)
        chosen.append({"galaxy": name, "index": names.index(name), "role": role, "detail": detail})
        return True

    btrace = json.loads((A10 / "logs/baryon_result.json").read_text())
    ntrace = json.loads((A10 / "logs/ptq-nu_result.json").read_text())
    b_index = int(btrace["terminal_max_index"])
    n_index = int(ntrace["terminal_max_index"])
    if not 0 <= b_index < 91:
        raise SystemExit("STOP baryon projected index is not a local Upsilon")
    if not 1 <= n_index < 92:
        raise SystemExit("STOP ptq-nu projected index is not a local Upsilon")
    add(names[b_index], "LARGE_LOCAL_GRADIENT_BARYON", {
        "a10_terminal_max_projected_index": b_index,
        "upsilon": float(baryon[b_index]),
        "projected_norm_at_vector": btrace["terminal_projected"],
    })
    upper = [names[i] for i in range(91) if baryon[i] >= UPS[1]]
    if not upper:
        raise SystemExit("STOP no baryon upper bound")
    add(upper[0], "UPPER_BOUND_BARYON", {"upsilon": float(baryon[names.index(upper[0])]), "n_upper_baryon": len(upper), "rule": "first sorted name with Upsilon>=1.5"})
    lower = [names[i] for i in range(91) if ptqnu[1 + i] <= UPS[0]]
    if not lower:
        raise SystemExit("STOP no ptq-nu lower bound")
    add(lower[0], "LOWER_BOUND_PTQNU", {"upsilon": float(ptqnu[1 + names.index(lower[0])]), "n_lower_ptqnu": len(lower), "rule": "first sorted name with ptq-nu Upsilon<=0.05"})
    domain_name = sorted(DOMAIN_EXCLUSION_COUNTS, key=lambda key: (-DOMAIN_EXCLUSION_COUNTS[key], key))[0]
    add(domain_name, "DOMAIN13_TOUCHED", {"excluded_radii": DOMAIN_EXCLUSION_COUNTS[domain_name], "source": "A9 exclusion register counts"})
    interior = [(names[i], float(linear[1 + i])) for i in range(91) if UPS[0] < linear[1 + i] < UPS[1]]
    median = float(np.median([value for _, value in interior]))
    interior.sort(key=lambda item: (abs(item[1] - median), item[0]))
    for name, value in interior:
        if add(name, "INTERIOR_LINEAR", {"upsilon": value, "interior_median": median, "rule": "linear terminal Upsilon closest to the interior median"}):
            break
    add(names[n_index - 1], "LARGE_LOCAL_GRADIENT_PTQNU", {
        "a10_terminal_max_projected_index": n_index,
        "galaxy_offset": n_index - 1,
        "upsilon": float(ptqnu[n_index]),
        "projected_norm_at_vector": ntrace["terminal_projected"],
    })
    if not 5 <= len(chosen) <= 6:
        raise SystemExit("STOP selection count " + str(len(chosen)))
    return {
        "rule_published_before_likelihood": True,
        "rule": [
            "Use only A10 terminal native vectors. Do not call the likelihood while selecting.",
            "Include the baryon terminal maximum projected-gradient coordinate when it is a galaxy Upsilon.",
            "Include the first sorted-name baryon galaxy sitting at Upsilon>=1.5.",
            "Include the first sorted-name ptq-nu galaxy sitting at Upsilon<=0.05. Baryon has no lower-bound contact.",
            "Include the S_DOMAIN13 galaxy with the most excluded radii in the frozen A9 register.",
            "Include the linear-PTQ interior Upsilon closest to the median of interior Upsilons.",
            "Include the ptq-nu terminal maximum projected-gradient galaxy when that coordinate is a Upsilon.",
            "Skip a later rule when the galaxy is already chosen. Do not replace the set after seeing new likelihood values.",
        ],
        "galaxies": chosen,
        "nfw_2d_conditional": {
            "max_galaxies": 2,
            "members": [chosen[0]["galaxy"], chosen[2]["galaxy"]],
            "execute_only_if": "1D and baryon paired quadrature finished and at least 250 marginals plus 3 minutes remain",
        },
    }


class Budget:
    def __init__(self):
        self.n = 0
        self.t0 = time.perf_counter()
        self.events = []

    def charge(self, count: int, label: str) -> None:
        if self.n + count > BUDGET_CAP or time.perf_counter() - self.t0 > WALL_CAP:
            raise RuntimeError("BUDGET_LIMIT")
        self.n += count
        self.events.append({"label": label, "count": count, "total": self.n})


def loglik(galaxy, gi, pars, model, order=192) -> float:
    gg = as_g(galaxy)
    pp = dict(pars, model=model, _H0=H0_SI, _c0=10.0, _c_slope=-0.1)
    record = integrate(gg, pp, gi, "segmented_quantile", order, floors=("B0",))
    value = float(record["logL"]["B0"])
    if not math.isfinite(value):
        raise RuntimeError("nonfinite logL")
    return value


def project_1d(x, gradient, lo, hi) -> float:
    if x <= lo and gradient > 0.0:
        return 0.0
    if x >= hi and gradient < 0.0:
        return 0.0
    return float(gradient)


def nominal_max(galaxy, pars, gi, model) -> float:
    if model == "baryon":
        speed = model_v_baryon(float(pars["U"][gi]), galaxy.r_kpc, galaxy.v_disk, galaxy.v_bulge, galaxy.v_gas)
    elif model == "ptq":
        speed = model_v_ptq(float(pars["U"][gi]), float(pars["eps"]), galaxy.r_kpc, galaxy.v_disk, galaxy.v_bulge, galaxy.v_gas, H0_si=H0_SI)
    elif model == "ptq-nu":
        speed = model_v_ptq_nu(float(pars["U"][gi]), float(pars["eps"]), galaxy.r_kpc, galaxy.v_disk, galaxy.v_bulge, galaxy.v_gas, H0_si=H0_SI)
    else:
        speed = model_v_nfw1p(float(pars["U"][gi]), float(pars["lM"][gi]), galaxy.r_kpc, galaxy.v_disk, galaxy.v_bulge, galaxy.v_gas, H0_si=H0_SI, c0=10.0, c_slope=-0.1)
    return float(np.nanmax(speed))


def node_max(galaxy, pars, gi, model) -> float:
    gg = as_g(galaxy)
    pp = dict(pars, model=model, _H0=H0_SI, _c0=10.0, _c_slope=-0.1)
    ds, _, _, _, _, _ = nuisance_nodes(gg, "segmented_quantile", 192)
    vd = np.asarray(velocity(gg, pp, gi, ds[:, None]), float)
    return float(np.nanmax(vd))


def validate_point(galaxy, gi, pars, model, budget, label) -> dict:
    budget.charge(2, label + ":jacobian")
    raw = galaxy_jacobian_row(galaxy, pars, gi, model)
    gap = abs(float(raw["official_logL"]) - float(raw["rederived_logL"]))
    d_u = raw["dlogL"].get("upsilon")
    finite = d_u is not None and math.isfinite(float(d_u)) and math.isfinite(float(raw["official_logL"]))
    return {
        "official_logL": float(raw["official_logL"]),
        "rederived_logL": float(raw["rederived_logL"]),
        "abs_gap": gap,
        "identity_pass": bool(gap <= 1e-9 and finite and raw["domain"] == "INTERIOR_ANALYTICALLY_DIFFERENTIABLE"),
        "domain": raw["domain"],
        "flags": list(raw["flags"]),
        "dlogL_upsilon": None if d_u is None else float(d_u),
        "dF_dUpsilon": None if d_u is None else -float(d_u),
        "nominal_max_kms": nominal_max(galaxy, pars, gi, model),
        "quadrature_node_max_kms": node_max(galaxy, pars, gi, model),
    }


def refine_interval(cache, a, b, limit=12):
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = float(a), float(b)
    if b < a:
        a, b = b, a
    if b - a < 1e-8:
        return {"u": a, "F": cache(a), "evaluations": 0, "width": b - a}
    c = b - golden * (b - a)
    d = a + golden * (b - a)
    fc, fd = cache(c), cache(d)
    used = 2
    while used < limit and b - a > 1e-5:
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - golden * (b - a)
            fc = cache(c)
        else:
            a, c, fc = c, d, fd
            d = a + golden * (b - a)
            fd = cache(d)
        used += 1
    best_u, best_f = min([(c, fc), (d, fd)], key=lambda item: item[1])
    return {"u": best_u, "F": best_f, "evaluations": used, "width": b - a}


def one_dimensional(galaxy, gi, base, model, budget) -> dict:
    origin = float(base["U"][gi])
    cache_values = {}

    def cache(u):
        key = round(float(u), 12)
        if key not in cache_values:
            trial = dict(base)
            trial["U"] = np.array(base["U"], float)
            trial["U"][gi] = float(u)
            budget.charge(1, f"{model}:{galaxy.name}:F")
            cache_values[key] = -loglik(galaxy, gi, trial, model)
        return cache_values[key]

    grid = np.linspace(UPS[0], UPS[1], 21)
    rows = [{"u": float(u), "F": cache(float(u)), "source": "grid"} for u in grid]
    if min(abs(origin - float(u)) for u in grid) > 1e-10:
        rows.append({"u": origin, "F": cache(origin), "source": "a10"})
    rows.sort(key=lambda item: item["u"])
    basins = []
    for i, row in enumerate(rows):
        left = rows[i - 1]["F"] if i else None
        right = rows[i + 1]["F"] if i + 1 < len(rows) else None
        is_min = (left is None or row["F"] <= left) and (right is None or row["F"] <= right)
        if is_min:
            lo = rows[i - 1]["u"] if i else row["u"]
            hi = rows[i + 1]["u"] if i + 1 < len(rows) else row["u"]
            refined = refine_interval(cache, lo, hi)
            basins.append({"grid_u": row["u"], "grid_F": row["F"], "bracket": [lo, hi], "refined": refined})
    best = min(basins, key=lambda item: item["refined"]["F"])
    barrier = False
    if len(basins) >= 2:
        ordered = sorted(basins, key=lambda item: item["grid_u"])
        for left, right in zip(ordered, ordered[1:]):
            between = [row for row in rows if left["grid_u"] < row["u"] < right["grid_u"]]
            crest = max(row["F"] for row in between) if between else max(left["grid_F"], right["grid_F"])
            if crest > max(left["grid_F"], right["grid_F"]) + 1e-6:
                barrier = True
    candidate_u = float(best["refined"]["u"])
    candidate = dict(base)
    candidate["U"] = np.array(base["U"], float)
    candidate["U"][gi] = candidate_u
    checked_a10 = validate_point(galaxy, gi, base, model, budget, model + ":" + galaxy.name + ":a10")
    checked = validate_point(galaxy, gi, candidate, model, budget, model + ":" + galaxy.name + ":candidate")
    f_a10 = -checked_a10["official_logL"]
    f_new = -checked["official_logL"]
    g_a10 = checked_a10["dF_dUpsilon"]
    g_new = checked["dF_dUpsilon"]
    return {
        "model": model,
        "galaxy": galaxy.name,
        "a10_upsilon": origin,
        "candidate_upsilon": candidate_u,
        "F_a10": f_a10,
        "F_candidate_grid": best["refined"]["F"],
        "F_candidate_validated": f_new,
        "delta_F": f_new - f_a10,
        "improved_within_1e-8": bool(f_new <= f_a10 + 1e-8),
        "projected_a10": None if g_a10 is None else project_1d(origin, g_a10, UPS[0], UPS[1]),
        "projected_candidate": None if g_new is None else project_1d(candidate_u, g_new, UPS[0], UPS[1]),
        "gradient_a10": g_a10,
        "gradient_candidate": g_new,
        "bound_a10": "lower" if origin <= UPS[0] else "upper" if origin >= UPS[1] else "interior",
        "bound_candidate": "lower" if candidate_u <= UPS[0] + 1e-8 else "upper" if candidate_u >= UPS[1] - 1e-8 else "interior",
        "n_basins": len(basins),
        "barrier_between_basins": barrier,
        "basins": basins,
        "n_unique_objective": len(cache_values),
        "validation_a10": checked_a10,
        "validation_candidate": checked,
        "kkt_sign_ok": bool(
            g_new is not None and (
                (candidate_u > UPS[0] + 1e-8 and candidate_u < UPS[1] - 1e-8 and abs(project_1d(candidate_u, g_new, UPS[0], UPS[1])) <= 1e-3)
                or (candidate_u <= UPS[0] + 1e-8 and g_new >= -1e-8)
                or (candidate_u >= UPS[1] - 1e-8 and g_new <= 1e-8)
            )
        ),
    }


def nfw_two_dimensional(galaxy, gi, base, budget) -> dict:
    origin_u = float(base["U"][gi])
    origin_m = float(base["lM"][gi])
    cache_values = {}

    def cache(u, logm):
        key = (round(float(u), 10), round(float(logm), 10))
        if key not in cache_values:
            trial = dict(base)
            trial["U"] = np.array(base["U"], float)
            trial["lM"] = np.array(base["lM"], float)
            trial["U"][gi] = float(u)
            trial["lM"][gi] = float(logm)
            budget.charge(1, "nfw1p:" + galaxy.name + ":F")
            cache_values[key] = -loglik(galaxy, gi, trial, "nfw1p")
        return cache_values[key]

    us = np.linspace(UPS[0], UPS[1], 7)
    ms = np.linspace(LOGM[0], LOGM[1], 7)
    grid = [{"u": float(u), "logm": float(m), "F": cache(u, m)} for u in us for m in ms]
    best = min(grid, key=lambda row: row["F"])
    a10_f = cache(origin_u, origin_m)

    def pull(start_u, start_m):
        u, m = float(start_u), float(start_m)
        for _ in range(2):
            left = UPS[0] if u - 0.2 < UPS[0] else u - 0.2
            right = UPS[1] if u + 0.2 > UPS[1] else u + 0.2
            refined = refine_interval(lambda x, mm=m: cache(x, mm), left, right, limit=8)
            u = refined["u"]
            left = LOGM[0] if m - 0.4 < LOGM[0] else m - 0.4
            right = LOGM[1] if m + 0.4 > LOGM[1] else m + 0.4
            refined = refine_interval(lambda x, uu=u: cache(uu, x), left, right, limit=8)
            m = refined["u"]
        return {"u": u, "logm": m, "F": cache(u, m)}

    from_grid = pull(best["u"], best["logm"])
    from_a10 = pull(origin_u, origin_m)
    winner = from_grid if from_grid["F"] <= from_a10["F"] else from_a10
    separated = abs(from_grid["F"] - from_a10["F"]) > 1e-4 and (
        abs(from_grid["u"] - from_a10["u"]) > 1e-3 or abs(from_grid["logm"] - from_a10["logm"]) > 1e-3
    )
    return {
        "galaxy": galaxy.name,
        "a10": {"u": origin_u, "logm": origin_m, "F": a10_f},
        "grid_best": best,
        "from_grid": from_grid,
        "from_a10": from_a10,
        "winner": winner,
        "delta_F": winner["F"] - a10_f,
        "initialization_dependent": separated,
        "n_unique_objective": len(cache_values),
        "certified_global": False,
    }


def baryon_paired(galaxies, budget) -> dict:
    manifest = json.loads((A10 / "state/baryon/ACCEPTED_MANIFEST.json").read_text())
    terminal = np.load(A10 / "state/baryon" / manifest["theta_file"])
    warm = np.load(A10 / "state/baryon/accepted_0_theta.npy")
    if theta_hash(terminal) != TERMINAL["baryon"] or theta_hash(warm) != WARM_BARYON:
        raise SystemExit("STOP baryon paired vectors")
    recorded = json.loads((A10 / "logs/baryon_result.json").read_text())
    f192_terminal = float(recorded["terminal_F"])
    f192_warm = float(recorded["initial_F"])
    f384_terminal = float(recorded["quadrature"]["refined_F"])
    pars = unpack(warm, "baryon")
    total = 0.0
    for gi, galaxy in enumerate(galaxies):
        budget.charge(1, "baryon384warm:" + galaxy.name)
        total -= loglik(galaxy, gi, pars, "baryon", order=384)
    delta192 = f192_terminal - f192_warm
    delta384 = f384_terminal - total
    return {
        "reused_a10_terminal_fine384": True,
        "reused_a10_fine192": True,
        "computed_warm_fine384": True,
        "F192_warm": f192_warm,
        "F192_terminal": f192_terminal,
        "F384_warm": total,
        "F384_terminal": f384_terminal,
        "Delta_F_192": delta192,
        "Delta_F_384": delta384,
        "Delta_difference": delta384 - delta192,
        "same_sign": bool(delta192 * delta384 > 0.0),
        "warm_theta_sha256": WARM_BARYON,
        "terminal_theta_sha256": TERMINAL["baryon"],
        "fine192_remains_target": True,
    }


def main() -> None:
    verify_sources()
    galaxies = load_galaxies()
    selection = select(galaxies)
    atomic_json(NEW / "SELECTED_GALAXIES.json", selection)
    print("SELECTION", [row["galaxy"] + ":" + row["role"] for row in selection["galaxies"]], flush=True)
    budget = Budget()
    results = []
    failure = None
    by_name = {g.name: (i, g) for i, g in enumerate(galaxies)}
    terminals = {model: unpack(load_terminal(model), model) for model in ("baryon", "ptq-nu", "ptq")}
    try:
        for row in selection["galaxies"]:
            gi, galaxy = by_name[row["galaxy"]]
            for model in ("baryon", "ptq-nu", "ptq"):
                print("LOCAL", model, galaxy.name, flush=True)
                results.append(one_dimensional(galaxy, gi, terminals[model], model, budget))
        paired = baryon_paired(galaxies, budget)
        atomic_json(NEW / "logs/baryon_paired_partial.json", paired)
        nfw_rows = []
        remaining = BUDGET_CAP - budget.n
        elapsed = time.perf_counter() - budget.t0
        if remaining >= 250 and elapsed < WALL_CAP - 180:
            nfw_base = unpack(load_terminal("nfw1p"), "nfw1p")
            for name in selection["nfw_2d_conditional"]["members"]:
                gi, galaxy = by_name[name]
                print("NFW2D", galaxy.name, flush=True)
                nfw_rows.append(nfw_two_dimensional(galaxy, gi, nfw_base, budget))
        else:
            nfw_rows.append({"status": "NOT_RUN", "remaining": remaining, "elapsed": elapsed})
    except Exception as exc:
        failure = {"type": type(exc).__name__, "message": str(exc)}
        paired = None
        nfw_rows = []
        print("STOP", failure, flush=True)
    atomic_json(NEW / "logs/local_results.json", {
        "results": results,
        "paired": paired if failure is None else None,
        "nfw": nfw_rows if failure is None else [],
        "failure": failure,
        "marginals": budget.n,
        "wall_seconds": time.perf_counter() - budget.t0,
    })
    print("A11_DONE", budget.n, failure, flush=True)


if __name__ == "__main__":
    main()
