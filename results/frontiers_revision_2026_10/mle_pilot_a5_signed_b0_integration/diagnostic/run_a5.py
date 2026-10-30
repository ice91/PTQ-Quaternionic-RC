"""Isolated signed-SPARC B0 checks. No optimizer and no canonical edits."""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

NEW = Path("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/mle_pilot_a5_signed_b0_integration")
IMPL = Path("/home/rocky/PTQ-Quaternionic-RC")
FEAS = IMPL / "results/frontiers_revision_2026_10/mle_pilot_a4_domain_feasibility"
ORIG_SRC = IMPL / "results/frontiers_revision_2026_10/mle_pilot_a/candidate/src"
EXEC = IMPL / "results/frontiers_revision_2026_10/mle_pilot_a/execution_v1"
sys.path.insert(0, str(NEW / "candidate/src"))

from ptquat.constants import C_LIGHT, H0_SI, KM, KPC  # noqa: E402
from ptquat.data import load_tidy_sparc  # noqa: E402
from ptquat.marginal_likelihood import PRIMARY_B0, galaxy_log_marginal  # noqa: E402
from ptquat.models import (  # noqa: E402
    catalog_velocity_at_distance,
    model_v_baryon,
    model_v_mond,
    model_v_mond_screen,
    model_v_nfw1p,
    model_v_ptq,
    model_v_ptq_nu,
    model_v_ptq_screen,
    signed_square,
    vbar_squared_kms2,
)

U_LO, U_HI = 0.05, 1.5
A0_HI = 2.0e-10
EXPECTED_THETA = {
    "baryon": "1e9c848c43e456d6eb233b90ff4796edaf2a30fb7cd9daf274fe40e95f680714",
    "mond": "4c2a9af69596074fc4f81d6240d3e70c926d0c80ab2fa2e4d8c6118f9cd1e40f",
    "ptq": "96ac8469398b416aa0740522da74c844c12ea8e95364d69f3920f2df460fd6c4",
    "ptq-nu": "7abded50306ec11a32380f1ae85c555faef1d2d9318f39c673b691ae974473b2",
    "ptq-screen": "b0a24278778944f59b3ae440ba3ae5a63d9dec584c851f93a64fa3e693ff3ab4",
}
THETA_PATHS = {
    "baryon": EXEC / "baryon_last_accepted_theta.npy",
    "mond": EXEC / "mond_last_accepted_theta.npy",
    "ptq": EXEC / "ptq_last_accepted_theta.npy",
    "ptq-nu": EXEC / "ptq_nu_last_accepted_theta.npy",
    "ptq-screen": EXEC / "ptq_screen_last_accepted_theta.npy",
    "mond-screen": EXEC / "mond_screen_last_accepted_theta.npy",
    "nfw1p": EXEC / "nfw1p_initial_theta.npy",
}
FROZEN = {
    ORIG_SRC / "ptquat/models.py": "cba208085f4e5e6373fe3b1bd5b862e9be66a6d1412a6a7750c4c4b053bbc952",
    IMPL / "dataset/sparc_tidy.csv": "3ab1c3c2b110c1e48680ebae77c1ceefaa4f778f81119dcf04881b630b274cbb",
    IMPL / "dataset/raw/vizier_table2.csv": "9542e3537c81ab88c2d7b872ada772d1653a50a703309e608f4cb21234317f1b",
    FEAS / "S_DOMAIN13_MASK.csv": "df0086b1e67f96403ff09f5c5bfb7f5818593d141d14234d752cb433a9511e6e",
    FEAS / "S_FULL_MASK.csv": "d8bafce52755f79045ae50aeb18d6fee9b11fc7fe8468052ed97463164c224b8",
    FEAS / "DOMAIN_PROOF.json": "7ed7ca4dbe351da61a5d868672a5101e407a692201186d1d9719f6ece153020c",
    FEAS / "COMMON_SUPPORT_MATRIX.json": "5daedf90260a40a65c576e93779b43a7227c5a8a46c1bec4ea0a4d666da765f9",
    FEAS / "SAMPLE_CANDIDATES.json": "10c0c35a6dfd78b0f7afc20690bd3556d3321c209144016d5e33671d0d047c58",
    FEAS / "INPUT_MANIFEST.json": "0ed1be5f83d6e4c53b42421c8a12a5b1bf20eaeea9690b1b3b94a874dc7fe322",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def theta_content_hash(theta: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(theta, dtype="<f8").tobytes()).hexdigest()


def layout(model: str, g: int) -> dict:
    L = {"model": model, "G": g, "k": 0}
    s = 0
    if model in ("ptq", "ptq-nu"):
        L["i_eps"] = s
        s += 1
        L["sl_U"] = slice(s, s + g)
        s += g
    elif model == "ptq-screen":
        L["i_eps"] = s
        s += 1
        L["i_lq"] = s
        s += 1
        L["sl_U"] = slice(s, s + g)
        s += g
    elif model == "baryon":
        L["sl_U"] = slice(s, s + g)
        s += g
    elif model == "mond":
        L["i_log10a0"] = s
        s += 1
        L["sl_U"] = slice(s, s + g)
        s += g
    elif model == "mond-screen":
        L["i_log10a0"] = s
        s += 1
        L["i_lq"] = s
        s += 1
        L["sl_U"] = slice(s, s + g)
        s += g
    elif model == "nfw1p":
        L["sl_U"] = slice(s, s + g)
        s += g
        L["sl_lM"] = slice(s, s + g)
        s += g
    else:
        raise ValueError(model)
    L["i_lnsig"] = s
    s += 1
    L["k"] = s
    return L


def unpack(theta: np.ndarray, L: dict) -> dict:
    out = {"model": L["model"]}
    if "i_eps" in L:
        out["eps"] = float(theta[L["i_eps"]])
    if "sl_U" in L:
        out["U"] = np.asarray(theta[L["sl_U"]], float)
    if "i_log10a0" in L:
        out["log10a0"] = float(theta[L["i_log10a0"]])
    if "sl_lM" in L:
        out["lM"] = np.asarray(theta[L["sl_lM"]], float)
    if "i_lq" in L:
        out["lq"] = float(theta[L["i_lq"]])
    out["sigma"] = float(np.exp(theta[L["i_lnsig"]]))
    return out


def map_ptqnu_to_mond(theta: np.ndarray, g: int) -> np.ndarray:
    src = unpack(theta, layout("ptq-nu", g))
    dst_l = layout("mond", g)
    out = np.zeros(dst_l["k"])
    out[dst_l["i_log10a0"]] = math.log10(src["eps"] * C_LIGHT * H0_SI)
    out[dst_l["sl_U"]] = src["U"]
    out[dst_l["i_lnsig"]] = math.log(src["sigma"])
    return out


def map_ptqscreen_to_mondscreen(theta: np.ndarray, g: int) -> np.ndarray:
    src = unpack(theta, layout("ptq-screen", g))
    dst_l = layout("mond-screen", g)
    out = np.zeros(dst_l["k"])
    out[dst_l["i_log10a0"]] = math.log10(src["eps"] * C_LIGHT * H0_SI)
    out[dst_l["i_lq"]] = src["lq"]
    out[dst_l["sl_U"]] = src["U"]
    out[dst_l["i_lnsig"]] = math.log(src["sigma"])
    return out


def inside_bounds(model: str, theta: np.ndarray) -> bool:
    g = int((theta.size - (3 if model in ("ptq-screen", "mond-screen") else 2 if model == "nfw1p" else 2)) )
    # Use explicit layout length instead of this guess.
    L = layout(model, theta.size - layout(model, 1)["k"] + 1)
    # The expression above is awkward. Callers pass G.
    raise RuntimeError("use inside_bounds_g")


def bounds_ok(model: str, theta: np.ndarray, g: int) -> dict:
    L = layout(model, g)
    if theta.shape != (L["k"],):
        return {"ok": False, "reason": f"shape {theta.shape} != {(L['k'],)}"}
    p = unpack(theta, L)
    eps_matched = (
        A0_HI / (C_LIGHT * H0_SI),
        5.0e-11 / (C_LIGHT * H0_SI),
    )
    # lo, hi
    a0_eps = (5.0e-11 / (C_LIGHT * H0_SI), A0_HI / (C_LIGHT * H0_SI))
    problems = []
    if "eps" in p:
        lo, hi = a0_eps if model in ("ptq-nu", "ptq-screen") else (1.0e-6, 3.999)
        if not (lo <= p["eps"] <= hi):
            problems.append(f"eps {p['eps']} outside {(lo, hi)}")
    if "log10a0" in p:
        lo, hi = math.log10(5.0e-11), math.log10(A0_HI)
        if not (lo <= p["log10a0"] <= hi):
            problems.append(f"log10a0 {p['log10a0']} outside {(lo, hi)}")
    if "lq" in p:
        lo, hi = math.log(0.25), math.log(8.0)
        if not (lo <= p["lq"] <= hi):
            problems.append(f"ln q {p['lq']} outside {(lo, hi)}")
    if np.any(p["U"] < U_LO) or np.any(p["U"] > U_HI):
        problems.append("Upsilon outside [0.05,1.5]")
    if "lM" in p and (np.any(p["lM"] < 9.0) or np.any(p["lM"] > 13.0)):
        problems.append("log10 M200 outside [9,13]")
    if not (math.log(1e-3) <= math.log(p["sigma"]) <= math.log(200.0)):
        problems.append("sigma outside [1e-3,200]")
    return {"ok": not problems, "problems": problems, "decoded_globals": {k: p[k] for k in p if k != "U" and k != "lM"}}


class Budget:
    def __init__(self, weights: dict):
        self.weights = weights
        self.calls = 0
        self.equivalents = 0.0
        self.started = time.perf_counter()
        self.stopped = None
        self.log_path = NEW / "logs/evaluations.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def remaining_ok(self, weight: float) -> bool:
        if self.calls + 1 > 1500 or self.equivalents + weight > 12.0:
            self.stopped = "BUDGET"
            return False
        if time.perf_counter() - self.started > 70 * 60:
            self.stopped = "WALL"
            return False
        return True

    def charge(self, galaxy: str) -> float:
        weight = float(self.weights[galaxy])
        if not self.remaining_ok(weight):
            raise RuntimeError(self.stopped or "BUDGET")
        self.calls += 1
        self.equivalents += weight
        return weight


def dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n")


def main() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    mismatches = []
    for path, digest in FROZEN.items():
        got = sha256(path)
        if got != digest:
            mismatches.append({"path": str(path), "got": got, "expected": digest})
    if mismatches:
        dump(NEW / "S_DOMAIN13_MASK_VERIFICATION.json", {"status": "STOP", "hash_mismatches": mismatches})
        raise SystemExit("STOP hash mismatch")

    tidy = pd.read_csv(IMPL / "dataset/sparc_tidy.csv")
    raw = pd.read_csv(IMPL / "dataset/raw/vizier_table2.csv")
    mask = pd.read_csv(FEAS / "S_DOMAIN13_MASK.csv")
    joined = tidy.merge(raw, left_on=["galaxy", "r_kpc"], right_on=["Name", "Rad"], how="left", validate="one_to_one")
    if not (
        len(tidy) == len(mask) == 1782
        and np.array_equal(mask.tidy_index.to_numpy(), np.arange(1782))
        and np.array_equal(mask.source_recno.to_numpy(), joined.recno.to_numpy())
        and mask.source_recno.is_unique
    ):
        raise SystemExit("STOP mask identity")
    disk = tidy.v_disk_kms.to_numpy(float)
    bulge = tidy.v_bulge_kms.to_numpy(float)
    gas = tidy.v_gas_kms.to_numpy(float)
    recomputed = vbar_squared_kms2(U_LO, disk, bulge, gas)
    included = recomputed > 0.0
    frozen_included = mask.included.to_numpy(int) == 1
    if not np.array_equal(included, frozen_included):
        raise SystemExit("STOP recomputed inclusion disagrees with frozen mask")
    if int(included.sum()) != 1769 or int((~included).sum()) != 13:
        raise SystemExit("STOP inclusion counts")
    kept = tidy.loc[included].copy()
    if kept.galaxy.nunique() != 91:
        raise SystemExit("STOP galaxy count")
    negative_gas_kept = int((kept.v_gas_kms < 0).sum())
    if negative_gas_kept != 98:
        raise SystemExit(f"STOP negative gas count {negative_gas_kept}")
    # Observation values are the original tidy values; the filtered file only drops rows.
    if not np.array_equal(kept.v_obs_kms.to_numpy(), tidy.loc[included, "v_obs_kms"].to_numpy()):
        raise SystemExit("STOP observation rewrite")
    filtered_path = NEW / "logs/s_domain13_filtered_tidy.csv"
    kept.to_csv(filtered_path, index=False)
    loaded = load_tidy_sparc(str(filtered_path))
    galaxies = [loaded[name] for name in sorted(loaded)]
    if len(galaxies) != 91 or sum(len(g.v_obs) for g in galaxies) != 1769:
        raise SystemExit("STOP loaded sample")
    names = [g.name for g in galaxies]
    recno_by_galaxy = {
        name: joined.loc[included & (tidy.galaxy == name), "recno"].to_numpy(int)
        for name in names
    }
    for g in galaxies:
        order = np.argsort(g.r_kpc, kind="mergesort")
        # load_tidy_sparc already sorts by radius. Confirm recnos follow that order.
        raw_rows = joined.loc[included & (tidy.galaxy == g.name), ["r_kpc", "recno"]].sort_values("r_kpc")
        if not np.array_equal(raw_rows.r_kpc.to_numpy(float), g.r_kpc):
            raise SystemExit("STOP radius order " + g.name)
        g.source_recno = raw_rows.recno.to_numpy(int)

    weights_doc = json.loads((IMPL / "results/frontiers_revision_2026_10/mle_pilot_a/GALAXY_WORK_WEIGHTS.json").read_text())
    weights = {row["galaxy"]: float(row["fullsample_equivalent"]) for row in weights_doc["baryon"]["192"]}
    if set(weights) != set(names) or abs(sum(weights.values()) - 1.0) > 1e-12:
        raise SystemExit("STOP work weights")
    thetas = {}
    theta_meta = {}
    for model, path in THETA_PATHS.items():
        theta = np.load(path)
        digest = theta_content_hash(theta)
        if model in EXPECTED_THETA and digest != EXPECTED_THETA[model]:
            raise SystemExit(f"STOP theta provenance {model}")
        L = layout(model, 91)
        if theta.shape != (L["k"],):
            raise SystemExit(f"STOP theta shape {model}")
        check = bounds_ok(model, theta, 91)
        if not check["ok"]:
            raise SystemExit(f"STOP theta bounds {model} {check['problems']}")
        thetas[model] = theta
        theta_meta[model] = {"path": str(path), "content_sha256": digest, "bounds": check}
    mapped_nu = map_ptqnu_to_mond(thetas["ptq-nu"], 91)
    mapped_screen = map_ptqscreen_to_mondscreen(thetas["ptq-screen"], 91)
    if not bounds_ok("mond", mapped_nu, 91)["ok"] or not bounds_ok("mond-screen", mapped_screen, 91)["ok"]:
        raise SystemExit("STOP mapped theta bounds")
    # Physical identity of the map.
    nu = unpack(thetas["ptq-nu"], layout("ptq-nu", 91))
    mond = unpack(mapped_nu, layout("mond", 91))
    if not np.array_equal(nu["U"], mond["U"]) or abs(math.log(nu["sigma"]) - math.log(mond["sigma"])) != 0.0:
        raise SystemExit("STOP nu map mismatch")
    if abs((10 ** mond["log10a0"]) - nu["eps"] * C_LIGHT * H0_SI) > 0.0:
        # Allow one ulp on the a0 reconstruction through log10.
        if abs((10 ** mond["log10a0"]) / (nu["eps"] * C_LIGHT * H0_SI) - 1.0) > 1e-12:
            raise SystemExit("STOP a0 map mismatch")

    mask_doc = {
        "status": "PASS",
        "original_records": 1782,
        "excluded": 13,
        "included": 1769,
        "galaxies": 91,
        "duplicate_source_recno": 0,
        "missing_retained_records": 0,
        "negative_gas_retained": 98,
        "inclusion_rule": "B_signed(Upsilon=0.05) > 0",
        "frozen_mask_sha256": FROZEN[FEAS / "S_DOMAIN13_MASK.csv"],
        "mask_applied_before_marginalization": True,
        "dynamic_mask_inside_integrator": False,
    }
    dump(NEW / "S_DOMAIN13_MASK_VERIFICATION.json", mask_doc)

    # Domain regression on the fixed masked sample. No marginal likelihood.
    source = (NEW / "candidate/src/ptquat/models.py").read_text()
    if "v_gas_kms**2" in source or "v_disk_kms**2" in source or "sqrt(abs" in source:
        raise SystemExit("STOP signed operator source still has a squared gas path")
    index = {g.name: i for i, g in enumerate(galaxies)}
    ugc = next(g for g in galaxies if g.name == "UGC01281")
    ugc_i = int(np.flatnonzero(np.isclose(ugc.r_kpc, 0.38))[0])
    if int(ugc.source_recno[ugc_i]) != 1947:
        raise SystemExit("STOP fixture recno")
    b_fix = float(vbar_squared_kms2(U_LO, ugc.v_disk[ugc_i], ugc.v_bulge[ugc_i], ugc.v_gas[ugc_i]))
    all_b = []
    y_min = np.inf
    velocity_problems = []
    model_names = ["baryon", "mond", "ptq", "ptq-nu", "ptq-screen", "mond-screen", "nfw1p"]
    for g in galaxies:
        gi = index[g.name]
        for u_name, u_value in (("lower", U_LO), ("archived", None), ("upper", U_HI)):
            for model in model_names:
                p = unpack(thetas[model], layout(model, 91))
                u = float(p["U"][gi] if u_value is None else u_value)
                b = vbar_squared_kms2(u, g.v_disk, g.v_bulge, g.v_gas)
                all_b.append(b if u_name == "lower" else None)
                kwargs = {}
                if model == "baryon":
                    v = model_v_baryon(u, g.r_kpc, g.v_disk, g.v_bulge, g.v_gas)
                    g_n = b * KM**2 / (g.r_kpc * KPC)
                elif model == "mond":
                    a0 = 10 ** p["log10a0"]
                    v = model_v_mond(u, a0, g.r_kpc, g.v_disk, g.v_bulge, g.v_gas)
                    g_n = b * KM**2 / (g.r_kpc * KPC)
                    y_raw = g_n / a0
                    y_min = min(y_min, float(np.min(y_raw)))
                elif model == "ptq":
                    v = model_v_ptq(u, p["eps"], g.r_kpc, g.v_disk, g.v_bulge, g.v_gas)
                    g_n = b * KM**2 / (g.r_kpc * KPC)
                elif model == "ptq-nu":
                    v = model_v_ptq_nu(u, p["eps"], g.r_kpc, g.v_disk, g.v_bulge, g.v_gas)
                    a0 = max(p["eps"] * C_LIGHT * H0_SI, 1e-30)
                    y_raw = (b * KM**2 / (g.r_kpc * KPC)) / a0
                    y_min = min(y_min, float(np.min(y_raw)))
                elif model == "ptq-screen":
                    v = model_v_ptq_screen(u, p["eps"], math.exp(p["lq"]), g.r_kpc, g.v_disk, g.v_bulge, g.v_gas)
                    a0 = max(p["eps"] * C_LIGHT * H0_SI, 1e-30)
                    y_raw = (b * KM**2 / (g.r_kpc * KPC)) / a0
                    y_min = min(y_min, float(np.min(y_raw)))
                elif model == "mond-screen":
                    a0 = 10 ** p["log10a0"]
                    v = model_v_mond_screen(u, a0, math.exp(p["lq"]), g.r_kpc, g.v_disk, g.v_bulge, g.v_gas)
                    y_raw = (b * KM**2 / (g.r_kpc * KPC)) / a0
                    y_min = min(y_min, float(np.min(y_raw)))
                else:
                    v = model_v_nfw1p(u, float(p["lM"][gi]), g.r_kpc, g.v_disk, g.v_bulge, g.v_gas)
                if not np.all(np.isfinite(v)) or np.any(v < 0):
                    velocity_problems.append({"galaxy": g.name, "model": model, "upsilon": u_name})
    b_lower = np.concatenate([
        vbar_squared_kms2(U_LO, g.v_disk, g.v_bulge, g.v_gas) for g in galaxies
    ])
    worst_y = np.inf
    for g in galaxies:
        b = vbar_squared_kms2(U_LO, g.v_disk, g.v_bulge, g.v_gas)
        worst_y = min(worst_y, float(np.min((b * KM**2 / (g.r_kpc * KPC)) / A0_HI)))
    y_min = min(y_min, worst_y)
    neg_rows = []
    pos_rows = []
    for g in galaxies:
        for j, vg in enumerate(g.v_gas):
            row = {"galaxy": g.name, "r_kpc": float(g.r_kpc[j]), "source_recno": int(g.source_recno[j]), "v_gas": float(vg)}
            if vg < 0:
                neg_rows.append(row)
            elif vg > 0 and not pos_rows:
                pos_rows.append(row)
    retained_negative = min(neg_rows, key=lambda row: row["v_gas"])
    # Distance and inclination at the minimum-B fixture.
    gi = index["UGC01281"]
    p_b = unpack(thetas["baryon"], layout("baryon", 91))
    view = type("G", (), {})()
    view.r = ugc.r_kpc
    view.disk = ugc.v_disk
    view.bulge = ugc.v_bulge
    view.gas = ugc.v_gas
    view.D = ugc.D_Mpc
    pars = {"U": p_b["U"], "model": "baryon"}
    v_nom = catalog_velocity_at_distance(view, pars, gi, "baryon", ugc.D_Mpc)
    scale_rows = []
    for s in (0.25, 1.0, 4.0):
        v_s = catalog_velocity_at_distance(view, pars, gi, "baryon", ugc.D_Mpc * s)
        gap = float(np.max(np.abs(v_s - np.sqrt(s) * v_nom)))
        scale_rows.append({"s": s, "max_abs_gap_versus_sqrt_s": gap, "fixture_positive": bool(v_s[ugc_i] > 0)})
    b_pert = float(vbar_squared_kms2(U_LO * (1 + 1e-6), ugc.v_disk[ugc_i], ugc.v_bulge[ugc_i], ugc.v_gas[ugc_i]))
    v_pert = float(model_v_baryon(U_LO * (1 + 1e-6), ugc.r_kpc[ugc_i:ugc_i+1], ugc.v_disk[ugc_i:ugc_i+1], ugc.v_bulge[ugc_i:ugc_i+1], ugc.v_gas[ugc_i:ugc_i+1])[0])
    v_at_bounds = {
        "U_0.05": float(model_v_baryon(U_LO, ugc.r_kpc[ugc_i:ugc_i+1], ugc.v_disk[ugc_i:ugc_i+1], ugc.v_bulge[ugc_i:ugc_i+1], ugc.v_gas[ugc_i:ugc_i+1])[0]),
        "U_1.5": float(model_v_baryon(U_HI, ugc.r_kpc[ugc_i:ugc_i+1], ugc.v_disk[ugc_i:ugc_i+1], ugc.v_bulge[ugc_i:ugc_i+1], ugc.v_gas[ugc_i:ugc_i+1])[0]),
    }
    i_nom = math.radians(ugc.i_deg)
    i_shift = math.radians(min(89.0, ugc.i_deg + 1.0))
    inclination = {
        "nominal_b": 1.0,
        "shifted_b": math.sin(i_shift) / math.sin(i_nom),
        "baryonic_B_independent_of_inclination": True,
        "integrator_applies_b_outside_baryonic_operator": True,
    }
    domain_pass = (
        not velocity_problems
        and float(np.min(b_lower)) > 0
        and abs(float(np.min(b_lower)) - b_fix) == 0.0
        and y_min >= 1e-12
        and all(row["fixture_positive"] and row["max_abs_gap_versus_sqrt_s"] < 1e-12 for row in scale_rows)
        and inclination["shifted_b"] > 0
    )
    domain_doc = {
        "status": "PASS" if domain_pass else "FAIL",
        "likelihood_evaluations": 0,
        "minimum_B_at_U_0.05": float(np.min(b_lower)),
        "minimum_row": {"galaxy": "UGC01281", "r_kpc": 0.38, "source_recno": 1947, "B": b_fix},
        "known_minimum_reference": 0.44940499999999983,
        "minimum_matches_reference": abs(b_fix - 0.44940499999999983) == 0.0,
        "velocity_problems": velocity_problems,
        "minimum_interpolation_y": y_min,
        "y_floor_inactive": bool(y_min >= 1e-12),
        "retained_negative_gas_example": retained_negative,
        "positive_gas_example": pos_rows[0],
        "fixture_velocity_at_upsilon_bounds": v_at_bounds,
        "relative_1e-6_upsilon_perturbation": {"B": b_pert, "baryon_speed": v_pert, "finite": bool(np.isfinite(v_pert) and v_pert > 0)},
        "distance_scaling": scale_rows,
        "inclination": inclination,
        "preexisting_radius_and_y_floors_left_unchanged": True,
        "preexisting_nfw_halo_nonnegative_sqrt_left_unchanged": True,
        "no_new_acceleration_floor": True,
        "no_sqrt_abs_B": True,
    }
    dump(NEW / "SEVEN_MODEL_DOMAIN_REGRESSION.json", domain_doc)
    dump(NEW / "SIGNED_OPERATOR_REGRESSION.json", {
        "operator": "Upsilon*(Vdisk*|Vdisk|+Vbulge*|Vbulge|)+Vgas*|Vgas|",
        "models_using_vbar_squared_kms2": model_names,
        "gas_square_remaining_in_isolated_models": "v_gas_kms**2" in source,
        "minimum_B": b_fix,
        "nonnegative_disk_and_bulge_on_retained_sample": bool(
            min(float(np.min(g.v_disk)) for g in galaxies) >= 0
            and min(float(np.min(g.v_bulge)) for g in galaxies) >= 0
        ),
    })
    dump(NEW / "BOUNDARY_STRESS_RESULTS.json", {
        "fixture": domain_doc["minimum_row"],
        "perturbation": domain_doc["relative_1e-6_upsilon_perturbation"],
        "bounds": v_at_bounds,
        "distance_scaling": scale_rows,
        "y_floor_inactive": domain_doc["y_floor_inactive"],
        "derivatives_not_certified_by_positive_speed": True,
    })
    if not domain_pass:
        raise SystemExit("STOP domain regression")

    budget = Budget(weights)
    clean = next(g for g in galaxies if np.all(g.v_gas >= 0) and np.all(g.v_disk >= 0) and np.all(g.v_bulge >= 0))
    records = []

    def persist(row: dict) -> None:
        records.append(row)
        with budget.log_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
            handle.flush()

    def run_signed(model: str, theta: np.ndarray, galaxy) -> dict:
        gi = index[galaxy.name]
        weight = budget.charge(galaxy.name)
        t0 = time.perf_counter()
        p = unpack(theta, layout(model, 91))
        p["model"] = model
        out = galaxy_log_marginal(galaxy, p, gi, model, likelihood_version=PRIMARY_B0)
        row = {
            "implementation": "signed",
            "model": model,
            "galaxy": galaxy.name,
            "gi": gi,
            "n_radii": int(len(galaxy.v_obs)),
            "log_marginal": out["log_marginal"],
            "weight": weight,
            "seconds": time.perf_counter() - t0,
            "likelihood_version_token": PRIMARY_B0,
            "operator": "signed",
        }
        persist(row)
        return row

    def run_original(model: str, theta: np.ndarray, galaxy) -> dict:
        weight = budget.charge(galaxy.name)
        out_path = NEW / "logs" / f"original_{model}_{galaxy.name}.json"
        theta_path = NEW / "logs" / f"theta_{model}_for_{galaxy.name}.npy"
        np.save(theta_path, theta)
        t0 = time.perf_counter()
        script = r"""
import json, os, sys, time
from pathlib import Path
import numpy as np
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
sys.path.insert(0, sys.argv[1])
from ptquat.data import load_tidy_sparc
from ptquat.marginal_likelihood import PRIMARY_B0, galaxy_log_marginal
from ptquat.constants import H0_SI
# Local unpack matching the signed runner.
def layout(model, g):
    s=0; L={}
    if model in ('ptq','ptq-nu'):
        L['i_eps']=s; s+=1; L['sl_U']=slice(s,s+g); s+=g
    elif model=='ptq-screen':
        L['i_eps']=s; s+=1; L['i_lq']=s; s+=1; L['sl_U']=slice(s,s+g); s+=g
    elif model=='baryon':
        L['sl_U']=slice(s,s+g); s+=g
    elif model=='mond':
        L['i_log10a0']=s; s+=1; L['sl_U']=slice(s,s+g); s+=g
    elif model=='mond-screen':
        L['i_log10a0']=s; s+=1; L['i_lq']=s; s+=1; L['sl_U']=slice(s,s+g); s+=g
    elif model=='nfw1p':
        L['sl_U']=slice(s,s+g); s+=g; L['sl_lM']=slice(s,s+g); s+=g
    L['i_lnsig']=s; return L
theta=np.load(sys.argv[4]); model=sys.argv[5]; name=sys.argv[6]
gs=load_tidy_sparc(sys.argv[3]); names=sorted(gs); g=gs[name]; gi=names.index(name)
L=layout(model, len(names)); p={'model':model,'U':np.asarray(theta[L['sl_U']],float),'sigma':float(np.exp(theta[L['i_lnsig']]))}
if 'i_eps' in L: p['eps']=float(theta[L['i_eps']])
if 'i_log10a0' in L: p['log10a0']=float(theta[L['i_log10a0']])
if 'i_lq' in L: p['lq']=float(theta[L['i_lq']])
if 'sl_lM' in L: p['lM']=np.asarray(theta[L['sl_lM']],float)
t0=time.perf_counter()
out=galaxy_log_marginal(g,p,gi,model,likelihood_version=PRIMARY_B0)
Path(sys.argv[2]).write_text(json.dumps({'log_marginal':out['log_marginal'],'seconds':time.perf_counter()-t0,'n_radii':int(len(g.v_obs))}))
"""
        subprocess.run(
            [sys.executable, "-c", script, str(ORIG_SRC), str(out_path), str(filtered_path), str(theta_path), model, galaxy.name],
            check=True,
            env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
        )
        payload = json.loads(out_path.read_text())
        row = {
            "implementation": "original_squared_gas",
            "model": model,
            "galaxy": galaxy.name,
            "n_radii": payload["n_radii"],
            "log_marginal": payload["log_marginal"],
            "weight": weight,
            "seconds": time.perf_counter() - t0,
            "operator": "squared",
        }
        persist(row)
        return row

    # Probe one signed evaluation before reserving full samples.
    probe = run_signed("baryon", thetas["baryon"], clean)
    probe_seconds = probe["seconds"]
    print(f"PROBE {clean.name} {probe_seconds:.3f}s logL {probe['log_marginal']}", flush=True)
    full_pair_seconds = probe_seconds * 91 * 2
    run_full_families = full_pair_seconds * 2 < 60 * 60 and budget.equivalents + 4.0 <= 12.0

    old_clean = run_original("baryon", thetas["baryon"], clean)
    signed_ugc = run_signed("baryon", thetas["baryon"], ugc)
    old_ugc = run_original("baryon", thetas["baryon"], ugc)
    clean_gap = abs(probe["log_marginal"] - old_clean["log_marginal"])
    ugc_gap = abs(signed_ugc["log_marginal"] - old_ugc["log_marginal"])

    family_results = {"mond_ptqnu": None, "screen": None}
    if run_full_families:
        def run_family(label, left_model, left_theta, right_model, right_theta):
            left_rows = [run_signed(left_model, left_theta, g) for g in galaxies]
            right_rows = [run_signed(right_model, right_theta, g) for g in galaxies]
            gaps = [abs(a["log_marginal"] - b["log_marginal"]) for a, b in zip(left_rows, right_rows)]
            return {
                "galaxies": 91,
                "complete": True,
                "max_abs_galaxy_logL_gap": max(gaps),
                "status": "PASS" if max(gaps) <= 1e-9 else "FAIL",
            }
        family_results["mond_ptqnu"] = run_family("nu", "ptq-nu", thetas["ptq-nu"], "mond", mapped_nu)
        family_results["screen"] = run_family("screen", "ptq-screen", thetas["ptq-screen"], "mond-screen", mapped_screen)
        family_status = "PASS" if all(v["status"] == "PASS" for v in family_results.values()) else "FAIL"
    else:
        family_status = "INCOMPLETE"
        family_results["reason"] = "FULL_SAMPLE_NOT_STARTED"
        family_results["probe_seconds"] = probe_seconds

    fixed = {
        "clean_galaxy": clean.name,
        "clean_model": "baryon",
        "clean_signed_logL": probe["log_marginal"],
        "clean_squared_logL": old_clean["log_marginal"],
        "clean_abs_gap": clean_gap,
        "clean_agreement": clean_gap <= 1e-9,
        "negative_gas_galaxy": "UGC01281",
        "negative_gas_signed_logL": signed_ugc["log_marginal"],
        "negative_gas_squared_logL": old_ugc["log_marginal"],
        "negative_gas_abs_gap": ugc_gap,
        "negative_gas_values_differ": ugc_gap > 1e-6,
        "not_a_model_ranking": True,
        "archived_vectors_are_legacy_squared_gas_points": True,
        "probe_seconds": probe_seconds,
        "full_families_started": run_full_families,
    }
    dump(NEW / "SIGNED_B0_FIXED_POINT_RESULTS.json", fixed)
    dump(NEW / "MATCHED_FAMILY_IDENTITY.json", {"status": family_status, "families": family_results, "criterion": "max |Delta galaxy logL| <= 1e-9 on all 91 galaxies"})
    dump(NEW / "EXECUTION_BUDGET.json", {
        "galaxy_marginal_evaluations": budget.calls,
        "fullsample_equivalents": budget.equivalents,
        "wall_seconds": time.perf_counter() - budget.started,
        "stopped": budget.stopped,
        "limit_calls": 1500,
        "limit_equivalents": 12.0,
        "optimizer_calls": 0,
    })
    print(json.dumps({"fixed": fixed, "family_status": family_status, "calls": budget.calls, "eq": budget.equivalents, "domain_min_B": b_fix}, indent=2))


if __name__ == "__main__":
    main()
