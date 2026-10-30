"""Bounded P1G-A-R1 diagnostics. No optimizer. Counts every galaxy marginal."""
import json, os, sys, time, hashlib, warnings
from pathlib import Path
import numpy as np

CAND = Path("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/mle_pilot_a/candidate/src")
sys.path.insert(0, str(CAND))
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

from ptquat.data import load_tidy_sparc
from ptquat.fit_global import _unpack, _layout
from ptquat.mle import theta_bounds, map_theta_ptqscreen_to_mondscreen, map_theta_ptqnu_to_mond
from ptquat.marginal_likelihood import galaxy_log_marginal, integrate, PRIMARY_B0
from ptquat.constants import H0_SI, C_LIGHT
from types import SimpleNamespace

EX = Path("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/mle_pilot_a/execution_v1")
OUT = Path("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/mle_pilot_a_r1_forensic/diagnostic")
IMPL = Path("/home/rocky/PTQ-Quaternionic-RC")
MAX_CALLS = 30000
MAX_EQ = 300.0
MAX_WALL = 60 * 60
calls = 0
eq = 0.0
t0 = time.monotonic()
weights = json.loads((IMPL / "results/frontiers_revision_2026_10/mle_pilot_a/GALAXY_WORK_WEIGHTS.json").read_text())

def charge(model, gi, order):
    global calls, eq
    if time.monotonic() - t0 >= MAX_WALL:
        raise RuntimeError("diagnostic wall cap")
    w = weights[model][str(order if order in (96, 192) else 192)][gi]
    unit = w["fullsample_equivalent"] * (1.0 if order in (96, 192) else 384 / 192)
    if calls + 1 > MAX_CALLS or eq + unit > MAX_EQ:
        raise RuntimeError("diagnostic work cap")
    calls += 1
    eq += unit

def load_gs():
    gd = load_tidy_sparc(str(IMPL / "dataset/sparc_tidy.csv"))
    return [gd[k] for k in sorted(gd)]

def eval_galaxy(g, theta, gi, model, order):
    charge(model, gi, order)
    L = _layout(model, 91, True, None)
    p = _unpack(theta, L)
    caught = []
    with warnings.catch_warnings(record=True) as got:
        warnings.simplefilter("always")
        r = galaxy_log_marginal(g, p, gi, model, likelihood_version=PRIMARY_B0, order=order) if order in (96, 192) else None
        if r is None:
            gg = SimpleNamespace(name=g.name, r=g.r_kpc, y=g.v_obs, e=g.v_err, disk=g.v_disk, bulge=g.v_bulge, gas=g.v_gas, D=float(g.D_Mpc), sD=float(g.D_err_Mpc), inc=float(np.deg2rad(g.i_deg)), si=float(np.deg2rad(g.i_err_deg)))
            pp = dict(p, model=model, _H0=H0_SI, _c0=10.0, _c_slope=-0.1)
            raw = integrate(gg, pp, gi, "segmented_quantile", order, floors=("B0",))
            r = dict(log_marginal=raw["logL"]["B0"], seconds=raw["seconds"], density_nodes=raw["density_nodes"], D_nodes=raw["D_nodes"], i_nodes=raw["i_nodes"], max_quadratic_roundoff_bound=raw["max_quadratic_roundoff_bound"])
        caught = [dict(category=w.category.__name__, message=str(w.message)) for w in got]
    r = dict(r)
    r["warnings"] = caught
    r["order"] = order
    return r

def main():
    gs = load_gs()
    names = [g.name for g in gs]
    cases = {
        "baryon": "UGC09133",
        "mond": "NGC0801",
        "ptq-screen": "NGC0801",
    }
    rows = []
    for model, galaxy in cases.items():
        theta = np.load(EX / f"{model.replace('-', '_')}_last_accepted_theta.npy")
        gi = names.index(galaxy)
        g = gs[gi]
        vals = {}
        times = {}
        warns = {}
        for order in (96, 192, 384):
            r = eval_galaxy(g, theta, gi, model, order)
            vals[order] = r["log_marginal"]
            times[order] = r["seconds"]
            warns[order] = r["warnings"]
        rep = eval_galaxy(g, theta, gi, model, 192)
        archived = json.loads((EX / f"{model.replace('-', '_')}_terminal_OBJECTIVE_PRECISION.json").read_text())
        af = archived["fine"]["galaxy_log_marginals"][gi]
        ac = archived["coarse"]["galaxy_log_marginals"][gi]
        rows.append(dict(
            model=model, galaxy=galaxy, gi=gi,
            D_Mpc=float(g.D_Mpc), D_err=float(g.D_err_Mpc), i_deg=float(g.i_deg), i_err=float(g.i_err_deg), n_points=int(len(g.v_obs)),
            archived_fine=af, archived_coarse=ac, archived_delta=af - ac,
            fresh_96=vals[96], fresh_192=vals[192], fresh_384=vals[384], fresh_192_repeat=rep["log_marginal"],
            fresh_192_minus_archived=vals[192] - af,
            fresh_96_minus_archived=vals[96] - ac,
            delta_192_96=vals[192] - vals[96],
            delta_384_192=vals[384] - vals[192],
            seconds=times,
            warnings=warns,
            threshold_per_galaxy=1e-4,
            failed_predicate="max_pergal_refinement > 1e-4",
            repeatable_192=abs(vals[192] - rep["log_marginal"]) == 0.0 or abs(vals[192] - rep["log_marginal"]) <= 1e-12,
        ))
        print("CASE", model, galaxy, "d192-96", rows[-1]["delta_192_96"], "d384-192", rows[-1]["delta_384_192"], "repro", rows[-1]["fresh_192_minus_archived"], flush=True)

    # raw versus wrapped timing on 20 baryon galaxies at the accepted vector
    theta = np.load(EX / "baryon_last_accepted_theta.npy")
    L = _layout("baryon", 91, True, None)
    p = _unpack(theta, L)
    raw_times = []
    wrap_times = []
    for gi in range(20):
        g = gs[gi]
        charge("baryon", gi, 192)
        gg = SimpleNamespace(name=g.name, r=g.r_kpc, y=g.v_obs, e=g.v_err, disk=g.v_disk, bulge=g.v_bulge, gas=g.v_gas, D=float(g.D_Mpc), sD=float(g.D_err_Mpc), inc=float(np.deg2rad(g.i_deg)), si=float(np.deg2rad(g.i_err_deg)))
        pp = dict(p, model="baryon", _H0=H0_SI, _c0=10.0, _c_slope=-0.1)
        t = time.perf_counter()
        integrate(gg, pp, gi, "segmented_quantile", 192, floors=("B0",))
        raw_times.append(time.perf_counter() - t)
        charge("baryon", gi, 192)
        t = time.perf_counter()
        galaxy_log_marginal(g, p, gi, "baryon", likelihood_version=PRIMARY_B0, order=192)
        wrap_times.append(time.perf_counter() - t)
    # one full baryon sample timing
    t = time.perf_counter()
    for gi, g in enumerate(gs):
        charge("baryon", gi, 192)
        galaxy_log_marginal(g, p, gi, "baryon", likelihood_version=PRIMARY_B0, order=192)
    full_seconds = time.perf_counter() - t

    # screen identity at ptq-screen accepted theta, not at mond-screen's own terminal
    th = np.load(EX / "ptq_screen_last_accepted_theta.npy")
    mapped = map_theta_ptqscreen_to_mondscreen(th, 91)
    bounds = np.asarray(theta_bounds("mond-screen", 91), float)
    inside = bool(np.all(mapped >= bounds[:, 0]) and np.all(mapped <= bounds[:, 1]))
    max_abs = None
    obj_abs = None
    if inside:
        a = []
        b = []
        Lp = _unpack(th, _layout("ptq-screen", 91, True, None))
        Lm = _unpack(mapped, _layout("mond-screen", 91, True, None))
        for gi, g in enumerate(gs):
            charge("ptq-screen", gi, 192)
            a.append(galaxy_log_marginal(g, Lp, gi, "ptq-screen", likelihood_version=PRIMARY_B0, order=192)["log_marginal"])
            charge("mond-screen", gi, 192)
            b.append(galaxy_log_marginal(g, Lm, gi, "mond-screen", likelihood_version=PRIMARY_B0, order=192)["log_marginal"])
        max_abs = float(np.max(np.abs(np.array(a) - np.array(b))))
        obj_abs = abs(-sum(a) + sum(b))
    # epsilon span for FD floor statement
    spans = {}
    for model, index in [("ptq", 0), ("ptq-nu", 0), ("ptq-nu", 92)]:
        b = np.asarray(theta_bounds(model, 91), float)
        spans[f"{model}:{index}"] = dict(lo=float(b[index, 0]), hi=float(b[index, 1]), span=float(b[index, 1] - b[index, 0]))
    report = dict(
        calls=calls, equivalents=eq, wall_seconds=time.monotonic() - t0,
        cases=rows,
        timing=dict(
            raw_integrate_median_s=float(np.median(raw_times)),
            wrapped_marginal_median_s=float(np.median(wrap_times)),
            raw_20=raw_times,
            wrap_20=wrap_times,
            full91_baryon_192_seconds=full_seconds,
            ledger_note="P1G-A ledger includes per-galaxy JSON checkpoint; these timings do not",
        ),
        screen_mapped_identity=dict(inside_bounds=inside, max_per_galaxy_abs=max_abs, objective_abs=obj_abs, tolerance=1e-8, point="ptq-screen last accepted, mapped to mond-screen"),
        a0_map="a0 = epsilon * c * H0",
        c_H0=C_LIGHT * H0_SI,
        native_spans=spans,
        imports={
            "marginal": __import__("ptquat.marginal_likelihood", fromlist=["x"]).__file__,
        },
    )
    (OUT / "DIAGNOSTIC_RESULTS.json").write_text(json.dumps(report, indent=2) + "\n")
    print("DONE calls", calls, "eq", round(eq, 3), "wall", round(report["wall_seconds"], 1), "screen", max_abs, flush=True)

if __name__ == "__main__":
    main()
