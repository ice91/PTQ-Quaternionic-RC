"""REV26-P1G-A2-ENG split precision diagnostic.

Engineering labels only. Does not activate a scientific acceptance rule.
No optimizer. Counts every galaxy marginal against the A2 caps.
"""
import hashlib, json, os, sys, time, warnings
from pathlib import Path
import numpy as np

CAND = Path("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/mle_pilot_a/candidate/src")
EX = Path("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/mle_pilot_a/execution_v1")
IMPL = Path("/home/rocky/PTQ-Quaternionic-RC")
OUT = Path("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/mle_pilot_a2_eng")
R1 = Path("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/mle_pilot_a_r1_forensic")
R1_REPORT = Path("/home/rocky/ptq-mond/docs/frontiers_major_revision_2026_10/rev26_p1g_a_r1_forensic")
LOG = OUT / "diagnostic" / "split_precision.log"

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
sys.path.insert(0, str(CAND))

MAX_CALLS = 3000
MAX_EQ = 50.0
MAX_WALL = 80 * 60
THRESHOLD = 1e-4
REPEAT_LIMIT = 1e-12

EXPECTED = {
    "models.py": "cba208085f4e5e6373fe3b1bd5b862e9be66a6d1412a6a7750c4c4b053bbc952",
    "fit_global.py": "489a3c6394ffbf41ead56aa88d46cca664a934274b18c508cbbc8c5df82bbe46",
    "mle.py": "1cc2b3eadcc7ae3f713a8c60a891670621d84d39fa18ab3a0d1edde2db14df6a",
    "marginal_likelihood.py": "3766ef2c1385316d1825ea62d72979829e00267f09e951644c58760783b6cbb7",
    "evaluation_contract.py": "b30445b9fd7edefe64f502c3f33f5e39dd6ad737f8de05d0069be7b2bbb6d4e2",
    "dependency_gradient.py": "4117a4c95753a0fbd251513d2556ddd79a0c4b40b7b735401ef89e6883d9039f",
}
PROTECTED = {
    "FIT_PROTOCOL.json": "381b1351c9e75fd93be557cc69fb67378c44a8cbad7d3ff1c1f86110e6e8a0b5",
    "execution_v1/RUN_SUMMARY.json": "600ec887e25155ba5e3ae651b0affc4fcd30fa36b4de0e7b3eeb7af4dd025cf0",
    "execution_v1/baryon_RESULT.json": "f5064c604c24e536a0d2f00172157bdc61743319618db6d6ea2c4e0edc0e0d3f",
    "execution_v1/mond_RESULT.json": "a1e9c5a0c6ce88b2f7508d3c84ed79568ecb2078351c20571baeff44745e15fd",
    "execution_v1/mond_screen_RESULT.json": "ac535a7617454d055f4e4ea031d35f3532a3281c61741b39cc73af24182a34b2",
    "execution_v1/nfw1p_RESULT.json": "4f94c02be4ec355cda29bb7b309714c238bb194104c3e1e012dde597e298ab60",
    "execution_v1/ptq_RESULT.json": "e1d3087dccc872e05663626acef4aac2abd58bcd9416cff0f5b19dcd4969f80f",
    "execution_v1/ptq_nu_RESULT.json": "86ccd5fdd2fe5e86473d2bc01525dd3091724cc94e0e9afc04863a75a75bf314",
    "execution_v1/ptq_screen_RESULT.json": "5c94531d184445f4c4b62528ee4ca5a97344716e10ae673461d9b414873e89d1",
    "execution_v1/B1_FIXED_POINT.json": "3807e2c652d93e6f6cc90b9fae530035224d6dd7f44144896c1c66b4b09e395a",
}
THETA_CONTENT = {
    "baryon": "1e9c848c43e456d6eb233b90ff4796edaf2a30fb7cd9daf274fe40e95f680714",
    "mond": "4c2a9af69596074fc4f81d6240d3e70c926d0c80ab2fa2e4d8c6118f9cd1e40f",
    "ptq-screen": "b0a24278778944f59b3ae440ba3ae5a63d9dec584c851f93a64fa3e693ff3ab4",
}
R1_DIAG = "f4d38824e993784cbb2935c9a1d5836d4bf3aacf9c1e5ff1e3eca26c3222ac8d"

calls = 0
eq = 0.0
t0 = time.monotonic()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def theta_hash(t):
    return hashlib.sha256(np.asarray(t, dtype="<f8").tobytes()).hexdigest()


def log(msg):
    line = msg + "\n"
    with LOG.open("a") as f:
        f.write(line)
        f.flush()
    print(msg, flush=True)


def verify_or_stop():
    mismatches = []
    root = IMPL / "results/frontiers_revision_2026_10/mle_pilot_a"
    for rel, digest in PROTECTED.items():
        got = sha256(root / rel)
        if got != digest:
            mismatches.append({"path": str(root / rel), "expected": digest, "got": got})
    for name, digest in EXPECTED.items():
        path = CAND / "ptquat" / name
        got = sha256(path)
        if got != digest:
            mismatches.append({"path": str(path), "expected": digest, "got": got})
    diag = R1 / "diagnostic" / "DIAGNOSTIC_RESULTS.json"
    got = sha256(diag)
    if got != R1_DIAG:
        mismatches.append({"path": str(diag), "expected": R1_DIAG, "got": got})
    sums = (R1_REPORT / "SHA256SUMS").read_text().splitlines()
    for line in sums:
        digest, rel = line.split(None, 1)
        path = R1_REPORT / rel
        got = sha256(path)
        if got != digest:
            mismatches.append({"path": str(path), "expected": digest, "got": got})
    for model, digest in THETA_CONTENT.items():
        path = EX / f"{model.replace('-', '_')}_last_accepted_theta.npy"
        if not path.is_file():
            mismatches.append({"path": str(path), "expected": digest, "got": "MISSING"})
            continue
        got = theta_hash(np.load(path))
        if got != digest:
            mismatches.append({"path": str(path), "expected": digest, "got": got})
    if mismatches:
        (OUT / "SOURCE_IDENTITY_FAILURE.json").write_text(json.dumps(mismatches, indent=2) + "\n")
        raise SystemExit("source identity failed")
    return {"protected_ok": True, "r1_report_files": len(sums), "theta_content_ok": True}


def main():
    LOG.write_text("")
    identity = verify_or_stop()
    log("IDENTITY OK")
    from ptquat.data import load_tidy_sparc
    from ptquat.fit_global import _unpack, _layout
    from ptquat.marginal_likelihood import galaxy_log_marginal, integrate, PRIMARY_B0
    from ptquat.constants import H0_SI
    import ptquat.marginal_likelihood as ml
    if Path(ml.__file__).resolve() != (CAND / "ptquat" / "marginal_likelihood.py").resolve():
        raise SystemExit(f"import authority failed: {ml.__file__}")
    if PRIMARY_B0 != "REV26-P1E-UNIFORM-EXPLICIT-B0-PHYSICAL-v1":
        raise SystemExit(PRIMARY_B0)
    weights = json.loads((IMPL / "results/frontiers_revision_2026_10/mle_pilot_a/GALAXY_WORK_WEIGHTS.json").read_text())
    gd = load_tidy_sparc(str(IMPL / "dataset/sparc_tidy.csv"))
    gs = [gd[k] for k in sorted(gd)]
    names = [g.name for g in gs]
    n_obs = int(sum(len(g.v_obs) for g in gs))
    if len(gs) != 91 or n_obs != 1782:
        raise SystemExit(f"sample scope {len(gs)} {n_obs}")

    def charge(model, gi, order):
        global calls, eq
        if time.monotonic() - t0 >= MAX_WALL:
            raise RuntimeError("diagnostic wall cap")
        base = weights[model][str(192 if order == 384 else order)][gi]["fullsample_equivalent"]
        unit = base * (1.0 if order in (96, 192) else 384 / 192)
        if calls + 1 > MAX_CALLS or eq + unit > MAX_EQ:
            raise RuntimeError("diagnostic work cap")
        calls += 1
        eq += unit

    def one(g, theta, gi, model, order):
        charge(model, gi, order)
        p = _unpack(theta, _layout(model, 91, True, None))
        with warnings.catch_warnings(record=True) as got:
            warnings.simplefilter("always")
            if order in (96, 192):
                r = galaxy_log_marginal(g, p, gi, model, likelihood_version=PRIMARY_B0, order=order)
                value = float(r["log_marginal"])
            else:
                from types import SimpleNamespace
                gg = SimpleNamespace(
                    name=g.name, r=g.r_kpc, y=g.v_obs, e=g.v_err,
                    disk=g.v_disk, bulge=g.v_bulge, gas=g.v_gas,
                    D=float(g.D_Mpc), sD=float(g.D_err_Mpc),
                    inc=float(np.deg2rad(g.i_deg)), si=float(np.deg2rad(g.i_err_deg)),
                )
                pp = dict(p, model=model, _H0=H0_SI, _c0=10.0, _c_slope=-0.1)
                raw = integrate(gg, pp, gi, "segmented_quantile", order, floors=("B0",))
                value = float(raw["logL"]["B0"])
            warns = [f"{w.category.__name__}: {w.message}" for w in got]
        return value, warns

    fine_rows = []
    coarse = {"threshold_per_galaxy": THRESHOLD, "scientific_objective_order": 192, "order_384_role": "diagnostic_only", "models": {}}
    for model in ("baryon", "mond", "ptq-screen"):
        theta = np.load(EX / f"{model.replace('-', '_')}_last_accepted_theta.npy")
        archived = json.loads((EX / f"{model.replace('-', '_')}_terminal_OBJECTIVE_PRECISION.json").read_text())
        if archived["theta_sha256"] != theta_hash(theta):
            raise SystemExit(f"archived precision theta mismatch {model}")
        arch_fine = archived["fine"]["galaxy_log_marginals"]
        arch_coarse = archived["coarse"]["galaxy_log_marginals"]
        if len(arch_fine) != 91 or len(arch_coarse) != 91:
            raise SystemExit(f"archived scope {model}")
        v96, v192, vrep = [], [], []
        warn_n = 0
        for gi, g in enumerate(gs):
            a, w = one(g, theta, gi, model, 96)
            b, w2 = one(g, theta, gi, model, 192)
            c, w3 = one(g, theta, gi, model, 192)
            v96.append(a)
            v192.append(b)
            vrep.append(c)
            warn_n += len(w) + len(w2) + len(w3)
        v96 = np.asarray(v96)
        v192 = np.asarray(v192)
        vrep = np.asarray(vrep)
        arch_f = np.asarray(arch_fine, float)
        arch_c = np.asarray(arch_coarse, float)
        d_arch = np.max(np.abs(v192 - arch_f))
        d_rep = np.max(np.abs(v192 - vrep))
        d96 = np.max(np.abs(v96 - arch_c))
        obj = float(-np.sum(v192))
        obj_err = abs(obj - float(archived["fine"]["objective"]))
        galaxies = []
        labels = []
        for gi, g in enumerate(gs):
            d = float(v192[gi] - v96[gi])
            rec = {
                "galaxy": g.name,
                "gi": gi,
                "n_points": int(len(g.v_obs)),
                "ell96": float(v96[gi]),
                "ell192": float(v192[gi]),
                "ell192_repeat": float(vrep[gi]),
                "abs_192_96": abs(d),
                "abs_192_repeat": abs(float(v192[gi] - vrep[gi])),
                "abs_192_minus_archived": abs(float(v192[gi] - arch_f[gi])),
            }
            if abs(d) <= THRESHOLD:
                rec["classification"] = "COARSE_REFINEMENT_PASS"
            else:
                e384, w384 = one(g, theta, gi, model, 384)
                warn_n += len(w384)
                d384 = abs(e384 - float(v192[gi]))
                rec.update(ell384=e384, abs_384_192=d384, original_refinement_threshold=THRESHOLD)
                if abs(float(v192[gi] - vrep[gi])) <= REPEAT_LIMIT and d384 <= THRESHOLD:
                    rec["classification"] = "COARSE_INSUFFICIENT_384_DIAGNOSTIC_PASS"
                else:
                    rec["classification"] = "UNRESOLVED_REFINEMENT"
            labels.append(rec["classification"])
            if rec["classification"] != "COARSE_REFINEMENT_PASS":
                galaxies.append(rec)
        fine_label = (
            "REPEATABLE_FINE_TARGET"
            if d_arch <= REPEAT_LIMIT and d_rep <= REPEAT_LIMIT and obj_err <= REPEAT_LIMIT and len(gs) == 91
            else "UNRESOLVED_REFINEMENT"
        )
        fine_rows.append({
            "model": model,
            "theta_sha256": theta_hash(theta),
            "source": "last_accepted_theta.npy",
            "n_galaxies": 91,
            "n_observations": n_obs,
            "likelihood_version": PRIMARY_B0,
            "order": 192,
            "fresh_objective": obj,
            "archived_objective": float(archived["fine"]["objective"]),
            "objective_abs_error": obj_err,
            "max_abs_galaxy_192_minus_archived": float(d_arch),
            "max_abs_192_repeat": float(d_rep),
            "max_abs_96_minus_archived": float(d96),
            "warning_count": warn_n,
            "label": fine_label,
            "scientific_mle_acceptance": "NOT_GRANTED",
            "official_precision_status_unchanged": archived["status"],
        })
        coarse["models"][model] = {
            "n_galaxies": 91,
            "n_coarse_pass": labels.count("COARSE_REFINEMENT_PASS"),
            "n_coarse_insufficient_384": labels.count("COARSE_INSUFFICIENT_384_DIAGNOSTIC_PASS"),
            "n_unresolved": labels.count("UNRESOLVED_REFINEMENT"),
            "failures": galaxies,
            "note": "384 comparison is pointwise and is not a uniform quadrature error bound",
        }
        log(f"MODEL {model} fine={fine_label} fail={len(galaxies)} calls={calls} eq={eq:.4f}")

    budget = {
        "calls": calls,
        "equivalents": eq,
        "wall_seconds": time.monotonic() - t0,
        "caps": {"calls": MAX_CALLS, "equivalents": MAX_EQ, "wall_seconds": MAX_WALL},
        "optimizer_calls": 0,
    }
    env = {
        "python": sys.version,
        "numpy": np.__version__,
        "executable": sys.executable,
        "import_file": str(Path(ml.__file__).resolve()),
        "blas_threads": 1,
        "identity": identity,
    }
    (OUT / "FINE192_REPEAT_RESULTS.json").write_text(json.dumps({
        "task": "REV26-P1G-A2-ENG",
        "record": "FINE192_REPRODUCIBILITY",
        "authoritative_scientific_acceptance": False,
        "environment": env,
        "budget": budget,
        "models": fine_rows,
    }, indent=2) + "\n")
    (OUT / "COARSE_96_192_384_DIAGNOSTICS.json").write_text(json.dumps({
        "task": "REV26-P1G-A2-ENG",
        "record": "COARSE_REFINEMENT_CLASSIFICATION",
        "authoritative_scientific_acceptance": False,
        "labels_are_engineering_diagnostics": True,
        "budget": budget,
        **coarse,
    }, indent=2) + "\n")
    (OUT / "diagnostic" / "RUN_ENVIRONMENT.json").write_text(json.dumps(env, indent=2) + "\n")
    log(f"DONE calls={calls} eq={eq:.4f} wall={budget['wall_seconds']:.1f}")


if __name__ == "__main__":
    main()
