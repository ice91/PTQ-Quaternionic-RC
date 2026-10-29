"""Freeze FIT_PROTOCOL.json and gate records before any optimizer call."""
import hashlib, json, os, subprocess
from pathlib import Path

IMPL = Path("/home/rocky/PTQ-Quaternionic-RC")
PROJ = Path("/home/rocky/ptq-mond")
BUNDLE = IMPL / "results/frontiers_revision_2026_10/mle_pilot_a"
DOCS = PROJ / "docs/frontiers_major_revision_2026_10/rev26_p1g_a_mle_pilot"
OLD = "/Users/rocky/ptq-mond/PTQ-Quaternionic-RC"
NEW = str(IMPL)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def theta_content_hash(path):
    import numpy as np
    t = np.load(path)
    return hashlib.sha256(np.asarray(t, dtype="<f8").tobytes()).hexdigest(), int(t.shape[0])

def git_state(root):
    def run(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()
    head = run("rev-parse", "HEAD")
    branch = run("rev-parse", "--abbrev-ref", "HEAD")
    upstream = run("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}")
    remote = run("remote", "get-url", "origin")
    toplevel = run("rev-parse", "--show-toplevel")
    status = run("status", "--porcelain=v1", "-b")
    ahead = run("rev-list", "--left-right", "--count", "HEAD...@{upstream}")
    return dict(toplevel=toplevel, branch=branch, head=head, upstream=upstream, remote=remote, status=status, ahead_behind=ahead)

starts = {}
names = {
    "baryon": "baryon", "mond": "mond", "ptq": "ptq", "ptq-nu": "ptq_nu",
    "ptq-screen": "ptq_screen", "mond-screen": "mond_screen", "nfw1p": "nfw1p",
}
expected_k = {"baryon": 92, "mond": 93, "ptq": 93, "ptq-nu": 93, "ptq-screen": 94, "mond-screen": 94, "nfw1p": 183}
for model, folder in names.items():
    path = IMPL / "results/frontiers_revision_2026_10/mle_recovery/original_recovered/01_mle" / folder / "theta_mle.npy"
    content, k = theta_content_hash(path)
    if k != expected_k[model]:
        raise SystemExit(f"k mismatch {model}")
    starts[model] = dict(path=str(path), sha256=sha(path), theta_sha256=content, k=k, origin="P1A original_recovered theta_mle.npy")

ref = IMPL / "results/frontiers_revision_2026_10/uniform_target_gate/acceptance_v1/full_sample_all1274.csv"
driver = BUNDLE / "candidate/src/ptquat/scientific_pilot_driver.py"
coords = [
    ("baryon", 0, "first stellar Upsilon, CamB"),
    ("baryon", 9, "F574-1 stellar Upsilon"),
    ("baryon", 91, "ln sigma_sys"),
    ("mond", 0, "log10 a0"),
    ("mond", 10, "F574-1 stellar Upsilon"),
    ("mond", 92, "ln sigma_sys"),
    ("ptq", 0, "epsilon"),
    ("ptq", 10, "F574-1 stellar Upsilon"),
    ("ptq", 92, "ln sigma_sys"),
    ("ptq-nu", 0, "epsilon"),
    ("ptq-nu", 10, "F574-1 stellar Upsilon"),
    ("ptq-nu", 92, "ln sigma_sys"),
    ("ptq-screen", 0, "epsilon; R2 frozen index"),
    ("ptq-screen", 1, "ln q; R2 frozen index"),
    ("ptq-screen", 11, "F574-1 stellar Upsilon; R2 frozen index"),
    ("ptq-screen", 93, "ln sigma_sys; R2 frozen index"),
    ("mond-screen", 0, "log10 a0"),
    ("mond-screen", 1, "ln q"),
    ("mond-screen", 11, "F574-1 stellar Upsilon"),
    ("mond-screen", 93, "ln sigma_sys"),
    ("nfw1p", 0, "first stellar Upsilon; R2 frozen index"),
    ("nfw1p", 91, "first halo log10 M200; R2 frozen index"),
    ("nfw1p", 182, "ln sigma_sys; R2 frozen index"),
]
per = {}
for model, k in expected_k.items():
    nfw = model == "nfw1p"
    per[model] = dict(
        k=k,
        max_gradient_batches=600 if nfw else 400,
        max_accepted_iterations=500 if nfw else 350,
        max_optimizer_wall_seconds=10800 if nfw else 7200,
    )
protocol = {
    "id": "REV26-P1G-A",
    "target": "REV26-P1E-UNIFORM-EXPLICIT-B0-PHYSICAL-v1",
    "sensitivity_target": "REV26-P1E-UNIFORM-EXPLICIT-B1-PHYSICAL-v1",
    "impl_root": str(IMPL),
    "candidate_src": str(BUNDLE / "candidate/src"),
    "requested_macos_impl_root": "/Users/rocky/PTQ-Quaternionic-RC",
    "host_resolution": "This host has no /Users/rocky. Git remotes identify the same repositories under /home/rocky.",
    "execution_order": ["baryon", "mond", "ptq", "ptq-nu", "ptq-screen", "mond-screen", "nfw1p"],
    "starts": starts,
    "start_policy": "Exactly one P1A original recovered vector per native model. R2 smoke terminals are not starts. No replacement after failure.",
    "optimizer": {"method": "L-BFGS-B", "jac": True, "maxls": 20, "maxcor": 10, "ftol": 1e-12, "gtol": 1e-6},
    "gradient": {"evaluator": "P1F-R1 dependency-aware", "relative_step": 1e-4, "scheme": "auto central or inward", "order": 192},
    "per_model": per,
    "budget": {"max_fullsample_equivalents": 25000, "max_actual_galaxy_calls": 3000000, "max_aggregate_seconds": 43200},
    "budget_policy": "Per-model batch, accepted-iteration and wall caps bind the optimizer loop. Preflight, terminal validation and B1 fixed-point evaluations count against the aggregate caps and are not extra optimizer starts. No retry and no budget increase.",
    "precision": {
        "pergal_finecoarse_abs": 1e-4,
        "sample_sumabs": 1e-2,
        "claimed_objective_abs": 1e-8,
        "repeat_pergal_abs": 1e-12,
        "source": "R2 SMOKE_PROTOCOL precision and directions.repeat_pergal_abs; no relaxation",
    },
    "directions": {
        "steps": [1e-4, 5e-5],
        "candidate_vs_independent_scaled_abs": 2e-5,
        "candidate_vs_independent_relative": 1e-6,
        "refinement_scaled_abs": 1e-3,
        "refinement_relative": 1e-3,
        "noise_unresolved_ratio": 0.1,
        "coordinates": [{"model": m, "index": i, "role": role} for m, i, role in coords],
        "selection": "Structural roles and the R2 index set. Not selected from gradient magnitude or fit outcome.",
    },
    "stationarity": {
        "native_projected_gradient_inf_max": 1e-6,
        "scipy_success_is_not_sufficient": True,
        "single_start_is_not_global_mle": True,
        "nfw_multistart_still_required": True,
    },
    "family_identity": {"max_per_galaxy_abs": 1e-8, "pairs": ["ptq-nu~mond", "ptq-screen~mond-screen"]},
    "b1_policy": "After all B0 endpoints are recorded, evaluate every finite in-bound native terminal under B1. Identification B1_AT_B0_TERMINAL_FIXED_POINT. Do not optimize B1. Do not drop a model because its B0 objective is worse. Do not publish a ranking.",
    "p1e_b0_reference": str(ref),
    "p1e_b0_reference_sha256": sha(ref),
    "driver_sha256": sha(driver),
    "verified_dependencies_sha256": sha(BUNDLE / "VERIFIED_DEPENDENCIES.json"),
    "work_weights_sha256": sha(BUNDLE / "GALAXY_WORK_WEIGHTS.json"),
    "dataset_sha256": sha(IMPL / "dataset/sparc_tidy.csv"),
}
(BUNDLE / "FIT_PROTOCOL.json").write_text(json.dumps(protocol, indent=2) + "\n")
(BUNDLE / "FIT_PROTOCOL_SHA256.txt").write_text(hashlib.sha256((BUNDLE / "FIT_PROTOCOL.json").read_bytes()).hexdigest() + "\n")
print("protocol", (BUNDLE / "FIT_PROTOCOL_SHA256.txt").read_text().strip())
print("driver", protocol["driver_sha256"])
print("dataset", protocol["dataset_sha256"])
for m, s in starts.items():
    print(m, s["k"], s["sha256"][:12], s["theta_sha256"][:12])
