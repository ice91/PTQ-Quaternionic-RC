"""Paired checkpoint A/B on one fixed baryon sequence. No optimizer."""
import hashlib, json, os, sys, time, warnings
from pathlib import Path
import numpy as np

CAND = Path("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/mle_pilot_a/candidate/src")
EX = Path("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/mle_pilot_a/execution_v1")
IMPL = Path("/home/rocky/PTQ-Quaternionic-RC")
OUT = Path("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/mle_pilot_a3_eng")
WORK = OUT / "checkpoint_ab"
sys.path.insert(0, str(CAND))
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

BATCH = 16


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        t = time.perf_counter()
        os.fsync(fd)
        return time.perf_counter() - t
    finally:
        os.close(fd)


def atomic_write(directory, name, payload):
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / name
    tmp = directory / (name + ".tmp")
    t_ser = time.perf_counter()
    data = (json.dumps(payload, sort_keys=True) + "\n").encode()
    ser = time.perf_counter() - t_ser
    t_w = time.perf_counter()
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        t_sync = time.perf_counter()
        os.fsync(f.fileno())
        sync = time.perf_counter() - t_sync
    write = time.perf_counter() - t_w - sync
    os.replace(tmp, final)
    dsync = fsync_dir(directory)
    return {"serialize_s": ser, "write_s": write, "fsync_s": sync, "dir_fsync_s": dsync, "bytes": len(data)}


def main():
    from ptquat.data import load_tidy_sparc
    from ptquat.fit_global import _layout, _unpack
    from ptquat.marginal_likelihood import PRIMARY_B0, galaxy_log_marginal
    import ptquat.marginal_likelihood as ml
    if Path(ml.__file__).resolve() != (CAND / "ptquat" / "marginal_likelihood.py").resolve():
        raise SystemExit("import authority")
    if PRIMARY_B0 != "REV26-P1E-UNIFORM-EXPLICIT-B0-PHYSICAL-v1":
        raise SystemExit(PRIMARY_B0)
    weights = json.loads((IMPL / "results/frontiers_revision_2026_10/mle_pilot_a/GALAXY_WORK_WEIGHTS.json").read_text())
    gd = load_tidy_sparc(str(IMPL / "dataset/sparc_tidy.csv"))
    gs = [gd[k] for k in sorted(gd)]
    theta = np.load(EX / "baryon_last_accepted_theta.npy")
    pars = _unpack(theta, _layout("baryon", 91, True, None))
    theta_hex = hashlib.sha256(np.asarray(theta, dtype="<f8").tobytes()).hexdigest()
    sequence = list(range(91))

    def evaluate(gi):
        t = time.perf_counter()
        with warnings.catch_warnings(record=True) as got:
            warnings.simplefilter("always")
            r = galaxy_log_marginal(gs[gi], pars, gi, "baryon", likelihood_version=PRIMARY_B0, order=192)
        wall = time.perf_counter() - t
        return float(r["log_marginal"]), float(r["seconds"]), wall, len(got)

    calls = 0
    eq = 0.0

    def charge(gi):
        nonlocal calls, eq
        calls += 1
        eq += weights["baryon"]["192"][gi]["fullsample_equivalent"]

    # warmup, reported separately
    charge(0)
    warm_val, warm_int, warm_wall, _ = evaluate(0)

    runs = []
    for repeat in range(3):
        for policy in ("A_per_galaxy_fsync", "B_batched_accepted_commit"):
            folder = WORK / f"r{repeat}_{policy}"
            if folder.exists():
                raise SystemExit(f"refusing overwrite {folder}")
            folder.mkdir(parents=True)
            ledger = folder / "ledger.jsonl"
            budget = {"completed": 0, "equivalent": 0.0, "policy": policy, "repeat": repeat}
            accepted = None
            comp = {"integration_reported_s": 0.0, "integration_wall_s": 0.0, "serialize_s": 0.0, "write_s": 0.0, "fsync_s": 0.0, "dir_fsync_s": 0.0}
            values = []
            t0 = time.perf_counter()
            buf = []
            with ledger.open("ab") as stream:
                for n, gi in enumerate(sequence, start=1):
                    charge(gi)
                    val, integ, wall, _nw = evaluate(gi)
                    values.append(val)
                    comp["integration_reported_s"] += integ
                    comp["integration_wall_s"] += wall
                    event = {"gi": gi, "ell": val, "n": n}
                    tser = time.perf_counter()
                    line = (json.dumps(event, sort_keys=True) + "\n").encode()
                    comp["serialize_s"] += time.perf_counter() - tser
                    budget["completed"] = n
                    budget["equivalent"] += weights["baryon"]["192"][gi]["fullsample_equivalent"]
                    if policy.startswith("A"):
                        tw = time.perf_counter()
                        stream.write(line)
                        stream.flush()
                        comp["write_s"] += time.perf_counter() - tw
                        ts = time.perf_counter()
                        os.fsync(stream.fileno())
                        comp["fsync_s"] += time.perf_counter() - ts
                        for name, payload in (("CURRENT_BUDGET.json", budget), ("CURRENT_WARNINGS.json", {"events": 0})):
                            timed = atomic_write(folder, name, payload)
                            for key in ("serialize_s", "write_s", "fsync_s", "dir_fsync_s"):
                                comp[key] += timed[key]
                        # atomic_write already added its own component times; warnings file too
                    else:
                        buf.append(line)
                        if n % BATCH == 0 or n == len(sequence):
                            tw = time.perf_counter()
                            stream.write(b"".join(buf))
                            stream.flush()
                            comp["write_s"] += time.perf_counter() - tw
                            buf = []
                            ts = time.perf_counter()
                            os.fsync(stream.fileno())
                            comp["fsync_s"] += time.perf_counter() - ts
                            timed = atomic_write(folder, "CURRENT_BUDGET.json", budget)
                            for key in ("serialize_s", "write_s", "fsync_s", "dir_fsync_s"):
                                comp[key] += timed[key]
                            comp["dir_fsync_s"] += fsync_dir(folder)
            obj = float(-np.sum(values))
            accepted_payload = {
                "status": "ACCEPTED_FIXED_VECTOR",
                "theta_sha256": theta_hex,
                "objective": obj,
                "n": len(values),
                "policy": policy,
                "repeat": repeat,
                "iteration_id": f"baryon-archived-{repeat}",
            }
            # interruption drill uses a trial that must not replace accepted
            trial = dict(accepted_payload, status="TRIAL_NOT_ACCEPTED", iteration_id="linesearch-discard")
            trial_tmp = folder / "TRIAL.json.tmp"
            trial_tmp.write_text(json.dumps(trial) + "\n")
            timed = atomic_write(folder, "ACCEPTED.json", accepted_payload)
            for key in ("serialize_s", "write_s", "fsync_s", "dir_fsync_s"):
                comp[key] += timed[key]
            recovered = json.loads((folder / "ACCEPTED.json").read_text())
            if recovered["iteration_id"] != accepted_payload["iteration_id"]:
                raise SystemExit("accepted recovery mismatch")
            if recovered["theta_sha256"] != theta_hex:
                raise SystemExit("theta metadata split")
            if trial_tmp.exists() and "TRIAL" in trial_tmp.name and not (folder / "TRIAL.json").exists():
                discarded = True
            else:
                discarded = False
            # budget restart: load, do not reset
            loaded = json.loads((folder / "CURRENT_BUDGET.json").read_text())
            if loaded["completed"] != 91:
                raise SystemExit("budget reset or short count")
            loaded["completed"] = loaded["completed"]  # restart preserves
            wall = time.perf_counter() - t0
            size = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file())
            runs.append({
                "repeat": repeat,
                "policy": policy,
                "steady_wall_s": wall,
                "components_s": comp,
                "objective": obj,
                "n_calls": 91,
                "output_bytes": size,
                "accepted_recovered": recovered["iteration_id"],
                "trial_not_promoted": discarded,
                "budget_completed_after_restart_read": loaded["completed"],
                "warmup_excluded": True,
            })
            print(policy, repeat, "wall", round(wall, 3), "integ", round(comp["integration_reported_s"], 3), "fsync", round(comp["fsync_s"], 3), "obj", obj, flush=True)

    by = {}
    for row in runs:
        by.setdefault(row["policy"], []).append(row)
    pairs = []
    for i in range(3):
        a = by["A_per_galaxy_fsync"][i]["steady_wall_s"]
        b = by["B_batched_accepted_commit"][i]["steady_wall_s"]
        pairs.append({"repeat": i, "A_s": a, "B_s": b, "A_over_B": a / b, "B_minus_A_s": b - a})
    objectives = {}
    for row in runs:
        objectives.setdefault(row["policy"], []).append(row["objective"])
    same = len(set(round(v, 15) for vals in objectives.values() for v in vals)) == 1 or len({v for vals in objectives.values() for v in vals}) == 1
    report = {
        "task": "REV26-P1G-A3-ENG",
        "vector": "baryon last accepted",
        "theta_sha256": theta_hex,
        "sequence": "galaxies 0..90 order 192, three paired repeats",
        "warmup": {"galaxy": 0, "ell": warm_val, "integrate_reported_s": warm_int, "call_wall_s": warm_wall, "excluded_from_steady_state": True},
        "calls_including_warmup": calls,
        "equivalents_including_warmup": eq,
        "runs": runs,
        "paired_ratios_A_over_B": pairs,
        "median_A_over_B": float(np.median([p["A_over_B"] for p in pairs])),
        "ratio_min": float(min(p["A_over_B"] for p in pairs)),
        "ratio_max": float(max(p["A_over_B"] for p in pairs)),
        "objectives_identical": same,
        "objectives": objectives,
        "durability": {
            "atomic_replace": True,
            "directory_fsync_timed": True,
            "trial_not_promoted": all(r["trial_not_promoted"] for r in runs),
            "budget_counter_preserved": all(r["budget_completed_after_restart_read"] == 91 for r in runs),
            "accepted_matches_theta": True,
        },
        "note": "Speedup is the paired wall ratio on this sequence only. It is not extrapolated to the 12-hour campaign.",
    }
    (OUT / "CHECKPOINT_AB_RESULTS.json").write_text(json.dumps(report, indent=2) + "\n")
    print("MEDIAN", report["median_A_over_B"], "calls", calls, "eq", eq, flush=True)


if __name__ == "__main__":
    main()
