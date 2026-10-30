"""No-likelihood scheduler simulation for REV26-P1G-A2-ENG.

Does not import ptquat and does not evaluate a scientific objective.
"""
import hashlib, json, os
from pathlib import Path

OUT = Path("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/mle_pilot_a2_eng")
MOCK = OUT / "scheduler_mock"
FAMILIES = ["baryon", "mond_ptqnu", "ptq_linear", "screen", "nfw1p"]
NON_NFW = ["baryon", "mond_ptqnu", "ptq_linear", "screen"]
OPT = {f: 90 * 60 for f in NON_NFW}
OPT["nfw1p"] = 180 * 60
VAL = {f: 15 * 60 for f in FAMILIES}
CONTINGENCY = 45 * 60
TOTAL = 660 * 60


def atomic_commit(directory, name, payload):
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / name
    tmp = directory / (name + ".tmp")
    data = json.dumps(payload, sort_keys=True).encode()
    with tmp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, final)
    return hashlib.sha256(final.read_bytes()).hexdigest()


class World:
    def __init__(self, tag):
        self.tag = tag
        self.opt = dict(OPT)
        self.val = dict(VAL)
        self.contingency = CONTINGENCY
        self.accepted = {f: None for f in FAMILIES}
        self.trial = {f: None for f in FAMILIES}
        self.events = []
        self.dir = MOCK / tag
        if self.dir.exists():
            raise RuntimeError(f"refusing to overwrite {self.dir}")

    def event(self, **kw):
        self.events.append(kw)

    def consume_opt(self, family, seconds):
        if family not in self.opt:
            raise KeyError(family)
        take = min(self.opt[family], seconds)
        self.opt[family] -= take
        borrowed = 0
        self.event(kind="opt", family=family, requested=seconds, taken=take, remaining_opt=self.opt[family], borrowed_from_nfw=borrowed, borrowed_from_validation=0)
        return take, seconds - take

    def consume_val(self, family, seconds):
        take = min(self.val[family], seconds)
        self.val[family] -= take
        self.event(kind="validation", family=family, requested=seconds, taken=take, remaining_val=self.val[family])
        return take, seconds - take

    def try_borrow(self, dest, source, bucket, seconds):
        self.event(kind="borrow_rejected", dest=dest, source=source, bucket=bucket, seconds=seconds)
        return False

    def propose_trial(self, family, token):
        self.trial[family] = token
        self.event(kind="trial", family=family, token=token, accepted=False)

    def accept(self, family, token, objective):
        if self.trial[family] != token:
            raise RuntimeError("refusing to accept a token that is not the current trial")
        payload = {"family": family, "token": token, "objective": objective, "status": "ACCEPTED"}
        digest = atomic_commit(self.dir, f"{family}_ACCEPTED.json", payload)
        self.accepted[family] = payload
        self.trial[family] = None
        self.event(kind="accepted_commit", family=family, token=token, sha256=digest)

    def budget_limit_during_trial(self, family, token):
        self.propose_trial(family, token)
        self.trial[family] = None
        self.event(kind="budget_limit_discards_trial", family=family, discarded=token, terminal=None if self.accepted[family] is None else self.accepted[family]["token"])

    def crash_after_accept(self, family):
        durable = json.loads((self.dir / f"{family}_ACCEPTED.json").read_text())
        self.event(kind="crash_recovery", family=family, recovered=durable["token"], volatile_trial=self.trial[family])
        return durable

    def snapshot(self):
        return {
            "opt_remaining_s": dict(self.opt),
            "val_remaining_s": dict(self.val),
            "contingency_remaining_s": self.contingency,
            "accepted": {f: None if self.accepted[f] is None else self.accepted[f]["token"] for f in FAMILIES},
            "trial": dict(self.trial),
            "nfw_opt_intact": self.opt["nfw1p"] == OPT["nfw1p"],
            "validation_intact": {f: self.val[f] == VAL[f] for f in FAMILIES},
        }


def run():
    assert sum(OPT.values()) + sum(VAL.values()) + CONTINGENCY == TOTAL
    scenarios = {}
    invariants = []

    def check(name, ok, detail):
        invariants.append({"name": name, "pass": bool(ok), "detail": detail})
        if not ok:
            raise SystemExit(name)

    w = World("normal_completion")
    for fam in FAMILIES:
        w.consume_opt(fam, 60 * 60 if fam != "nfw1p" else 120 * 60)
        w.propose_trial(fam, f"{fam}-A")
        w.accept(fam, f"{fam}-A", 1.0)
        w.consume_val(fam, 10 * 60)
    w.contingency -= 20 * 60
    scenarios["normal_completion"] = {"events": w.events, "end": w.snapshot()}
    check("normal_keeps_nfw_reserve_until_nfw_runs", scenarios["normal_completion"]["events"][0]["family"] == "baryon", "baryon precedes nfw")
    check("normal_accepted_all", all(w.accepted[f]["token"].endswith("-A") for f in FAMILIES), w.snapshot()["accepted"])

    w = World("first_family_cap")
    taken, short = w.consume_opt("baryon", OPT["baryon"] + 60)
    scenarios["first_family_cap"] = {"events": w.events, "end": w.snapshot(), "short_s": short}
    check("first_cap_does_not_borrow", short == 60 and w.opt["nfw1p"] == OPT["nfw1p"] and w.val["baryon"] == VAL["baryon"], w.snapshot())

    w = World("multiple_caps")
    for fam in ("baryon", "mond_ptqnu", "ptq_linear"):
        w.consume_opt(fam, OPT[fam])
    scenarios["multiple_caps"] = {"end": w.snapshot()}
    check("multiple_caps_leave_nfw", w.opt["nfw1p"] == 180 * 60 and w.opt["screen"] == 90 * 60, w.snapshot()["opt_remaining_s"])

    w = World("early_stop")
    w.consume_opt("mond_ptqnu", 10 * 60)
    w.propose_trial("mond_ptqnu", "mond-early")
    w.accept("mond_ptqnu", "mond-early", 2.0)
    leftover = w.opt["mond_ptqnu"]
    scenarios["early_stop"] = {"end": w.snapshot(), "mond_opt_leftover_s": leftover}
    check("early_stop_does_not_redistribute", leftover == 80 * 60 and w.opt["nfw1p"] == 180 * 60 and w.opt["baryon"] == 90 * 60, leftover)

    w = World("aggregate_nearly_exhausted")
    for fam in NON_NFW:
        w.consume_opt(fam, OPT[fam])
        w.consume_val(fam, VAL[fam])
    w.contingency = 0
    used = TOTAL - w.opt["nfw1p"] - w.val["nfw1p"]
    borrow = w.try_borrow("baryon", "nfw1p", "opt", 60)
    scenarios["aggregate_nearly_exhausted"] = {"end": w.snapshot(), "used_s": used, "borrow": borrow}
    check("near_exhaustion_preserves_nfw_and_its_validation", (not borrow) and w.opt["nfw1p"] == 180 * 60 and w.val["nfw1p"] == 15 * 60, w.snapshot())

    w = World("budget_limit_during_gradient")
    w.propose_trial("ptq_linear", "ptq-accepted")
    w.accept("ptq_linear", "ptq-accepted", 3.0)
    w.budget_limit_during_trial("ptq_linear", "ptq-linesearch-not-accepted")
    recovered = json.loads((w.dir / "ptq_linear_ACCEPTED.json").read_text())
    scenarios["budget_limit_during_gradient"] = {"events": w.events, "durable": recovered}
    check("budget_limit_keeps_accepted_not_trial", recovered["token"] == "ptq-accepted" and w.trial["ptq_linear"] is None, recovered)

    w = World("crash_after_accept")
    w.propose_trial("screen", "screen-1")
    w.accept("screen", "screen-1", 4.0)
    w.propose_trial("screen", "screen-torn-trial")
    durable = w.crash_after_accept("screen")
    scenarios["crash_after_accept"] = {"events": w.events, "recovered": durable, "volatile_left_uncommitted": "screen-torn-trial"}
    check("crash_recovers_last_accepted", durable["token"] == "screen-1" and not (w.dir / "screen_ACCEPTED.json.tmp").exists(), durable)

    w = World("validation_reservation")
    w.consume_opt("baryon", OPT["baryon"])
    _, short = w.consume_opt("baryon", 30)
    vtake, vshort = w.consume_val("baryon", 30)
    scenarios["validation_reservation"] = {"end": w.snapshot(), "opt_short": short, "val_taken": vtake}
    check("validation_not_spent_by_optimizer", short == 30 and vtake == 30 and w.val["baryon"] == VAL["baryon"] - 30, w.snapshot())

    w = World("nfw_scheduled_late")
    for fam in NON_NFW:
        w.consume_opt(fam, OPT[fam])
        w.propose_trial(fam, fam + "-final")
        w.accept(fam, fam + "-final", 5.0)
    before = w.opt["nfw1p"]
    w.consume_opt("nfw1p", 60 * 60)
    w.propose_trial("nfw1p", "nfw-accepted")
    w.accept("nfw1p", "nfw-accepted", 6.0)
    scenarios["nfw_scheduled_late"] = {"nfw_before_its_turn_s": before, "end": w.snapshot()}
    check("nfw_late_starts_from_full_reserve", before == 180 * 60 and w.accepted["nfw1p"]["token"] == "nfw-accepted", before)

    durable_token = json.loads((MOCK / "budget_limit_during_gradient" / "ptq_linear_ACCEPTED.json").read_text())["token"]
    check("no_unaccepted_trial_is_durable_terminal", durable_token != "ptq-linesearch-not-accepted", durable_token)
    check("all_invariants", all(row["pass"] for row in invariants), len(invariants))

    (OUT / "SCHEDULER_SIMULATION.json").write_text(json.dumps({
        "task": "REV26-P1G-A2-ENG",
        "likelihood_calls": 0,
        "optimizer_calls": 0,
        "total_seconds": TOTAL,
        "allocation_seconds": {"opt": OPT, "validation": VAL, "contingency": CONTINGENCY},
        "order": FAMILIES,
        "scenarios": scenarios,
    }, indent=2) + "\n")
    (OUT / "SCHEDULER_INVARIANTS.json").write_text(json.dumps({
        "all_pass": all(row["pass"] for row in invariants),
        "checks": invariants,
    }, indent=2) + "\n")
    print("SCHEDULER", len(invariants), "checks", flush=True)


if __name__ == "__main__":
    run()
