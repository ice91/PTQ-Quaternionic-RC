"""REV26-P1G-A isolated scientific pilot.

Byte-identical P1F-R2 likelihood and dependency gradient. One original P1A
start per native model. No canonical installation and no legacy import.
"""
from pathlib import Path
import os, json, time, hashlib, traceback, signal, csv
from collections import Counter
import numpy as np
from scipy.optimize import minimize
from .data import load_tidy_sparc
from .fit_global import _layout, _unpack, log_likelihood_per_galaxy
from .mle import (
    theta_bounds, map_theta_ptqnu_to_mond, map_theta_mond_to_ptqnu,
    map_theta_ptqscreen_to_mondscreen, map_theta_mondscreen_to_ptqscreen,
)
from .evaluation_contract import validate_native
from .dependency_gradient import DependencyGradient, step_stencil, source_hashes
from .marginal_likelihood import PRIMARY_B0, SENSITIVITY_B1, galaxy_log_marginal
from .warning_telemetry import WarningCollector, collect_warnings

LABEL = "REV26-P1G-A-SINGLE-START — NOT GLOBAL MLE"
MODELS = ("baryon", "mond", "ptq", "ptq-nu", "ptq-screen", "mond-screen", "nfw1p")
FAMILY = {
    "ptq-nu": ("mond", map_theta_ptqnu_to_mond),
    "mond": ("ptq-nu", map_theta_mond_to_ptqnu),
    "ptq-screen": ("mond-screen", map_theta_ptqscreen_to_mondscreen),
    "mond-screen": ("ptq-screen", map_theta_mondscreen_to_ptqscreen),
}


class BudgetLimit(RuntimeError):
    pass


class PrecisionFailure(RuntimeError):
    pass


class SourceFailure(RuntimeError):
    pass


def theta_hash(t):
    return hashlib.sha256(np.asarray(t, dtype="<f8").tobytes()).hexdigest()


def serial(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    raise TypeError(type(v).__name__)


def save(path, record):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w") as f:
        json.dump(record, f, indent=2, allow_nan=False, default=serial)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def decode_record(t, gs, model):
    t, L = validate_native(t, gs, model, likelihood_version=PRIMARY_B0, hard_bounds=True)
    p = _unpack(t, L)
    bounds = np.asarray(theta_bounds(model, len(gs)), float)
    return dict(
        theta=t.tolist(), theta_sha256=theta_hash(t),
        decoded={k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in p.items()},
        bounds=bounds.tolist(),
        exact_bound_contacts=np.flatnonzero((t == bounds[:, 0]) | (t == bounds[:, 1])).tolist(),
        likelihood_version=PRIMARY_B0, parameter_count=int(L["k"]),
    )


def kkt(theta, gradient, model, n_galaxies):
    x = np.asarray(theta, float)
    g = np.asarray(gradient, float)
    bounds = np.asarray(theta_bounds(model, n_galaxies), float)
    span = bounds[:, 1] - bounds[:, 0]
    pg = g.copy()
    lower = x <= bounds[:, 0]
    upper = x >= bounds[:, 1]
    pg[lower & (g > 0)] = 0.0
    pg[upper & (g < 0)] = 0.0
    scaled = pg * span
    return dict(
        native_projected_gradient_inf_norm=float(np.max(np.abs(pg))),
        scaled_projected_gradient_inf_norm=float(np.max(np.abs(scaled))),
        native_projected_gradient=pg.tolist(),
        scaled_projected_gradient=scaled.tolist(),
        exact_bound_contacts=np.flatnonzero(lower | upper).tolist(),
        lower_contacts=np.flatnonzero(lower).tolist(),
        upper_contacts=np.flatnonzero(upper).tolist(),
        stationarity_rule="bound-aware projected gradient; active lower and g>0 or upper and g<0 set to zero",
    )


class Telemetry(WarningCollector):
    def __init__(self):
        super().__init__()
        self.active_native_hash = None

    def capture(self, *args, **kwargs):
        super().capture(*args, **kwargs)
        for r in self.records.values():
            if "representative_native_theta_hash" not in r:
                r["representative_native_theta_hash"] = self.active_native_hash


class WorkLedger:
    def __init__(self, bundle, out, protocol, collector):
        self.out = out
        self.protocol = protocol
        self.collector = collector
        self.weights = json.loads((bundle / "GALAXY_WORK_WEIGHTS.json").read_text())
        self.start = time.monotonic()
        self.actual = 0
        self.completed = 0
        self.equivalent = 0.0
        self.nodes = 0
        self.completed_nodes = 0
        self.pending = None
        self.requests = []
        self.request = None
        self.optimizer_batches = Counter()
        self.objective_requests = 0
        self.gradient_requests = 0
        self.model_counts = Counter()
        self.phase_counts = Counter()
        self.last_verified = None
        self.last_numerically_completed = None
        self.model_origin = None
        self.stream = (out / "galaxy_work_ledger.jsonl").open("a")
        self.checkpoint()

    def guard(self, model=None):
        elapsed = time.monotonic() - self.start
        if elapsed >= self.protocol["budget"]["max_aggregate_seconds"]:
            raise BudgetLimit("aggregate wall cap")
        if model is not None and self.request is not None and self.request["phase"] == "optimizer":
            origin = self.model_origin
            cap = self.protocol["per_model"][model]["max_optimizer_wall_seconds"]
            if origin is not None and time.monotonic() - origin >= cap:
                raise BudgetLimit("per-model optimizer wall cap")

    def checkpoint(self):
        save(self.out / "CURRENT_BUDGET.json", self.summary())
        save(self.out / "CURRENT_WARNINGS.json", self.collector.as_dict())

    def summary(self):
        return dict(
            actual_galaxy_calls=self.actual,
            completed_galaxy_calls=self.completed,
            charged_fullsample_equivalents=self.equivalent,
            charged_density_nodes=self.nodes,
            completed_density_nodes=self.completed_nodes,
            elapsed_wall_seconds=time.monotonic() - self.start,
            objective_requests=self.objective_requests,
            gradient_requests=self.gradient_requests,
            optimizer_batches=dict(self.optimizer_batches),
            model_galaxy_calls=dict(self.model_counts),
            phase_galaxy_calls=dict(self.phase_counts),
            last_pending=self.pending,
            last_verified_native_input=self.last_verified,
            last_numerically_completed_theta=self.last_numerically_completed,
            limit_definitions=self.protocol["budget"],
        )

    def begin(self, model, theta, phase, kind, order):
        self.guard(model if phase == "optimizer" else None)
        if self.pending is not None:
            raise RuntimeError("Unfinished prior marginal before new request")
        if phase == "optimizer":
            cap = self.protocol["per_model"][model]["max_gradient_batches"]
            if self.optimizer_batches[model] >= cap:
                raise BudgetLimit("per-model optimizer gradient-batch cap")
            self.optimizer_batches[model] += 1
        self.objective_requests += 1
        if kind == "gradient":
            self.gradient_requests += 1
        self.request = dict(
            id=len(self.requests) + 1, model=model, phase=phase, kind=kind, order=order,
            theta_sha256=theta_hash(theta), theta=np.asarray(theta).tolist(),
            likelihood_version=PRIMARY_B0, start_elapsed=time.monotonic() - self.start,
            start_actual=self.actual, start_equivalent=self.equivalent, status="STARTED",
        )
        self.requests.append(self.request)
        self.last_verified = dict(model=model, theta_hash=theta_hash(theta), theta=np.asarray(theta).tolist(), phase=phase)
        save(self.out / "CURRENT_REQUEST.json", self.request)
        self.checkpoint()
        return self.request

    def charge(self, model, gi, order, native_hash, likelihood_version=PRIMARY_B0):
        if self.pending is not None:
            self.completed += 1
            self.completed_nodes += self.pending["density_nodes"]
            self.pending = None
        self.guard(model if self.request["phase"] == "optimizer" else None)
        w = self.weights[model][str(order)][gi]
        if self.actual + 1 > self.protocol["budget"]["max_actual_galaxy_calls"]:
            raise BudgetLimit("aggregate actual-galaxy cap")
        if self.equivalent + w["fullsample_equivalent"] > self.protocol["budget"]["max_fullsample_equivalents"]:
            raise BudgetLimit("aggregate full-sample-equivalent cap")
        self.actual += 1
        self.equivalent += w["fullsample_equivalent"]
        nodes = w["nD"] * w["ni"]
        self.nodes += nodes
        self.model_counts[model] += 1
        self.phase_counts[self.request["phase"]] += 1
        self.collector.active_native_hash = native_hash
        event = dict(
            label=LABEL, id=self.actual, request_id=self.request["id"], model=model, gi=gi,
            galaxy=w["galaxy"], order=order, native_theta_hash=native_hash,
            likelihood_version=likelihood_version, equivalent=w["fullsample_equivalent"],
            density_nodes=nodes, status="CALL_STARTED", elapsed=time.monotonic() - self.start,
        )
        self.pending = event
        self.stream.write(json.dumps(event, allow_nan=False) + "\n")
        self.stream.flush()
        self.checkpoint()

    def finish(self, result=None, error=None):
        if error is None and self.pending is not None:
            self.completed += 1
            self.completed_nodes += self.pending["density_nodes"]
            self.pending = None
        self.request.update(
            status="COMPLETE" if error is None else "FAILED",
            end_elapsed=time.monotonic() - self.start,
            actual_galaxy_calls=self.actual - self.request["start_actual"],
            equivalent_work=self.equivalent - self.request["start_equivalent"],
        )
        if error is not None:
            self.request["error"] = repr(error)
        if result is not None and "objective" in result:
            self.request["objective"] = result["objective"]
            self.last_numerically_completed = dict(
                model=self.request["model"], theta_sha256=self.request["theta_sha256"],
                theta=self.request["theta"], objective=result["objective"],
            )
        save(self.out / "CURRENT_REQUEST.json", self.request)
        save(self.out / "REQUESTS.json", dict(records=self.requests))
        self.checkpoint()

    def close(self):
        self.checkpoint()
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()


class Runtime:
    def __init__(self, bundle, out, protocol, gs, collector):
        self.bundle = bundle
        self.out = out
        self.protocol = protocol
        self.gs = gs
        self.collector = collector
        self.ledger = WorkLedger(bundle, out, protocol, collector)
        self.engines = {m: DependencyGradient(gs, m, bundle / "VERIFIED_DEPENDENCIES.json") for m in MODELS}
        self.gradient_records = {}
        self.results = []
        self.point_vectors = {}
        self.preflight = {}

    def gradient(self, theta, model, phase, order=192):
        x, _ = validate_native(np.asarray(theta, float).copy(), self.gs, model, likelihood_version=PRIMARY_B0, hard_bounds=True)
        engine = self.engines[model]
        req = self.ledger.begin(model, x, phase, "gradient", order)

        def plan():
            yield from ((gi, x) for gi in range(91))
            for row in engine.rows:
                j = row["index"]
                st, _ = step_stencil(x[j], row["bounds"], 1e-4)
                for name, value in st:
                    if name == "base":
                        continue
                    y = x.copy()
                    y[j] = value
                    for gi in row["affected_indices"]:
                        yield gi, y

        planned = iter(plan())

        def tick(m, gi, o):
            expected_gi, y = next(planned)
            if m != model or gi != expected_gi or o != order:
                raise SourceFailure("Unexpected dependency evaluation path")
            h = theta_hash(y)
            self.point_vectors.setdefault(h, y.tolist())
            self.ledger.charge(m, gi, o, h)

        try:
            r = engine.evaluate(x, relative_step=1e-4, scheme="auto", order=order, work_callback=tick)
            if next(planned, None) is not None:
                raise SourceFailure("Incomplete dependency evaluation path")
            g = np.asarray(r["gradient_native"], float)
            if g.shape != x.shape or not np.isfinite(g).all() or not np.isfinite(r["objective"]) or r["requested_indices"] != list(range(len(x))):
                raise PrecisionFailure("Invalid native objective/gradient shape/order/value")
            r.update(input_theta=x.tolist(), phase=phase, request_id=req["id"], native_theta_hash=theta_hash(x))
            self.ledger.finish(r)
            slim = {k: v for k, v in r.items() if k != "evaluations"}
            save(self.out / f"gradient_{req['id']:04d}_{model.replace('-', '_')}.json", slim)
            self.gradient_records[(model, theta_hash(x), phase)] = r
            self.gradient_records[(model, theta_hash(x))] = r
            self.ledger.guard()
            return r
        except BaseException as exc:
            self.ledger.finish(error=exc)
            raise

    def pure(self, theta, model, phase, order, likelihood_version=PRIMARY_B0):
        x, L = validate_native(np.asarray(theta, float).copy(), self.gs, model, likelihood_version=likelihood_version, hard_bounds=True)
        req = self.ledger.begin(model, x, phase, "objective", order)
        if likelihood_version != PRIMARY_B0:
            req["likelihood_version"] = likelihood_version
        p = _unpack(x, L)
        values = []
        try:
            for gi, g in enumerate(self.gs):
                self.ledger.charge(model, gi, order, theta_hash(x), likelihood_version)
                r = galaxy_log_marginal(g, p, gi, model, likelihood_version=likelihood_version, order=order)
                values.append(r["log_marginal"])
            a = np.asarray(values, float)
            if a.shape != (91,) or not np.isfinite(a).all():
                raise PrecisionFailure("Invalid full objective decomposition")
            result = dict(
                objective=-float(a.sum()), galaxy_log_marginals=values, theta=x.tolist(),
                theta_sha256=theta_hash(x), likelihood_version=likelihood_version, order=order,
                request_id=req["id"], decomposition_error=abs(-float(a.sum()) + float(a.sum())),
            )
            self.ledger.finish(result)
            save(self.out / f"objective_{req['id']:04d}_{model.replace('-', '_')}.json", result)
            self.ledger.guard()
            return result
        except BaseException as exc:
            self.ledger.finish(error=exc)
            raise

    def objective_precision(self, theta, model, phase, candidate_objective):
        x = np.asarray(theta, float)
        fine = self.pure(x, model, phase + "_fine", 192)
        coarse = self.pure(x, model, phase + "_coarse", 96)
        repeat = self.pure(x, model, phase + "_repeat", 192)
        a = np.array(fine["galaxy_log_marginals"])
        b = np.array(coarse["galaxy_log_marginals"])
        delta = a - b
        p = self.protocol["precision"]
        rep = float(np.max(np.abs(a - repeat["galaxy_log_marginals"])))
        metrics = dict(
            max_pergal_refinement=float(np.max(np.abs(delta))),
            sumabs_refinement=float(np.sum(np.abs(delta))),
            repeat_max=rep,
            claimed_objective_error=abs(fine["objective"] - candidate_objective),
            decomposition_error=abs(fine["objective"] + float(a.sum())),
        )
        ok = (
            metrics["max_pergal_refinement"] <= p["pergal_finecoarse_abs"]
            and metrics["sumabs_refinement"] <= p["sample_sumabs"]
            and rep <= p["repeat_pergal_abs"]
            and metrics["claimed_objective_error"] <= p["claimed_objective_abs"]
            and metrics["decomposition_error"] <= p["claimed_objective_abs"]
        )
        record = dict(model=model, phase=phase, theta_sha256=theta_hash(x), metrics=metrics, status="PASS" if ok else "FAIL", fine=fine, coarse=coarse, repeat=repeat)
        save(self.out / f"{model.replace('-', '_')}_{phase}_OBJECTIVE_PRECISION.json", record)
        if not ok:
            raise PrecisionFailure("Independent objective/refinement/repeat check failed")
        return record

    def direction_audit(self, theta, model, phase, candidate_gradient):
        x = np.asarray(theta, float)
        ct = self.protocol["directions"]
        indices = [d["index"] for d in ct["coordinates"] if d["model"] == model]
        engine = self.engines[model]
        directions = []
        base = {}
        base[192] = self.pure(x, model, phase + "_dirbase", 192)
        base[96] = self.pure(x, model, phase + "_dirbase_coarse", 96)
        stencilcache = {}
        span_of = {row["index"]: row["bounds"][1] - row["bounds"][0] for row in engine.rows}
        for j in indices:
            row = engine.rows[j]
            span = span_of[j]
            records = {}
            for h in ct["steps"]:
                st, kind = step_stencil(x[j], row["bounds"], h)
                for order in (192, 96):
                    f = []
                    affected = []
                    for name, value in st:
                        if name == "base":
                            r = base[order]
                        else:
                            key = (j, float(value), order)
                            if key not in stencilcache:
                                y = x.copy()
                                y[j] = value
                                stencilcache[key] = self.pure(y, model, phase + "_direction", order)
                            r = stencilcache[key]
                        f.append(r["objective"])
                        affected.append(-float(np.sum(np.array(r["galaxy_log_marginals"])[row["affected_indices"]])))
                    denom = st[1][1] - st[0][1]
                    g = (f[1] - f[0]) / denom
                    localg = (affected[1] - affected[0]) / denom
                    records[(h, order)] = dict(
                        index=j, name=row["name"], h_scaled=h, order=order, scheme=kind,
                        native_gradient=g, scaled_gradient=g * span, affected_scaled_gradient=localg * span,
                        stencil_affected_objectives=affected, denominator=denom,
                    )
            c = float(candidate_gradient[j]) * span
            ind = records[(1e-4, 192)]["scaled_gradient"]
            lim = ct["candidate_vs_independent_scaled_abs"] + ct["candidate_vs_independent_relative"] * max(abs(c), abs(ind))
            candidate_ok = abs(c - ind) <= lim
            checks = []
            for h in ct["steps"]:
                r1 = records[(h, 192)]
                r2 = records[(h, 96)]
                v1 = r1["affected_scaled_gradient"]
                v2 = r2["affected_scaled_gradient"]
                lim2 = ct["refinement_scaled_abs"] + ct["refinement_relative"] * max(abs(v1), abs(v2))
                noise = sum(abs(np.array(r1["stencil_affected_objectives"]) - r2["stencil_affected_objectives"]))
                numerator = abs(r1["stencil_affected_objectives"][1] - r1["stencil_affected_objectives"][0])
                ratio = noise / max(numerator, np.finfo(float).tiny)
                checks.append(dict(kind="fine_coarse", h=h, difference=abs(v1 - v2), limit=lim2, noise_ratio=ratio, pass_test=bool(abs(v1 - v2) <= lim2 and ratio < ct["noise_unresolved_ratio"])))
            for order in (192, 96):
                v1 = records[(ct["steps"][0], order)]["affected_scaled_gradient"]
                v2 = records[(ct["steps"][1], order)]["affected_scaled_gradient"]
                lim2 = ct["refinement_scaled_abs"] + ct["refinement_relative"] * max(abs(v1), abs(v2))
                checks.append(dict(kind="two_steps", order=order, difference=abs(v1 - v2), limit=lim2, pass_test=bool(abs(v1 - v2) <= lim2)))
            directions.append(dict(
                index=j, name=row["name"], candidate_scaled_gradient=c, independent_scaled_gradient=ind,
                candidate_difference=abs(c - ind), candidate_limit=lim, candidate_pass=bool(candidate_ok),
                checks=checks, pass_direction=bool(candidate_ok and all(z["pass_test"] for z in checks)),
            ))
        status = "PASS" if directions and all(d["pass_direction"] for d in directions) else "FAIL"
        result = dict(model=model, phase=phase, theta_sha256=theta_hash(x), directions=directions, status=status)
        save(self.out / f"{model.replace('-', '_')}_{phase}_DIRECTION_AUDIT.json", result)
        return result

    def family_pair(self, theta, model, phase):
        if model not in FAMILY:
            return dict(model=model, status="NOT_A_MATCHED_FAMILY", phase=phase)
        other, mapper = FAMILY[model]
        mapped = np.asarray(mapper(np.asarray(theta, float), len(self.gs)), float)
        try:
            validate_native(mapped, self.gs, other, likelihood_version=PRIMARY_B0, hard_bounds=True)
        except ValueError as exc:
            record = dict(model=model, counterpart=other, phase=phase, status="OUTSIDE_COUNTERPART_BOUNDS", error=str(exc))
            save(self.out / f"{model.replace('-', '_')}_{phase}_FAMILY.json", record)
            return record
        a = self.pure(theta, model, phase + "_family", 192)
        b = self.pure(mapped, other, phase + "_family_mapped", 192)
        da = np.array(a["galaxy_log_marginals"])
        db = np.array(b["galaxy_log_marginals"])
        err = float(np.max(np.abs(da - db)))
        tol = self.protocol["family_identity"]["max_per_galaxy_abs"]
        record = dict(
            model=model, counterpart=other, phase=phase, max_per_galaxy_abs=err,
            objective_abs=abs(a["objective"] - b["objective"]), tolerance=tol,
            status="PASS" if err <= tol else "FAIL", mapped_theta_sha256=theta_hash(mapped),
        )
        save(self.out / f"{model.replace('-', '_')}_{phase}_FAMILY.json", record)
        return record

    def optimize_once(self, t0, model, initial_objective):
        start = time.monotonic()
        self.ledger.model_origin = start
        t0 = np.array(t0, float, copy=True)
        accepted = []
        trials = []
        result = None
        failure = None
        terminal = t0.copy()
        status = "NOT_STARTED"
        cap = self.protocol["per_model"][model]["max_accepted_iterations"]
        save(self.out / (model.replace("-", "_") + "_INITIAL.json"), decode_record(t0, self.gs, model))
        np.save(self.out / (model.replace("-", "_") + "_initial_theta.npy"), t0)

        def fun(theta):
            attempt = dict(theta=np.array(theta).tolist(), theta_sha256=theta_hash(theta), status="REQUESTED", elapsed=time.monotonic() - self.ledger.start)
            trials.append(attempt)
            save(self.out / (model.replace("-", "_") + "_TRIALS.json"), dict(records=trials))
            try:
                r = self.gradient(theta, model, "optimizer", 192)
            except BaseException as exc:
                attempt.update(status="FAILED", error=repr(exc))
                save(self.out / (model.replace("-", "_") + "_TRIALS.json"), dict(records=trials))
                raise
            attempt.update(request_id=r["request_id"], objective=r["objective"], gradient=r["gradient_native"], status="COMPLETE_NOT_YET_ACCEPTED")
            save(self.out / (model.replace("-", "_") + "_TRIALS.json"), dict(records=trials))
            print("OPTIMIZER G", model, len(trials), "F", r["objective"], "eq", round(self.ledger.equivalent, 3), "wall", round(time.monotonic() - self.ledger.start, 2), flush=True)
            return r["objective"], np.array(r["gradient_native"], float)

        def callback(xk):
            x = np.array(xk, float, copy=True)
            record = self.gradient_records.get((model, theta_hash(x)))
            if record is None or record["phase"] != "optimizer":
                raise SourceFailure("Accepted callback has no exact optimizer request")
            accepted.append(dict(theta=x.tolist(), theta_sha256=theta_hash(x), request_id=record["request_id"], objective=record["objective"], advanced=bool(np.any(x != t0)), elapsed=time.monotonic() - self.ledger.start))
            save(self.out / (model.replace("-", "_") + "_ACCEPTED_CALLBACKS.json"), dict(records=accepted))
            np.save(self.out / (model.replace("-", "_") + "_last_accepted_theta.npy"), x)
            print("ACCEPTED", model, len(accepted), "F", record["objective"], flush=True)
            if len(accepted) > cap:
                raise BudgetLimit("accepted iteration cap exceeded")

        opts = self.protocol["optimizer"]
        try:
            result = minimize(
                fun, t0, method="L-BFGS-B", jac=True, bounds=theta_bounds(model, 91), callback=callback,
                options=dict(maxiter=cap, maxls=opts["maxls"], maxcor=opts["maxcor"], ftol=opts["ftol"], gtol=opts["gtol"]),
            )
            terminal = np.array(result.x, float, copy=True)
            record = self.gradient_records.get((model, theta_hash(terminal)))
            if record is None or record["phase"] != "optimizer":
                raise SourceFailure("Returned terminal is not an exact completed optimizer point")
            if accepted and theta_hash(terminal) != accepted[-1]["theta_sha256"]:
                raise SourceFailure("Returned terminal differs from accepted callback")
            status = "OPTIMIZER_RETURNED"
            summary = dict(
                model=model, status=status, accepted_callbacks=len(accepted),
                terminal=decode_record(terminal, self.gs, model),
                scipy=dict(success=bool(result.success), status=int(result.status), message=str(result.message), nit=int(result.nit), nfev=int(result.nfev), njev=int(result.njev), fun=float(result.fun)),
                optimizer_wall_seconds=time.monotonic() - start,
                initial_objective=initial_objective,
                terminal_request_objective=record["objective"],
                gradient_native=record["gradient_native"],
            )
        except BaseException as exc:
            failure = dict(type=type(exc).__name__, error=repr(exc), traceback=traceback.format_exc())
            if accepted:
                terminal = np.array(accepted[-1]["theta"], float)
            status = "BUDGET_LIMITED" if isinstance(exc, BudgetLimit) else ("BLOCKED" if isinstance(exc, (SourceFailure, PrecisionFailure)) else "OPTIMIZER_FAILURE")
            record = self.gradient_records.get((model, theta_hash(terminal)))
            summary = dict(
                model=model, status=status, accepted_callbacks=len(accepted), failure=failure,
                terminal=decode_record(terminal, self.gs, model),
                optimizer_wall_seconds=time.monotonic() - start,
                initial_objective=initial_objective,
                terminal_request_objective=None if record is None else record["objective"],
                gradient_native=None if record is None else record["gradient_native"],
                scipy=None if result is None else dict(success=bool(result.success), status=int(result.status), message=str(result.message), nit=int(result.nit), nfev=int(result.nfev), njev=int(result.njev), fun=float(result.fun)),
            )
        finally:
            for t in trials:
                t["acceptance"] = (
                    "ACCEPTED_CALLBACK" if any(t["theta_sha256"] == a["theta_sha256"] for a in accepted)
                    else ("INITIAL_REQUEST" if t["theta_sha256"] == theta_hash(t0) else "NOT_ACCEPTED_LINESEARCH_TRIAL")
                )
            save(self.out / (model.replace("-", "_") + "_TRIALS.json"), dict(records=[{k: v for k, v in t.items() if k != "gradient"} for t in trials]))
            self.ledger.checkpoint()
        summary["trials_retained"] = len(trials)
        self._accept_terminal(summary, terminal, model, initial_objective)
        self.results.append(summary)
        save(self.out / (model.replace("-", "_") + "_RESULT.json"), summary)
        print("RUN RESULT", model, summary["status"], summary.get("classification"), flush=True)
        return summary

    def _accept_terminal(self, summary, terminal, model, initial_objective):
        record = self.gradient_records.get((model, theta_hash(terminal)))
        summary["terminal_precision_validated"] = False
        summary["two_step_pass"] = None
        summary["family_status"] = None
        if record is None or record.get("gradient_native") is None:
            summary["classification"] = "OPTIMIZER_FAILURE" if summary["status"] == "OPTIMIZER_FAILURE" else summary["status"]
            if summary["status"] == "BUDGET_LIMITED":
                summary["classification"] = "BUDGET_LIMITED"
            return
        try:
            prec = self.objective_precision(terminal, model, "terminal", record["objective"])
            summary["terminal_precision_validated"] = True
            summary["terminal_objective"] = prec["fine"]["objective"]
            summary["objective_improvement"] = float(initial_objective - prec["fine"]["objective"])
            directions = self.direction_audit(terminal, model, "terminal", record["gradient_native"])
            summary["two_step_pass"] = directions["status"] == "PASS"
            family = self.family_pair(terminal, model, "terminal")
            summary["family_status"] = family["status"]
            diag = kkt(terminal, record["gradient_native"], model, len(self.gs))
            summary["kkt"] = diag
            gtol = self.protocol["stationarity"]["native_projected_gradient_inf_max"]
            stationary = diag["native_projected_gradient_inf_norm"] <= gtol
            summary["stationary_under_frozen_gtol"] = bool(stationary)
            np.save(self.out / (model.replace("-", "_") + "_terminal_theta.npy"), np.asarray(terminal, float))
            with (self.out / (model.replace("-", "_") + "_terminal_galaxy_log_marginals.csv")).open("w") as f:
                w = csv.writer(f)
                w.writerow(["galaxy", "log_marginal", "likelihood_version", "order"])
                for g, val in zip(self.gs, prec["fine"]["galaxy_log_marginals"]):
                    w.writerow([g.name, f"{val:.16g}", PRIMARY_B0, 192])
            family_ok = family["status"] in ("PASS", "NOT_A_MATCHED_FAMILY", "OUTSIDE_COUNTERPART_BOUNDS")
            if summary["status"] == "BUDGET_LIMITED":
                summary["classification"] = "BUDGET_LIMITED"
            elif not summary["two_step_pass"] or not family_ok:
                summary["classification"] = "GRADIENT_OR_STATIONARITY_UNRESOLVED"
            elif stationary and family_ok:
                summary["classification"] = "SCIENTIFICALLY_VALIDATED_SINGLE_START_CANDIDATE"
            else:
                summary["classification"] = "NUMERICALLY_VALID_PROVISIONAL_TERMINAL"
            summary["global_mle"] = False
        except BudgetLimit as exc:
            summary["status"] = "BUDGET_LIMITED"
            summary["classification"] = "BUDGET_LIMITED"
            summary["terminal_validation_error"] = repr(exc)
        except PrecisionFailure as exc:
            summary["classification"] = "NUMERICAL_PRECISION_FAILURE"
            summary["terminal_validation_error"] = repr(exc)
        except SourceFailure as exc:
            summary["status"] = "BLOCKED"
            summary["classification"] = "BLOCKED_START"
            summary["terminal_validation_error"] = repr(exc)
            raise


def load_galaxies(protocol):
    root = Path(protocol["impl_root"])
    gd = load_tidy_sparc(str(root / "dataset" / "sparc_tidy.csv"))
    gs = [gd[k] for k in sorted(gd)]
    if len(gs) != 91 or sum(len(g.v_obs) for g in gs) != 1782:
        raise SourceFailure("Sample mismatch")
    vectors = {}
    for model in MODELS:
        spec = protocol["starts"][model]
        path = Path(spec["path"])
        blob = path.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        if digest != spec["sha256"]:
            raise SourceFailure("Start vector hash mismatch " + model)
        t = np.load(path)
        if int(t.shape[0]) != protocol["per_model"][model]["k"] or theta_hash(t) != spec["theta_sha256"]:
            raise SourceFailure("Start vector shape or content hash mismatch " + model)
        vectors[model] = t
    return gs, vectors


def reference_b0(protocol):
    path = Path(protocol["p1e_b0_reference"])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != protocol["p1e_b0_reference_sha256"]:
        raise SourceFailure("P1E reference hash mismatch")
    table = {m: {} for m in MODELS}
    with path.open() as f:
        for row in csv.DictReader(f):
            if row["floor"] == "B0" and row["model"] in table:
                table[row["model"]][row["galaxy"]] = float(row["explicit_logL"])
    return table


def assert_import_authority(candidate_src):
    import ptquat, ptquat.marginal_likelihood, ptquat.models, ptquat.dependency_gradient, ptquat.mle, ptquat.data, ptquat.fit_global
    roots = {
        "ptquat": Path(ptquat.__file__).resolve(),
        "marginal_likelihood": Path(ptquat.marginal_likelihood.__file__).resolve(),
        "models": Path(ptquat.models.__file__).resolve(),
        "dependency_gradient": Path(ptquat.dependency_gradient.__file__).resolve(),
        "mle": Path(ptquat.mle.__file__).resolve(),
        "data": Path(ptquat.data.__file__).resolve(),
        "fit_global": Path(ptquat.fit_global.__file__).resolve(),
        "driver": Path(__file__).resolve(),
    }
    src = Path(candidate_src).resolve()
    legacy = "ptq-mond/PTQ-Quaternionic-RC"
    canonical = Path("/home/rocky/PTQ-Quaternionic-RC/src/ptquat").resolve()
    for name, path in roots.items():
        if legacy in str(path):
            raise SourceFailure("Legacy import " + name)
        if name != "driver" and canonical in path.parents:
            raise SourceFailure("Canonical import " + name)
        if src not in path.parents and path != src:
            raise SourceFailure("Import outside candidate " + name + " " + str(path))
    return {k: str(v) for k, v in roots.items()}


def run(bundle):
    bundle = Path(bundle).resolve()
    protocol = json.loads((bundle / "FIT_PROTOCOL.json").read_text())
    out = bundle / "execution_v1"
    out.mkdir(exist_ok=False)
    collector = Telemetry()
    runtime = None
    exception = None
    preflight = {}
    old = {}

    def interrupted(signum, frame):
        raise InterruptedError("Signal " + str(signum))

    for sig in (signal.SIGTERM, signal.SIGINT):
        old[sig] = signal.signal(sig, interrupted)
    try:
        with collect_warnings(collector):
            imports = assert_import_authority(protocol["candidate_src"])
            save(out / "PYTHON_IMPORT_AUTHORITY.json", dict(modules=imports, source_hashes=source_hashes()))
            try:
                log_likelihood_per_galaxy()
            except TypeError:
                raise SourceFailure("Legacy per-galaxy likelihood is callable")
            except ValueError as exc:
                if "Radius-wise" not in str(exc):
                    raise SourceFailure("Unexpected legacy firewall: " + str(exc))
            gs, vectors = load_galaxies(protocol)
            ref = reference_b0(protocol)
            runtime = Runtime(bundle, out, protocol, gs, collector)
            layouts = {}
            for model in protocol["execution_order"]:
                L = _layout(model, 91, True, None)
                if L["k"] != protocol["per_model"][model]["k"]:
                    raise SourceFailure("Parameter count mismatch " + model)
                layouts[model] = L["k"]
                t = vectors[model]
                decode_record(t, gs, model)
                begin = time.monotonic()
                grad = runtime.gradient(t, model, "preflight", 192)
                actual = np.array(grad["baseline_log_marginals"])
                expect = np.array([ref[model][g.name] for g in gs])
                err = float(np.max(np.abs(actual - expect)))
                if err > 1e-8:
                    raise PrecisionFailure("Preflight P1E reference parity " + model)
                prec = runtime.objective_precision(t, model, "initial", grad["objective"])
                preflight[model] = dict(status="PASS", reference_pergal_max=err, k=L["k"], objective=grad["objective"], wall_seconds=time.monotonic() - begin, repeat_max=prec["metrics"]["repeat_max"])
                save(out / "PREFLIGHT_GATE.json", dict(status="IN_PROGRESS", models=preflight, layouts=layouts))
                print("PREFLIGHT", model, "ref", err, "F", grad["objective"], flush=True)
            interior = {}
            for model in ("ptq-nu", "ptq-screen"):
                bounds = np.asarray(theta_bounds(model, 91), float)
                mid = 0.5 * (bounds[:, 0] + bounds[:, 1])
                interior[model] = runtime.family_pair(mid, model, "preflight_interior")
                if interior[model]["status"] != "PASS":
                    raise PrecisionFailure("Matched-family interior identity failed " + model)
            save(out / "PREFLIGHT_GATE.json", dict(status="PASS", models=preflight, layouts=layouts, interior_family=interior, imports=imports))
            for model in protocol["execution_order"]:
                runtime.optimize_once(vectors[model], model, preflight[model]["objective"])
                if runtime.results[-1]["status"] == "BLOCKED":
                    break
            b1_rows = []
            for summary in runtime.results:
                model = summary["model"]
                if summary.get("terminal") is None:
                    continue
                theta = np.array(summary["terminal"]["theta"], float)
                if summary["classification"] == "BLOCKED_START":
                    continue
                try:
                    b1 = runtime.pure(theta, model, "b1_fixed_point", 192, SENSITIVITY_B1)
                    b0 = summary.get("terminal_objective")
                    row = dict(
                        identification="B1_AT_B0_TERMINAL_FIXED_POINT", model=model,
                        theta_sha256=theta_hash(theta), b1_objective=b1["objective"],
                        b0_objective=b0, classification_at_b0_endpoint=summary["classification"],
                        optimized_b1=False,
                    )
                    b1_rows.append(row)
                    with (out / (model.replace("-", "_") + "_b1_galaxy_log_marginals.csv")).open("w") as f:
                        w = csv.writer(f)
                        w.writerow(["galaxy", "log_marginal", "likelihood_version"])
                        for g, val in zip(gs, b1["galaxy_log_marginals"]):
                            w.writerow([g.name, f"{val:.16g}", SENSITIVITY_B1])
                except BudgetLimit as exc:
                    b1_rows.append(dict(model=model, status="BUDGET_LIMITED", error=repr(exc), optimized_b1=False))
                    break
            save(out / "B1_FIXED_POINT.json", dict(policy=protocol["b1_policy"], rows=b1_rows, ranking_published=False))
    except BaseException as exc:
        exception = dict(type=type(exc).__name__, error=repr(exc), traceback=traceback.format_exc())
        save(out / "TERMINAL_FAILURE.json", exception)
        print("TERMINAL FAILURE", exception["type"], exception["error"], flush=True)
    finally:
        if runtime is not None:
            runtime.ledger.close()
            save(out / "RUN_SUMMARY.json", dict(
                label=LABEL, results=runtime.results, preflight=preflight, exception=exception,
                budget=runtime.ledger.summary(), warnings=collector.as_dict(),
                global_mle=False, b1_optimization=False,
            ))
        elif exception is not None:
            save(out / "BOOTSTRAP_FAILURE.json", dict(exception=exception, warnings=collector.as_dict()))
        for sig, handler in old.items():
            signal.signal(sig, handler)
    return runtime.results if runtime is not None else []
