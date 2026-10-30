"""Fixed-vector gradient study and NFW fixed point. No optimizer."""
import hashlib, json, os, sys, time, warnings
from pathlib import Path
import numpy as np

CAND = Path("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/mle_pilot_a/candidate/src")
EX = Path("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/mle_pilot_a/execution_v1")
IMPL = Path("/home/rocky/PTQ-Quaternionic-RC")
OUT = Path("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/mle_pilot_a3_eng")
LOG = OUT / "diagnostic" / "gradient_nfw.log"
sys.path.insert(0, str(CAND))
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

MAX_CALLS = 10900
MAX_EQ = 137.0
MAX_WALL = 100 * 60
GTOL = 1e-6
REL_STEPS = (8e-4, 4e-4, 2e-4, 1e-4, 5e-5)


def log(msg):
    line = msg + "\n"
    with LOG.open("a") as f:
        f.write(line)
        f.flush()
    print(msg, flush=True)


def main():
    from ptquat.constants import C_LIGHT, H0_SI, KPC, KM
    from ptquat.data import load_tidy_sparc
    from ptquat.fit_global import _layout, _unpack
    from ptquat.marginal_likelihood import (
        PRIMARY_B0, galaxy_log_marginal, integrate, nuisance_nodes, velocity,
    )
    from ptquat.mle import theta_bounds
    from ptquat.models import model_v_ptq, model_v_ptq_nu, vbar_squared_kms2
    import ptquat.marginal_likelihood as ml
    if Path(ml.__file__).resolve() != (CAND / "ptquat" / "marginal_likelihood.py").resolve():
        raise SystemExit("import authority")
    weights = json.loads((IMPL / "results/frontiers_revision_2026_10/mle_pilot_a/GALAXY_WORK_WEIGHTS.json").read_text())
    gd = load_tidy_sparc(str(IMPL / "dataset/sparc_tidy.csv"))
    gs = [gd[k] for k in sorted(gd)]
    if len(gs) != 91 or int(sum(len(g.v_obs) for g in gs)) != 1782:
        raise SystemExit("sample scope")
    t0 = time.monotonic()
    calls = 0
    eq = 0.0
    stopped = None

    def charge(model, gi, order):
        nonlocal calls, eq
        if time.monotonic() - t0 > MAX_WALL:
            raise RuntimeError("wall cap")
        base_order = 192 if order == 384 else order
        unit = weights[model][str(base_order)][gi]["fullsample_equivalent"]
        if order == 384:
            unit *= 2.0
        if calls + 1 > MAX_CALLS or eq + unit > MAX_EQ:
            raise RuntimeError("work cap")
        calls += 1
        eq += unit

    def namespace(g):
        return type("G", (), {})()  # placeholder replaced below

    def as_g(g):
        from types import SimpleNamespace
        return SimpleNamespace(
            name=g.name, r=g.r_kpc, y=g.v_obs, e=g.v_err, disk=g.v_disk, bulge=g.v_bulge, gas=g.v_gas,
            D=float(g.D_Mpc), sD=float(g.D_err_Mpc), inc=float(np.deg2rad(g.i_deg)), si=float(np.deg2rad(g.i_err_deg)),
        )

    def ell(g, pars, gi, model, order):
        charge(model, gi, order)
        with warnings.catch_warnings(record=True) as got:
            warnings.simplefilter("always")
            if order in (96, 192):
                r = galaxy_log_marginal(g, pars, gi, model, likelihood_version=PRIMARY_B0, order=order)
                val = float(r["log_marginal"])
                warns = [f"{w.category.__name__}: {w.message}" for w in got]
                return val, warns
            gg = as_g(g)
            pp = dict(pars, model=model, _H0=H0_SI, _c0=10.0, _c_slope=-0.1)
            raw = integrate(gg, pp, gi, "segmented_quantile", order, floors=("B0",))
            warns = [f"{w.category.__name__}: {w.message}" for w in got]
            return float(raw["logL"]["B0"]), warns

    def objective(theta, model, order, galaxies=None):
        L = _layout(model, 91, True, None)
        pars = _unpack(theta, L)
        ids = range(91) if galaxies is None else galaxies
        total = 0.0
        warns = []
        for gi in ids:
            val, w = ell(gs[gi], pars, gi, model, order)
            total += val
            warns.extend(w)
        return -float(total), warns

    def dv_and_v(g, pars, D, model):
        scale = np.asarray(D, float) / g.D
        radius = g.r * scale
        disk = g.disk * np.sqrt(scale)
        bulge = g.bulge * np.sqrt(scale)
        gas = g.gas * np.sqrt(scale)
        U = float(pars["U"][0] if np.ndim(pars["U"]) else pars["U"])
        # caller passes a one-galaxy U slice via pars copy
        eps = float(pars["eps"])
        flags = []
        if model == "ptq":
            v = model_v_ptq(U, eps, radius, disk, bulge, gas, H0_si=H0_SI)
            lin = eps and (model_v_ptq(U, 1.0, radius, disk, bulge, gas, H0_si=H0_SI) * 0 + (
                (C_LIGHT * H0_SI * radius * KPC) / (KM ** 2)
            ))
            additive = (C_LIGHT * H0_SI * (radius * KPC)) / (KM ** 2)
            dv = additive / (2.0 * v)
            return v, dv, flags
        if model == "ptq-nu":
            a0 = eps * C_LIGHT * H0_SI
            if a0 <= 1e-30:
                flags.append("a0_floor")
            vN2 = vbar_squared_kms2(U, disk, bulge, gas)
            r_m = np.maximum(radius * KPC, 1e-12)
            gN = (vN2 * (KM ** 2)) / r_m
            y = gN / max(a0, 1e-30)
            if np.any(y < 1e-11):
                flags.append("y_near_clip")
            root = np.sqrt(0.25 + 1.0 / np.maximum(y, 1e-12))
            nu = 0.5 + root
            v = np.sqrt(vN2 * nu)
            dnu_dy = -0.5 / (root * np.maximum(y, 1e-12) ** 2)
            dv = (v / (2.0 * nu)) * dnu_dy * (-y / max(a0, 1e-30)) * (C_LIGHT * H0_SI)
            return v, dv, flags
        raise ValueError(model)

    def analytic_galaxy(g, pars, gi, model, order):
        gg = as_g(g)
        pp = dict(pars, model=model, _H0=H0_SI, _c0=10.0, _c_slope=-0.1)
        charge(model, gi, order)
        base = integrate(gg, pp, gi, "segmented_quantile", order, floors=("B0",))
        ds, wd, incs, wi, _, _ = nuisance_nodes(gg, "segmented_quantile", order)
        sigma = float(pp["sigma"])
        # one-galaxy parameter view for dv: U is full vector
        vd = velocity(gg, pp, gi, ds[:, None])
        scale = ds / gg.D
        radius = gg.r * scale[:, None]
        disk = gg.disk * np.sqrt(scale)[:, None]
        bulge = gg.bulge * np.sqrt(scale)[:, None]
        gas = gg.gas * np.sqrt(scale)[:, None]
        U = float(pp["U"][gi])
        eps = float(pp["eps"])
        flags = []
        if model == "ptq":
            v_formula = model_v_ptq(U, eps, radius, disk, bulge, gas, H0_si=H0_SI)
            if np.max(np.abs(v_formula - vd)) > 1e-8 * max(1.0, float(np.max(np.abs(vd)))):
                flags.append("velocity_formula_mismatch")
            additive = (C_LIGHT * H0_SI * (radius * KPC)) / (KM ** 2)
            dvd = additive / (2.0 * vd)
        elif model == "ptq-nu":
            v_formula = model_v_ptq_nu(U, eps, radius, disk, bulge, gas, H0_si=H0_SI)
            if np.max(np.abs(v_formula - vd)) > 1e-8 * max(1.0, float(np.max(np.abs(vd)))):
                flags.append("velocity_formula_mismatch")
            a0 = eps * C_LIGHT * H0_SI
            if a0 <= 1e-30:
                flags.append("a0_floor")
            vN2 = vbar_squared_kms2(U, disk, bulge, gas)
            r_m = np.maximum(radius * KPC, 1e-12)
            gN = (vN2 * (KM ** 2)) / r_m
            y = gN / max(a0, 1e-30)
            if np.any(y < 1e-11):
                flags.append("y_near_clip")
            root = np.sqrt(0.25 + 1.0 / np.maximum(y, 1e-12))
            nu = 0.5 + root
            dnu_dy = -0.5 / (root * np.maximum(y, 1e-12) ** 2)
            dvd = (vd / (2.0 * nu)) * dnu_dy * (-y / max(a0, 1e-30)) * (C_LIGHT * H0_SI)
        else:
            raise ValueError(model)
        if np.max(np.abs(vd)) == 0:
            flags.append("zero_velocity")
        b = np.sin(incs) / np.sin(gg.inc)
        var = gg.e ** 2 + sigma ** 2
        inv = 1.0 / var
        inv2 = inv / var
        norm = float(np.sum(np.log(2 * np.pi * var)))
        Y2 = float(np.sum(gg.y ** 2 * inv))
        Y2q = float(np.sum(gg.y ** 2 * inv2))
        sum_inv = float(np.sum(inv))
        lls = []
        dsig = []
        deps = []
        logw = []
        for start in range(0, len(ds), 128):
            end = min(start + 128, len(ds))
            v = vd[start:end]
            dv = dvd[start:end]
            cross = v @ (gg.y * inv)
            vv = (v * v) @ inv
            Q = Y2 - 2.0 * b[None, :] * cross[:, None] + (b ** 2)[None, :] * vv[:, None]
            ll = -0.5 * (Q + norm)
            crossq = v @ (gg.y * inv2)
            vvq = (v * v) @ inv2
            resid_over = Y2q - 2.0 * b[None, :] * crossq[:, None] + (b ** 2)[None, :] * vvq[:, None]
            dll_sigma = sigma * (resid_over - sum_inv)
            cy = dv @ (gg.y * inv)
            cv = (dv * v) @ inv
            dll_eps = b[None, :] * cy[:, None] - (b ** 2)[None, :] * cv[:, None]
            lls.append(ll)
            dsig.append(dll_sigma)
            deps.append(dll_eps)
            logw.append(np.log(wd[start:end])[:, None] + np.log(wi)[None, :])
        from scipy.special import logsumexp
        ll = np.concatenate([a.ravel() for a in lls])
        ds_ = np.concatenate([a.ravel() for a in dsig])
        de = np.concatenate([a.ravel() for a in deps])
        lw = np.concatenate([a.ravel() for a in logw])
        s = ll + lw
        logL = float(logsumexp(s))
        p = np.exp(s - logL)
        return {
            "official_logL": float(base["logL"]["B0"]),
            "rederived_logL": logL,
            "dlogL_dsigma": float(np.sum(p * ds_)),
            "dlogL_depsilon": float(np.sum(p * de)),
            "flags": flags,
            "max_abs_logL_gap": abs(logL - float(base["logL"]["B0"])),
        }

    def study_global(model, theta, index, name, order, steps):
        bounds = np.asarray(theta_bounds(model, 91), float)
        lo, hi = bounds[index]
        span = float(hi - lo)
        x = float(theta[index])
        rows = []
        base = None
        for rel in steps:
            h = span * rel
            if not (x - h >= lo and x + h <= hi):
                rows.append({"relative_step": rel, "scheme": "central_not_inside_bounds"})
                continue
            plus = theta.copy(); plus[index] = x + h
            minus = theta.copy(); minus[index] = x - h
            fp, wp = objective(plus, model, order)
            fm, wm = objective(minus, model, order)
            native = (fp - fm) / (2.0 * h)
            rows.append({
                "relative_step": rel,
                "native_step": h,
                "scheme": "central",
                "objective_plus": fp,
                "objective_minus": fm,
                "objective_difference": fp - fm,
                "native_gradient": native,
                "scaled_gradient": native * span,
                "warnings": len(wp) + len(wm),
            })
            log(f"FD {model} {name} order {order} rel {rel} g {native}")
        fc = None
        if order == 192:
            h = span * 1e-4
            plus = theta.copy(); plus[index] = x + h
            fp, _ = objective(plus, model, order)
            fc, wc = objective(theta, model, order)
            forward = (fp - fc) / h
            rows.append({
                "relative_step": 1e-4,
                "scheme": "forward_one_sided",
                "native_step": h,
                "native_gradient": forward,
                "scaled_gradient": forward * span,
                "center_objective": fc,
                "warnings": len(wc),
            })
        return {"index": index, "name": name, "value": x, "lo": float(lo), "hi": float(hi), "span": span, "at_bound": bool(x == lo or x == hi), "rows": rows, "center_objective": fc}

    results = {"task": "REV26-P1G-A3-ENG", "gtol": GTOL, "models": {}}
    try:
        for model, theta_name, audit_name in (
            ("ptq", "ptq_last_accepted_theta.npy", "ptq_terminal_DIRECTION_AUDIT.json"),
            ("ptq-nu", "ptq_nu_last_accepted_theta.npy", "ptq_nu_terminal_DIRECTION_AUDIT.json"),
        ):
            theta = np.load(EX / theta_name)
            audit = json.loads((EX / audit_name).read_text())
            L = _layout(model, 91, True, None)
            pars = _unpack(theta, L)
            # analytic over full sample at 192, one pass
            pieces = []
            flags = []
            for gi, g in enumerate(gs):
                piece = analytic_galaxy(g, pars, gi, model, 192)
                pieces.append(piece)
                flags.extend(piece["flags"])
            gap = max(p["max_abs_logL_gap"] for p in pieces)
            dF_sigma = -sum(p["dlogL_dsigma"] for p in pieces) * float(pars["sigma"])  # dF/d ln_sigma
            dF_eps = -sum(p["dlogL_depsilon"] for p in pieces)
            log(f"ANALYTIC {model} gap {gap} dF_eps {dF_eps} dF_lnsig {dF_sigma}")
            coords = []
            for index, name in ((L["i_eps"], "epsilon"), (L["i_lnsig"], "ln_sigma_sys")):
                block = study_global(model, theta, index, name, 192, REL_STEPS)
                block_384 = study_global(model, theta, index, name, 384, (1e-4,))
                archived = next(d for d in audit["directions"] if d["index"] == index)
                span = block["span"]
                analytic_native = dF_eps if name == "epsilon" else dF_sigma
                central = [r for r in block["rows"] if r.get("scheme") == "central"]
                match = next(r for r in central if r["relative_step"] == 1e-4)
                g384 = next(r for r in block_384["rows"] if r.get("scheme") == "central")
                fwd = next((r for r in block["rows"] if r.get("scheme") == "forward_one_sided"), None)
                # Richardson on successive halvings
                rich = []
                by_rel = {r["relative_step"]: r for r in central}
                for big, small in ((8e-4, 4e-4), (4e-4, 2e-4), (2e-4, 1e-4), (1e-4, 5e-5)):
                    if big in by_rel and small in by_rel:
                        g_r = (4.0 * by_rel[small]["native_gradient"] - by_rel[big]["native_gradient"]) / 3.0
                        rich.append({
                            "h_big": big, "h_small": small,
                            "richardson_native": g_r,
                            "abs_change_from_smaller": abs(g_r - by_rel[small]["native_gradient"]),
                        })
                step_spread = max(r["native_gradient"] for r in central) - min(r["native_gradient"] for r in central)
                analytic_gap = abs(analytic_native - match["native_gradient"])
                quad_gap = abs(g384["native_gradient"] - match["native_gradient"])
                err = max(analytic_gap, quad_gap, max((r["abs_change_from_smaller"] for r in rich), default=0.0))
                native_g = match["native_gradient"]
                labels = []
                if abs(match["scaled_gradient"] - archived["candidate_scaled_gradient"]) <= max(1e-6, 1e-8 * abs(archived["candidate_scaled_gradient"])):
                    labels.append("IMPLEMENTATION_CONSISTENT")
                if step_spread > max(1.0, 0.05 * abs(native_g)):
                    labels.append("DERIVATIVE_APPROXIMATION_UNSTABLE")
                if quad_gap > GTOL:
                    labels.append("QUADRATURE_SENSITIVITY")
                if abs(native_g) > err + GTOL and "DERIVATIVE_APPROXIMATION_UNSTABLE" not in labels:
                    labels.append("NONSTATIONARY_WITH_RESOLVED_GRADIENT")
                if err > GTOL or abs(native_g) > GTOL:
                    labels.append("STATIONARITY_NOT_CERTIFIABLE")
                coords.append({
                    "coordinate": name,
                    "index": index,
                    "native_parameterization": "epsilon" if name == "epsilon" else "ln_sigma_sys",
                    "physical": "epsilon multiplies c*H0 in the model law" if name == "epsilon" else "sigma_sys = exp(ln_sigma_sys)",
                    "at_bound": block["at_bound"],
                    "native_gradient_fd_1e-4": native_g,
                    "scaled_gradient_fd_1e-4": match["scaled_gradient"],
                    "archived_scaled_gradient": archived["candidate_scaled_gradient"],
                    "analytic_native_gradient": analytic_native,
                    "analytic_minus_fd": analytic_native - native_g,
                    "fd_384_native": g384["native_gradient"],
                    "fd_384_minus_192": g384["native_gradient"] - native_g,
                    "forward_minus_central": None if fwd is None else fwd["native_gradient"] - native_g,
                    "step_rows_192": block["rows"],
                    "richardson": rich,
                    "step_spread_native": step_spread,
                    "error_indicator_native": err,
                    "rederived_logL_gap_max": gap,
                    "clip_flags": sorted(set(flags)),
                    "gtol_resolvable": bool(err <= GTOL),
                    "labels": labels,
                    "projected_kkt": "interior_component_equals_native" if not block["at_bound"] else "bound",
                    "assumptions": "Analytic derivative differentiates the same B0 conditional Gaussian and the same segmented quadrature. max() floors are inactive only if clip_flags is empty. Differences are diagnostics, not rigorous bounds.",
                })
            # local M/L, galaxy index 9 -> native index 10, may be one galaxy
            gi = 9
            j = L["sl_U"].start + gi
            bounds = np.asarray(theta_bounds(model, 91), float)
            lo, hi = bounds[j]
            span = float(hi - lo)
            x = float(theta[j])
            local_rows = []
            h = span * 1e-4
            if x - h >= lo and x + h <= hi:
                plus = theta.copy(); plus[j] = x + h
                minus = theta.copy(); minus[j] = x - h
                fp, _ = objective(plus, model, 192, galaxies=[gi])
                fm, _ = objective(minus, model, 192, galaxies=[gi])
                native = (fp - fm) / (2 * h)
                scheme = "central"
            elif x + h <= hi:
                plus = theta.copy(); plus[j] = x + h
                fp, _ = objective(plus, model, 192, galaxies=[gi])
                fc, _ = objective(theta, model, 192, galaxies=[gi])
                native = (fp - fc) / h
                scheme = "forward_inward"
            else:
                minus = theta.copy(); minus[j] = x - h
                fm, _ = objective(minus, model, 192, galaxies=[gi])
                fc, _ = objective(theta, model, 192, galaxies=[gi])
                native = (fc - fm) / h
                scheme = "backward_inward"
            side = "lower" if x == lo else ("upper" if x == hi else "interior")
            projected = 0.0 if (side == "lower" and native > 0) or (side == "upper" and native < 0) else native
            local_rows.append({
                "galaxy": gs[gi].name, "index": j, "scheme": scheme, "bound_side": side,
                "native_gradient": native, "scaled_gradient": native * span,
                "projected_native": projected,
                "note": "Unprojected gradient at an active bound with the outward sign is not a KKT violation.",
            })
            results["models"][model] = {
                "theta_sha256": hashlib.sha256(np.asarray(theta, dtype="<f8").tobytes()).hexdigest(),
                "analytic_logL_gap_max": gap,
                "coordinates": coords,
                "local_ml": local_rows,
                "model_stationarity_certificate": False,
            }
        # NFW fixed point at the archived start
        theta = np.load(EX / "nfw1p_initial_theta.npy")
        t_n = time.perf_counter()
        f_nfw, w_nfw = objective(theta, "nfw1p", 192)
        nfw_obj_s = time.perf_counter() - t_n
        L = _layout("nfw1p", 91, True, None)
        bounds = np.asarray(theta_bounds("nfw1p", 91), float)
        inside = bool(np.all(theta >= bounds[:, 0]) and np.all(theta <= bounds[:, 1]))
        # two interior halo-mass directions if possible, else the first that fits a central step
        halo = []
        for gi in range(91):
            j = L["sl_lM"].start + gi
            lo, hi = bounds[j]
            span = float(hi - lo)
            x = float(theta[j])
            h = span * 1e-4
            if x - h >= lo and x + h <= hi:
                plus = theta.copy(); plus[j] = x + h
                minus = theta.copy(); minus[j] = x - h
                t1 = time.perf_counter()
                fp, wp = objective(plus, "nfw1p", 192, galaxies=[gi])
                fm, wm = objective(minus, "nfw1p", 192, galaxies=[gi])
                halo.append({
                    "galaxy": gs[gi].name, "index": j, "log10_M200": x,
                    "native_gradient": (fp - fm) / (2 * h),
                    "scaled_gradient": ((fp - fm) / (2 * h)) * span,
                    "seconds": time.perf_counter() - t1,
                    "warnings": len(wp) + len(wm),
                    "at_bound": False,
                })
            if len(halo) == 2:
                break
        j = L["i_lnsig"]
        lo, hi = bounds[j]
        span = float(hi - lo)
        x = float(theta[j])
        h = span * 1e-4
        plus = theta.copy(); plus[j] = x + h
        minus = theta.copy(); minus[j] = x - h
        t1 = time.perf_counter()
        fp, wp = objective(plus, "nfw1p", 192)
        fm, wm = objective(minus, "nfw1p", 192)
        nfw_sigma = {
            "index": j, "value": x, "at_bound": bool(x == lo or x == hi),
            "native_gradient": (fp - fm) / (2 * h),
            "scaled_gradient": ((fp - fm) / (2 * h)) * span,
            "seconds": time.perf_counter() - t1,
            "warnings": len(wp) + len(wm),
            "objective_plus": fp, "objective_minus": fm,
        }
        f_rep, w_rep = objective(theta, "nfw1p", 192)
        results["nfw"] = {
            "source": "nfw1p_initial_theta.npy",
            "theta_sha256": hashlib.sha256(np.asarray(theta, dtype="<f8").tobytes()).hexdigest(),
            "likelihood_version": PRIMARY_B0,
            "objective_finite": bool(np.isfinite(f_nfw)),
            "objective": f_nfw,
            "objective_repeat_abs": abs(f_rep - f_nfw),
            "inside_original_bounds": inside,
            "warnings": sorted(set(w_nfw + w_rep)),
            "full_sample_seconds": nfw_obj_s,
            "halo_directions": halo,
            "ln_sigma_direction": nfw_sigma,
            "optimizer_steps": 0,
            "global_optimum_claimed": False,
            "checkpoint_note": "Per-galaxy fsync overhead is taken from the baryon A/B, not retimed on NFW.",
        }
        log(f"NFW F {f_nfw} repeat {abs(f_rep-f_nfw)} warnings {len(w_nfw)}")
    except (RuntimeError, Exception) as exc:
        stopped = repr(exc)
        log("STOPPED " + stopped)
        if not isinstance(exc, RuntimeError):
            import traceback
            log(traceback.format_exc())
    results["budget"] = {"calls": calls, "equivalents": eq, "wall_seconds": time.monotonic() - t0, "stopped": stopped}
    (OUT / "GRADIENT_STEP_STUDY.json").write_text(json.dumps(results, indent=2) + "\n")
    # compact audit
    audit_out = {"task": "REV26-P1G-A3-ENG", "gtol": GTOL, "stopped": stopped, "models": {}}
    for model, block in results.get("models", {}).items():
        audit_out["models"][model] = [{
            "coordinate": c["coordinate"],
            "labels": c["labels"],
            "gtol_resolvable": c["gtol_resolvable"],
            "native_gradient_fd_1e-4": c["native_gradient_fd_1e-4"],
            "analytic_native_gradient": c["analytic_native_gradient"],
            "analytic_minus_fd": c["analytic_minus_fd"],
            "fd_384_minus_192": c["fd_384_minus_192"],
            "error_indicator_native": c["error_indicator_native"],
            "step_spread_native": c["step_spread_native"],
            "at_bound": c["at_bound"],
        } for c in block["coordinates"]]
    audit_out["nfw_objective_finite"] = results.get("nfw", {}).get("objective_finite")
    audit_out["model_stationarity_certificate"] = False
    (OUT / "GRADIENT_ACCURACY_AUDIT.json").write_text(json.dumps(audit_out, indent=2) + "\n")
    if "nfw" in results:
        (OUT / "NFW_FIXED_POINT_RESULTS.json").write_text(json.dumps(results["nfw"], indent=2) + "\n")
    log(f"DONE calls {calls} eq {eq:.3f} wall {results['budget']['wall_seconds']:.1f}")


if __name__ == "__main__":
    LOG.write_text("")
    main()
