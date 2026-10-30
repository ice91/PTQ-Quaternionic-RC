"""Draft gradient-certification labels from saved P1G-A audits. No new evaluations."""
import json
from pathlib import Path
import numpy as np

EX = Path("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/mle_pilot_a/execution_v1")
OUT = Path("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/mle_pilot_a2_eng")
GTOL = 1e-6


def load(name):
    return json.loads((EX / name).read_text())


def classify(direction, native_projected, at_bound, bound_side):
    checks = { (c["kind"], c.get("order"), c.get("h")): c for c in direction["checks"] }
    two = [c for c in direction["checks"] if c["kind"] == "two_steps"]
    fine = [c for c in direction["checks"] if c["kind"] == "fine_coarse"]
    same = direction["candidate_difference"]
    floor = max([c["difference"] for c in two + fine] + [same])
    g = abs(direction["candidate_scaled_gradient"])
    implementation = same <= direction["candidate_limit"]
    record = {
        "index": direction["index"],
        "name": direction["name"],
        "at_bound": at_bound,
        "bound_side": bound_side,
        "candidate_scaled_gradient": direction["candidate_scaled_gradient"],
        "independent_scaled_gradient": direction["independent_scaled_gradient"],
        "same_step_difference": same,
        "same_step_limit": direction["candidate_limit"],
        "native_projected_component": native_projected,
        "observed_derivative_discrepancy": floor,
        "discrepancy_is_not_a_rigorous_error_bound": True,
        "driver_direction_pass": direction["pass_direction"],
        "checks": direction["checks"],
    }
    if at_bound and native_projected == 0.0:
        record["primary"] = "GRADIENT_IMPLEMENTATION_CONSISTENT"
        record["kkt_reading"] = "ACTIVE_BOUND_PROJECTED_COMPONENT_ZERO"
        record["stationarity_1e-6"] = "NOT_APPLICABLE_ACTIVE_BOUND_SIGN_CONDITION"
        record["note"] = "Unprojected scaled gradient is not a KKT violation when the projected component is zero."
        return record
    record["implementation"] = "GRADIENT_IMPLEMENTATION_CONSISTENT" if implementation else "UNRESOLVED"
    resolves_1e6 = floor <= GTOL
    clearly_nonzero = g > floor + GTOL
    if not implementation:
        primary = "UNRESOLVED"
    elif resolves_1e6 and g <= GTOL:
        primary = "STATIONARITY_CERTIFIABLE"
    elif clearly_nonzero and all(c["pass_test"] for c in fine) and all(c["pass_test"] for c in two):
        primary = "NONSTATIONARY_WITH_RESOLVED_GRADIENT"
    elif clearly_nonzero and all(c["pass_test"] for c in two) and not all(c["pass_test"] for c in fine):
        primary = "DERIVATIVE_RESOLUTION_INSUFFICIENT"
        record["coarse_nonzero_relative_to_observed_discrepancy"] = True
    else:
        primary = "DERIVATIVE_RESOLUTION_INSUFFICIENT"
    record["primary"] = primary
    record["stationarity_1e-6"] = "CERTIFIABLE" if primary == "STATIONARITY_CERTIFIABLE" else "NOT_CERTIFIABLE"
    record["gtol"] = GTOL
    return record


def model_block(model, audit_name, theta_name):
    result = load(model.replace("-", "_") + "_RESULT.json")
    audit = load(audit_name)
    theta = np.load(EX / theta_name)
    bounds = np.asarray(result["terminal"]["bounds"] if "bounds" in result.get("terminal", {}) else result["bounds"], float)
    contacts = set(result["kkt"]["exact_bound_contacts"])
    native = result["kkt"]["native_projected_gradient"]
    rows = []
    for direction in audit["directions"]:
        i = direction["index"]
        lo, hi = bounds[i]
        value = float(theta[i])
        if i in contacts and value == lo:
            side = "lower"
        elif i in contacts and value == hi:
            side = "upper"
        elif i in contacts:
            side = "contact_without_exact_endpoint_match"
        else:
            side = "interior"
        rows.append(classify(direction, native[i], i in contacts, side))
    primaries = {r["primary"] for r in rows}
    if "STATIONARITY_CERTIFIABLE" in primaries and primaries <= {"STATIONARITY_CERTIFIABLE", "GRADIENT_IMPLEMENTATION_CONSISTENT"}:
        model_primary = "STATIONARITY_CERTIFIABLE"
    elif any(r["primary"] == "NONSTATIONARY_WITH_RESOLVED_GRADIENT" for r in rows):
        model_primary = "NONSTATIONARY_WITH_RESOLVED_GRADIENT"
    elif any(r["primary"] == "DERIVATIVE_RESOLUTION_INSUFFICIENT" for r in rows):
        model_primary = "DERIVATIVE_RESOLUTION_INSUFFICIENT"
    else:
        model_primary = "UNRESOLVED"
    return {
        "model": model,
        "theta_sha256": audit["theta_sha256"],
        "audit_status": audit["status"],
        "scipy_result_present": result["scipy"] is not None,
        "native_projected_inf": result["kkt"]["native_projected_gradient_inf_norm"],
        "scaled_projected_inf": result["kkt"]["scaled_projected_gradient_inf_norm"],
        "stationary_under_frozen_gtol": result["stationary_under_frozen_gtol"],
        "two_step_pass": result["two_step_pass"],
        "n_exact_bound_contacts": len(contacts),
        "coordinates": rows,
        "model_primary": model_primary,
        "official_gtol_unchanged": GTOL,
        "activated": False,
    }


def main():
    ptq = model_block("ptq", "ptq_terminal_DIRECTION_AUDIT.json", "ptq_last_accepted_theta.npy")
    nu = model_block("ptq-nu", "ptq_nu_terminal_DIRECTION_AUDIT.json", "ptq_nu_last_accepted_theta.npy")
    doc = {
        "task": "REV26-P1G-A2-ENG",
        "document": "GRADIENT_CERTIFICATION_DRAFT",
        "version": "REV26-P1G-A2-ENG-DRAFT-1",
        "activated": False,
        "official_mle_acceptance_unchanged": True,
        "gtol": GTOL,
        "gtol_changed": False,
        "new_finite_difference_evaluations": 0,
        "classes": [
            "GRADIENT_IMPLEMENTATION_CONSISTENT",
            "DERIVATIVE_RESOLUTION_INSUFFICIENT",
            "NONSTATIONARY_WITH_RESOLVED_GRADIENT",
            "STATIONARITY_CERTIFIABLE",
            "UNRESOLVED",
        ],
        "decision_order": [
            "Active bound with projected component exactly zero is a KKT sign condition, not a stationarity failure.",
            "Same-step candidate versus independent difference within the frozen limit is implementation consistency only.",
            "STATIONARITY_CERTIFIABLE requires the projected component and every saved derivative discrepancy to be at or below gtol.",
            "A discrepancy above gtol blocks 1e-6 certification even when the gradient is obviously large.",
            "NONSTATIONARY_WITH_RESOLVED_GRADIENT requires same-step consistency, passing two-step and 96/192 checks, and |gradient| larger than the observed discrepancy plus gtol.",
            "A failed 96/192 derivative check yields DERIVATIVE_RESOLUTION_INSUFFICIENT. A large gradient beside that failure is recorded as a coarse indication, not a resolved certificate.",
            "Derivative differences are diagnostics, not rigorous error bounds.",
        ],
        "examples": [ptq, nu],
        "proposal_for_author_review": "Keep gtol=1e-6 as the frozen official tolerance. Do not mark stationarity unless the saved discrepancy floor is also <= 1e-6. Where the floor is higher, report DERIVATIVE_RESOLUTION_INSUFFICIENT or NONSTATIONARY_WITH_RESOLVED_GRADIENT and leave the official acceptance rule unchanged until a separate human decision.",
    }
    (OUT / "GRADIENT_CERTIFICATION_DRAFT.json").write_text(json.dumps(doc, indent=2) + "\n")
    print("DRAFT", ptq["model_primary"], nu["model_primary"], flush=True)


if __name__ == "__main__":
    main()
