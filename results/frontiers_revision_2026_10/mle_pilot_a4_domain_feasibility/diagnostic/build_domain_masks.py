"""Isolated mask construction for REV26-P1G-A4-DOMAIN-DECISION-FEASIBILITY.

Reads frozen tables and the signed-domain census. Does not evaluate a likelihood
or call an optimizer. Does not activate a sample.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

IMPL = Path("/home/rocky/PTQ-Quaternionic-RC")
PREV = IMPL / "results/frontiers_revision_2026_10/mle_pilot_a4_sign_spec_domain"
NEW = IMPL / "results/frontiers_revision_2026_10/mle_pilot_a4_domain_feasibility"
sys.path.insert(0, str(IMPL / "results/frontiers_revision_2026_10/mle_pilot_a/candidate/src"))
sys.path.insert(0, str(PREV / "diagnostic"))

from ptquat.constants import H0_SI, KM, KPC, C_LIGHT  # noqa: E402
from signed_sparc_operator import baryonic_signed_v2, signed_square  # noqa: E402

U_LO = 0.05
U_HI = 1.5
A0_HI = 2.0e-10
Y_FLOOR = 1.0e-12


def qstats(x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    qs = np.quantile(x, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {
        "n": int(x.size),
        "min": float(qs[0]),
        "q25": float(qs[1]),
        "median": float(qs[2]),
        "q75": float(qs[3]),
        "max": float(qs[4]),
        "mean": float(np.mean(x)),
    }


def main() -> None:
    tidy = pd.read_csv(IMPL / "dataset/sparc_tidy.csv")
    raw = pd.read_csv(IMPL / "dataset/raw/vizier_table2.csv")
    table1 = pd.read_csv(IMPL / "dataset/raw/vizier_table1.csv")
    dom = pd.read_csv(PREV / "NEGATIVE_BARYONIC_DOMAIN.csv")
    if not (
        len(tidy) == len(dom) == 1782
        and np.array_equal(tidy.r_kpc.to_numpy(), dom.r_kpc.to_numpy())
        and (tidy.galaxy.to_numpy() == dom.galaxy.to_numpy()).all()
    ):
        raise SystemExit("Frozen tidy and domain census are not row-aligned")
    joined = tidy.merge(
        raw,
        left_on=["galaxy", "r_kpc"],
        right_on=["Name", "Rad"],
        how="left",
        indicator=True,
        validate="one_to_one",
    )
    if not (joined["_merge"] == "both").all() or joined["recno"].duplicated().any():
        raise SystemExit("VizieR recno join is not one-to-one")
    meta = table1.set_index("Name")
    if not tidy.galaxy.isin(meta.index).all():
        raise SystemExit("A tidy galaxy is missing from SPARC Table 1")

    disk = tidy.v_disk_kms.to_numpy(float)
    bulge = tidy.v_bulge_kms.to_numpy(float)
    gas = tidy.v_gas_kms.to_numpy(float)
    radius = tidy.r_kpc.to_numpy(float)
    stellar = signed_square(disk) + signed_square(bulge)
    gas_s = signed_square(gas)
    b_lo = baryonic_signed_v2(U_LO, disk, bulge, gas)
    b_mid = baryonic_signed_v2(0.5, disk, bulge, gas)
    b_hi = baryonic_signed_v2(U_HI, disk, bulge, gas)
    stored_b = dom["B_at_U_0.05"].to_numpy(float)
    csv_gap = float(np.max(np.abs(b_lo - stored_b)))
    if int(np.sum(stored_b <= 0.0)) != 13 or not np.array_equal(stored_b <= 0.0, b_lo <= 0.0):
        raise SystemExit("Frozen census and recomputed B(0.05) disagree on the nonpositive rows")
    if np.min(stellar) < 0.0:
        raise SystemExit("Stellar signed contribution is negative")
    if np.any(radius <= 0.0):
        raise SystemExit("Nonpositive radius")

    permanent = dom.interval_class.to_numpy() == "NEGATIVE_ON_ENTIRE_ADMISSIBLE_INTERVAL"
    conditional = dom.interval_class.to_numpy() == "NEGATIVE_ON_A_SUBINTERVAL"
    if int(permanent.sum()) != 2 or int(conditional.sum()) != 11:
        raise SystemExit("Unexpected permanent/conditional counts")
    if not (
        set(zip(tidy.galaxy[permanent], tidy.r_kpc[permanent]))
        == {("UGC01281", 0.08), ("UGC01281", 0.23)}
    ):
        raise SystemExit("Permanent rows are not the two UGC01281 fixtures")

    rdisk = tidy.galaxy.map(meta["Rdisk"]).to_numpy(float)
    qual = tidy.galaxy.map(meta["Qual"]).to_numpy(int)
    r_over = radius / rdisk

    base = pd.DataFrame(
        {
            "tidy_index": np.arange(len(tidy), dtype=int),
            "source_recno": joined["recno"].to_numpy(int),
            "galaxy": tidy.galaxy.to_numpy(),
            "r_kpc": radius,
            "Rdisk_kpc": rdisk,
            "r_over_Rdisk": r_over,
            "Qual": qual,
            "v_obs_kms": tidy.v_obs_kms.to_numpy(float),
            "v_err_kms": tidy.v_err_kms.to_numpy(float),
            "v_gas_kms": gas,
            "gas_sign": np.sign(gas).astype(int),
            "stellar_signed_v2": stellar,
            "gas_signed_v2": gas_s,
            "B_at_U_0.05": b_lo,
            "interval_class": dom.interval_class.to_numpy(),
            "zero_crossing_upsilon": dom.zero_crossing_upsilon.to_numpy(float),
        }
    )

    masks = {
        "S_FULL": np.ones(len(base), dtype=bool),
        "S_HARD2": ~permanent,
        "S_DOMAIN13": b_lo > 0.0,
        "S_NO_UGC01281": tidy.galaxy.to_numpy() != "UGC01281",
    }
    reasons = {
        "S_FULL": np.full(len(base), "INCLUDE_ORIGINAL_91_1782"),
        "S_HARD2": np.where(
            permanent,
            "EXCLUDE_B_NEGATIVE_ON_ENTIRE_ADMISSIBLE_UPSILON_INTERVAL",
            "INCLUDE_NOT_PERMANENTLY_UNDEFINED",
        ),
        "S_DOMAIN13": np.where(
            b_lo > 0.0,
            "INCLUDE_B_STRICTLY_POSITIVE_AT_UPSILON_0.05",
            "EXCLUDE_B_AT_UPSILON_0.05_NOT_STRICTLY_POSITIVE",
        ),
        "S_NO_UGC01281": np.where(
            tidy.galaxy.to_numpy() == "UGC01281",
            "EXCLUDE_ENTIRE_GALAXY_LITERATURE_SENSITIVITY_NOT_ACTIVATED",
            "INCLUDE_GALAXY_RETAINED",
        ),
    }
    domain_safe = masks["S_NO_UGC01281"] & masks["S_DOMAIN13"]
    domain_safe_reason = np.where(
        tidy.galaxy.to_numpy() == "UGC01281",
        "EXCLUDE_ENTIRE_GALAXY_LITERATURE_SENSITIVITY_NOT_ACTIVATED",
        np.where(
            b_lo > 0.0,
            "INCLUDE_GALAXY_RETAINED_AND_B_STRICTLY_POSITIVE_AT_UPSILON_0.05",
            "EXCLUDE_B_AT_UPSILON_0.05_NOT_STRICTLY_POSITIVE",
        ),
    )

    counts = {}
    for name, mask in masks.items():
        counts[name] = {
            "galaxies": int(base.loc[mask, "galaxy"].nunique()),
            "radii": int(mask.sum()),
            "excluded_radii": int((~mask).sum()),
            "excluded_galaxies_touched": int(base.loc[~mask, "galaxy"].nunique()) if (~mask).any() else 0,
        }
    counts["S_NO_UGC01281_DOMAIN_SAFE"] = {
        "galaxies": int(base.loc[domain_safe, "galaxy"].nunique()),
        "radii": int(domain_safe.sum()),
        "excluded_radii": int((~domain_safe).sum()),
        "rule": "Exclude every UGC01281 radius, then require B(0.05)>0. Not activated.",
    }
    expected_counts = {
        "S_FULL": (91, 1782),
        "S_HARD2": (91, 1780),
        "S_DOMAIN13": (91, 1769),
    }
    for name, (ng, nr) in expected_counts.items():
        if (counts[name]["galaxies"], counts[name]["radii"]) != (ng, nr):
            raise SystemExit(f"{name} counts {counts[name]} differ from the predeclared expectation")

    # Full-box proof for S_DOMAIN13.
    keep = masks["S_DOMAIN13"]
    star_k = stellar[keep]
    gas_k = gas_s[keep]
    bmin = b_lo[keep]
    ugrid = np.linspace(U_LO, U_HI, 1001)
    grid = ugrid[:, None] * star_k[None, :] + gas_k[None, :]
    cond_lo = (np.abs(U_LO * star_k) + np.abs(gas_k)) / bmin
    cond_hi = (np.abs(U_HI * star_k) + np.abs(gas_k)) / (U_HI * star_k + gas_k)
    cross = np.full(keep.sum(), np.nan)
    neg_gas = gas_k < 0.0
    cross[neg_gas] = -gas_k[neg_gas] / star_k[neg_gas]
    scale_checks = []
    for s in (0.25, 1.0, 4.0):
        root = np.sqrt(s)
        left = (
            U_LO * (signed_square(root * disk[keep]) + signed_square(root * bulge[keep]))
            + signed_square(root * gas[keep])
        )
        right = s * bmin
        scale_checks.append(
            {
                "s": s,
                "min_scaled_B": float(np.min(left)),
                "max_abs_gap_versus_s_times_B": float(np.max(np.abs(left - right))),
                "all_positive": bool(np.all(left > 0.0)),
            }
        )
    g_n = bmin * (KM ** 2) / (radius[keep] * KPC)
    y_at_a0_max = g_n / A0_HI
    proof = {
        "stellar_signed_contribution_min": float(np.min(stellar)),
        "stellar_nonnegative_on_all_1782": bool(np.min(stellar) >= 0.0),
        "B_affine_nondecreasing_because_stellar_nonnegative": True,
        "minimum_B_at_U_0.05": float(np.min(bmin)),
        "max_abs_gap_versus_frozen_csv_B": csv_gap,
        "row_of_minimum_B": {
            "galaxy": str(base.loc[keep, "galaxy"].to_numpy()[int(np.argmin(bmin))]),
            "r_kpc": float(base.loc[keep, "r_kpc"].to_numpy()[int(np.argmin(bmin))]),
            "source_recno": int(base.loc[keep, "source_recno"].to_numpy()[int(np.argmin(bmin))]),
        },
        "grid_minimum_B_on_1001_upsilon_nodes": float(np.min(grid)),
        "endpoints_positive": {
            "U_0.05": bool(np.all(b_lo[keep] > 0.0)),
            "U_0.5": bool(np.all(b_mid[keep] > 0.0)),
            "U_1.5": bool(np.all(b_hi[keep] > 0.0)),
        },
        "zero_crossings_inside_interval": int(np.sum((cross >= U_LO) & (cross <= U_HI))),
        "maximum_zero_crossing_among_retained_negative_gas": (
            None if not np.any(neg_gas) else float(np.max(cross[neg_gas]))
        ),
        "retained_negative_gas_radii": int(np.sum(gas[keep] < 0.0)),
        "retained_negative_gas_galaxies": int(base.loc[keep & (gas < 0.0), "galaxy"].nunique()),
        "max_condition_number_at_U_0.05": float(np.max(cond_lo)),
        "max_condition_number_at_U_1.5": float(np.max(cond_hi)),
        "machine_epsilon": float(np.finfo(float).eps),
        "distance_rescaling": scale_checks,
        "minimum_newtonian_g_at_U_0.05_m_s2": float(np.min(g_n)),
        "minimum_y_at_maximum_matched_a0": float(np.min(y_at_a0_max)),
        "y_floor_1e-12_inactive_on_matched_a0_box": bool(np.min(y_at_a0_max) >= Y_FLOOR),
        "nuisance_sign_convention": {
            "distance_nodes_open_above_zero": True,
            "velocity_factor": "sqrt(D_node/D_nominal) > 0",
            "inclination_factor_applies_to_predicted_speed_not_baryonic_sign": True,
            "integrator_called": False,
        },
    }
    domain13_pass = (
        proof["stellar_nonnegative_on_all_1782"]
        and proof["endpoints_positive"]["U_0.05"]
        and proof["grid_minimum_B_on_1001_upsilon_nodes"] > 0.0
        and proof["zero_crossings_inside_interval"] == 0
        and all(row["all_positive"] for row in scale_checks)
        and proof["max_condition_number_at_U_0.05"] * np.finfo(float).eps < 1e-8
    )

    hard_rows = base.loc[conditional, [
        "source_recno", "galaxy", "r_kpc", "zero_crossing_upsilon", "B_at_U_0.05", "stellar_signed_v2", "gas_signed_v2"
    ]].copy()
    galaxy_gate = (
        hard_rows.groupby("galaxy", as_index=False)
        .agg(
            conditional_radii=("r_kpc", "size"),
            upsilon_threshold_exclusive=("zero_crossing_upsilon", "max"),
            most_restrictive_radius_kpc=("r_kpc", "max"),
        )
        .sort_values("galaxy")
    )
    # The restrictive radius is the one with the maximum threshold, not the largest radius.
    idx = hard_rows.groupby("galaxy")["zero_crossing_upsilon"].idxmax()
    galaxy_gate = hard_rows.loc[idx, ["galaxy", "r_kpc", "source_recno", "zero_crossing_upsilon"]].rename(
        columns={
            "r_kpc": "most_restrictive_r_kpc",
            "source_recno": "most_restrictive_source_recno",
            "zero_crossing_upsilon": "upsilon_must_exceed",
        }
    )
    galaxy_gate["conditional_radii"] = hard_rows.groupby("galaxy").size().reindex(galaxy_gate.galaxy).to_numpy()
    galaxy_gate["original_box_entirely_feasible"] = False
    galaxy_gate["feasible_open_interval"] = galaxy_gate["upsilon_must_exceed"].map(
        lambda u: f"({u}, 1.5]"
    )

    def support_row(sample: str, model: str, status: str, reason: str) -> dict:
        return {"sample": sample, "model": model, "status": status, "reason": reason}

    models = ["baryon", "mond", "ptq-nu", "mond-screen", "ptq-screen", "ptq", "nfw1p"]
    matrix = []
    for model in models[:5]:
        matrix.append(support_row(
            "S_FULL", model, "NOT_DEFINED_ON_ORIGINAL_BOX",
            "Two radii have B<0 for every Upsilon in [0.05,1.5]; eleven more are negative below an interior threshold. nu stays positive, so v^2 has the sign of B.",
        ))
        matrix.append(support_row(
            "S_HARD2", model, "NOT_DEFINED_ON_ORIGINAL_BOX",
            "Eleven retained radii are negative on a subinterval of the original Upsilon box.",
        ))
        matrix.append(support_row(
            "S_DOMAIN13", model, "DEFINED_ON_ORIGINAL_BOX",
            "B>0 on the whole Upsilon interval, g_N>0, and y at the maximum matched a0 stays above the 1e-12 floor. Fractional q of a positive y is real.",
        ))
        matrix.append(support_row(
            "S_NO_UGC01281", model, "NOT_DEFINED_ON_ORIGINAL_BOX",
            "Six radii outside UGC01281 remain negative on a Upsilon subinterval.",
        ))
        matrix.append(support_row(
            "S_NO_UGC01281_DOMAIN_SAFE", model, "DEFINED_ON_ORIGINAL_BOX",
            "This optional mask is a subset of S_DOMAIN13, so the same strict positivity applies. The extra galaxy deletion is not required by the domain rule.",
        ))
    matrix.append(support_row(
        "S_FULL", "ptq", "NOT_DEFINED_ON_ORIGINAL_BOX",
        "At epsilon=1e-6 and Upsilon=0.05 the thirteen negative-B radii remain negative. High epsilon can rescue them, which is a parameter restriction.",
    ))
    matrix.append(support_row(
        "S_HARD2", "ptq", "NOT_DEFINED_ON_ORIGINAL_BOX",
        "The eleven conditional radii are still negative at the epsilon lower bound when Upsilon is at its lower bound.",
    ))
    matrix.append(support_row(
        "S_DOMAIN13", "ptq", "DEFINED_ON_ORIGINAL_BOX",
        "B>0 and the linear term epsilon*c*H0*r is nonnegative for epsilon in (1e-6, 3.999) and r>0, so the sum stays positive.",
    ))
    matrix.append(support_row(
        "S_NO_UGC01281", "ptq", "NOT_DEFINED_ON_ORIGINAL_BOX",
        "Six conditional radii remain.",
    ))
    matrix.append(support_row(
        "S_NO_UGC01281_DOMAIN_SAFE", "ptq", "DEFINED_ON_ORIGINAL_BOX",
        "Subset of S_DOMAIN13.",
    ))
    matrix.append(support_row(
        "S_FULL", "nfw1p", "CONDITIONAL",
        "The earlier nine-point and 401-point mass scans found a positive slack, and the worst tested radius increased with mass. That is not a continuous proof for every mass in [1e9, 1e13] when B can be negative.",
    ))
    matrix.append(support_row(
        "S_HARD2", "nfw1p", "CONDITIONAL",
        "Removing the two worst radii does not convert the previous discrete halo scan into a continuous proof. B can still be negative on eleven radii.",
    ))
    matrix.append(support_row(
        "S_DOMAIN13", "nfw1p", "DEFINED_ON_ORIGINAL_BOX",
        "For r>0 and log10 M200 in [9,13], c(M)>0 and the halo speed is a real square root of a nonnegative quantity. B>0 then implies B+v_halo^2>0 at every mass, without a grid.",
    ))
    matrix.append(support_row(
        "S_NO_UGC01281", "nfw1p", "CONDITIONAL",
        "Six radii can still have B<0, so halo positivity is not guaranteed by B alone.",
    ))
    matrix.append(support_row(
        "S_NO_UGC01281_DOMAIN_SAFE", "nfw1p", "DEFINED_ON_ORIGINAL_BOX",
        "Subset of S_DOMAIN13.",
    ))

    def bias_block(name: str, mask: np.ndarray) -> dict:
        retained = base.loc[mask]
        excluded = base.loc[~mask]
        galaxies_losing = sorted(excluded.galaxy.unique().tolist()) if len(excluded) else []
        return {
            "galaxies_retained": int(retained.galaxy.nunique()) if len(retained) else 0,
            "radii_retained": int(len(retained)),
            "radii_excluded": int(len(excluded)),
            "galaxies_with_any_excluded_radius": galaxies_losing,
            "retained_radius_kpc": qstats(retained.r_kpc.to_numpy()),
            "excluded_radius_kpc": qstats(excluded.r_kpc.to_numpy()),
            "retained_r_over_Rdisk": qstats(retained.r_over_Rdisk.to_numpy()),
            "excluded_r_over_Rdisk": qstats(excluded.r_over_Rdisk.to_numpy()),
            "excluded_inside_one_Rdisk": int((excluded.r_over_Rdisk < 1.0).sum()) if len(excluded) else 0,
            "retained_gas_sign_counts": {
                "negative": int((retained.gas_sign < 0).sum()) if len(retained) else 0,
                "zero": int((retained.gas_sign == 0).sum()) if len(retained) else 0,
                "positive": int((retained.gas_sign > 0).sum()) if len(retained) else 0,
            },
            "excluded_gas_sign_counts": {
                "negative": int((excluded.gas_sign < 0).sum()) if len(excluded) else 0,
                "zero": int((excluded.gas_sign == 0).sum()) if len(excluded) else 0,
                "positive": int((excluded.gas_sign > 0).sum()) if len(excluded) else 0,
            },
            "retained_stellar_signed_v2": qstats(retained.stellar_signed_v2.to_numpy()),
            "excluded_stellar_signed_v2": qstats(excluded.stellar_signed_v2.to_numpy()),
            "retained_gas_signed_v2": qstats(retained.gas_signed_v2.to_numpy()),
            "excluded_gas_signed_v2": qstats(excluded.gas_signed_v2.to_numpy()),
            "retained_v_obs_kms": qstats(retained.v_obs_kms.to_numpy()),
            "excluded_v_obs_kms": qstats(excluded.v_obs_kms.to_numpy()),
            "retained_v_err_kms": qstats(retained.v_err_kms.to_numpy()),
            "excluded_v_err_kms": qstats(excluded.v_err_kms.to_numpy()),
            "retained_galaxy_Qual_counts": {int(k): int(v) for k, v in retained.drop_duplicates("galaxy").Qual.value_counts().sort_index().items()} if len(retained) else {},
            "excluded_point_galaxy_Qual_counts": {int(k): int(v) for k, v in excluded.drop_duplicates("galaxy").Qual.value_counts().sort_index().items()} if len(excluded) else {},
            "mask_uses_observed_response_residuals": False,
            "missingness_depends_on_baryonic_domain_variables": name != "S_FULL",
        }

    bias = {name: bias_block(name, mask) for name, mask in masks.items()}
    bias["S_NO_UGC01281_DOMAIN_SAFE"] = bias_block("S_NO_UGC01281_DOMAIN_SAFE", domain_safe)
    bias["interpretation"] = {
        "S_DOMAIN13_retains_negative_Vgas": True,
        "percent_of_radii_removed_by_S_DOMAIN13": 13.0 / 1782.0,
        "small_fraction_is_not_evidence_of_no_bias": True,
        "statistically_unbiased": False,
    }

    for name, mask in masks.items():
        out = base.copy()
        out.insert(0, "candidate", name)
        out.insert(1, "included", mask.astype(int))
        out.insert(2, "reason", reasons[name])
        if name == "S_NO_UGC01281":
            out["domain_safe_included"] = domain_safe.astype(int)
            out["domain_safe_reason"] = domain_safe_reason
        out.to_csv(NEW / f"{name}_MASK.csv", index=False)

    proof_doc = {
        "task": "REV26-P1G-A4-DOMAIN-DECISION-FEASIBILITY",
        "likelihood_evaluations": 0,
        "optimizer_calls": 0,
        "S_DOMAIN13_FULL_BOX": "PASS" if domain13_pass else "FAIL",
        "S_HARD2_FULL_BOX": "FAIL",
        "S_HARD2_note": "FAIL means the original parameter boxes are not entirely feasible. It does not mean the eleven rows are undefined for every admissible Upsilon.",
        "domain13_proof": proof,
        "hard2_permanent_rows": base.loc[permanent, ["source_recno", "galaxy", "r_kpc", "zero_crossing_upsilon", "B_at_U_0.05"]].to_dict(orient="records"),
        "hard2_conditional_rows": hard_rows.to_dict(orient="records"),
        "hard2_galaxy_parameter_constraints": galaxy_gate.to_dict(orient="records"),
        "distinctions": {
            "fixed_common_sample_unchanged_boxes": "S_DOMAIN13 satisfies this for the signed operator. S_HARD2 and S_NO_UGC01281 do not.",
            "parameter_constrained_feasible_domains": "For each of the five galaxies, Upsilon must exceed the listed threshold before every retained conditional radius has B>0. Bounds were not changed.",
            "permanently_undefined_observations": "UGC01281 source rows at 0.08 and 0.23 kpc.",
        },
        "linear_coefficient_sign": "epsilon*c*H0*r/(km^2) > 0 for epsilon>0 and r>0",
        "H0_SI": H0_SI,
        "C_LIGHT": C_LIGHT,
    }
    def _jsonable(value):
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(type(value))

    (NEW / "DOMAIN_PROOF.json").write_text(json.dumps(proof_doc, indent=2, default=_jsonable) + "\n")
    (NEW / "COMMON_SUPPORT_MATRIX.json").write_text(json.dumps({
        "task": "REV26-P1G-A4-DOMAIN-DECISION-FEASIBILITY",
        "native_code_still_uses_squared_gas": True,
        "analysis": "Analytical substitution of the isolated signed operator into the seven native formulas. No model function was evaluated as a likelihood.",
        "rows": matrix,
    }, indent=2, default=_jsonable) + "\n")
    (NEW / "SELECTION_BIAS_SUMMARY.json").write_text(json.dumps({
        "task": "REV26-P1G-A4-DOMAIN-DECISION-FEASIBILITY",
        "predeclared_inner_rule": "A radius is inside one disk scale length when r_kpc/Rdisk < 1. Rdisk is SPARC Table 1 Rdisk in kpc, joined by galaxy name.",
        "candidates": bias,
    }, indent=2, default=_jsonable) + "\n")
    (NEW / "SAMPLE_CANDIDATES.json").write_text(json.dumps({
        "task": "REV26-P1G-A4-DOMAIN-DECISION-FEASIBILITY",
        "activated_sample": None,
        "primary_sample": "PENDING_INDEPENDENT_REVIEW",
        "candidates": counts,
        "rules": {
            "S_FULL": "All 91 galaxies and 1782 radii.",
            "S_HARD2": "Remove only radii with B<0 for every Upsilon in [0.05,1.5].",
            "S_DOMAIN13": "Keep radii with B(0.05)>0. Stellar signed contribution is nonnegative, so B stays positive on [0.05,1.5].",
            "S_NO_UGC01281": "Exclude the whole galaxy as an unactivated literature-sensitivity candidate.",
            "S_NO_UGC01281_DOMAIN_SAFE": "Optional and unactivated: S_NO_UGC01281 intersect S_DOMAIN13.",
        },
    }, indent=2, default=_jsonable) + "\n")
    print(json.dumps({
        "counts": counts,
        "domain13_pass": domain13_pass,
        "min_B": proof["minimum_B_at_U_0.05"],
        "min_row": proof["row_of_minimum_B"],
        "max_cond": proof["max_condition_number_at_U_0.05"],
        "y_min": proof["minimum_y_at_maximum_matched_a0"],
        "neg_gas_retained": proof["retained_negative_gas_radii"],
        "neg_gas_galaxies": proof["retained_negative_gas_galaxies"],
        "max_cross": proof["maximum_zero_crossing_among_retained_negative_gas"],
        "scale": scale_checks,
        "galaxy_gate": galaxy_gate.to_dict(orient="records"),
        "bias_domain13_excluded_r": bias["S_DOMAIN13"]["excluded_radius_kpc"],
        "bias_domain13_excluded_rdisk": bias["S_DOMAIN13"]["excluded_r_over_Rdisk"],
        "inside": bias["S_DOMAIN13"]["excluded_inside_one_Rdisk"],
        "gas_retained": bias["S_DOMAIN13"]["retained_gas_sign_counts"],
        "vobs_ex": bias["S_DOMAIN13"]["excluded_v_obs_kms"],
        "vobs_re": bias["S_DOMAIN13"]["retained_v_obs_kms"],
        "verr_ex": bias["S_DOMAIN13"]["excluded_v_err_kms"],
        "verr_re": bias["S_DOMAIN13"]["retained_v_err_kms"],
        "star_ex": bias["S_DOMAIN13"]["excluded_stellar_signed_v2"],
        "gas_ex": bias["S_DOMAIN13"]["excluded_gas_signed_v2"],
        "qual_full": bias["S_FULL"]["retained_galaxy_Qual_counts"],
        "hard_galaxies": bias["S_HARD2"]["galaxies_with_any_excluded_radius"],
        "nougc_conditional_remaining": int((conditional & masks["S_NO_UGC01281"]).sum()),
    }, indent=2, default=_jsonable))


if __name__ == "__main__":
    main()
