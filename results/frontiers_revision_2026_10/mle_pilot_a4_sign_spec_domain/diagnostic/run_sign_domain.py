"""Nominal-distance signed-baryon audit. No likelihood and no optimizer."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NEW = Path("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/mle_pilot_a4_sign_spec_domain")
sys.path.insert(0, str(NEW / "diagnostic"))
CAND = Path("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/mle_pilot_a/candidate/src")
sys.path.insert(0, str(CAND))

from signed_sparc_operator import (
    STATUS_DEFINED,
    STATUS_INVALID,
    baryonic_radial_acceleration,
    baryonic_signed_v2,
    distance_scaled_signed_square,
    model_circular_speed_domain,
    signed_square,
)
from ptquat.constants import C_LIGHT, H0_SI, KM, KPC
from ptquat.models import c_powerlaw, linear_term_kms2, nfw_v_kms_M200_c

IMPL = Path("/home/rocky/PTQ-Quaternionic-RC")
EX = IMPL / "results/frontiers_revision_2026_10/mle_pilot_a/execution_v1"
U_LO, U_HI = 0.05, 1.5
U_MID = 0.5
EPS_LO, EPS_HI = 1.0e-6, 3.999
LOGM_LO, LOGM_HI = 9.0, 13.0
EXPECTED = {
    "galaxies": 91,
    "radii": 1782,
    "negative_gas_galaxies": 18,
    "negative_gas_radii": 111,
    "positive_gas_radii": 1590,
    "zero_gas_radii": 81,
    "max_diff": 527.4751999999999,
    "sum_diff": 11050.037799999996,
}


def sign_label(x):
    if x < 0.0:
        return "B_signed<0"
    if x == 0.0:
        return "B_signed=0"
    return "B_signed>0"


def main():
    raw = pd.read_csv(IMPL / "dataset/raw/vizier_table2.csv")
    tidy = pd.read_csv(IMPL / "dataset/sparc_tidy.csv")
    merged = tidy.merge(raw, left_on=["galaxy", "r_kpc"], right_on=["Name", "Rad"], how="left", indicator=True)
    if len(tidy) != 1782 or tidy.galaxy.nunique() != 91 or (merged["_merge"] != "both").any() is False:
        raise SystemExit("sample shape")
    if not (merged["_merge"] == "both").all():
        raise SystemExit("raw join incomplete")
    copy_gap = float(np.max(np.abs(merged.v_gas_kms - merged.Vgas)))
    if copy_gap != 0.0 or float(np.max(np.abs(merged.v_disk_kms - merged.Vdisk))) != 0.0:
        raise SystemExit("tidy is not an exact copy of the raw velocities")

    disk = tidy.v_disk_kms.to_numpy(float)
    bulge = tidy.v_bulge_kms.to_numpy(float)
    gas = tidy.v_gas_kms.to_numpy(float)
    radius = tidy.r_kpc.to_numpy(float)
    names = tidy.galaxy.to_numpy()
    old = U_MID * (disk ** 2 + bulge ** 2) + gas ** 2
    signed = baryonic_signed_v2(U_MID, disk, bulge, gas)
    diff = old - signed
    neg = gas < 0.0
    pos = gas > 0.0
    zero = gas == 0.0
    outside = ~neg
    expected_diff = np.where(neg, 2.0 * gas ** 2, 0.0)
    residual = np.abs(diff - expected_diff)
    # Stellar terms cancel. A few ulps remain in the subtracted (km/s)^2 values.
    mismatch = np.flatnonzero(residual > 1e-9)
    disk_neg = int(np.sum(disk < 0.0))
    bulge_neg = int(np.sum(bulge < 0.0))
    recon_pass = (
        len(mismatch) == 0
        and int(neg.sum()) == EXPECTED["negative_gas_radii"]
        and int(pd.unique(names[neg]).size) == EXPECTED["negative_gas_galaxies"]
        and int(pos.sum()) == EXPECTED["positive_gas_radii"]
        and int(zero.sum()) == EXPECTED["zero_gas_radii"]
        and disk_neg == 0
        and bulge_neg == 0
        and float(np.max(np.abs(diff[outside]))) == 0.0
        and float(residual.max()) <= 1e-9
    )

    comp = tidy[["galaxy", "r_kpc"]].copy()
    comp["v_disk_kms"] = disk
    comp["v_bulge_kms"] = bulge
    comp["v_gas_kms"] = gas
    comp["comparison_upsilon"] = U_MID
    comp["old_squared_v2"] = old
    comp["signed_v2"] = signed
    comp["old_minus_signed"] = diff
    comp["gas_sign"] = np.where(neg, "negative", np.where(pos, "positive", "zero"))
    comp.to_csv(NEW / "BARYONIC_SIGN_DIFFERENCES.csv", index=False)

    # Distance-scaling fixtures. No integrator.
    fixtures = {
        "positive": np.array([2.0, 10.0]),
        "zero": np.array([0.0, 0.0]),
        "negative": np.array([-5.85, -16.24]),
    }
    scale_rows = []
    scale_pass = True
    for label, values in fixtures.items():
        for s in (0.25, 1.0, 4.0):
            left, right = distance_scaled_signed_square(values, s)
            gap = float(np.max(np.abs(left - right)))
            scale_rows.append({"component": label, "scale": s, "max_abs_gap": gap})
            if gap != 0.0:
                scale_pass = False
    # Catalog scaling uses sqrt(s) on every component, then one Upsilon.
    s = 1.7
    left = baryonic_signed_v2(U_MID, np.sqrt(s) * disk[:4], np.sqrt(s) * bulge[:4], np.sqrt(s) * gas[:4])
    right = s * baryonic_signed_v2(U_MID, disk[:4], bulge[:4], gas[:4])
    catalog_gap = float(np.max(np.abs(left - right)))
    # Real arithmetic gives equality. sqrt(s)**2 differs from s by a few ulps.
    if catalog_gap > 1e-12:
        scale_pass = False

    stellar = signed_square(disk) + signed_square(bulge)
    sgas = signed_square(gas)
    b_lo = baryonic_signed_v2(U_LO, disk, bulge, gas)
    b_mid = signed
    b_hi = baryonic_signed_v2(U_HI, disk, bulge, gas)
    threshold = np.full(len(tidy), np.nan)
    has_star = stellar > 0.0
    gas_neg = sgas < 0.0
    threshold[gas_neg & has_star] = -sgas[gas_neg & has_star] / stellar[gas_neg & has_star]
    all_negative = (b_lo < 0.0) & (b_hi < 0.0)
    partial = (b_lo < 0.0) & (b_hi >= 0.0)
    nonegative = b_lo >= 0.0

    order = sorted(tidy.galaxy.unique())
    if len(order) != 91 or order[9] != "F574-1":
        raise SystemExit("galaxy order does not match the frozen theta index")
    index = {name: i for i, name in enumerate(order)}
    archived = {}
    theta_files = {
        "baryon": EX / "baryon_last_accepted_theta.npy",
        "mond": EX / "mond_last_accepted_theta.npy",
        "ptq": EX / "ptq_last_accepted_theta.npy",
        "ptq-nu": EX / "ptq_nu_last_accepted_theta.npy",
        "ptq-screen": EX / "ptq_screen_last_accepted_theta.npy",
        "mond-screen": EX / "mond_screen_last_accepted_theta.npy",
        "nfw1p": EX / "nfw1p_initial_theta.npy",
    }
    u_offset = {"baryon": 0, "mond": 1, "ptq": 1, "ptq-nu": 1, "ptq-screen": 2, "mond-screen": 2, "nfw1p": 0}
    for model, path in theta_files.items():
        if not path.is_file():
            archived[model] = None
            continue
        theta = np.load(path)
        archived[model] = np.asarray(theta[u_offset[model]:u_offset[model] + 91], float)

    domain = tidy[["galaxy", "r_kpc"]].copy()
    domain["stellar_signed_v2"] = stellar
    domain["gas_signed_v2"] = sgas
    domain["B_at_U_0.05"] = b_lo
    domain["B_at_U_0.5"] = b_mid
    domain["B_at_U_1.5"] = b_hi
    domain["sign_at_U_0.05"] = [sign_label(x) for x in b_lo]
    domain["sign_at_U_0.5"] = [sign_label(x) for x in b_mid]
    domain["sign_at_U_1.5"] = [sign_label(x) for x in b_hi]
    domain["zero_crossing_upsilon"] = threshold
    domain["interval_class"] = np.where(all_negative, "NEGATIVE_ON_ENTIRE_ADMISSIBLE_INTERVAL", np.where(partial, "NEGATIVE_ON_A_SUBINTERVAL", "NONEGATIVE_ON_ADMISSIBLE_INTERVAL"))
    for model, values in archived.items():
        if values is None:
            domain[f"U_{model}"] = np.nan
            domain[f"B_{model}"] = np.nan
            continue
        u = np.array([values[index[g]] for g in names])
        domain[f"U_{model}"] = u
        domain[f"B_{model}"] = u * stellar + sgas
    domain.to_csv(NEW / "NEGATIVE_BARYONIC_DOMAIN.csv", index=False)

    # Acceleration sign follows B because r>0 on this sample.
    g_lo = baryonic_radial_acceleration(b_lo, radius, kpc_m=KPC, km_m=KM)
    if int(np.sum(~np.isfinite(g_lo))) != 0:
        raise SystemExit("nonfinite acceleration")
    if int(np.sum((g_lo < 0.0) != (b_lo < 0.0))) != 0:
        raise SystemExit("acceleration sign diverged from B")

    # Linear PTQ rescue. Coefficient is (km/s)^2 per unit epsilon.
    coeff = linear_term_kms2(1.0, radius, H0_si=H0_SI)
    def rescued(b, eps):
        total = b + eps * coeff
        return model_circular_speed_domain(total)["defined"]

    lin_defined_lo_epsmin = int(np.sum(rescued(b_lo, EPS_LO)))
    lin_defined_lo_epsmax = int(np.sum(rescued(b_lo, EPS_HI)))
    lin_defined_hi_epsmax = int(np.sum(rescued(b_hi, EPS_HI)))
    lin_nowhere = int(np.sum(~rescued(b_hi, EPS_HI)))

    # NFW halo squared speed on the mass bounds and a short interior grid.
    logm_grid = np.linspace(LOGM_LO, LOGM_HI, 9)
    halo = []
    for logm in logm_grid:
        mass = 10.0 ** float(logm)
        c = c_powerlaw(mass, c0=10.0, beta=-0.1)
        halo.append(nfw_v_kms_M200_c(radius, mass, c, H0_si=H0_SI) ** 2)
    halo = np.vstack(halo)
    halo_min = halo.min(axis=0)
    halo_max = halo.max(axis=0)
    nfw_nowhere = int(np.sum(b_hi + halo_max < 0.0))
    nfw_all_defined = int(np.sum(b_lo + halo_min >= 0.0))
    nfw_some = int(np.sum((b_hi + halo_max >= 0.0) & (b_lo + halo_min < 0.0)))

    def counts(mask):
        return {"radii": int(np.sum(mask)), "galaxies": int(pd.unique(names[mask]).size) if np.any(mask) else 0}

    ugc = domain[(domain.galaxy == "UGC01281") & (np.isin(domain.r_kpc, [0.08, 0.23]))]
    ugc_rows = ugc.to_dict(orient="records")

    nu_nowhere = counts(all_negative)
    nu_partial_galaxies = int(pd.unique(names[partial | all_negative]).size)

    summary = {
        "task": "REV26-P1G-A4-SIGN-SPEC-DOMAIN",
        "likelihood_evaluations": 0,
        "optimizer_calls": 0,
        "reconciliation_pass": bool(recon_pass),
        "mismatched_rows": [int(i) for i in mismatch[:20]],
        "n_mismatch": int(len(mismatch)),
        "counts": {
            "galaxies": int(tidy.galaxy.nunique()),
            "radii": int(len(tidy)),
            "negative_gas_galaxies": int(pd.unique(names[neg]).size),
            "negative_gas_radii": int(neg.sum()),
            "positive_gas_radii": int(pos.sum()),
            "zero_gas_radii": int(zero.sum()),
            "disk_negative_radii": disk_neg,
            "bulge_negative_radii": bulge_neg,
            "max_old_minus_signed": float(diff.max()),
            "sum_old_minus_signed": float(diff.sum()),
            "max_abs_diff_outside_negative_gas": float(np.max(np.abs(diff[outside]))),
            "max_abs_residual_versus_2_vgas2": float(residual.max()),
        },
        "distance_scaling_pass": bool(scale_pass),
        "distance_scaling_rows": scale_rows,
        "catalog_scale_gap": catalog_gap,
        "domain_counts": {
            "negative_at_0.05": counts(b_lo < 0.0),
            "zero_at_0.05": counts(b_lo == 0.0),
            "positive_at_0.05": counts(b_lo > 0.0),
            "negative_at_0.5": counts(b_mid < 0.0),
            "negative_at_1.5": counts(b_hi < 0.0),
            "entire_interval_negative": counts(all_negative),
            "subinterval_negative": counts(partial),
            "nonegative_interval": counts(nonegative),
        },
        "ugc01281": ugc_rows,
        "linear_ptq": {
            "epsilon_bounds": [EPS_LO, EPS_HI],
            "radii_defined_at_U0.05_eps_min": lin_defined_lo_epsmin,
            "radii_defined_at_U0.05_eps_max": lin_defined_lo_epsmax,
            "radii_defined_at_U1.5_eps_max": lin_defined_hi_epsmax,
            "radii_invalid_for_every_admissible_U_and_epsilon": lin_nowhere,
        },
        "nfw": {
            "log10_M200_bounds": [LOGM_LO, LOGM_HI],
            "radii_defined_for_every_admissible_U_and_M": nfw_all_defined,
            "radii_defined_for_some_but_not_all": nfw_some,
            "radii_invalid_for_every_admissible_U_and_M": nfw_nowhere,
            "halo_grid": logm_grid.tolist(),
            "minimum_B_at_U0.05_plus_grid_min_halo_v2": float(np.min(b_lo + halo_min)),
        },
        "nu_like_nowhere_defined_radii": nu_nowhere,
        "galaxies_with_any_negative_B_inside_bounds": nu_partial_galaxies,
        "archived_theta_files_present": {k: v is not None for k, v in archived.items()},
        "circular_speed_status_labels": [STATUS_DEFINED, STATUS_INVALID],
    }
    # Archived sign counts.
    arch_summary = {}
    for model, values in archived.items():
        if values is None:
            arch_summary[model] = {"present": False}
            continue
        column = domain[f"B_{model}"].to_numpy(float)
        arch_summary[model] = {"present": True, "negative_radii": int(np.sum(column < 0.0)), "negative_galaxies": int(pd.unique(names[column < 0.0]).size) if np.any(column < 0.0) else 0}
    summary["archived_signed_B"] = arch_summary
    (NEW / "SIGNED_OPERATOR_TESTS.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("reconciliation_pass", "counts", "distance_scaling_pass", "domain_counts", "linear_ptq", "nfw", "archived_signed_B", "ugc01281")}, indent=2))


if __name__ == "__main__":
    main()
