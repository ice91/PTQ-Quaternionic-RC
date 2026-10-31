"""Profiles, cross-validation and galaxy influence from the cached grid only."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

TEST = Path("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/a15_conditional_sparc_evidence")
LR = 0.5 * 3.841458820694124  # uncalibrated one-parameter chi-square 0.95 threshold on Delta F


def classify(F):
    F = np.asarray(F, float)
    if not np.isfinite(F).all():
        return "NONFINITE_OR_DOMAIN_FAILURE"
    i = int(np.argmin(F))
    if i == 0 or i == len(F) - 1:
        diffs = np.diff(F)
        if np.all(diffs >= -1e-8) or np.all(diffs <= 1e-8):
            return "CONDITIONAL_MONOTONICITY"
        return "CONDITIONAL_BOUNDARY_PREFERENCE"
    left = float(F[i - 1] - F[i])
    right = float(F[i + 1] - F[i])
    inside = np.where(F - F[i] <= LR)[0]
    width = int(inside.max() - inside.min())
    if left < 0.05 and right < 0.05:
        return "GRID_UNRESOLVED"
    if width < 1:
        return "GRID_UNRESOLVED"
    return "CONDITIONAL_INTERIOR_MINIMUM"


def main():
    study = json.loads((TEST / "PREDECLARED_STUDY.json").read_text())
    store = np.load(TEST / "PER_GALAXY_LOGL_CACHE.npz")
    if not bool(store["done"].all()):
        raise SystemExit("cache incomplete")
    logL = store["logL"]
    names = study["galaxies"]
    assign = study["fold_assignment"]
    models = ["ptq-nu", "ptq"]
    ups = [0.5, 0.7]
    sigs = [5.0, 10.0]
    profiles = []
    for im, model in enumerate(models):
        a0 = np.asarray(study["grids"][model]["a0_si"], float)
        eps = np.asarray(study["grids"][model]["epsilon"], float)
        for iu, U in enumerate(ups):
            for iss, sigma in enumerate(sigs):
                F = -logL[im, iu, iss].sum(axis=1)
                i = int(np.argmin(F))
                inside = np.where(np.isfinite(F) & (F - F[i] <= LR))[0]
                profiles.append({
                    "model": model,
                    "upsilon": U,
                    "sigma_sys_kms": sigma,
                    "status": classify(F),
                    "best_node": i,
                    "best_epsilon": float(eps[i]),
                    "best_a0_si": float(a0[i]),
                    "F": float(F[i]),
                    "neighbor_F": {
                        "left": None if i == 0 else float(F[i - 1]),
                        "center": float(F[i]),
                        "right": None if i == len(F) - 1 else float(F[i + 1]),
                    },
                    "F_profile": [float(x) for x in F],
                    "logL_profile": [float(x) for x in logL[im, iu, iss].sum(axis=1)],
                    "uncalibrated_lr_nodes": [int(k) for k in inside],
                    "uncalibrated_a0_span_si": [float(a0[inside.min()]), float(a0[inside.max()])] if len(inside) else [],
                    "note": "Delta F <= 1.9207 is an uncalibrated one-parameter illustration with Upsilon and sigma fixed. It is not a confidence interval.",
                })
    (TEST / "CONDITIONAL_PROFILE_RESULTS.json").write_text(json.dumps(profiles, indent=2) + "\n")

    cv = []
    held = {}
    for im, model in enumerate(models):
        a0 = np.asarray(study["grids"][model]["a0_si"], float)
        eps = np.asarray(study["grids"][model]["epsilon"], float)
        for iu, U in enumerate(ups):
            for iss, sigma in enumerate(sigs):
                fold_rows = []
                scores = np.full(91, np.nan)
                chosen = np.full(91, -1, dtype=int)
                for fold in range(5):
                    test = np.array([assign[n] == fold for n in names])
                    train = ~test
                    Ftrain = -logL[im, iu, iss][:, train].sum(axis=1)
                    node = int(np.argmin(Ftrain))
                    scores[test] = logL[im, iu, iss][node, test]
                    chosen[test] = node
                    fold_rows.append({
                        "fold": fold,
                        "n_train": int(train.sum()),
                        "n_test": int(test.sum()),
                        "training_node": node,
                        "training_a0_si": float(a0[node]),
                        "training_epsilon": float(eps[node]),
                        "training_F": float(Ftrain[node]),
                        "heldout_logscore": float(logL[im, iu, iss][node, test].sum()),
                    })
                key = (model, U, sigma)
                held[key] = scores
                cv.append({
                    "model": model,
                    "upsilon": U,
                    "sigma_sys_kms": sigma,
                    "folds": fold_rows,
                    "total_oos_logscore": float(np.nansum(scores)),
                    "unique_training_a0_si": sorted({row["training_a0_si"] for row in fold_rows}),
                })
    paired = []
    for U in ups:
        for sigma in sigs:
            diff = held[("ptq-nu", U, sigma)] - held[("ptq", U, sigma)]
            order = np.argsort(-np.abs(diff))
            paired.append({
                "upsilon": U,
                "sigma_sys_kms": sigma,
                "total_oos_ptq_nu": float(held[("ptq-nu", U, sigma)].sum()),
                "total_oos_linear": float(held[("ptq", U, sigma)].sum()),
                "total_paired_difference_nu_minus_linear": float(diff.sum()),
                "n_galaxies_nu_higher": int(np.sum(diff > 0)),
                "n_galaxies_linear_higher": int(np.sum(diff < 0)),
                "n_ties": int(np.sum(diff == 0)),
                "median_paired_difference": float(np.median(diff)),
                "largest_positive": [
                    {"galaxy": names[i], "difference": float(diff[i])} for i in order if diff[i] > 0
                ][:8],
                "largest_negative": [
                    {"galaxy": names[i], "difference": float(diff[i])} for i in order if diff[i] < 0
                ][:8],
            })
    (TEST / "CROSS_VALIDATION_RESULTS.json").write_text(json.dumps({"folds": cv, "paired": paired}, indent=2) + "\n")

    stability = []
    rng = np.random.default_rng(20261031)
    for item in paired:
        U, sigma = item["upsilon"], item["sigma_sys_kms"]
        diff = held[("ptq-nu", U, sigma)] - held[("ptq", U, sigma)]
        total = float(diff.sum())
        loo = []
        for i, name in enumerate(names):
            loo.append({"galaxy": name, "total_without_galaxy": float(total - diff[i]), "contribution": float(diff[i])})
        flips = [row for row in loo if (total > 0) != (row["total_without_galaxy"] > 0)]
        boots = np.array([rng.choice(diff, size=91, replace=True).sum() for _ in range(1000)])
        stability.append({
            "upsilon": U,
            "sigma_sys_kms": sigma,
            "total_paired_difference": total,
            "galaxies_whose_removal_flips_sign": flips,
            "descriptive_resample_sum_percentiles": {
                "p05": float(np.percentile(boots, 5)),
                "p50": float(np.percentile(boots, 50)),
                "p95": float(np.percentile(boots, 95)),
                "note": "Resampling of already fixed cross-validated scores. Not a confidence interval.",
            },
            "contributions": loo,
        })
    (TEST / "GALAXY_STABILITY_RESULTS.json").write_text(json.dumps(stability, indent=2) + "\n")
    print("analysis written")


if __name__ == "__main__":
    main()
