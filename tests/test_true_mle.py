"""R1B true-MLE tests: objective excludes priors; reparameterization invariance."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from ptquat.constants import C_LIGHT, H0_SI
from ptquat.data import load_tidy_sparc
from ptquat.fit_global import _layout, log_likelihood, log_prior
from ptquat.mle import (
    a0_to_epsilon,
    evaluate_loglike,
    load_historical_median_theta,
    map_theta_mond_to_ptqnu,
    map_theta_mondscreen_to_ptqscreen,
    map_theta_ptqnu_to_mond,
    map_theta_ptqscreen_to_mondscreen,
    matched_epsilon_bounds,
    objective_neg_loglike,
    optimize_mle,
    parameter_domain_manifest,
    theta_bounds,
)

IMPL_ROOT = Path(__file__).resolve().parents[1]
TIDY = IMPL_ROOT / "dataset" / "sparc_tidy.csv"


@pytest.fixture(scope="module")
def galaxies():
    gdict = load_tidy_sparc(str(TIDY))
    return [gdict[k] for k in sorted(gdict.keys())]


def test_parameter_counts_r1b(galaxies):
    G = len(galaxies)
    assert G == 91
    expected = {
        "baryon": 92,
        "mond": 93,
        "ptq": 93,
        "ptq-nu": 93,
        "ptq-screen": 94,
        "mond-screen": 94,
        "nfw1p": 183,
    }
    for m, k in expected.items():
        assert _layout(m, G, True, None)["k"] == k


def test_mle_objective_excludes_priors(galaxies):
    """Changing prior density must not change the MLE objective value."""
    G = len(galaxies)
    theta, _ = load_historical_median_theta(IMPL_ROOT, "ptq-nu", galaxies, H0_SI)
    L = _layout("ptq-nu", G, True, None)
    ll = log_likelihood(theta, galaxies, 4.0, H0_SI, L, 10.0, -0.1, None, "gauss", 8.0)
    # Prior values differ for galaxies-only vs evaluating prior at same point,
    # but objective_neg_loglike must equal -ll exactly (no prior addend).
    neg = objective_neg_loglike(theta, galaxies, "ptq-nu", H0_SI)
    assert abs(neg - (-ll)) < 1e-9
    lp = log_prior(theta, galaxies, "galaxies-only", L, (9.0, 13.0), (5e-11, 2e-10))
    # If priors were wrongly included, neg would be -(ll+lp)
    assert abs(neg - (-(ll + lp))) > 1.0  # prior contribution is material


def test_full_covariance_in_loglike_matches_fit_global(galaxies):
    theta, meta = load_historical_median_theta(IMPL_ROOT, "mond", galaxies, H0_SI)
    ll = evaluate_loglike(theta, galaxies, "mond", H0_SI)
    assert abs(ll - meta["historical_logL_total_full"]) < 1e-8


def test_matched_domain_mapping():
    man = parameter_domain_manifest(H0_SI)
    lo, hi = man["matched_epsilon_bounds"]
    assert abs(lo - 5e-11 / (C_LIGHT * H0_SI)) < 1e-18
    assert abs(hi - 2e-10 / (C_LIGHT * H0_SI)) < 1e-18


def test_ptqnu_mond_mle_invariance(galaxies):
    """Matched-domain MLE max logL must agree within 1e-5."""
    # Fast path: optimize mond from historical, map to ptq-nu, and also optimize ptq-nu.
    starts_m = [("hist", load_historical_median_theta(IMPL_ROOT, "mond", galaxies)[0])]
    starts_p = [("hist", load_historical_median_theta(IMPL_ROOT, "ptq-nu", galaxies)[0])]
    # Add mapped start
    starts_p.append(("mapped", map_theta_mond_to_ptqnu(starts_m[0][1], len(galaxies))))
    starts_m.append(("mapped", map_theta_ptqnu_to_mond(starts_p[0][1], len(galaxies))))

    res_m = optimize_mle("mond", galaxies, starts_m, maxiter=8)
    res_p = optimize_mle("ptq-nu", galaxies, starts_p, maxiter=8)
    d = abs(res_p["max_loglike"] - res_m["max_loglike"])
    assert d < 1e-5, f"delta_logL={d}"

    # Mapped optima must evaluate identically
    th_map = map_theta_mond_to_ptqnu(np.asarray(res_m["theta"]), len(galaxies))
    ll = evaluate_loglike(th_map, galaxies, "ptq-nu")
    assert abs(ll - res_m["max_loglike"]) < 1e-8


def test_ptqscreen_mondscreen_mle_invariance(galaxies):
    starts_m = [("hist", load_historical_median_theta(IMPL_ROOT, "mond-screen", galaxies)[0])]
    starts_p = [("hist", load_historical_median_theta(IMPL_ROOT, "ptq-screen", galaxies)[0])]
    starts_p.append(("mapped", map_theta_mondscreen_to_ptqscreen(starts_m[0][1], len(galaxies))))
    starts_m.append(("mapped", map_theta_ptqscreen_to_mondscreen(starts_p[0][1], len(galaxies))))

    res_m = optimize_mle("mond-screen", galaxies, starts_m, maxiter=8)
    res_p = optimize_mle("ptq-screen", galaxies, starts_p, maxiter=8)
    d = abs(res_p["max_loglike"] - res_m["max_loglike"])
    assert d < 1e-5, f"delta_logL={d}"


def test_interior_optimum_mond(galaxies):
    th, _ = load_historical_median_theta(IMPL_ROOT, "mond", galaxies)
    # Single-start local refinement
    res = optimize_mle("mond", galaxies, [("hist", th)], maxiter=8)
    a0 = res["parameter_vector"]["a0"]
    lo, hi = 5e-11, 2e-10
    assert lo < a0 < hi
