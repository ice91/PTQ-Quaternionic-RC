"""
Deterministic identity under code implementation:
  PTQ-screen(epsilon, q) == MOND-screen(a0 = epsilon * c * H0, q)

NOTE (R1A audit): manuscript eq:screen_function / eq:ptq_screen_model describe a
Hill-type geometric kernel S(r/Rd; q), while the code implements nu_q(y) on
acceleration ratio y = gN/a0. This test verifies the *implemented* matched-kernel
reparameterization identity. Manuscript/code equation mismatch is classified
separately as FATAL_MODEL_EQUATION_MISMATCH.
"""
from __future__ import annotations

import numpy as np
import pytest

from ptquat.constants import C_LIGHT, H0_SI
from ptquat.models import model_v_mond_screen, model_v_ptq_screen, nu_q


def _random_galaxy(rng: np.random.Generator, n: int = 24):
    r = np.sort(rng.uniform(0.3, 35.0, size=n))
    v_disk = rng.uniform(10.0, 180.0, size=n)
    v_bulge = rng.uniform(0.0, 80.0, size=n) * (rng.random(size=n) < 0.4)
    v_gas = rng.uniform(0.0, 60.0, size=n)
    Upsilon = float(rng.uniform(0.1, 1.2))
    epsilon = float(rng.uniform(1e-3, 3.5))
    q = float(rng.uniform(0.25, 8.0))
    H0 = float(H0_SI * rng.uniform(0.9, 1.1))
    return Upsilon, epsilon, q, r, v_disk, v_bulge, v_gas, H0


@pytest.mark.parametrize("seed", list(range(15)))
def test_ptqscreen_equals_mondscreen_under_a0_map(seed: int):
    rng = np.random.default_rng(1000 + seed)
    Upsilon, epsilon, q, r, vd, vb, vg, H0 = _random_galaxy(rng)
    a0 = epsilon * C_LIGHT * H0
    v_ptq = model_v_ptq_screen(Upsilon, epsilon, q, r, vd, vb, vg, H0_si=H0)
    v_mond = model_v_mond_screen(Upsilon, a0, q, r, vd, vb, vg)
    np.testing.assert_allclose(v_ptq, v_mond, rtol=0.0, atol=1e-10)


def test_nu_q_reduces_to_simple_nu_at_q1():
    y = np.array([1e-3, 0.1, 1.0, 10.0, 1e3])
    np.testing.assert_allclose(nu_q(y, 1.0), 0.5 + np.sqrt(0.25 + 1.0 / y), atol=1e-12)


def test_ptqscreen_is_not_hill_S_of_r_over_Rd():
    """
    Guard: implemented PTQ-screen must NOT equal the manuscript Hill S(r/Rd;q)
    linear-channel form for generic inputs (documents equation mismatch).
    """
    rng = np.random.default_rng(7)
    Upsilon, epsilon, q, r, vd, vb, vg, H0 = _random_galaxy(rng)
    # Manuscript Hill form would require Rd; use a plausible Rd and compare shapes.
    Rd = float(np.median(r))
    a0 = epsilon * C_LIGHT * H0
    S = (r / Rd) ** q / (1.0 + (r / Rd) ** q)
    # Manuscript eq:ptq_screen_model: v^2 = v_bar^2 + a0 * r * S  (SI-consistent via linear_term)
    from ptquat.models import vbar_squared_kms2, linear_term_kms2

    vbar2 = vbar_squared_kms2(Upsilon, vd, vb, vg)
    lin = linear_term_kms2(epsilon, r, H0_si=H0)  # = (epsilon c H0) r in (km/s)^2
    v_hill = np.sqrt(np.maximum(vbar2 + lin * S, 0.0))
    v_code = model_v_ptq_screen(Upsilon, epsilon, q, r, vd, vb, vg, H0_si=H0)
    # They must differ on this generic draw; if they coincide, mismatch classification changes.
    assert float(np.max(np.abs(v_code - v_hill))) > 1.0, (
        "Unexpected near-equality between nu_q implementation and Hill S(r/Rd) form; "
        "revisit FATAL_MODEL_EQUATION_MISMATCH classification."
    )
