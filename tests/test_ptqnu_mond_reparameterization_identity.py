"""Deterministic identity: PTQ-nu(epsilon) == MOND(a0 = epsilon * c * H0)."""
from __future__ import annotations

import numpy as np
import pytest

from ptquat.constants import C_LIGHT, H0_SI
from ptquat.models import model_v_mond, model_v_ptq_nu


def _random_galaxy(rng: np.random.Generator, n: int = 24):
    r = np.sort(rng.uniform(0.3, 35.0, size=n))
    v_disk = rng.uniform(10.0, 180.0, size=n)
    v_bulge = rng.uniform(0.0, 80.0, size=n) * (rng.random(size=n) < 0.4)
    v_gas = rng.uniform(0.0, 60.0, size=n)
    Upsilon = float(rng.uniform(0.1, 1.2))
    epsilon = float(rng.uniform(1e-3, 3.5))
    H0 = float(H0_SI * rng.uniform(0.9, 1.1))
    return Upsilon, epsilon, r, v_disk, v_bulge, v_gas, H0


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 17, 19, 23])
def test_ptqnu_equals_mond_under_a0_map(seed: int):
    rng = np.random.default_rng(seed)
    Upsilon, epsilon, r, vd, vb, vg, H0 = _random_galaxy(rng)
    a0 = epsilon * C_LIGHT * H0
    v_ptq = model_v_ptq_nu(Upsilon, epsilon, r, vd, vb, vg, H0_si=H0)
    v_mond = model_v_mond(Upsilon, a0, r, vd, vb, vg)
    np.testing.assert_allclose(v_ptq, v_mond, rtol=0.0, atol=1e-10)


def test_ptqnu_mond_identity_grid_dense():
    """Broader grid covering edge cases (small/large y, bulge-heavy, gas-heavy)."""
    rng = np.random.default_rng(42)
    max_abs = 0.0
    for _ in range(200):
        Upsilon, epsilon, r, vd, vb, vg, H0 = _random_galaxy(rng, n=int(rng.integers(8, 40)))
        a0 = epsilon * C_LIGHT * H0
        v_ptq = model_v_ptq_nu(Upsilon, epsilon, r, vd, vb, vg, H0_si=H0)
        v_mond = model_v_mond(Upsilon, a0, r, vd, vb, vg)
        max_abs = max(max_abs, float(np.max(np.abs(v_ptq - v_mond))))
    assert max_abs < 1e-10, f"max abs delta = {max_abs}"
