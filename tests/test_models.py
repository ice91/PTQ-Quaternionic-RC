import numpy as np
from ptquat.models import vbar_squared_kms2, linear_term_kms2, model_v_kms

def test_linear_term_units_monotonic():
    r = np.array([1.0, 5.0, 10.0])  # kpc
    lin = linear_term_kms2(1.0, r)  # epsilon=1
    assert np.all(np.diff(lin) > 0.0)
    # Order of magnitude sanity: at 10 kpc should be > at 1 kpc
    assert lin[-1] > 5 * lin[0]

def test_model_shapes():
    r = np.linspace(1, 20, 5)
    vd = np.full_like(r, 100.0)
    vb = np.zeros_like(r)
    vg = np.full_like(r, 30.0)
    v = model_v_kms(0.5, 1.0, r, vd, vb, vg)
    assert v.shape == r.shape
    assert np.all(v > 0)
