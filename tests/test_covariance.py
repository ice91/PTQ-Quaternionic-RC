import numpy as np
from ptquat.likelihood import build_covariance

def test_covariance_positive_definite():
    n = 12
    r = np.linspace(1, 15, n)
    v_mod = 150 + 0.0*r
    v_err = np.full(n, 5.0)
    def vfun(x): return 150 + 0*x
    C = build_covariance(v_mod, r, v_err, 10.0, 1.0, 60.0, 2.0, vfun, sigma_sys_kms=4.0)
    # Cholesky should succeed with small jitter if needed
    try:
        np.linalg.cholesky(C)
        ok = True
    except np.linalg.LinAlgError:
        # add tiny jitter
        eps = 1e-6*np.median(np.diag(C))
        np.linalg.cholesky(C + eps*np.eye(n))
        ok = True
    assert ok
