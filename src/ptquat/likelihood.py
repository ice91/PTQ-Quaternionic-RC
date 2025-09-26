from __future__ import annotations
import numpy as np
from numpy.linalg import slogdet, solve
from .constants import DEG2RAD, DR_NUM_KPC
from .models import model_v_kms

def _num_grad_v_wrt_r(r_kpc, v_fun, h=DR_NUM_KPC):
    """
    Central finite difference derivative dv/dr at each radius.
    v_fun takes r_k as input and returns v(r_k).
    """
    r_plus  = r_kpc + h
    r_minus = np.clip(r_kpc - h, a_min=1e-6, a_max=None)
    v_plus  = v_fun(r_plus)
    v_minus = v_fun(r_minus)
    return (v_plus - v_minus) / (r_plus - r_minus)

def build_covariance(v_model_kms: np.ndarray,
                     r_kpc: np.ndarray,
                     v_err_meas_kms: np.ndarray,
                     D_Mpc: float, D_err_Mpc: float,
                     i_deg: float, i_err_deg: float,
                     v_fun_for_grad,                 # callable: r_kpc -> v_model(r)
                     sigma_sys_kms: float = 4.0) -> np.ndarray:
    """
    C = diag(meas^2) + sigma_D^2 J_D J_D^T + sigma_i^2 J_i J_i^T + sigma_sys^2 I
    with
      J_D = dv/dD ≈ (dv/dr) * (r/D)
      J_i = dv/di ≈ v * cot(i)
    where i is in radians. All in velocity units [km/s].
    """
    n = len(v_model_kms)
    C = np.diag(np.asarray(v_err_meas_kms)**2)

    # Distance term
    if D_err_Mpc > 0:
        dv_dr = _num_grad_v_wrt_r(r_kpc, v_fun_for_grad)
        J_D = dv_dr * (r_kpc / D_Mpc)  # since r = D * theta => dr/dD = r/D
        C += (D_err_Mpc**2) * np.outer(J_D, J_D)

    # Inclination term
    if i_err_deg > 0:
        i_rad = i_deg * DEG2RAD
        cot_i = np.cos(i_rad) / np.sin(i_rad)
        J_i = v_model_kms * cot_i
        C += (i_err_deg * DEG2RAD)**2 * np.outer(J_i, J_i)

    # Velocity floor
    if sigma_sys_kms > 0:
        C += (sigma_sys_kms**2) * np.eye(n)

    return C

def gaussian_loglike(v_obs_kms: np.ndarray,
                     v_model_kms: np.ndarray,
                     C: np.ndarray) -> float:
    """
    -1/2 [ r^T C^{-1} r + ln det C + n ln 2pi ]
    """
    r = v_obs_kms - v_model_kms
    sign, logdet = slogdet(C)
    if sign <= 0:
        # Regularize softly if numerical issues
        eps = 1e-6 * np.median(np.diag(C))
        sign, logdet = slogdet(C + eps*np.eye(C.shape[0]))
    alpha = solve(C, r, assume_a='pos')
    quad = r @ alpha
    n = C.shape[0]
    return -0.5 * (quad + logdet + n*np.log(2*np.pi))
