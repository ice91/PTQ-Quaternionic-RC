# src/ptquat/likelihood.py
from __future__ import annotations
import numpy as np
from numpy.linalg import cholesky
from .constants import DEG2RAD, DR_NUM_KPC


def _num_grad_v_wrt_r(r_kpc, v_fun, h=DR_NUM_KPC):
    """
    中心差分估計 dv/dr。v_fun 需可向量化地回傳 v(r_kpc)。
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
    其中
      J_D = dv/dD ≈ (dv/dr) * (r/D)      (因 r = D*theta, dr/dD = r/D)
      J_i = dv/di ≈ v * cot(i)           (已在 deprojected 速度下回推)
    單位一致皆為 [km/s]。
    """
    n = len(v_model_kms)
    C = np.diag(np.asarray(v_err_meas_kms, float)**2)

    # Distance term
    if D_Mpc > 0 and D_err_Mpc > 0:
        dv_dr = _num_grad_v_wrt_r(r_kpc, v_fun_for_grad)
        J_D = dv_dr * (r_kpc / D_Mpc)
        C += (D_err_Mpc**2) * np.outer(J_D, J_D)

    # Inclination term
    if i_err_deg > 0:
        i_rad = float(i_deg) * DEG2RAD
        # 避免 i→0 的數值不穩（品質切已有 i>30°）
        sin_i = np.maximum(np.sin(i_rad), 1e-6)
        cot_i = np.cos(i_rad) / sin_i
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
    高斯對數概似：
      -1/2 [ r^T C^{-1} r + ln det C + n ln(2π) ]

    純 NumPy 的穩定作法：
    - 先強制對稱 C，以減少數值殘餘造成的非正定。
    - 逐步加入輕微抖動（jitter）直到可 Cholesky。
    - 以同一份正則化後的 C_reg 同時計算 logdet 與解線性方程。
    """
    r = np.asarray(v_obs_kms, float) - np.asarray(v_model_kms, float)

    # 以同一個 C_reg 做所有後續運算
    C_reg = 0.5 * (C + C.T)  # 強制對稱
    n = C_reg.shape[0]

    # 設定初始 jitter（取對角中位數做尺度）
    diag_med = float(np.median(np.diag(C_reg))) if np.isfinite(np.median(np.diag(C_reg))) else 1.0
    jitter0 = max(1e-12 * diag_med, 1e-12)

    # 嘗試多次 Cholesky，必要時逐次放大 jitter
    L = None
    for k in range(4):  # 1e-12, 1e-11, 1e-10, 1e-9 倍階梯
        try:
            L = cholesky(C_reg)
            break
        except np.linalg.LinAlgError:
            C_reg = C_reg + (10.0**k) * jitter0 * np.eye(n)
    if L is None:
        # 最後保底再加大一些，理論上不太會走到這裡
        C_reg = C_reg + 1e-6 * np.eye(n)
        L = cholesky(C_reg)

    # log det C = 2 * sum(log(diag(L)))
    logdet = 2.0 * np.sum(np.log(np.diag(L)))

    # 以前代/回代解 alpha = C^{-1} r
    y = np.linalg.solve(L, r)
    alpha = np.linalg.solve(L.T, y)
    quad = float(r @ alpha)

    return -0.5 * (quad + logdet + n * np.log(2.0 * np.pi))

# --- Append to src/ptquat/likelihood.py ---

def student_t_loglike(y_obs, y_mod, C, nu: float):
    """
    Multivariate Student-t log-likelihood with scale matrix C and dof=nu.
    """
    import numpy as _np
    from math import lgamma as _lgamma, log as _log, pi as _pi
    r = _np.asarray(y_obs) - _np.asarray(y_mod)
    d = r.size
    if nu <= 2.0:
        nu = 2.0001
    sign, logdet = _np.linalg.slogdet(C)
    if sign <= 0 or not _np.isfinite(logdet):
        return -_np.inf
    alpha = _np.linalg.solve(C, r)
    z2 = float(r @ alpha)
    c0 = _lgamma(0.5*(nu + d)) - _lgamma(0.5*nu) - 0.5*d*_log(nu*_pi) - 0.5*logdet
    return c0 - 0.5*(nu + d)*_np.log1p(z2/nu)

