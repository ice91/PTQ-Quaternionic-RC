from __future__ import annotations
import numpy as np
from .constants import C_LIGHT, H0_SI, KPC, KM

def vbar_squared_kms2(Upsilon: float,
                      v_disk_kms: np.ndarray,
                      v_bulge_kms: np.ndarray,
                      v_gas_kms: np.ndarray) -> np.ndarray:
    """
    v_bar^2 = Upsilon * (v_disk^2 + v_bulge^2) + v_gas^2   [ (km/s)^2 ]
    """
    v_star2 = v_disk_kms**2 + v_bulge_kms**2
    return Upsilon * v_star2 + v_gas_kms**2

def linear_term_kms2(epsilon: float, r_kpc: np.ndarray, H0_si: float = H0_SI) -> np.ndarray:
    """
    (epsilon * c * H0) * r   in (km/s)^2.
    r is in kpc.
    """
    r_m = r_kpc * KPC
    term_m2s2 = epsilon * C_LIGHT * H0_si * r_m
    return term_m2s2 / (KM**2)

def model_v_kms(Upsilon: float,
                epsilon: float,
                r_kpc: np.ndarray,
                v_disk_kms: np.ndarray,
                v_bulge_kms: np.ndarray,
                v_gas_kms: np.ndarray,
                H0_si: float = H0_SI) -> np.ndarray:
    """
    v_model = sqrt( v_bar^2 + (epsilon c H0) r )  [km/s]
    """
    vbar2 = vbar_squared_kms2(Upsilon, v_disk_kms, v_bulge_kms, v_gas_kms)
    lin   = linear_term_kms2(epsilon, r_kpc, H0_si=H0_si)
    return np.sqrt(vbar2 + lin)

# 現有的 model_v_kms 保持不變

def model_v_kms_split(U_disk: float,
                      U_bulge: float,
                      epsilon: float,
                      r_kpc: np.ndarray,
                      v_disk_kms: np.ndarray,
                      v_bulge_kms: np.ndarray,
                      v_gas_kms: np.ndarray,
                      H0_si: float = H0_SI) -> np.ndarray:
    """
    拆分 M/L：v_bar^2 = U_disk * v_disk^2 + U_bulge * v_bulge^2 + v_gas^2
    v_model = sqrt( v_bar^2 + (epsilon c H0) r )
    """
    vbar2 = (U_disk * (v_disk_kms**2)
             + U_bulge * (v_bulge_kms**2)
             + (v_gas_kms**2))
    lin   = linear_term_kms2(epsilon, r_kpc, H0_si=H0_si)
    return np.sqrt(vbar2 + lin)
