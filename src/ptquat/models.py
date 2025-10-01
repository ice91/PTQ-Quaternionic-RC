from __future__ import annotations
import numpy as np
from .constants import (
    C_LIGHT, H0_SI, KPC, KM, G_SI, MSUN
)

# ---------- 共用：恆星/氣體的貢獻 ----------

def vbar_squared_kms2(Upsilon: float,
                      v_disk_kms: np.ndarray,
                      v_bulge_kms: np.ndarray,
                      v_gas_kms: np.ndarray) -> np.ndarray:
    """
    v_bar^2 = Upsilon * (v_disk^2 + v_bulge^2) + v_gas^2   [ (km/s)^2 ]
    """
    v_star2 = v_disk_kms**2 + v_bulge_kms**2
    return Upsilon * v_star2 + v_gas_kms**2


# ---------- PTQ 模型（線性項） ----------

def linear_term_kms2(epsilon: float,
                     r_kpc: np.ndarray,
                     H0_si: float = H0_SI) -> np.ndarray:
    """
    (epsilon * c * H0) * r   in (km/s)^2  with r in kpc.
    """
    r_m = r_kpc * KPC
    term_m2s2 = epsilon * C_LIGHT * H0_si * r_m
    return term_m2s2 / (KM**2)

def model_v_ptq(Upsilon: float,
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

def model_v_ptq_split(U_disk: float,
                      U_bulge: float,
                      epsilon: float,
                      r_kpc: np.ndarray,
                      v_disk_kms: np.ndarray,
                      v_bulge_kms: np.ndarray,
                      v_gas_kms: np.ndarray,
                      H0_si: float = H0_SI) -> np.ndarray:
    """
    拆分 M/L：
      v_bar^2 = U_disk * v_disk^2 + U_bulge * v_bulge^2 + v_gas^2
      v_model = sqrt( v_bar^2 + (epsilon c H0) r )
    """
    vbar2 = (U_disk * (v_disk_kms**2)
             + U_bulge * (v_bulge_kms**2)
             + (v_gas_kms**2))
    lin   = linear_term_kms2(epsilon, r_kpc, H0_si=H0_si)
    return np.sqrt(vbar2 + lin)


# ---------- 基線：只有重子（epsilon=0） ----------

def model_v_baryon(Upsilon: float,
                   r_kpc: np.ndarray,
                   v_disk_kms: np.ndarray,
                   v_bulge_kms: np.ndarray,
                   v_gas_kms: np.ndarray) -> np.ndarray:
    """
    v_model = sqrt( v_bar^2 )  [km/s]
    """
    vbar2 = vbar_squared_kms2(Upsilon, v_disk_kms, v_bulge_kms, v_gas_kms)
    return np.sqrt(vbar2)


# ---------- MOND（simple ν） ----------

def model_v_mond(Upsilon: float,
                 a0_si: float,
                 r_kpc: np.ndarray,
                 v_disk_kms: np.ndarray,
                 v_bulge_kms: np.ndarray,
                 v_gas_kms: np.ndarray) -> np.ndarray:
    """
    MOND (simple μ):
      g_N = v_N^2 / r,  v_N^2 = Upsilon*(v_disk^2 + v_bulge^2) + v_gas^2
      ν(y) = 0.5 + sqrt(0.25 + 1/y), y = g_N / a0
      v^2 = v_N^2 * ν(y)
    """
    vN2 = Upsilon * (v_disk_kms**2 + v_bulge_kms**2) + v_gas_kms**2      # (km/s)^2
    r_m = r_kpc * KPC
    gN = (vN2 * (KM**2)) / np.maximum(r_m, 1e-12)                        # m/s^2
    y  = np.maximum(gN / np.maximum(a0_si, 1e-16), 1e-12)
    nu = 0.5 + np.sqrt(0.25 + 1.0 / y)
    v2 = vN2 * nu
    return np.sqrt(v2)


# ---------- NFW-1p（每星系只放 M200，c 由 c–M 關係給出） ----------

def nfw_v_kms_M200_c(r_kpc: np.ndarray,
                     M200_Msun: float,
                     c: float,
                     H0_si: float = H0_SI) -> np.ndarray:
    """
    NFW halo circular velocity with (M200, c).
    """
    M200 = M200_Msun * MSUN
    rho_crit = 3.0 * H0_si**2 / (8.0 * np.pi * G_SI)
    R200 = (3.0 * M200 / (4.0 * np.pi * 200.0 * rho_crit))**(1.0/3.0)  # [m]
    rs = R200 / np.maximum(c, 1e-6)
    x = (r_kpc * KPC) / np.maximum(rs, 1e-30)
    g  = np.log1p(x) - x/(1.0 + x)
    gc = np.log1p(c) - c/(1.0 + c)
    gc = np.maximum(gc, 1e-20)
    M_r = M200 * (g / gc)
    v2  = G_SI * M_r / np.maximum(r_kpc * KPC, 1e-30)   # [m^2 s^-2]
    return np.sqrt(np.maximum(v2, 0.0)) / KM            # [km/s]

def c_powerlaw(M200_Msun: float,
               c0: float = 10.0,
               beta: float = -0.1,
               Mpiv: float = 1.0e12) -> float:
    return float(c0) * (M200_Msun / Mpiv)**float(beta)

def model_v_nfw1p(Upsilon: float,
                  log10_M200: float,
                  r_kpc: np.ndarray,
                  v_disk_kms: np.ndarray,
                  v_bulge_kms: np.ndarray,
                  v_gas_kms: np.ndarray,
                  H0_si: float = H0_SI,
                  c0: float = 10.0,
                  c_slope: float = -0.1) -> np.ndarray:
    """
    v_model = sqrt( v_bar^2 + v_nfw^2(M200, c(M200)) ),  M200 in Msun (given as log10).
    """
    M200 = 10.0**log10_M200
    c    = c_powerlaw(M200, c0=c0, beta=c_slope)
    vbar2 = vbar_squared_kms2(Upsilon, v_disk_kms, v_bulge_kms, v_gas_kms)
    vhalo = nfw_v_kms_M200_c(r_kpc, M200, c, H0_si=H0_si)
    return np.sqrt(vbar2 + vhalo**2)
