# src/ptquat/constants.py
import numpy as np

# Physical constants (SI)
C_LIGHT = 299_792_458.0                 # m/s
KM      = 1_000.0                       # m
KPC     = 3.085677581491367e19          # m
MPC     = 1_000.0 * KPC                 # m
DEG2RAD = np.pi / 180.0

# H0 default: Planck 2018 (可由 CLI 覆蓋)
H0_KMS_MPC_DEFAULT = 67.4               # km/s/Mpc
H0_SI              = (H0_KMS_MPC_DEFAULT * KM) / MPC  # s^-1

# Finite-difference step for dv/dr [kpc]
DR_NUM_KPC = 0.01

# add
G_SI   = 6.67430e-11            # m^3 kg^-1 s^-2
MSUN   = 1.98847e30             # kg
