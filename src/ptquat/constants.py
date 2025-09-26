import numpy as np

# Physical constants
C_LIGHT = 299_792_458.0             # m/s
KM = 1_000.0                        # m
KPC = 3.085677581491367e19          # m
MPC = 1_000.0 * KPC                 # m
DEG2RAD = np.pi / 180.0

# H0: Planck 2018 baseline (can be changed by CLI if desired)
H0_KMS_MPC = 67.4                   # km/s/Mpc
H0_SI = (H0_KMS_MPC * KM) / MPC     # s^-1

# Small finite-difference step in kpc for numerical gradients
DR_NUM_KPC = 0.01
