from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class GalaxyData:
    name: str
    r_kpc: np.ndarray        # radii [kpc]
    v_obs: np.ndarray        # observed (deprojected) velocities [km/s]
    v_err: np.ndarray        # measurement errors [km/s]
    v_disk: np.ndarray       # disk contribution [km/s]
    v_bulge: np.ndarray      # bulge contribution [km/s]
    v_gas: np.ndarray        # gas contribution [km/s]
    D_Mpc: float
    D_err_Mpc: float
    i_deg: float
    i_err_deg: float

def load_tidy_sparc(path_csv: str) -> Dict[str, GalaxyData]:
    df = pd.read_csv(path_csv)
    required = ["galaxy","r_kpc","v_obs_kms","v_err_kms",
                "v_disk_kms","v_bulge_kms","v_gas_kms","D_Mpc","D_err_Mpc","i_deg","i_err_deg"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path_csv}: {missing}")

    galaxies: Dict[str, GalaxyData] = {}
    for name, sub in df.groupby("galaxy"):
        sub = sub.sort_values("r_kpc")
        gd = GalaxyData(
            name=name,
            r_kpc=sub["r_kpc"].to_numpy(float),
            v_obs=sub["v_obs_kms"].to_numpy(float),
            v_err=sub["v_err_kms"].to_numpy(float),
            v_disk=sub["v_disk_kms"].to_numpy(float),
            v_bulge=sub["v_bulge_kms"].to_numpy(float),
            v_gas=sub["v_gas_kms"].to_numpy(float),
            D_Mpc=float(sub["D_Mpc"].iloc[0]),
            D_err_Mpc=float(sub["D_err_Mpc"].iloc[0]),
            i_deg=float(sub["i_deg"].iloc[0]),
            i_err_deg=float(sub["i_err_deg"].iloc[0]),
        )
        galaxies[name] = gd
    return galaxies
