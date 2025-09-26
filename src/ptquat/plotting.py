from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

def plot_rc(gal_name: str,
            r_kpc: np.ndarray,
            v_obs: np.ndarray,
            v_err: np.ndarray,
            v_mod: np.ndarray,
            outpath_png):
    plt.figure(figsize=(6.5,4.5))
    plt.errorbar(r_kpc, v_obs, yerr=v_err, fmt='o', ms=4, label='data', alpha=0.8)
    plt.plot(r_kpc, v_mod, lw=2, label='model')
    plt.xlabel('r [kpc]')
    plt.ylabel('v [km s$^{-1}$]')
    plt.title(gal_name)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath_png, dpi=160)
    plt.close()
