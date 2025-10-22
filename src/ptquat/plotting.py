# src/ptquat/plotting.py
from __future__ import annotations
import numpy as np
from pathlib import Path

# 使用無頭後端，避免 CI/無顯示環境錯誤
import matplotlib
matplotlib.use("Agg")
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

def plot_ppc_hist(z_values: np.ndarray, outpath: Path):
    fig = plt.figure(figsize=(5,4))
    ax = fig.add_subplot(111)
    ax.hist(z_values, bins=50, density=True, alpha=0.8)
    ax.set_xlabel("Standardized residual z")
    ax.set_ylabel("PDF")
    ax.axvline(0.0, lw=1)
    for k in [1,2]:
        ax.axvline(+k, ls="--", lw=1)
        ax.axvline(-k, ls="--", lw=1)
    ax.set_title("Posterior(-like) predictive residuals")
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)

def plot_residual_plateau(df_points, df_binned, outpath: Path):
    """
    df_points: columns [r_kpc, delta_a]
    df_binned: columns [r_mid_kpc, q16, q50, q84, n]
    """
    fig = plt.figure(figsize=(6.5,4.2))
    ax = fig.add_subplot(111)
    # 淡淡散點
    ax.scatter(df_points["r_kpc"], df_points["delta_a"], s=6, alpha=0.15)
    # 分箱帶狀
    x = df_binned["r_mid_kpc"].values
    y = df_binned["q50"].values
    y1 = df_binned["q16"].values
    y2 = df_binned["q84"].values
    ax.plot(x, y, lw=2, label="median")
    ax.fill_between(x, y1, y2, alpha=0.25, label="16–84%")
    ax.set_xlabel("r  [kpc]")
    ax.set_ylabel(r"$\Delta a = v^2/r - a_{\rm bar}(r)$  [m s$^{-2}$]")
    ax.legend(loc="best", frameon=False)
    ax.set_title("Stacked residual-acceleration plateau")
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)

# --- 新增：BTFR 在正確座標（y=V_f^4, x=M_b），理論斜率=1 ---
def plot_btfr_one2one(Mb: np.ndarray,
                      Vf: np.ndarray,
                      outpath_png: str,
                      Mb_label: str = r"$M_b\,[M_\odot]$",
                      epsilon: float | None = None):
    """
    以理論座標繪製 BTFR：y=V_f^4、x=M_b。參考線固定斜率=1。
    若提供 epsilon=a0/(cH0)，會在圖中標註（僅作資訊顯示）。
    """
    x = np.asarray(Mb, float)
    y = np.asarray(Vf, float)**4

    fig, ax = plt.subplots(figsize=(6.2, 5.0), dpi=160)
    ax.scatter(x, y, s=22, alpha=0.85, label=f"galaxies (N={len(x)})")

    xmin = max(np.nanmin(x[x>0]), 1e6)
    xmax = np.nanmax(x) * 1.2
    xx = np.logspace(np.log10(xmin), np.log10(xmax), 256)
    ax.plot(xx, xx, lw=1.6, label="theory: slope = 1")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(Mb_label)
    ax.set_ylabel(r"$V_f^4\,[{\rm km}^4\,{\rm s}^{-4}]$")
    if epsilon is not None and np.isfinite(epsilon):
        ax.text(0.03, 0.06, rf"$\epsilon=a_0/(cH_0)\approx {epsilon:.3f}$",
                transform=ax.transAxes)

    ax.grid(True, which="both", ls=":", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outpath_png)
    plt.close(fig)
