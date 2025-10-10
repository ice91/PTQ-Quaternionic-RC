# tests/test_make_paper_artifacts.py
from __future__ import annotations
import subprocess, sys, os
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]

def _sh(*cmd, cwd=None):
    subprocess.check_call(list(map(str, cmd)), cwd=cwd or ROOT)

def _make_fake_tidy(path: Path):
    rng = np.random.default_rng(0)
    rows = []
    for g in ["FAKE_A","FAKE_B"]:
        D = 10.0 + (5.0 if g=="FAKE_B" else 0.0)
        for r in np.linspace(0.5, 8.0, 8):
            vdisk = 100.0*(1-np.exp(-r/2.5))
            vbul  = 40.0*np.exp(-r/1.0)
            vgas  = 20.0*(1-np.exp(-r/3.0))
            vbar2 = (vdisk**2 + vbul**2) + vgas**2
            vobs  = np.sqrt(vbar2) + rng.normal(0, 3.0)
            rows.append(dict(
                galaxy=g, r_kpc=r,
                v_obs_kms=vobs, v_err_kms=3.0,
                v_disk_kms=vdisk, v_bulge_kms=vbul, v_gas_kms=vgas,
                D_Mpc=D, D_err_Mpc=0.1*D,
                i_deg=45.0, i_err_deg=3.0
            ))
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def test_end2end_make_artifacts(tmp_path: Path):
    # 1) fake data
    tidy = tmp_path/"fake_sparc_tidy.csv"
    _make_fake_tidy(tidy)

    # 2) quick fits (baryon & ptq-screen)
    out1 = tmp_path/"results/baryon_gauss"
    out2 = tmp_path/"results/ptq-screen_gauss"
    _sh(sys.executable, "-m", "ptquat.cli", "fit", "--data", str(tidy),
        "--outdir", str(out1), "--model", "baryon", "--steps","50","--nwalkers","2x","--prior","galaxies-only")
    _sh(sys.executable, "-m", "ptquat.cli", "fit", "--data", str(tidy),
        "--outdir", str(out2), "--model", "ptq-screen", "--steps","50","--nwalkers","2x","--prior","galaxies-only")

    # 3) diagnostics
    _sh(sys.executable, "-m", "ptquat.cli", "exp", "plateau", "--results", str(out2), "--data", str(tidy), "--nbins","8")
    _sh(sys.executable, "-m", "ptquat.cli", "exp", "ppc", "--results", str(out2), "--data", str(tidy))
    _sh(sys.executable, "-m", "ptquat.cli", "exp", "kappa-gal", "--results", str(out2), "--data", str(tidy), "--eta","0.15")
    _sh(sys.executable, "-m", "ptquat.cli", "exp", "kappa-prof", "--results", str(out2), "--data", str(tidy), "--eta","0.15","--nbins","6")

    # 4) make_paper_artifacts.py (fast)
    script = ROOT/"scripts/make_paper_artifacts.py"
    _sh(sys.executable, str(script),
        "--data", str(tidy),
        "--out", str(tmp_path/"ejpc_run"),
        "--figdir", str(tmp_path/"figs"),
        "--models", "baryon", "ptq-screen",
        "--fast", "--skip-fetch")

    # 5) asserts: check main outputs
    run_root = tmp_path/"ejpc_run"
    assert (run_root/"ejpc_model_compare.csv").exists()
    cmp_df = pd.read_csv(run_root/"ejpc_model_compare.csv")
    assert set(cmp_df["model"]) == {"baryon","ptq-screen"}

    # figures
    figs = tmp_path/"figs"
    assert any(figs.glob("plateau*.png"))
    assert any(figs.glob("kappa_*.png"))

    # summaries exist
    for d in ["baryon_gauss","ptq-screen_gauss"]:
        y = yaml.safe_load((run_root/d/"global_summary.yaml").read_text())
        assert "AIC_full" in y and "BIC_full" in y
