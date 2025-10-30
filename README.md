
# PTQ-Quaternionic-RC — SPARC Rotation-Curve Pipeline

**TL;DR.**  
This repo reproduces the experiments in our paper where **PTQ is positioned as the physical origin of MOND**.  
We compare **Baryon-only**, **NFW-1p**, **MOND(ν)**, **PTQ (linear)**, **PTQ-ν**, and **PTQ-screen**, report **full-likelihood AIC/BIC**, and run a robustness/closure suite including **PPC**, **error stress (×2)**, **inner-masking**, **H₀ scan**, and **κ-checks** (cross-scale tests).  
We also include a **κ–h regression** linking disk scale-height (h) with **epicyclic frequency** (\kappa) and **surface density** (\Sigma).

Key dictionary:  
$$[  
\Omega_\Lambda(\epsilon)=\frac{\epsilon^2}{1+\epsilon^2},\qquad  
a_0(\epsilon)=\epsilon,c,H_0 \quad (\text{optionally } a_0=\kappa,\epsilon,c,H_0).  
]$$

---

## 0. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
````

Test:

```bash
pytest -q
```

---

## 1. Data

### 1.1 Download raw VizieR tables (SPARC; Lelli+ 2016)

```bash
ptquat fetch --out dataset/raw
```

This produces `dataset/raw/vizier_table1.csv` and `dataset/raw/vizier_table2.csv`.

### 1.2 Build tidy CSV with quality cuts

```bash
ptquat preprocess \
  --raw dataset/raw \
  --out dataset/sparc_tidy.csv \
  --i-min 30.0 --reldmax 0.20 --qual-max 2
```

Columns in the tidy CSV include:  
`galaxy, r_kpc, v_obs_kms, v_err_kms, v_disk_kms, v_bulge_kms, v_gas_kms, D_Mpc, D_err_Mpc, i_deg, i_err_deg`.

### 1.3 S4G disk scale-heights (h) & merge into SPARC

This step builds the **S4G** disk thickness scale (h) catalog, including distances, and merges it with **SPARC** data to produce a single file for the **κ–h** experiment.

```bash
# Archive the VizieR query and perform minimal ETL
bash scripts/vizier_s4g_query.sh
python scripts/etl_s4g_h.py --sparc dataset/sparc_tidy.csv
```

This produces `dataset/geometry/h_catalog.csv` and `dataset/geometry/sparc_with_h.csv`.

SHA-256 checksums are written alongside the outputs to verify byte-level identity.

---

## 2. Models

- `baryon`: (v=\sqrt{v_{\rm bar}^2}).
    
- `mond`: MOND (simple-ν). If `--a0-si` not given, (a_0) is sampled with prior `--a0-range`.
    
- `nfw1p`: one halo parameter ((M_{200})) per galaxy; concentration from a c–M power law (`--c0`, `--c-slope`).
    
- `ptq` (linear): (v=\sqrt{v_{\rm bar}^2 + (\epsilon cH_0),r}). **Negative control**.
    
- `ptq-nu`: reuse MOND shape but set (a_0=\epsilon cH_0) (ε global).
    
- `ptq-screen`: generalized MOND-like (\nu_q(y)=0.5+\sqrt{0.25+y^{-q}}) with **global** (q) and (a_0=\epsilon cH_0).
    

**Priors** (`--prior`):

- `galaxies-only` (flat ε on (0,4)).
    
- `planck-anchored` (Gaussian ε prior around 1.47±0.05).
    

**Likelihood:** `--likelihood gauss|t` (default: gauss). Student-t (ν>2) improves outlier robustness.

---

## 3. Reproduce the main table (AIC/BIC)

Run the six models with default settings (SPARC90 / N≈1734 points):

```bash
# PTQ-screen (main model)
ptquat fit --model ptq-screen --data dataset/sparc_tidy.csv --outdir results/ptq_screen

# MOND baseline
ptquat fit --model mond --data dataset/sparc_tidy.csv --outdir results/ejpc_mond

# PTQ-ν
ptquat fit --model ptq-nu --data dataset/sparc_tidy.csv --outdir results/ptq_nu

# NFW-1p
ptquat fit --model nfw1p --data dataset/sparc_tidy.csv --outdir results/ejpc_nfw1p

# PTQ (linear) — negative control
ptquat fit --model ptq --data dataset/sparc_tidy.csv --outdir results/ejpc_main

# Baryon-only — negative control
ptquat fit --model baryon --data dataset/sparc_tidy.csv --outdir results/ejpc_baryon
```

Each run writes:

- `global_summary.yaml` (includes `chi2_total`, **`AIC_full/BIC_full`**, `k_parameters`, `N_total`, median posteriors `epsilon_median/q_median/a0_median`, etc.)
    
- `per_galaxy_summary.csv`
    
- `plot_*.png` for each galaxy
    

**Compile a comparison CSV:**

```bash
python - <<'PY'
import yaml, pandas as pd, pathlib as P
runs = {
  "PTQ-screen": "results/ptq_screen/global_summary.yaml",
  "PTQ-v":      "results/ptq_nu/global_summary.yaml",
  "PTQ":        "results/ejpc_main/global_summary.yaml",
  "Baryon":     "results/ejpc_baryon/global_summary.yaml",
  "NFW-1p":     "results/ejpc_nfw1p/global_summary.yaml",
  "MOND":       "results/ejpc_mond/global_summary.yaml",
}
rows=[]
for name, yml in runs.items():
    s=yaml.safe_load(open(yml))
    rows.append(dict(model=name,k=s["k_parameters"],chi2=s["chi2_total"],
                     AIC_full=s["AIC_full"],BIC_full=s["BIC_full"],
                     AIC_quad=s["AIC_quad"],BIC_quad=s["BIC_quad"],
                     sigma_sys_med=s["sigma_sys_median"],
                     epsilon_med=s.get("epsilon_median"),
                     a0_med=s.get("a0_median"),
                     N=s["N_total"]))
df=(pd.DataFrame(rows).set_index("model").sort_values("BIC_full"))
df.to_csv("results/ejpc_model_compare.csv")
print(df[["k","chi2","AIC_full","BIC_full","AIC_quad","BIC_quad","sigma_sys_med","epsilon_med","a0_med"]])
print("Saved results/ejpc_model_compare.csv")
PY
```

---

## 4. Robustness suite (as in the paper)

### S1. Posterior-(like) Predictive Coverage (PPC)

```bash
ptquat exp ppc --results results/ptq_screen --data dataset/sparc_tidy.csv
```

Outputs `{results}/ppc_coverage.json` with 68/95% coverage.

### S2. Error stress test (×2 on i_err & D_err)

```bash
ptquat exp stress --model ptq-screen --data dataset/sparc_tidy.csv \
  --scale-i 2 --scale-D 2 --outroot results/stress --prior galaxies-only
```

Re-fits on a perturbed CSV and writes a fresh `global_summary.yaml`.

---

## 5. Cross-scale tests

### 5.1 Closure test: (\epsilon_{\rm cos}) vs (\epsilon_{\rm RC})

```bash
# using ΩΛ=0.69  → ε_cos = sqrt(ΩΛ/(1-ΩΛ))
ptquat exp closure --results results/ptq_screen --omega-lambda 0.69
# or explicitly:
ptquat exp closure --results results/ptq_screen --epsilon-cos 1.47
```

Writes `{results}/closure_test.yaml` with `epsilon_RC`, `epsilon_cos`, `sigma_RC`, and `pass_within_3sigma`.

---

## 6. Residual-acceleration plateau (main figure)

```bash
ptquat exp plateau --results results/ptq_screen --data dataset/sparc_tidy.csv
```

Produces `{results}/plateau_per_point.csv`, `{results}/plateau_binned.csv`, and a PNG.

---

## 7. Thickness–κ–Σ regression at ( R_\star ) (per galaxy)

### Main command (WLS, LOO-CV, bootstrap)

```bash
python -m ptq.experiments.kappa_h \
  --sparc-with-h dataset/geometry/sparc_with_h.csv \
  --per-galaxy --rstar vdisk-peak --wls \
  --ml36 0.5 --rgas-mult 1.7 --gas-helium 1.33 \
  --loo --bootstrap 5000 --cv-by-galaxy \
  --out-csv dataset/geometry/kappa_h_used.csv \
  --report-json dataset/geometry/kappa_h_report.json \
  --plot-out ${RESULTS}/ptq-screen_gauss
```

Key outputs:

- `kappa_h_scatter.png`, `kappa_h_scatter_sigma.png` (in the family dir)
    
- `kappa_h_report.json` (AICc, $\Delta$AICc among ${\kappa{\rm\text{-only}},\Sigma{\rm\text{-only}},\kappa{+}\Sigma}$, $R^2$, LOO, bootstrap)
    
- `kappa_h_used.csv`
    

### Distance-invariant sanity check (optional)

```bash
python -m ptq.experiments.kappa_h \
  --sparc-with-h dataset/geometry/sparc_with_h.csv \
  --per-galaxy --rstar vdisk-peak --wls --dist-inv \
  --ml36 0.5 --rgas-mult 1.7 --gas-helium 1.33 \
  --out-csv dataset/geometry/kappa_h_used_distinv.csv \
  --report-json dataset/geometry/kappa_h_report_distinv.json \
  --plot-out ${RESULTS}/ptq-screen_gauss
```

---

## 8. Reproducibility

- Fix seeds via `--seed`; enable HDF5 backends via `--backend-hdf5 results/<run>/chain.h5 --resume`.
    
- Record environment (Python version, `requirements.txt`) and include produced `global_summary.yaml` and CSVs.
    

---

## 9. Field glossary (what the YAML/JSON fields mean)

**Fit YAML (`global_summary.yaml`)**

- `AIC_full, BIC_full`: from the **chosen likelihood** (`gauss` or `t`); **use these in the paper**.
    

---

## 10. Optional ablations & knobs

- **Student-t likelihood** (`--likelihood t --t-dof 8`) for RC fits.
    
- **Planck-anchored prior** (`--prior planck-anchored`) vs `galaxies-only`.
    

---

## 11. Citation

If you use this code or reproduce our results, please cite the accompanying paper (TBD) and this repository.

