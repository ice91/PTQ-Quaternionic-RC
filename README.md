# PTQ-Quaternionic-RC — SPARC Rotation-Curve Pipeline

**TL;DR.**  
This repository reproduces the experiments for our PTQ weak-field rotation-curve program on **SPARC**.  
The codebase compares **Baryon-only**, **NFW-1p**, **MOND**, **MOND-screen**, **PTQ (linear)**, **PTQ-ν**, and **PTQ-screen**, reports **full-likelihood AIC/BIC**, and supports **Bayesian model comparison** via **WAIC** and **PSIS-LOO** (`ptquat exp compare`, `ptquat exp loo`) together with **pWAIC diagnostics** (per-galaxy / per-radius).  

**Current manuscript-facing stance of this repo:**
- **`ptq-nu`** is the current **primary RC-level PTQ realization** for paper-facing analyses.
- **`ptq-screen`** is retained as a **generalized comparison model** with a global response-kernel parameter.
- **Strict cosmology–galaxy closure is treated as a diagnostic test, not a claimed success condition.**
- The repo includes **geometry audits** such as **κ diagnostics**, **residual-acceleration plateau**, **z-profile**, and **κ–h / κ+Σ thickness regression**.

A **κ–h regression** links disk scale-height \(h\) with **epicyclic frequency** \(\kappa\) and **surface density** \(\Sigma\), serving as an external geometry audit rather than a by-construction fit improvement.

Key dictionary:
\[
$\Omega_\Lambda(\epsilon)=\frac{\epsilon^2}{1+\epsilon^2},\qquad a_0(\epsilon)=\epsilon\,c\,H_0.$
\]

For generalized or interpretive extensions, an effective geometric efficiency factor may be introduced at the analysis level, but it should be read as a **reinterpretation of mismatch**, not as a passed closure claim.

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

This produces:

- `dataset/raw/vizier_table1.csv`
    
- `dataset/raw/vizier_table2.csv`
    

### 1.2 Build tidy CSV with quality cuts

```bash
ptquat preprocess \
  --raw dataset/raw \
  --out dataset/sparc_tidy.csv \
  --i-min 30.0 --reldmax 0.20 --qual-max 2
```

Columns in the tidy CSV include:

- `galaxy`
    
- `r_kpc`
    
- `v_obs_kms`
    
- `v_err_kms`
    
- `v_disk_kms`
    
- `v_bulge_kms`
    
- `v_gas_kms`
    
- `D_Mpc`
    
- `D_err_Mpc`
    
- `i_deg`
    
- `i_err_deg`
    

### 1.3 S4G disk scale-heights (h) & merge into SPARC

This step builds the **S4G** disk thickness catalog and merges it with **SPARC** for the **κ–h / κ+Σ** experiments.

**Option A — ptquat CLI (recommended):**

```bash
ptquat geom s4g-hcat --sparc dataset/sparc_tidy.csv --out dataset/geometry/h_catalog.csv
ptquat geom s4g-join --sparc dataset/sparc_tidy.csv --h dataset/geometry/h_catalog.csv --out dataset/geometry/sparc_with_h.csv
```

**Option B — scripts (VizieR + ETL):**

```bash
bash scripts/vizier_s4g_query.sh
python scripts/etl_s4g_h.py --sparc dataset/sparc_tidy.csv
```

Outputs:

- `dataset/geometry/h_catalog.csv`
    
- `dataset/geometry/sparc_with_h.csv`
    

SHA-256 checksums are written alongside to verify byte-level identity.

---

## 2. Models

### Main model family

- **`baryon`**:  
    [  
    $v=\sqrt{v_{\rm bar}^2}.$  
    ]
    
- **`mond`**:  
    Standard MOND $(`simple-\nu`)$. If `--a0-si` is not given, $(a_0)$ is sampled with prior `--a0-range`.
    
- **`mond-screen`**:  
    Generalized MOND-like screening model with free global (q) and free $(a_0)$. This is primarily used as a **matched-kernel comparison model**.
    
- **`nfw1p`**:  
    One halo parameter ((M_{200})) per galaxy; concentration from a c–M power law (`--c0`, `--c-slope`).
    
- **`ptq` (linear)**:  
    [  
    $v=\sqrt{v_{\rm bar}^2 + (\epsilon cH_0),r}.$  
    ]  
    **Negative control**.
    
- **`ptq-nu`**:  
    Reuses the MOND interpolation shape but sets  
    [  
    $a_0=\epsilon cH_0,$  
    ]  
    with a **global** $(\epsilon)$.  
    This is the current **primary RC-level PTQ realization** in the manuscript-facing workflow.
    
- **`ptq-screen`**:  
    Generalized PTQ-screened realization  
    [  
    $\nu_q(y)=\frac12+\sqrt{\frac14+y^{-q}},$  
    ]  
    with **global** (q) and  
    [  
    $a_0=\epsilon cH_0.$  
    ]  
    This is retained as a **comparison / extension model**, not the default hero model.
    

### Priors (`--prior`)

- `galaxies-only`: flat (\epsilon) prior on ((0,4))
    
- `planck-anchored`: Gaussian (\epsilon) prior around (1.47\pm0.05)
    

### Likelihood

```text
--likelihood gauss|t
```

Default: `gauss`

Student-t likelihood (`--likelihood t --t-dof 8`) may improve robustness to outliers.

---

## 3. Reproduce the six-model baseline (AIC/BIC)

The recommended paper-facing baseline is the **six-model long-chain baseline**:

- `ptq-screen`
    
- `mond`
    
- `ptq-nu`
    
- `nfw1p`
    
- `ptq`
    
- `baryon`
    

### 3.1 Recommended production script

```bash
bash scripts/run_baseline_sixmodels_12000.sh
```

This runs fresh long-chain fits and then computes:

- six-model WAIC comparison
    
- six-model PSIS-LOO comparison
    
- baseline artifact summaries
    

### 3.2 Aggregate the baseline outputs

```bash
python scripts/summarize_baseline_sixmodels.py
```

Expected outputs include:

- `results/revision_baseline_sixmodels_12000/aggregate/baseline_sixmodels_fit_metrics.csv`
    
- `results/revision_baseline_sixmodels_12000/aggregate/baseline_sixmodels_compare_metrics.csv`
    
- `results/revision_baseline_sixmodels_12000/aggregate/baseline_sixmodels_summary.md`
    
- `results/revision_baseline_sixmodels_12000/aggregate/baseline_sixmodels_table.tex`
    

### 3.3 Manual per-model fit example

If you want to run one model manually:

```bash
ptquat fit \
  --model ptq-nu \
  --data dataset/sparc_tidy.csv \
  --outdir results/ptq_nu_gauss \
  --nwalkers 192 \
  --steps 12000 \
  --seed 0 \
  --backend-hdf5 results/ptq_nu_gauss/chain.h5 \
  --thin-by 1
```

Each run writes:

- `global_summary.yaml`
    
- `per_galaxy_summary.csv`
    
- `plot_*.png` for each galaxy
    

`global_summary.yaml` includes fields such as:

- `chi2_total`
    
- `AIC_full`, `BIC_full`
    
- `k_parameters`
    
- `N_total`
    
- median posteriors such as `epsilon_median`, `q_median`, `a0_median`
    

---

## 4. Bayesian model comparison

### 4.1 WAIC model comparison (`ptquat exp compare`)

Compute **WAIC** from posterior samples.

Example for a paper-facing baseline:

```bash
ptquat exp compare \
  --models ptq-screen mond ptq-nu nfw1p ptq baryon \
  --data dataset/sparc_tidy.csv \
  --fit-root results/revision_baseline_sixmodels_12000 \
  --outdir results/revision_baseline_sixmodels_12000/paper_extra/model_compare_baseline \
  --seed 0
```

Typical outputs:

- `compare_table.csv`
    
- `compare_table.tex`
    
- `breakdown.csv`
    
- `rank_plot.png`
    
- `rank_plot.pdf`
    
- `manifest.yaml`
    

When enabled, pWAIC diagnostics may also write:

- `pwaic_diagnostics/var_loglik_points.csv`
    
- `pwaic_diagnostics/pwaic_by_galaxy.csv`
    
- `pwaic_diagnostics/top20_galaxies_pwaic.csv`
    
- `pwaic_diagnostics/pwaic_by_radius.csv`
    

### 4.2 PSIS-LOO (`ptquat exp loo`)

Compute **PSIS-LOO** via `arviz`.

Example:

```bash
ptquat exp loo \
  --models ptq-screen mond ptq-nu nfw1p ptq baryon \
  --data dataset/sparc_tidy.csv \
  --fit-root results/revision_baseline_sixmodels_12000 \
  --outdir results/revision_baseline_sixmodels_12000/paper_extra/loo_compare_baseline \
  --seed 0
```

Typical outputs:

- `loo_table.csv`
    
- `pareto_k.csv`
    
- `pareto_k_hist.png`
    
- `elpd_difference_plot.png`
    

Interpretation note:

- Prefer `pareto_k < 0.7` for most points when reading PSIS-LOO.
    
- If influential observations exist, treat LOO as a diagnostic rather than a sole decision criterion.
    

---

## 5. Multi-seed robustness audit

The recommended robustness layer is a **multi-seed audit** over the main PTQ-family and MOND-family models.

### 5.1 Recommended production script

```bash
bash scripts/run_robustness_core_multiseed12000.sh
```

This runs fresh long-chain fits for multiple seeds and then computes per-seed:

- WAIC compare
    
- PSIS-LOO compare
    

### 5.2 Aggregate the robustness outputs

```bash
python scripts/summarize_robustness_core_multiseed.py
```

Expected outputs include:

- `results/revision_robustness_core_multiseed12000/aggregate/multiseed_fit_metrics_long.csv`
    
- `results/revision_robustness_core_multiseed12000/aggregate/multiseed_compare_metrics_long.csv`
    
- `results/revision_robustness_core_multiseed12000/aggregate/multiseed_model_summary.csv`
    
- `results/revision_robustness_core_multiseed12000/aggregate/multiseed_rank_stability.csv`
    
- `results/revision_robustness_core_multiseed12000/aggregate/multiseed_robustness_table.tex`
    
- `results/revision_robustness_core_multiseed12000/aggregate/multiseed_summary.md`
    

This robustness layer is intended to assess:

- seed sensitivity
    
- chain-path sensitivity
    
- ranking stability among the main candidate models
    

---

## 6. Diagnostic and geometry-audit experiments

These experiments are designed as **post-fit diagnostics** and **external geometry audits**. They should not be interpreted as by-construction improvements to the RC fit.

### 6.1 Posterior-(like) Predictive Coverage (PPC)

```bash
ptquat exp ppc --results results/ptq_nu_gauss --data dataset/sparc_tidy.csv
```

Outputs:

- `ppc_coverage.json`
    

### 6.2 Error stress test

```bash
ptquat exp stress \
  --model ptq-nu \
  --data dataset/sparc_tidy.csv \
  --scale-i 2 --scale-D 2 \
  --outroot results/stress_ptqnu \
  --prior galaxies-only
```

### 6.3 Residual-acceleration plateau

```bash
ptquat exp plateau \
  --results results/ptq_nu_gauss \
  --data dataset/sparc_tidy.csv
```

Outputs:

- `plateau_per_point.csv`
    
- `plateau_binned.csv`
    
- `plateau.png`
    

### 6.4 Per-galaxy κ diagnostic

Example (obs-based (r_\star), obs-debiased):

```bash
ptquat exp kappa-gal \
  --results results/ptq_nu_gauss \
  --data dataset/sparc_tidy.csv \
  --y-source obs-debias \
  --eps-norm cos \
  --rstar-from obs \
  --frac-vmax 0.95 \
  --regression deming \
  --deming-lambda 1.0 \
  --prefix kappa_gal_obsdebiased_cos_f095
```

Example negative/control variant:

```bash
ptquat exp kappa-gal \
  --results results/ptq_nu_gauss \
  --data dataset/sparc_tidy.csv \
  --y-source obs-debias \
  --eps-norm cos \
  --rstar-from model \
  --frac-vmax 0.90 \
  --regression deming \
  --deming-lambda 1.0 \
  --prefix kappa_gal_obsdebiased_cos_rstar_model
```

### 6.5 Radius-resolved κ profile

```bash
ptquat exp kappa-prof \
  --results results/ptq_nu_gauss \
  --data dataset/sparc_tidy.csv \
  --eta 0.15 \
  --nbins 24 \
  --min-per-bin 20 \
  --x-kind r_over_Rd \
  --eps-norm cos \
  --prefix kappa_profile_cos
```

Fit the stacked profile:

```bash
ptquat exp kappa-fit \
  --results results/ptq_nu_gauss \
  --prefix kappa_profile_cos \
  --eps-norm cos \
  --bootstrap 2000 \
  --seed 1234
```

### 6.6 z-profile

```bash
ptquat exp zprof \
  --results results/ptq_nu_gauss \
  --data dataset/sparc_tidy.csv \
  --nbins 24 \
  --min-per-bin 20 \
  --eps-norm cos \
  --prefix z_profile
```

### 6.7 Thickness–κ–Σ regression at (R_\star)

Main command (WLS, LOO-CV, bootstrap):

```bash
python -m ptq.experiments.kappa_h \
  --sparc-with-h dataset/geometry/sparc_with_h.csv \
  --per-galaxy --rstar vdisk-peak --wls \
  --ml36 0.5 --rgas-mult 1.7 --gas-helium 1.33 \
  --loo --bootstrap 5000 --cv-by-galaxy \
  --out-csv dataset/geometry/kappa_h_used.csv \
  --report-json dataset/geometry/kappa_h_report.json \
  --plot-out results/ptq_nu_gauss
```

Key outputs:

- `kappa_h_scatter.png`
    
- `kappa_h_scatter_sigma.png`
    
- `kappa_h_report.json`
    
- `kappa_h_used.csv`
    

Interpretation note:

- This is a **geometry audit** and **external consistency test**.
    
- In current manuscript-facing usage, it should be described as supporting evidence rather than as the primary fit criterion.
    

---

## 7. Cross-scale tests

### 7.1 Strict closure test

The strict closure test compares the fitted galactic (\epsilon_{\rm RC}) against a cosmology-implied (\epsilon_{\rm cos}).

```bash
ptquat exp closure \
  --results results/ptq_nu_gauss \
  --omega-lambda 0.69
```

or explicitly:

```bash
ptquat exp closure \
  --results results/ptq_nu_gauss \
  --epsilon-cos 1.47
```

Output:

- `closure_test.yaml`
    

### 7.2 Closure scan

```bash
ptquat exp closure-scan \
  --results results/ptq_nu_gauss \
  --outdir results/ptq_nu_gauss/closure_scan \
  --omega-lambda-min 0.65 \
  --omega-lambda-max 0.75 \
  --n 11
```

Outputs:

- `closure_sensitivity.csv`
    
- `closure_sensitivity_plot.png`
    
- `closure_sensitivity.yaml`
    
- `manifest.yaml`
    

**Important interpretation note:**  
Strict closure is treated as a **diagnostic test**.  
A failed closure scan does **not** invalidate the RC-level utility of `ptq-nu`, but it does mean that the repo does **not** claim a completed cosmology–galaxy closure.

A derived effective geometric efficiency may still be discussed at the paper-analysis level as a **reinterpretation of mismatch**, not as a passed closure criterion.

---

## 8. Paper artifacts

The script `scripts/make_paper_artifacts.py` aggregates comparison results and writes LaTeX tables, figures, and summary markdown into a paper-facing output directory.

### Example

```bash
python scripts/make_paper_artifacts.py \
  --data dataset/sparc_tidy.csv \
  --results-dir results/revision_baseline_sixmodels_12000 \
  --out results/revision_baseline_sixmodels_12000/paper_run \
  --models ptq-screen mond ptq-nu nfw1p ptq baryon
```

Typical outputs include:

- `ejpc_model_compare.csv`
    
- `paper_artifacts/waic_compare_table.tex`
    
- `paper_artifacts/loo_compare_table.tex`
    
- `paper_artifacts/pwaic_top_galaxies_table.tex`
    
- `paper_artifacts/model_compare_waic.pdf`
    
- `paper_artifacts/model_compare_loo.pdf`
    
- `paper_artifacts/paper_results_summary.md`
    

The script now serves as a paper-facing collector and should be interpreted relative to the chosen `--results-dir`.

---

## 9. Submission-bundle workflow

The current recommended manuscript-facing packaging script is:

```bash
bash scripts/run_ptqnu_submission_bundle.sh
```

This script is intended to:

1. refresh baseline and robustness aggregate tables
    
2. run the additional `ptq-nu` hero-model diagnostics
    
3. assemble a clean submission bundle directory
    

Typical output target:

```text
results/paper_submission_ptqnu_YYYYMMDD/
```

This bundle is designed for:

- manuscript drafting
    
- supplement preparation
    
- archiving / sharing with collaborators
    
- final journal submission support
    

---

## 10. Reproducibility

Recommended practices:

- Fix seeds via `--seed`
    
- Use HDF5 backends via:
    
    ```bash
    --backend-hdf5 results/<run>/chain.h5
    ```
    
- Resume chains when needed with:
    
    ```bash
    --resume
    ```
    

Please record:

- Python version
    
- `requirements.txt`
    
- git commit hash
    
- `global_summary.yaml`
    
- comparison CSVs
    
- robustness aggregate CSVs
    

---

## 11. Field glossary

### `global_summary.yaml`

Important fields include:

- `AIC_full`, `BIC_full`:  
    from the chosen likelihood (`gauss` or `t`); use these for paper-facing full-likelihood comparison.
    
- `chi2_total`
    
- `k_parameters`
    
- `N_total`
    
- `epsilon_median`, `q_median`, `a0_median`
    
- `sigma_sys_median`
    

### Compare tables

`compare_table.csv` typically contains:

- `model`
    
- `waic`
    
- `delta_waic`
    
- `p_waic`
    
- `n_params`
    
- `n_data`
    
- `rank`
    

`loo_table.csv` typically contains:

- `model`
    
- `elpd_loo`
    
- `p_loo`
    
- `looic`
    
- `delta_looic`
    
- `rank`
    

---

## 12. Optional knobs and ablations

- Student-t likelihood:
    
    ```bash
    --likelihood t --t-dof 8
    ```
    
- Planck-anchored prior:
    
    ```bash
    --prior planck-anchored
    ```
    
- Distance-invariant thickness sanity check:  
    available in the `kappa_h` workflow where supported.
    

---

## 13. Project layout

Key paths relevant to fitting, comparison, and extensions:

| Path                                             | Purpose                                                                                            |
| ------------------------------------------------ | -------------------------------------------------------------------------------------------------- |
| `src/ptquat/experiments.py`                      | WAIC/LOO comparison, pWAIC diagnostics, PPC, stress, closure, plateau, κ-gal, κ-profile, z-profile |
| `src/ptquat/fit_global.py`                       | MCMC fitting and likelihood evaluation                                                             |
| `src/ptquat/likelihood.py`                       | Likelihood and covariance construction                                                             |
| `src/ptquat/cli.py`                              | CLI entry points (`fit`, `exp compare`, `exp loo`, etc.)                                           |
| `src/ptq/experiments/kappa_h.py`                 | Thickness regression / κ–h / κ+Σ audit                                                             |
| `scripts/make_paper_artifacts.py`                | Collect paper-facing tables, figures, and summaries                                                |
| `scripts/run_baseline_sixmodels_12000.sh`        | Six-model long-chain baseline                                                                      |
| `scripts/summarize_baseline_sixmodels.py`        | Aggregate baseline outputs                                                                         |
| `scripts/run_robustness_core_multiseed12000.sh`  | Multi-seed robustness audit                                                                        |
| `scripts/summarize_robustness_core_multiseed.py` | Aggregate robustness outputs                                                                       |
| `scripts/run_ptqnu_submission_bundle.sh`         | Hero-model diagnostics + submission bundle assembly                                                |

---

## 14. Extending with new experiments

When adding a new diagnostic or comparison experiment:

1. Implement the computation in `experiments.py` (or the relevant module).
    
2. Keep the output layout consistent with:
    
    - `global_summary.yaml`
        
    - `chain.h5`
        
    - `results/<run>/<experiment_name>/...`
        
3. For information-criterion methods, ensure `log_lik` shape is consistent:
    
    - per-point: `(n_samples, n_data)`
        
    - per-galaxy: `(n_samples, n_galaxies)`
        
4. If the experiment is paper-facing, add a corresponding collector hook in `scripts/make_paper_artifacts.py`.
    

---

## 15. Interpretation policy for this repo

This repository is designed to support **auditable PTQ-family phenomenology**.

That means:

- strong RC-level performance may identify a useful PTQ realization
    
- geometry audits may provide supporting structure
    
- failed strict closure tests should be reported transparently
    
- comparison models (`mond-screen`, `ptq-screen`) should remain available even when not selected as the manuscript-facing hero model
    

The repo therefore prioritizes:

- reproducibility
    
- model comparability
    
- diagnostic transparency
    
- explicit separation between **fit success**, **geometry audit**, and **closure claims**
    

---

## 16. Citation

If you use this code or reproduce results from this repository, please cite:

1. the accompanying paper(s)
    
2. this repository / archived artifact bundle when available
    

A formal citation block can be added once the manuscript and archive DOI are finalized.
