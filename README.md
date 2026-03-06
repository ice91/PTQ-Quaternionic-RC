# PTQ-Quaternionic-RC — SPARC Rotation-Curve Pipeline

**TL;DR.**  
This repo reproduces the experiments in our paper where **PTQ is positioned as the physical origin of MOND**.  
We compare **Baryon-only**, **NFW-1p**, **MOND(ν)**, **PTQ (linear)**, **PTQ-ν**, and **PTQ-screen**, report **full-likelihood AIC/BIC**, and run a robustness/closure suite including **PPC**, **error stress (×2)**, **inner-masking**, **H₀ scan**, and **κ-checks** (cross-scale tests).  
We also include **Bayesian model comparison** via **WAIC** and **PSIS-LOO** (`ptquat exp compare`, `ptquat exp loo`), **pWAIC diagnostics** (per-galaxy / per-radius), and **paper-artifact aggregation** (tables, figures, summary) via `scripts/make_paper_artifacts.py`.  
A **κ–h regression** links disk scale-height (h) with **epicyclic frequency** (κ) and **surface density** (Σ).

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

This step builds the **S4G** disk thickness scale (h) catalog and merges it with **SPARC** for the **κ–h** experiment.

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

Outputs: `dataset/geometry/h_catalog.csv`, `dataset/geometry/sparc_with_h.csv`. SHA-256 checksums are written alongside to verify byte-level identity.

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
import yaml, pandas as pd
from pathlib import Path
import math

def read_summary(yml_path: str) -> dict:
    p = Path(yml_path)
    s = yaml.safe_load(p.read_text(encoding="utf-8")) or {}

    # 取得/回推 k
    k = s.get("k_parameters")
    logL = s.get("logL_total_full")
    N = s.get("N_total") or s.get("N")
    AIC_full = s.get("AIC_full")
    BIC_full = s.get("BIC_full")

    # 若缺 k 但有 AIC_full、logL → 由 AIC = 2k - 2 logL 回推
    if k is None and AIC_full is not None and logL is not None:
        k = 0.5 * (AIC_full + 2.0*logL)
        # 四捨五入為整數參數數目
        k = int(round(k))

    # 若缺 AIC_full 可由 k、logL 回推
    if AIC_full is None and (k is not None) and (logL is not None):
        AIC_full = 2*k - 2*logL

    # 若缺 BIC_full 可由 k、N、logL 回推
    if BIC_full is None and (k is not None) and (N is not None) and (logL is not None):
        BIC_full = k*math.log(N) - 2*logL

    row = dict(
        k = k,
        chi2 = s.get("chi2_total"),
        AIC_full = AIC_full,
        BIC_full = BIC_full,
        AIC_quad = s.get("AIC_quad"),
        BIC_quad = s.get("BIC_quad"),
        sigma_sys_med = s.get("sigma_sys_median"),
        epsilon_med = s.get("epsilon_median"),
        a0_med = s.get("a0_median"),
        N = N,
    )
    return row

runs = {
  "PTQ-screen": "results/full_20251014_084358/ptq-screen_gauss/global_summary.yaml",
  "PTQ-v":      "results/full_20251014_084358/ptq-nu_gauss/global_summary.yaml",
  "PTQ":        "results/full_20251014_084358/ptq_gauss/global_summary.yaml",
  "Baryon":     "results/full_20251014_084358/baryon_gauss/global_summary.yaml",
  "NFW-1p":     "results/full_20251014_084358/nfw1p_gauss/global_summary.yaml",
  "MOND":       "results/full_20251014_084358/mond_gauss/global_summary.yaml",
}

rows=[]
for name, yml in runs.items():
    try:
        row = read_summary(yml)
        row["model"] = name
        rows.append(row)
    except FileNotFoundError:
        rows.append(dict(model=name))  # 故障時保留占位
        print(f"[WARN] missing file: {yml}")
    except Exception as e:
        rows.append(dict(model=name))
        print(f"[WARN] {name}: {e}")

df = pd.DataFrame(rows).set_index("model")
# 僅在 BIC_full 存在時排序，否則跳過排序避免錯誤
df = df.sort_values("BIC_full", na_position="last") if "BIC_full" in df.columns else df
df.to_csv("results/all_model_compare.csv")
print(df[["k","chi2","AIC_full","BIC_full","AIC_quad","BIC_quad","sigma_sys_med","epsilon_med","a0_med"]])
print("Saved results/all_model_compare.csv")
PY
```

### 3.1 WAIC model comparison (`ptquat exp compare`)

From posterior samples, compute **WAIC** (Watanabe–Akaike Information Criterion) for model comparison. Outputs go to a chosen `--outdir` (e.g. `results/paper_extra/model_compare/`).

```bash
ptquat exp compare --models ptq-screen mond nfw1p baryon \
  --data dataset/sparc_tidy.csv \
  --fit-root results \
  --outdir results/paper_extra/model_compare \
  --seed 0
```

- **Outputs:** `compare_table.csv`, `compare_table.tex`, `rank_plot.png` / `rank_plot.pdf`, `manifest.yaml`, and optionally `breakdown.csv`, `compare_table_covfull_pergal.csv`, `compare_table_all_modes.csv`.  
- **pWAIC diagnostics** (when run): `pwaic_diagnostics/var_loglik_points.csv`, `pwaic_by_galaxy.csv`, `top20_galaxies_pwaic.csv`, and plots (`pwaic_by_galaxy_bar.png`, `radius_vs_varloglik.png`).
- **Options:** `--run-fits` to run short MCMC fits if `chain.h5` or `global_summary.yaml` are missing; `--burn-frac`, `--thin` for chain post-processing.

`compare_table.csv` columns: `model`, `waic`, `delta_waic`, `p_waic`, `n_params`, `n_data`, `rank`, and optionally `lppd`, `mean_lppd_per_point`, `mean_loglik_per_point`.

### 3.2 PSIS-LOO (`ptquat exp loo`)

Compare models via **Pareto-smoothed importance sampling LOO** (requires `arviz`).

```bash
ptquat exp loo --models ptq-screen mond nfw1p baryon \
  --data dataset/sparc_tidy.csv --fit-root results \
  --outdir results/paper_extra/loo_compare --seed 0
```

- **Outputs:** `loo_table.csv` (model, elpd_loo, p_loo, looic, delta_looic, rank), `pareto_k.csv`, `pareto_k_hist.png`, `elpd_difference_plot.png`.
- Use `--run-fits` if chains are missing. Prefer `pareto k < 0.7` for most points when interpreting LOO.

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

## 8. Paper artifacts (tables & figures for the paper)

The script `scripts/make_paper_artifacts.py` aggregates comparison results and writes LaTeX tables, PDF figures, and a short summary into `results/paper_artifacts/`. Run after `ptquat exp compare` and `ptquat exp loo` so that WAIC/LOO/pWAIC outputs exist under `results/paper_extra/`.

```bash
python scripts/make_paper_artifacts.py --data dataset/sparc_tidy.csv --out results/paper_run
```

- **Tables (`.tex`):** `waic_compare_table.tex`, `loo_compare_table.tex`, `pwaic_top_galaxies_table.tex` (when corresponding compare/diagnostic outputs exist).
- **Figures (`.pdf`):** `model_compare_waic.pdf`, `model_compare_loo.pdf`, `pwaic_by_galaxy_bar.pdf`.
- **Summary:** `paper_results_summary.md` — WAIC comparison, LOO comparison, pWAIC diagnostics, and key conclusions.

The script infers `results/` from `--out` or `--data`; it also writes `ejpc_model_compare.csv` and optional `--figdir` figure copies. It automatically populates `results/paper_artifacts/` under the chosen results root whenever the corresponding WAIC/LOO/pWAIC outputs exist (e.g. under `results/paper_extra/model_compare/` and `results/paper_extra/loo_compare/`). Use `--results-dir` to point at a specific results root.

---

## 9. Reproducibility

- Fix seeds via `--seed`; enable HDF5 backends via `--backend-hdf5 results/<run>/chain.h5 --resume`.
    
- Record environment (Python version, `requirements.txt`) and include produced `global_summary.yaml` and CSVs.
    

---

## 10. Field glossary (what the YAML/JSON fields mean)

**Fit YAML (`global_summary.yaml`)**

- `AIC_full, BIC_full`: from the **chosen likelihood** (`gauss` or `t`); **use these in the paper**.
    

---

## 11. Optional ablations & knobs

- **Student-t likelihood** (`--likelihood t --t-dof 8`) for RC fits.
    
- **Planck-anchored prior** (`--prior planck-anchored`) vs `galaxies-only`.
    

---

## 12. Project layout & extending with new experiments

Directory layout relevant to **statistical analysis and new experiments**:

| Path | Purpose |
|------|--------|
| `src/ptquat/experiments.py` | WAIC/LOO comparison (`compare_models_waic`, `loo_compare_models`), pWAIC diagnostics, PPC, stress, closure, plateau, κ–gal |
| `src/ptquat/fit_global.py` | MCMC fitting; `log_likelihood_per_galaxy` (per-point, diag cov for WAIC), `log_likelihood_full_per_galaxy` (per-galaxy, full cov) |
| `src/ptquat/likelihood.py` | Likelihood and covariance construction |
| `src/ptquat/cli.py` | CLI: `exp compare`, `exp loo`, `fit`, `geom`, etc. |
| `scripts/make_paper_artifacts.py` | Aggregates WAIC/LOO/pWAIC outputs → `results/paper_artifacts/` (tables, figures, summary) |

**Adding a new statistical or comparison experiment:**

1. **New metric (e.g. another information criterion):** Implement the computation in `experiments.py` (e.g. from posterior samples or `log_lik` matrices). Use the same `fit_root` / `global_summary.yaml` + `chain.h5` layout as WAIC/LOO.
2. **New CLI subcommand:** In `cli.py`, add a subparser under `exp` and call the new function in `experiments.py`; keep outputs under `results/<run_id>/<experiment_name>/` with a small `manifest.yaml` (git hash, command, seed).
3. **Log-likelihood shape:** WAIC/LOO expect `log_lik` of shape `(n_samples, n_data)` (per-point) or `(n_samples, n_galaxies)` for per-galaxy; ensure your new metric is consistent with the same convention.
4. **Paper artifacts:** If the new experiment produces tables or figures, extend `make_paper_artifacts.py` with a `_try_build_<name>_artifacts(results_root, paper_root)` and call it from `main()` so `results/paper_artifacts/` stays the single entry point for the paper.

---

## 13. Citation

If you use this code or reproduce our results, please cite the accompanying paper (TBD) and this repository.

