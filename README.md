# PTQ-Quaternionic-RC — SPARC Rotation-Curve Pipeline

**TL;DR.**  
This repo reproduces the experiments in our paper where **PTQ is positioned as the physical origin of MOND**.  
We compare **Baryon-only**, **NFW-1p**, **MOND(ν)**, **PTQ (linear)**, **PTQ-ν**, and **PTQ-screen**, report **full-likelihood AIC/BIC**, and run a robustness/closure suite including **PPC**, **error stress (×2)**, **inner-masking**, **H₀ scan**, and two **κ-checks** that diagnose a universal renormalization (\kappa).

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
```

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

---

## 2. Models

- `baryon`: $(v=\sqrt{v_{\rm bar}^2}).$
    
- `mond`: MOND (simple-ν). If `--a0-si` not given, (a_0) is sampled with prior `--a0-range`.
    
- `nfw1p`: one halo parameter ($M_{200}$) per galaxy; concentration from a c–M power law (`--c0`, `--c-slope`).
    
- `ptq` (linear): $(v=\sqrt{v_{\rm bar}^2 + (\epsilon cH_0),r})$. **Negative control** (ruled out by data).
    
- `ptq-nu`: reuse MOND shape but set $(a_0=\epsilon cH_0)$ (ε global).
    
- `ptq-screen`: generalized MOND-like $(\nu_q(y)=0.5+\sqrt{0.25+y^{-q}})$ with **global** q and $(a_0=\epsilon cH_0)$.
    

**Priors:** choose via `--prior`:

- `galaxies-only` (flat ε on (0,4)).
    
- `planck-anchored` (Gaussian ε prior around 1.47±0.05).
    

**Likelihood:** `--likelihood gauss|t` (default: gauss). Student-t (ν>2) improves robustness to outliers.

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
    
- `plot_*.png` for each galaxy.
    

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

> **Reporting rule:** In the paper, **always cite `AIC_full`/`BIC_full`** (computed from the chosen likelihood), not `AIC_quad/BIC_quad`.  
> Use ΔAIC/ΔBIC relative to the **best (lowest)** model; you can convert ΔAIC into Akaike weights to discuss strength.

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

### S3. Inner-disk masking

```bash
ptquat exp mask --model ptq-screen --rmin-kpc 2.0 \
  --data dataset/sparc_tidy.csv --outroot results/mask
```

### S4. H₀ sensitivity scan

```bash
ptquat exp H0 --model ptq-screen --data dataset/sparc_tidy.csv \
  --outroot results/H0_scan --H0-list 60 67.4 70 73 76
```

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

> In our runs, the **simplest closure (a_0=\epsilon cH_0)** fails in amplitude (|Δε|≫3σ).  
> This motivates a **universal renormalization** (\kappa) tested below.

### 5.2 κ-checks (observational validation of a global (\kappa))

**(A) Per-galaxy regression (kappa-gal):**

```bash
ptquat exp kappa-gal --results results/ptq_screen \
  --omega-lambda 0.69 --eta 0.15 --frac-vmax 0.9 --nsamp 300
```

Outputs a CSV and PNG; the fit reports slope≈κ and (R^2).  
**Sensitivity:** also run `--frac-vmax 0.8` and `1.0` to show slope stability.

**(B) Radius-resolved stack (kappa-prof):**

```bash
ptquat exp kappa-prof --results results/ptq_screen \
  --omega-lambda 0.69 --eta 0.15 --nbins 24
```

The stacked median/band follows ~(\propto 1/x). For the paper figure, overlay **(\kappa\eta/x)** (κ≈0.136).

---

## 6. Residual-acceleration plateau (main figure)

```bash
ptquat exp plateau --results results/ptq_screen --data dataset/sparc_tidy.csv
```

Produces `{results}/plateau_per_point.csv`, `{results}/plateau_binned.csv`, and a PNG.

---

## 7. What to report in the paper

- **Information criteria:** use `AIC_full/BIC_full` and **ΔAIC/ΔBIC** relative to the best model.  
    Typical outcome (may vary slightly with seeds/likelihood):  
    **PTQ-screen** < **MOND** < **PTQ-ν** ≪ **NFW-1p** ≪ **PTQ (linear)** ≪ **Baryon**.  
    It is acceptable to write:  
    _“PTQ-screen outperforms MOND at nearly identical complexity (k≈93 vs 92), with ΔAIC≈10 and ΔBIC≈4.8.”_
    
- **Parameters:** report `epsilon_median` (PTQ family), `q_median` (PTQ-screen), `a0_median` (MOND if sampled), `sigma_sys_median`.
    
- **PPC / stress / mask / H₀:** one-line summaries: ranking preserved; key posteriors shift minimally.
    
- **Closure & κ:** be explicit: simplest closure fails; **κ≈0.13** from κ-checks restores amplitude continuity across scales.
    

---

## 8. Reproducibility

- Fix seeds via `--seed`; enable HDF5 backends via `--backend-hdf5 results/<run>/chain.h5 --resume`.
    
- Record environment (Python version, `requirements.txt`) and include the produced `global_summary.yaml` and CSVs.
    

---

## 9. Field glossary (what the YAML fields mean)

- `AIC_full, BIC_full`: computed from the **chosen likelihood** (`gauss` or `t`); **use these in the paper**.
    
- `AIC_quad, BIC_quad`: χ²-based quadratic approximation (diagnostic only).
    
- `epsilon_median`: global ε (PTQ family).
    
- `a0_median`: MOND (a_0) (if sampled or fixed via `--a0-si`).
    
- `q_median`: PTQ-screen global shape parameter (q).
    
- `sigma_sys_median`: global velocity floor [km/s].
    
- `k_parameters`: number of free parameters used in AIC/BIC.
    
- `N_total`: number of RC points used.
    

---

## 10. Optional ablations

- **Student-t likelihood** (`--likelihood t --t-dof 8`) for robustness.
    
- **NFW c–M variants** via `--c0` and `--c-slope`.
    
- **Planck-anchored prior** (`--prior planck-anchored`) vs `galaxies-only`.
    

---

## 11. Citation


---

## 12. Paper figures & tables — **one-command build**

在完成「主六模型擬合」與（可選）各 robustness/closure/κ 實驗後，可以用單一腳本把**論文用表格與圖片**全部匯出到 `paper_outputs/`：

```bash
python scripts/make_paper_artifacts.py
```

**輸出包含：**

- `model_compare.csv` / `model_compare.tex`（主表；含 ΔAIC/ΔBIC 與 Akaike weight）
    
- `fig_delta_AIC.png`, `fig_delta_BIC.png`
    
- `fig_ppc_coverage.png`（若已執行 `exp ppc`）
    
- `fig_h0_AIC.png`, `fig_h0_epsilon.png`（若已執行 `exp H0` 產生掃描 CSV）
    
- `fig_kappa_gal.png`, `fig_kappa_profile.png`（若已執行 `kappa-gal`/`kappa-prof`）
    
- `fig_plateau.png`（若已執行 `exp plateau`）
    
- `stress_mask_summary.csv` / `.tex`（若有 `results/stress` 與/或 `results/mask` 子目錄）
    
- `closure_summary.tex`（若有 `closure_test.yaml`）
    

**自訂模型—結果目錄對映（如果你的 results 路徑不同）：**

```bash
python scripts/make_paper_artifacts.py \
  --model "PTQ-screen=results/ptq_screen" \
  --model "MOND=results/ejpc_mond" \
  --model "PTQ-v=results/ptq_nu" \
  --model "NFW-1p=results/ejpc_nfw1p" \
  --model "PTQ=results/ejpc_main" \
  --model "Baryon=results/ejpc_baryon" \
  --outdir paper_outputs
```

**其他常用參數：**

- `--ppc-json results/ptq_screen/ppc_coverage.json`
    
- `--h0-scan-csv results/H0_scan/ptq-screen_H0_scan.csv`
    
- `--kappa-gal-csv results/ptq_screen/kappa_gal_per_galaxy.csv`
    
- `--kappa-gal-summary results/ptq_screen/kappa_gal_summary.json`
    
- `--kappa-prof-binned results/ptq_screen/kappa_profile_binned.csv`
    
- `--plateau-binned results/ptq_screen/plateau_binned.csv`
    
- `--stress-root results/stress --mask-root results/mask`
    
- `--closure-yaml results/ptq_screen/closure_test.yaml`
    
- `--outdir paper_outputs_v2`
    

> **Reminder:** 圖表皆以 **full-likelihood** 的 `AIC_full/BIC_full` 為準；手稿撰寫請引用 ΔAIC/ΔBIC（相對最佳模型）。

---

### 12.1 Manual per-figure recipes (quick cheatsheet)

若你想針對單張圖快速重跑，流程如下（先產生各實驗的中間檔，再用生成腳本繪圖）：

**ΔAIC/ΔBIC 柱狀圖（主表同時輸出）**

```bash
python scripts/make_paper_artifacts.py --outdir paper_outputs
```

**PPC 覆蓋率**

```bash
ptquat exp ppc --results results/ptq_screen --data dataset/sparc_tidy.csv
python scripts/make_paper_artifacts.py \
  --ppc-json results/ptq_screen/ppc_coverage.json --outdir paper_outputs
```

**H₀ 掃描圖**

```bash
ptquat exp H0 --model ptq-screen --data dataset/sparc_tidy.csv \
  --outroot results/H0_scan --H0-list 60 67.4 70 73 76
python scripts/make_paper_artifacts.py \
  --h0-scan-csv results/H0_scan/ptq-screen_H0_scan.csv --outdir paper_outputs
```

**κ 檢核（per-galaxy 與半徑解析）**

```bash
ptquat exp kappa-gal  --results results/ptq_screen --omega-lambda 0.69 --eta 0.15
ptquat exp kappa-prof --results results/ptq_screen --omega-lambda 0.69 --eta 0.15 --nbins 24
python scripts/make_paper_artifacts.py \
  --kappa-gal-csv results/ptq_screen/kappa_gal_per_galaxy.csv \
  --kappa-gal-summary results/ptq_screen/kappa_gal_summary.json \
  --kappa-prof-binned results/ptq_screen/kappa_profile_binned.csv \
  --outdir paper_outputs
```

**Residual-acceleration plateau**

```bash
ptquat exp plateau --results results/ptq_screen --data dataset/sparc_tidy.csv
python scripts/make_paper_artifacts.py \
  --plateau-binned results/ptq_screen/plateau_binned.csv --outdir paper_outputs
```

**Stress / Mask 摘要表**

```bash
ptquat exp stress --model ptq-screen --data dataset/sparc_tidy.csv \
  --scale-i 2 --scale-D 2 --outroot results/stress
ptquat exp mask --model ptq-screen --data dataset/sparc_tidy.csv \
  --rmin-kpc 2.0 --outroot results/mask
python scripts/make_paper_artifacts.py \
  --stress-root results/stress --mask-root results/mask --outdir paper_outputs
```

**Closure 摘要表**

```bash
ptquat exp closure --results results/ptq_screen --omega-lambda 0.69
python scripts/make_paper_artifacts.py \
  --closure-yaml results/ptq_screen/closure_test.yaml --outdir paper_outputs
```
## Reproducing paper figures & tables

```bash
# install
pip install -e .
# one-shot (full)
python scripts/make_paper_artifacts.py
# fast mode (for CI / tests)
python scripts/make_paper_artifacts.py --fast --skip-fetch --data dataset/sparc_tidy.csv
# customize models
python scripts/make_paper_artifacts.py --models baryon mond ptq-screen
