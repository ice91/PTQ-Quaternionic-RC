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

### 1.3 S4G disk scale-heights (h) & merge into SPARC

這一步把 **S4G** 的盤厚度尺度 (h)（含距離）建成 catalog，並與 **SPARC** 每半徑資料合併到單一檔案，用於 **κ–h** 實驗。

```bash
# Build h-catalog from S4G table (Vizier export)
python -m ptq.data.s4g_h_pipeline build-h \
  --src dataset/raw/vizier_table1.csv \
  --out dataset/geometry/h_catalog.csv \
  --outliers dataset/geometry/h_z_outliers.csv

# Merge h into SPARC tidy rows (name canonicalization + alias mapping)
python -m ptq.data.s4g_h_pipeline merge-sparc-h \
  --sparc dataset/sparc_tidy.csv \
  --h dataset/geometry/h_catalog.csv \
  --out dataset/geometry/sparc_with_h.csv \
  --alias dataset/geometry/aliases.csv \
  --unmatched dataset/geometry/unmatched_galaxies.csv
```

Notes:

- `aliases.csv`（兩欄：`alias,canonical`）可提升 join 成功率。
    
- `h_z_outliers.csv` 列出疑似的 (h) outliers（高相對誤差或距離異常）。κ–h 實驗可用 `--drop-h-outliers` 排除。
    

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
> This motivates a **universal renormalization** ((\kappa)) tested below.

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

The stacked median/band follows (\propto 1/x). For the paper figure, overlay **(\kappa\eta/x)** (κ≈0.136).

### 5.3 κ–h regression (disk thickness vs. (\kappa) & (\Sigma))

我們以  
[  
\log_{10} h ;=; a ;+; b,\log_{10}\kappa ;+; c,\log_{10}\Sigma_{\rm tot}  
]  
做加權線性回歸（WLS）。若缺少每半徑的 (\Sigma)，本版本會在**代表半徑 (R^\star)**（預設 **vdisk-peak**）以 galaxy-level 量（`L36_*`, `M_HI`）合成單一點的 (\Sigma_{\rm tot})，回歸即為「每星系一筆」。

**Quick start（per-galaxy；R★=vdisk-peak；WLS + LOO + Bootstrap）：**

```bash
python -m ptq.experiments.kappa_h \
  --sparc-with-h dataset/geometry/sparc_with_h.csv \
  --per-galaxy --rstar vdisk-peak --wls \
  --ml36 0.5 --rgas-mult 1.7 --gas-helium 1.33 \
  --loo --bootstrap 5000 --cv-by-galaxy \
  --out-csv     dataset/geometry/kappa_h_used.csv \
  --report-json dataset/geometry/kappa_h_report.json
```

**目前資料上的典型結果（N=19；per-galaxy；R★=vdisk-peak；WLS）：**

- **κ-only**：(b=-0.64\pm0.51)，(R^2=0.09)（解釋力極弱）
    
- **κ+Σ**：(a=4.68\pm0.65)、(b_\kappa=-2.56\pm0.33)、(c_\Sigma=1.12\pm0.14)，**(R^2=0.82)**
    
- **模型選擇（AICc）**：κ+Σ = **-48.35**，優於 κ-only（-25.98）與 Σ-only（-28.93）  
    → **ΔAICc(κ-only→κ+Σ)=22.36**, **ΔAICc(Σ-only→κ+Σ)=19.42**（≫10，屬強力證據）
    
- **交叉驗證（by-galaxy LOO）**：(\beta_\kappa\approx-2.56\pm0.11)、(\beta_\Sigma\approx1.12\pm0.04)，**穩定**
    
- **Bootstrap(5000)**：(\beta_\kappa\in[-2.95,-2.19])、(\beta_\Sigma\in[0.99,1.28])，**不跨 0**
    
- **距離不變（DIST-INV）**：(\log(h/D)=0.181 - 0.512\log(\kappa D) + 0.409\log\Sigma)，(R^2\approx0.22)
    

輸出：

- `kappa_h_used.csv`：回歸使用到的樣本（log-space）。
    
- `kappa_h_report.json`：完整係數、AICc、LOO、Bootstrap、DIST-INV 摘要。
    

可選參數與敏感度：

- `--rstar {vdisk-peak|2.2Rd|flat-rc}`：代表半徑定義。
    
- `--drop-h-outliers --h-outliers-csv dataset/geometry/h_z_outliers.csv`：排除 S4G 可疑 (h)。
    
- `--deproject-velocity --deriv-window 7`：若跑每半徑模式時計算 (\kappa(R)) 需要（per-galaxy 模式不必）。
    

---

## 6. Residual-acceleration plateau (main figure)

```bash
ptquat exp plateau --results results/ptq_screen --data dataset/sparc_tidy.csv
```

Produces `{results}/plateau_per_point.csv`, `{results}/plateau_binned.csv`, and a PNG.

---

## 7. What to report in the paper

**Model comparison（rotation-curve fits）**

- **Information criteria:** use `AIC_full/BIC_full` and **ΔAIC/ΔBIC** relative to the best model.  
    Typical outcome (may vary slightly with seeds/likelihood):  
    **PTQ-screen** < **MOND** < **PTQ-ν** ≪ **NFW-1p** ≪ **PTQ (linear)** ≪ **Baryon**.  
    It is acceptable to write:  
    _“PTQ-screen outperforms MOND at nearly identical complexity (k≈93 vs 92), with ΔAIC≈10 and ΔBIC≈4.8.”_
    
- **Parameters:** report `epsilon_median` (PTQ family), `q_median` (PTQ-screen), `a0_median` (MOND if sampled), `sigma_sys_median`.
    
- **PPC / stress / mask / H₀:** one-line summaries: ranking preserved; key posteriors shift minimally.
    
- **Closure & κ:** be explicit: simplest closure fails; **κ≈0.13** from κ-checks restores amplitude continuity across scales.
    

**κ–h regression（disk thickness vs. κ & Σ）**

- 同步報告 **κ-only、Σ-only、κ+Σ** 三模型之 (R^2)、AICc 與 **ΔAICc**（相對 κ+Σ）。目前資料顯示 κ+Σ 明顯優勢（ΔAICc≥19）。
    
- 係數（WLS）方向與顯著性：(b_\kappa<0)、(c_\Sigma>0)；LOO 與 Bootstrap **不跨 0**。建議主文給點估與 16–84 百分位，方法節列出 LOO 平均/標準差。
    
- **距離不變（DIST-INV）** 的 (R^2) 目前偏低（≈0.2）：誠實註明限制（樣本小、距離誤差共用、(\Sigma) fallback 近似），主結論以 κ+Σ 的 **ΔAICc/LOO/Bootstrap** 為主。
    
- 附錄納入敏感度：`--rstar 2.2Rd` / `flat-rc` 的重跑（係數與 AICc 穩定），以及 (\mathrm{corr}(\log\kappa,\log\Sigma)) 與 VIF（排除共線性主導）。
    

---

## 8. Reproducibility

- Fix seeds via `--seed`; enable HDF5 backends via `--backend-hdf5 results/<run>/chain.h5 --resume`.
    
- Record environment (Python version, `requirements.txt`) and include produced `global_summary.yaml` and CSVs.
    
- 對 κ–h：將 `kappa_h_report.json` 與 `kappa_h_used.csv` 一併存檔、上傳附檔。
    

---

## 9. Field glossary (what the YAML/JSON fields mean)

**Fit YAML (`global_summary.yaml`)**

- `AIC_full, BIC_full`: from the **chosen likelihood** (`gauss` or `t`); **use these in the paper**.
    
- `AIC_quad, BIC_quad`: χ²-based quadratic approximation (diagnostic only).
    
- `epsilon_median`: global ε (PTQ family).
    
- `a0_median`: MOND (a_0) (if sampled or fixed via `--a0-si`).
    
- `q_median`: PTQ-screen global shape parameter (q).
    
- `sigma_sys_median`: global velocity floor [km/s].
    
- `k_parameters`: number of free parameters used in AIC/BIC.
    
- `N_total`: number of RC points used.
    

**κ–h JSON (`kappa_h_report.json`)**

- `model`: one of `kappa_only`, `sigma_only`, `kappa_sigma`.
    
- `beta`: ([a, b_\kappa, c_\Sigma]) for κ+Σ；κ-only/Σ-only 省略不在模型內的係數。
    
- `stderr`: WLS 標準誤。
    
- `R2`, `AICc`.
    
- `loo_mean`, `loo_std`: by-galaxy LOO 的係數均值/標準差（若指定 `--cv-by-galaxy`）。
    
- `bootstrap_median`, `p16`, `p84`: 係數的 BBC（5000 次為預設建議）。
    
- `dist_inv`: 距離不變回歸的係數與 (R^2)。
    
- `N_used`: 樣本數（per-galaxy 模式下等於星系數）。
    

---

## 10. Optional ablations & knobs

- **Student-t likelihood** (`--likelihood t --t-dof 8`) for RC fits.
    
- **NFW c–M variants** via `--c0` and `--c-slope`.
    
- **Planck-anchored prior** (`--prior planck-anchored`) vs `galaxies-only`.
    
- **κ–h knobs**：`--rstar`、`--ml36`（stellar ML，預設 0.5）、`--rgas-mult`（分子/金屬校正）、`--gas-helium`（氦因子）、`--wls`、`--loo`、`--bootstrap`、`--cv-by-galaxy`、`--drop-h-outliers`。
    

---

## 11. Citation

If you use this code or reproduce our results, please cite the accompanying paper (TBD) and this repository.


---

## 12. Paper figures & tables — build script

完成主模型擬合後，可用單一腳本產出**論文用表格**與**重點圖**到指定資料夾。**此腳本需要明確提供 `--data --out --figdir --models`**：

```bash
# Full (six models), Gaussian likelihood, quick smoke run
python scripts/make_paper_artifacts.py \
  --data dataset/sparc_tidy.csv \
  --out results/ejpc_run \
  --figdir paper_figs \
  --models baryon mond nfw1p ptq ptq-nu ptq-screen \
  --likelihood gauss \
  --fast
```

**會產出：**

- **Per-model results**（在 `--out/<model>_<likelihood>/`）  
    `global_summary.yaml`, `per_galaxy_summary.csv`, 每個星系的 `plot_*.png`；若執行診斷，另含 `ppc_coverage.json`, `kappa_*`, `plateau_*`, `closure_test.yaml`
    
- **比較表**（在 `--out/`）：`ejpc_model_compare.csv`（彙整 `AIC_full/BIC_full/χ²/k/N` 等）
    
- **圖檔彙整**（在 `--figdir/`）：`plateau_<modeldir>.png`、`kappa_gal_<modeldir>.png`、`kappa_profile_<modeldir>.png`
    

> 預設會對 `ptq-screen`（若包含於 `--models`）執行診斷（plateau / PPC / κ / closure）；若未包含，則改對最後一個模型執行。  
> 進階選項：`--likelihood t`、`--eta 0.15`、`--nbins 24`、`--omega-lambda 0.685`。  
> `--skip-fetch` 為相容舊介面，現版本會被忽略。  
> **κ–h** 不在此彙整腳本中，請依 §5.3 指令另行產出（建議將 JSON/CSV 一併存入 `paper_figs/`）。

### 12.1 Quick recipes (per-figure)

若想針對單圖快速重跑，可直接呼叫 `ptquat exp ...` 產出中間檔；上面的 build script 會自動把主要 PNG 收集到 `--figdir`。

**PPC 覆蓋率**

```bash
ptquat exp ppc --results results/ptq_screen --data dataset/sparc_tidy.csv
```

**H₀ 掃描**

```bash
ptquat exp H0 --model ptq-screen --data dataset/sparc_tidy.csv \
  --outroot results/H0_scan --H0-list 60 67.4 70 73 76
```

**κ 檢核（per-galaxy 與半徑解析）**

```bash
ptquat exp kappa-gal  --results results/ptq_screen --omega-lambda 0.69 --eta 0.15
ptquat exp kappa-prof --results results/ptq_screen --omega-lambda 0.69 --eta 0.15 --nbins 24
```

**Residual-acceleration plateau**

```bash
ptquat exp plateau --results results/ptq_screen --data dataset/sparc_tidy.csv
```

**Stress / Mask 摘要**

```bash
ptquat exp stress --model ptq-screen --data dataset/sparc_tidy.csv --scale-i 2 --scale-D 2 --outroot results/stress
ptquat exp mask   --model ptq-screen --data dataset/sparc_tidy.csv --rmin-kpc 2.0 --outroot results/mask
```

> 完成上述任一實驗後，再執行一次 `make_paper_artifacts.py`（同一組 `--out --figdir`）即可把新圖自動蒐集到 `--figdir`。

---

## Known notes / warnings

- **Pandas FutureWarning**：在建構 (\Sigma) 的 fallback 統計時，Pandas 可能印出  
    `DataFrameGroupBy.apply ... behavior is deprecated` 的 _FutureWarning_。  
    這不影響結果；後續版本將改用 `include_groups=False` 消除此訊息。
    
- **Data provenance**：請保存 `dataset/geometry/h_catalog.csv`, `sparc_with_h.csv`, `h_z_outliers.csv`, `unmatched_galaxies.csv`，以利審稿重現。
    
- **κ–h 可重現性**：將 `kappa_h_report.json` 與 `kappa_h_used.csv` 隨稿件提供。
    

---