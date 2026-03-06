# Revision results summary (reviewer briefing)

This document summarizes the **actual** result outputs present in the repo as of the revision pack. It is based only on existing files and numbers; gaps are stated explicitly.

---

## 1. Mond-screen vs ptq-screen / mond: fair comparison

**Status: not available.**

- **mond-screen** is implemented in code (matched-kernel MOND control, same ν_q as ptq-screen, free a0) but **no fit has been run** in this repo.
- There is no `results/mond-screen_gauss/` directory and no `mond-screen` entry in any comparison table.
- **Available comparison** (from `results/paper_extra/model_compare/`):
  - **mond** (standard MOND): WAIC = 20730.1 (rank 1), p_waic = 2044.0, k = 93.
  - **ptq-screen**: WAIC = 27142.2 (rank 2), ΔWAIC = 6412.1 vs mond, p_waic = 5272.9, k = 94.
- So the only current WAIC comparison is **standard MOND vs PTQ-screen**, not matched-kernel MOND (mond-screen) vs PTQ-screen. A fair comparison on the same kernel family is **not yet supported by existing outputs**.

**Data source:**  
`results/{ptq-screen,mond,nfw1p,baryon}_gauss/global_summary.yaml`; `results/paper_extra/model_compare/compare_table.csv`. No mond-screen fit directory.

---

## 2. WAIC / pWAIC diagnostics and main sources

### 2.1 WAIC comparison (actual numbers)

| Model      | WAIC (diag, per point) | ΔWAIC | p_waic | k   | N    |
|-----------|-------------------------|-------|--------|-----|------|
| mond      | 20730.1                 | 0.0   | 2044.0 | 93  | 1782 |
| ptq-screen| 27142.2                 | 6412.1| 5272.9 | 94  | 1782 |
| nfw1p     | 132596.7                | 111866.6 | 54734.2 | 183 | 1782 |
| baryon    | 302556.9                | 281826.8 | 128242.0 | 92  | 1782 |

- **Source:** `results/paper_extra/model_compare/compare_table.csv`, `compare_table.tex`.
- **Breakdown (Δlppd / ΔpWAIC vs best):**  
  ptq-screen vs mond: delta_lppd_vs_best ≈ +22.9, delta_pwaic_vs_best ≈ +3228.9, delta_waic_vs_best ≈ 6412.1.  
  Source: `results/paper_extra/model_compare/breakdown.csv`.
- **Full-covariance, per-galaxy WAIC** (N = 91): mond 20143.4, ptq-screen 27245.3 (Δ ≈ 7102).  
  Source: `results/paper_extra/model_compare/compare_table_covfull_pergal.csv`, `compare_table_all_modes.csv`.

**Note:** All fits in repo use **steps = 100** (short chains). WAIC and p_waic are computed from these chains; production-length runs may change values.

### 2.2 pWAIC diagnostics

- **Source directory:** `results/paper_extra/model_compare/pwaic_diagnostics/`.
- **Scope:** Diagnostics are produced for the **best-WAIC model only**. In the current run that is **mond**, not ptq-screen (manifest: `pwaic_diagnostics_model: mond`, `pwaic_diagnostics_cov_mode: diag`).
- **Files present:**
  - `diagnostics_summary.yaml`: model = mond, cov_mode = diag, n_points = 1782, n_galaxies = 91, p_waic_total ≈ 2043.96.
  - `pwaic_by_galaxy.csv`, `top20_galaxies_pwaic.csv`: per-galaxy pWAIC contribution (mond).
  - `var_loglik_points.csv`, `top20_points_varloglik.csv`, `pwaic_by_radius.csv`.
  - `pwaic_by_galaxy_bar.png`, `radius_vs_varloglik.png`.
- **Top galaxies by pWAIC (mond):** ESO563-G021 (≈263.8), NGC2841 (≈255.5), IC4202 (≈213.3), NGC7814 (≈166.6), NGC4217 (≈123.5), then others.  
  Source: `pwaic_diagnostics/top20_galaxies_pwaic.csv`.
- **By radius (mond):** pWAIC contribution is largest in mid-radius bins (e.g. 5–10 kpc, 10–20 kpc) in the provided `pwaic_by_radius.csv`.

**Gap:** There are **no pWAIC diagnostics for ptq-screen** (or for nfw1p/baryon) in this repo. Only the best model (mond) is diagnosed. The narrative “ptq-screen has higher p_waic” is supported by the compare table, but per-galaxy/per-radius breakdown for ptq-screen is missing.

### 2.3 LOO (PSIS-LOO)

- **Status: missing.** No `loo_table.csv` or `results/paper_extra/loo_compare/` (or equivalent) found in the repo. The `paper_artifacts/paper_results_summary.md` text correctly states that LOO comparison has not been found. LOO is not part of the current revision pack outputs.

**Sources checked:** `results/**/loo*.csv`, `results/paper_extra/` — none.

---

## 3. Closure sensitivity scan: results and limitations

### 3.1 Single-point closure (ptq-screen)

- **Source:** `results/ptq-screen_gauss/closure_test.yaml` (single run with ε_cos = 1.47).
- **Values:** epsilon_RC ≈ 0.1955, sigma_RC ≈ 0.0251, epsilon_cos = 1.47, diff ≈ −1.275, **pass_within_3sigma: false.**
- So at the default cosmology anchor, **strict closure (ε_RC ≈ ε_cos) does not hold** for this fit.

### 3.2 Closure sensitivity scan (actual outputs)

- **Scan 1 (epsilon_cos):**  
  `results/closure_scan/` — epsilon_cos linearly spaced in [1.2, 1.8], n = 13.  
  results_dir = `results/ptq-screen_gauss` (same fit for all points).
- **Scan 2 (omega_lambda):**  
  `results/closure_scan_ol/` — omega_lambda in [0.65, 0.75], n = 11.  
  Same results_dir (ptq-screen fit).

**Reported quantities (both scans):**

- epsilon_RC ≈ 0.1955, sigma_RC ≈ 0.0251 (constant: single fit).
- For each anchor: epsilon_cos (or omega_lambda), delta_epsilon = epsilon_RC − epsilon_cos, pass_within_1/2/3sigma.
- **Result:** For all 13 (resp. 11) points, **pass_within_1sigma, pass_within_2sigma, pass_within_3sigma are all False.**  
  delta_epsilon ranges from about −1.00 to −1.60 (epsilon_cos scan) and similarly negative for the omega_lambda scan. So no anchor in the scanned ranges brings the fit into closure within 1–3 σ.

**Sources:** `results/closure_scan/closure_sensitivity.csv`, `closure_sensitivity.yaml`, `manifest.yaml`; same for `results/closure_scan_ol/`. Plot: `closure_sensitivity_plot.png`.

### 3.3 Limitations

- **Single fit:** The scan varies only the cosmology anchor (epsilon_cos or omega_lambda). The RC fit (epsilon_RC, sigma_RC) is from **one** run (ptq-screen). There is no closure scan using mond or mond-screen fits.
- **No C-parameter scan:** The design is ε_RC vs ε_cos (or ΩΛ→ε_cos). There is no explicit C-parameter scan in the repo; if a C↔ε_cos mapping were introduced later, it would need to be documented and run separately.
- **Interpretation:** The current outputs show that for this ptq-screen fit, closure is not achieved at the default or scanned anchors; sensitivity is to the chosen cosmology band, not to a different model or different fit.

---

## 4. Supported / Not yet supported

**Supported by current revision pack:**

- WAIC comparison (diag per-point and full per-gal) for **mond, ptq-screen, nfw1p, baryon**; breakdown (Δlppd, ΔpWAIC, ΔWAIC) vs best model.
- pWAIC diagnostics for the **best-WAIC model (mond)**: per-galaxy, per-point, by-radius; top-20 tables and figures; source: `paper_extra/model_compare/pwaic_diagnostics/`.
- Single-point closure test for one fit (ptq-screen at ε_cos = 1.47); output in `ptq-screen_gauss/closure_test.yaml`.
- Closure sensitivity scan over **epsilon_cos** or **omega_lambda** for that same ptq-screen fit; CSV, YAML, and plot in `results/closure_scan/` and `results/closure_scan_ol/`.
- Paper-artifact aggregation: WAIC LaTeX table, pWAIC top-galaxies table, WAIC bar plot PDF (from `paper_artifacts/`). Summary text in `paper_results_summary.md` (with LOO/pWAIC path caveats as noted above).

**Not yet supported (gaps that can affect revision):**

- **mond-screen:** No fit run, no entry in compare tables. The intended “matched-kernel MOND vs PTQ-screen” comparison cannot be reported from current outputs.
- **LOO:** No LOO run or `loo_table.csv` in repo; PSIS-LOO comparison is not available.
- **pWAIC for ptq-screen (and others):** Diagnostics exist only for the best model (mond). No per-galaxy/per-radius pWAIC for ptq-screen in this pack.
- **Closure for other models:** Closure test and closure-scan outputs are for ptq-screen only. No closure or closure-scan for mond or mond-screen.
- **Production chains:** All `global_summary.yaml` files show steps = 100. For a final revision, longer chains and possibly separate production WAIC/LOO/closure runs may be needed.

---

*Generated from repo result paths and file contents only. No code or paper text was modified.*
