# 論文修稿：三項分析 — 理解摘要與最小修改計畫

## 一、理解摘要

### 1. Model registry / model dispatch

- **註冊方式**：沒有獨立的 registry 字典，而是**分散式 if/elif**。模型名稱是字串（`ptq`, `ptq-nu`, `ptq-screen`, `ptq-split`, `baryon`, `mond`, `nfw1p`），在以下位置必須一致列舉或擴充：
  - **`fit_global.py`**：`_layout()` 定義參數版型（參數索引、slice）；`_unpack()` 從 theta 向量取出參數；`log_prior()`、`log_likelihood()`、`log_likelihood_per_galaxy()`、`log_likelihood_full_per_galaxy()` 以及 run() 內寫 summary 的迴圈，皆用 `if m == "model_name"` 分派對應的 `model_v_*` 與 prior。
  - **`models.py`**：提供 `model_v_ptq`, `model_v_mond`, `model_v_ptq_screen` 等純函數，輸入 (U, a0 或 ε, r, v_disk, v_bulge, v_gas)，輸出 v_model 陣列。
  - **`cli.py`**：`fit` / `exp` 子命令的 `--model` 使用 `choices=[...]` 列舉合法模型。
  - **`experiments.py`**：`_make_vfun()`、`ppc_check()`、`residual_plateau()`、`compare_models_waic` 等從 `global_summary.yaml` 讀出 `model` 後，用一串 if/elif 呼叫對應的 `model_v_*`。
- **新增模型**：需在以上所有分派點加入新分支，並在 `_layout` 中定義新模型的參數順序（例如 `mond-screen` = log10a0 + lq + U[G] + lnsig）。

### 2. Fit 輸出 global_summary.yaml / per_galaxy_summary.csv 的流程

- **入口**：`cli.py` 的 `fit` 子命令組裝 argv，呼叫 `fit_global.run(argv)`。
- **流程**：`fit_global.run()` 依序：
  1. 載入 SPARC tidy CSV → `galaxies`；
  2. 依 `args.model` 呼叫 `_layout(model, G, sigma_learn=True, a0_fixed)` 得到 `L`；
  3. 初始化 walkers、可選 HDFBackend；
  4. 用 `log_posterior` 跑 emcee；
  5. burn + thin 後取 chain，用中位數參數 `_unpack(med, L)` 得到 `P`；
  6. 對每個 galaxy 用同一套 if/elif 算 v_mod、Cg、chi2_g、ll_g，累積成 `chi2_tot`、`logL_tot`，並寫入 `rows`（per_galaxy 欄位：galaxy, n, Upsilon_med, chi2, chi2_red, loglike 等）；
  7. 計算 AIC_full/BIC_full（用 full likelihood），寫入 `summary` dict；
  8. 寫入 `per_galaxy_summary.csv`、`global_summary.yaml`，並對每個 galaxy 呼叫 `plotting.plot_rc()` 輸出 `plot_{galaxy}.png`。
- **慣例**：結果目錄為 `--outdir`，通常為 `results/<run>/<model>_gauss`（例如 `results/ptq_screen/ptq-screen_gauss`），其內固定有 `global_summary.yaml`、`per_galaxy_summary.csv`、`plot_*.png`；若使用 `--backend-hdf5` 則有 `chain.h5`。

### 3. exp compare / exp loo / exp closure 的入口

- **exp compare**：`cli.py` 中 `exp_cmd == "compare"` → `EXP.compare_models_waic(data_path, fit_root, models, outdir, ...)`。`compare_models_waic` 對每個 `models` 項找到 `fit_root/<model>_gauss/chain.h5` 與 `global_summary.yaml`，從 chain 算 per-point（及 per-galaxy full-cov）log_lik，用 `_waic_from_chain` 得 WAIC，寫出 `compare_table.csv`、`breakdown.csv`、`compare_table.tex`、`rank_plot.png/.pdf`、`manifest.yaml`，以及 full-cov 的 `compare_table_covfull_pergal.csv`、`compare_table_all_modes.csv`。可選 `run_fits_if_missing` 時會對缺失的 chain 跑短鏈。
- **exp loo**：`exp_cmd == "loo"` → `EXP.loo_compare_models(...)`。同樣依 `fit_root/<model>_gauss/` 讀 chain，用 `log_likelihood_per_galaxy` 建 `log_lik_mat`，交給 ArviZ 的 `az.loo(pointwise=True)`，寫出 `loo_table.csv`、`pareto_k.csv`、`pareto_k_hist.png`、`elpd_difference_plot.png`。
- **exp closure**：`exp_cmd == "closure"` → 先由 `--omega-lambda` 或 `--epsilon-cos` 算出 `epsilon_cos`，再呼叫 `EXP.closure_test(results_dir, epsilon_cos=..., omega_lambda=...)`。`closure_test` 只讀單一 `results_dir` 的 `global_summary.yaml`，取出 `epsilon_median`、`epsilon_p16/p84`，與給定的 `epsilon_cos` 比較，寫出 `closure_test.yaml`（含 epsilon_RC, epsilon_cos, sigma_RC, diff, pass_within_3sigma）。若 CLI 傳入 `--plot`，會另呼叫 `plotting.plot_omega_eps_curve` 畫 Fig.3。

### 4. paper_artifacts 如何蒐集 compare / loo / pwaic 結果

- **腳本**：`scripts/make_paper_artifacts.py`。主流程用 `--data`、`--out`（或 `--results-dir`）推斷 `results` 根目錄，彙整 YAML 到 `out_dir/<model>_gauss/global_summary.yaml` 並寫出 `ejpc_model_compare.csv`；若指定 `--figdir` 會複製 `plateau*.png`、`kappa_*.png`。
- **paper_artifacts 子目錄**：在 `main()` 末尾呼叫 `_try_build_waic_artifacts(primary_results, paper_root)`、`_try_build_loo_artifacts(...)`、`_try_build_pwaic_artifacts(...)`、`_build_summary_markdown(...)`，其中 `paper_root = primary_results / "paper_artifacts"`。
  - **WAIC**：從 `results/paper_extra/model_compare/` 讀取 `compare_table.tex`、`rank_plot.pdf`（或 .png），複製/轉成 `paper_artifacts/waic_compare_table.tex`、`model_compare_waic.pdf`。
  - **LOO**：從 `results/paper_extra/loo_compare/loo_table.csv` 讀取，生成 `loo_compare_table.tex` 與 `model_compare_loo.pdf`（或從既有 elpd_difference_plot.png 轉 PDF）。
  - **pWAIC**：從 `results/paper_extra/model_compare/pwaic_diagnostics/pwaic_by_galaxy.csv` 讀取，生成 `pwaic_top_galaxies_table.tex` 與 `pwaic_by_galaxy_bar.pdf`。
  - **summary**：`_build_summary_markdown` 依上述 CSV 是否存在撰寫 `paper_artifacts/paper_results_summary.md`（WAIC/LOO/pWAIC 小結與 key conclusions）。
- **注意**：目前 `compare_models_waic` 並未寫入 `pwaic_diagnostics/` 目錄；README 與 make_paper_artifacts 已預期該目錄存在，故「B. 補 WAIC decomposition/localization diagnostics」需在 compare 流程中（或獨立子步驟）產出該目錄。

---

## 二、分步實作計畫（僅三件事）

### A. 新增 matched-kernel MOND control：mond-screen

**目標**：新增模型 `mond-screen`，與 PTQ-screen 使用**相同** interpolating function ν_q(y)=0.5+√(0.25+y^{-q})，但 a0 為**自由參數**（不綁 ε c H0），作為 MOND 的 matched-kernel control，方便與 ptq-screen 在相同 kernel 下比較 WAIC/LOO。

| 項目 | 內容 |
|------|------|
| **預計修改檔案** | `src/ptquat/models.py`（新增 `model_v_mond_screen`）；`src/ptquat/fit_global.py`（`_layout`、`_unpack`、`log_prior`、`log_likelihood`、`log_likelihood_per_galaxy`、`log_likelihood_full_per_galaxy`，以及 run() 內 init p0、summary 迴圈、summary 欄位）；`src/ptquat/cli.py`（fit / stress / mask / exp 等 `--model` 的 choices 加入 `mond-screen`）；`src/ptquat/experiments.py`（`_make_vfun`、`ppc_check`、`residual_plateau`、`compare_models_waic` 的 model 迴圈與任何依 model 分派之處，例如 get_vmod_vbar2、scan_H0 等若有用到 model 名）。 |
| **新增 CLI** | 無新子命令；僅在既有 `--model` 的 `choices` 中加入 `mond-screen`。 |
| **預期輸出** | 與現有 fit 完全相同：`<outdir>/global_summary.yaml`、`per_galaxy_summary.csv`、`plot_*.png`，以及可選 `chain.h5`。`global_summary.yaml` 需包含 `a0_median`、`q_median`（與 mond / ptq-screen 一致）。compare/loo 若在 `--models` 中加入 `mond-screen`，會多一列結果。 |
| **測試策略** | （1）在 `tests/test_models.py` 中對 `model_v_mond_screen` 做形狀與正性測試（若匯出該函數）；（2）在現有或新 test 中跑一次 `ptquat fit --model mond-screen --data <tiny_csv> --outdir <tmp> --steps 20 --nwalkers 2x`，檢查 `global_summary.yaml` 存在且含 `model: mond-screen`、`a0_median`、`q_median`，且 `per_galaxy_summary.csv` 欄位與他模型一致；（3）不改動既有 test_models 的既有測試，僅新增。 |

**參數版型（mond-screen）**：與 ptq-screen 類似，但用 a0 取代 ε。建議：`i_log10a0`, `i_lq`, `sl_U`，即 1 + 1 + G + 1(lnsig) = G+3。prior：a0 用現有 mond 的 a0_range（log-uniform）；q 用 ptq-screen 的 lq 正態 prior。

---

### B. 補 WAIC decomposition / localization diagnostics

**目標**：在執行 `ptquat exp compare` 時（或作為 compare 的一部分），產出 pWAIC 的**分解與定位**診斷：每資料點之 Var_s(log p_si)、每星系之 pWAIC 貢獻、以及依半徑/星系的圖表，方便回答「哪些 galaxy/半徑對 pWAIC 貢獻最大」。

| 項目 | 內容 |
|------|------|
| **預計修改檔案** | `src/ptquat/experiments.py`（在 `compare_models_waic` 內，在已有 `log_lik_mat` 的迴圈後，對**至少一個**模型［例如 WAIC 最佳或預設第一個］計算 per-point var(log_lik)，並寫入 `pwaic_diagnostics/`）；可選 `scripts/make_paper_artifacts.py`（若診斷檔路徑或欄位有擴充則適度調整讀取，否則無需改動）。 |
| **新增 CLI** | 可選：在 `ptquat exp compare` 增加 `--pwaic-diagnostics [model]` 或 `--no-pwaic-diagnostics`，預設為 True 且對「第一個模型」或「best WAIC model」寫診斷；若希望省時可設為 False。不新增子命令。 |
| **預期輸出** | 在 `outdir`（即 compare 的 `--outdir`）下新增子目錄 `pwaic_diagnostics/`，內含：<br>• `var_loglik_points.csv`：欄位 `galaxy`, `radius`, `var_loglik`, `v_obs`, `v_model_mean`（或等價）；<br>• `pwaic_by_galaxy.csv`：`galaxy`, `pwaic_contribution`, `n_points`, `mean_var_loglik`；<br>• `top20_galaxies_pwaic.csv`：同上，排序取 top 20；<br>• `pwaic_by_galaxy_bar.png`、`radius_vs_varloglik.png`（可選 .pdf）。<br>既有的 `compare_table.csv` / `breakdown.csv` 等**不**改欄位名，僅新增上述檔案與目錄。 |
| **測試策略** | （1）單元：對一小型 chain 與 log_lik_mat 呼叫撰寫好的「計算 var_loglik 並彙總 by galaxy」的 helper，檢查輸出 DataFrame 形狀與合計與總 p_waic 一致；（2）整合：跑一次 `ptquat exp compare --models ptq-screen mond --data <small> --fit-root <tmp> --outdir <out> --run-fits --fit-steps 30`，檢查 `out/pwaic_diagnostics/` 存在且含上述 CSV 與圖檔；（3）`test_make_paper_artifacts` 已預期 `pwaic_by_galaxy.csv` 存在時可生成 pwaic 表/圖，只要 compare 產出該目錄即可，必要時在 test 中先跑一次 compare 再跑 make_paper_artifacts。 |

**實作要點**：在 `compare_models_waic` 中，對選定模型在已有 `log_lik_mat` (S×N) 上計算 `var_loglik_i = np.nanvar(log_lik_mat, axis=0)`，再依現有 `galaxies` 順序與 `g.r_kpc` 拆成 galaxy/radius 對應，寫出 CSV；`pwaic_contribution_g = sum(var_loglik_i for i in g)`；重用現有 `_waic_from_chain` 的 p_waic 概念，不重寫 WAIC 公式。

---

### C. 補 closure sensitivity scan

**目標**：在單一 ε_cos（或 ΩΛ）的 closure test 之外，新增「對多個 ε_cos（或 ΩΛ）掃描」的實驗，產出 closure 指標隨 ε_cos 的變化，供論文討論 closure 的敏感度。

| 項目 | 內容 |
|------|------|
| **預計修改檔案** | `src/ptquat/experiments.py`（新增函數 `closure_sensitivity_scan(results_dir, epsilon_cos_list=None, omega_lambda_list=None, ...)`：對每個 ε_cos 呼叫既有 `closure_test` 的邏輯［或抽出共用的「單點 closure 計算」helper］，蒐集 epsilon_RC, epsilon_cos, sigma_RC, diff, pass_within_3sigma 等，寫成 CSV 與可選圖）；`src/ptquat/cli.py`（在 `exp` 下新增子命令 `closure-scan`，參數：`--results`、`--epsilon-cos-list` 或 `--omega-lambda-list`／`--omega-lambda-range`＋步數、`--out`、可選 `--plot`）。 |
| **新增 CLI** | `ptquat exp closure-scan --results <dir> --omega-lambda-range 0.65,0.75 --n 11 --out results/closure_scan/closure_scan.csv`（或 `--epsilon-cos-list 1.2 1.4 1.5 1.6 1.8`）。輸出路徑由 `--out` 指定，不改變既有 `closure` 的 `closure_test.yaml` 路徑。 |
| **預期輸出** | 單一 CSV（例如 `closure_scan.csv`）：欄位 `epsilon_cos`, `omega_lambda`, `epsilon_RC`, `sigma_RC`, `diff`, `pass_within_3sigma`（epsilon_RC 來自同一個 results_dir，故為常數；sigma_RC 亦然）；可選一張圖 `closure_scan.png`（x=epsilon_cos 或 ΩΛ，y=diff，標註 3σ 帶）。不覆寫既有 `results_dir/closure_test.yaml`（單點 closure 仍由 `exp closure` 寫入）。 |
| **測試策略** | （1）單元：mock 一個 `global_summary.yaml`（含 epsilon_median, epsilon_p16, epsilon_p84），對 3 個 epsilon_cos 呼叫 closure 計算，檢查輸出列數與數值合理；（2）整合：用既有 PTQ 結果目錄跑 `ptquat exp closure-scan --results <path> --epsilon-cos-list 1.4 1.47 1.5 --out <tmp>/scan.csv`，檢查 CSV 存在且欄位正確，且 epsilon_RC 三列相同。 |

**實作要點**：重用 `closure_test` 內「讀 summary、算 diff、pass_within_3sigma」的邏輯，可抽成 `_closure_one(results_dir, epsilon_cos)` 回傳 dict，再由 `closure_sensitivity_scan` 迴圈呼叫並寫 CSV；避免重複讀檔與重寫公式。

---

## 三、準備開始修改的檔案清單

依賴順序與最小侵入原則，建議修改順序與清單如下：

1. **A. mond-screen**
   - `src/ptquat/models.py` — 新增 `model_v_mond_screen`。
   - `src/ptquat/fit_global.py` — `_layout` / `_unpack` / `log_prior` / `log_likelihood` / `log_likelihood_per_galaxy` / `log_likelihood_full_per_galaxy` 及 run() 內 p0、summary 迴圈、summary 寫出。
   - `src/ptquat/cli.py` — 所有 `--model` 的 `choices` 加入 `mond-screen`。
   - `src/ptquat/experiments.py` — `_make_vfun`、`ppc_check`、`residual_plateau` 的 get_vmod_vbar2、以及 `compare_models_waic` / `loo_compare_models` 可接受 mond-screen（預設 `--models` 可選是否含 mond-screen）；其他依 model 字串分派之處（如 scan_H0、kappa 等）若需支援 mond-screen 則一併加入。
   - `tests/test_models.py` — 可選：對 `model_v_mond_screen` 的形狀/正性測試；或新增小型 fit 整合 test。

2. **B. WAIC diagnostics**
   - `src/ptquat/experiments.py` — 在 `compare_models_waic` 中，在寫完 `compare_table.csv` 等之後，對一個模型（如第一個或 best）計算 per-point var(log_lik)，組裝 galaxy/radius 對應，寫入 `out_path / "pwaic_diagnostics"` 下之 CSV 與圖檔；可選 `--pwaic-diagnostics` 開關。
   - `scripts/make_paper_artifacts.py` — 僅在 pwaic 診斷欄位或路徑有變更時調整（目前預期為 `pwaic_diagnostics/pwaic_by_galaxy.csv` 等，應可沿用）。

3. **C. closure sensitivity scan**
   - `src/ptquat/experiments.py` — 抽出 `_closure_one(results_dir, epsilon_cos, omega_lambda)`（或僅用 epsilon_cos），新增 `closure_sensitivity_scan(...)`，寫 CSV 與可選圖。
   - `src/ptquat/cli.py` — `exp` 下新增 `closure-scan` 子命令，綁定 `closure_sensitivity_scan`。

**不修改**：既有 CSV/YAML 的既有欄位名稱與路徑慣例；既有 `exp closure` 的介面與輸出；`README.md` 可於三項都完成後一併更新章節（A/B/C 的說明與範例指令）。

---

請確認以上計畫與檔案清單，確認後再開始改 code。
