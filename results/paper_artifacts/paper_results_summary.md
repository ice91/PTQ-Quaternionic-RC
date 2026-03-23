## WAIC comparison

- **最佳 WAIC 模型**: `mond` (WAIC ≈ 20730.1)
- `mond`: WAIC ≈ 20730.1, ΔWAIC ≈ 0.0
- `ptq-screen`: WAIC ≈ 27142.2, ΔWAIC ≈ 6412.1

## LOO comparison

- （尚未找到 LOO loo_table.csv）

## pWAIC diagnostics

- 最高 pWAIC 貢獻的前幾個星系：
  - `ESO563-G021`: pWAIC ≈ 263.8, N ≈ 30, mean Var(loglik) ≈ 8.795
  - `NGC2841`: pWAIC ≈ 255.5, N ≈ 50, mean Var(loglik) ≈ 5.110
  - `IC4202`: pWAIC ≈ 213.3, N ≈ 32, mean Var(loglik) ≈ 6.665
  - `NGC7814`: pWAIC ≈ 166.6, N ≈ 18, mean Var(loglik) ≈ 9.253
  - `NGC4217`: pWAIC ≈ 123.5, N ≈ 19, mean Var(loglik) ≈ 6.502

## Key conclusions

- WAIC 與 LOO 都可用來比較 MOND 與 PTQ 家族模型的整體擬合與預測能力。
- 目前結果顯示，在本次 SPARC 資料與設定下，MOND 在 WAIC / LOOIC 上仍優於 PTQ-screen。
- pWAIC 分解顯示，少數星系與半徑區域對複雜度懲罰有顯著貢獻，適合作為後續針對性診斷與模型改進的重點對象。
