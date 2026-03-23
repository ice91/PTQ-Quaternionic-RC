## WAIC comparison

- **最佳 WAIC 模型**: `mond` (WAIC ≈ 15201.9)
- `mond`: WAIC ≈ 15201.9, ΔWAIC ≈ 0.0, pWAIC ≈ 275.5
- `mond-screen`: WAIC ≈ 15261.2, ΔWAIC ≈ 59.3, pWAIC ≈ 300.7
- `ptq-screen`: WAIC ≈ 15279.1, ΔWAIC ≈ 77.2, pWAIC ≈ 319.7

## LOO comparison

- **最佳 LOO 模型**: `mond` (LOOIC ≈ 15215.1)
- `mond`: elpd_loo ≈ -7607.5, LOOIC ≈ 15215.1, ΔLOOIC ≈ 0.0
- `mond-screen`: elpd_loo ≈ -7640.2, LOOIC ≈ 15280.3, ΔLOOIC ≈ 65.3
- `ptq-screen`: elpd_loo ≈ -7651.4, LOOIC ≈ 15302.8, ΔLOOIC ≈ 87.8

## pWAIC diagnostics

- 最高 pWAIC 貢獻的前幾個星系：
  - `NGC2841`: pWAIC ≈ 28.9, N ≈ 50, mean Var(loglik) ≈ 0.578
  - `IC4202`: pWAIC ≈ 28.7, N ≈ 32, mean Var(loglik) ≈ 0.896
  - `NGC6195`: pWAIC ≈ 27.7, N ≈ 23, mean Var(loglik) ≈ 1.204
  - `ESO563-G021`: pWAIC ≈ 26.0, N ≈ 30, mean Var(loglik) ≈ 0.868
  - `NGC0801`: pWAIC ≈ 16.7, N ≈ 13, mean Var(loglik) ≈ 1.285

## Key conclusions

- **WAIC** 最佳模型：`mond`
- **LOO** 最佳模型：`mond`
- WAIC 與 LOO 一致選出同一個最佳模型：`mond`。
- 在 **WAIC** 排名中，關注模型的相對順序為：mond (rank 1, WAIC ≈ 15201.9); ptq-screen (rank 3, WAIC ≈ 15279.1); mond-screen (rank 2, WAIC ≈ 15261.2)。
- 在 **LOO** 排名中，關注模型的相對順序為：mond (rank 1, LOOIC ≈ 15215.1); ptq-screen (rank 3, LOOIC ≈ 15302.8); mond-screen (rank 2, LOOIC ≈ 15280.3)。
- pWAIC 診斷可用；目前顯示 `NGC2841` 對複雜度懲罰的貢獻最高 (pWAIC ≈ 28.9)。
