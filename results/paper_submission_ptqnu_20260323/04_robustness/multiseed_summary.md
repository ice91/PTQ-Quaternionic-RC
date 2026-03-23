# Multi-seed robustness summary (Layer B)

- This robustness layer uses three models (`ptq-screen`, `mond`, `mond-screen`) with seeds {0, 7, 13}, each run with steps = 12000 and nwalkers = 192.
- Its role is to audit seed sensitivity and ranking stability, not to replace the six-model baseline landscape (Layer A).

## Rank stability (WAIC / LOO)

- `ptq-screen`: WAIC rank mean ≈ 3.6666666666666665, std ≈ 0.4714045207910317; LOO rank mean ≈ 3.6666666666666665, std ≈ 0.4714045207910317 (n_seeds = 3).
- `mond`: WAIC rank mean ≈ 2.3333333333333335, std ≈ 1.247219128924647; LOO rank mean ≈ 2.3333333333333335, std ≈ 1.247219128924647 (n_seeds = 3).
- `mond-screen`: WAIC rank mean ≈ 2.6666666666666665, std ≈ 0.4714045207910317; LOO rank mean ≈ 2.6666666666666665, std ≈ 0.4714045207910317 (n_seeds = 3).

## Seed sensitivity for key models

- `ptq-screen` WAIC across seeds: min ≈ 15175.2, max ≈ 15323.3, spread ≈ 148.1.
  LOOIC across seeds: min ≈ 15189.3, max ≈ 15352.6, spread ≈ 163.3.
- `mond` WAIC across seeds: min ≈ 15081.3, max ≈ 15195.0, spread ≈ 113.7.
  LOOIC across seeds: min ≈ 15105.5, max ≈ 15220.3, spread ≈ 114.8.
- `mond-screen` WAIC across seeds: min ≈ 15153.1, max ≈ 15304.7, spread ≈ 151.6.
  LOOIC across seeds: min ≈ 15171.0, max ≈ 15341.8, spread ≈ 170.8.

## Relationship to baseline (Layer A)

- The six-model baseline (Layer A) provides a single long-chain comparison at seed 0; this multi-seed audit checks whether that snapshot is representative across seeds.
- Results here should be interpreted as evidence for ranking stability or variability under different seeds, not as new single 'best' runs.
