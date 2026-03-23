# Baseline six-model summary (Layer A)

- This baseline uses a single long chain per model (steps = 12000, nwalkers = 192, seed = 0) on the same SPARC dataset.
- It provides the main six-model landscape for the paper; multi-seed robustness is handled separately in Layer B.

## AIC/BIC ordering

- `mond`: AIC_full ≈ 13768.7, BIC_full ≈ 14278.9, k ≈ 93
- `ptq-nu`: AIC_full ≈ 13784.8, BIC_full ≈ 14294.9, k ≈ 93
- `ptq-screen`: AIC_full ≈ 13779.8, BIC_full ≈ 14295.4, k ≈ 94
- `ptq`: AIC_full ≈ 14842.8, BIC_full ≈ 15353.0, k ≈ 93
- `nfw1p`: AIC_full ≈ 14979.4, BIC_full ≈ 15983.2, k ≈ 183
- `baryon`: AIC_full ≈ 19155.9, BIC_full ≈ 19660.6, k ≈ 92

## WAIC / LOO ordering

- `ptq-nu`: WAIC ≈ 15156.6 (ΔWAIC ≈ 0.0, rank=1), LOOIC ≈ 15172.2 (ΔLOOIC ≈ 0.0, rank=1)
- `ptq-screen`: WAIC ≈ 15194.9 (ΔWAIC ≈ 38.3, rank=2), LOOIC ≈ 15213.7 (ΔLOOIC ≈ 41.5, rank=2)
- `mond`: WAIC ≈ 15207.9 (ΔWAIC ≈ 51.3, rank=3), LOOIC ≈ 15220.5 (ΔLOOIC ≈ 48.3, rank=3)
- `ptq`: WAIC ≈ 16045.3 (ΔWAIC ≈ 888.7, rank=4), LOOIC ≈ 16055.5 (ΔLOOIC ≈ 883.3, rank=4)
- `baryon`: WAIC ≈ 19615.4 (ΔWAIC ≈ 4458.7, rank=5), LOOIC ≈ 19617.2 (ΔLOOIC ≈ 4445.0, rank=5)
- `nfw1p`: WAIC ≈ 35982.9 (ΔWAIC ≈ 20826.3, rank=6), LOOIC ≈ 31279.0 (ΔLOOIC ≈ 16106.9, rank=6)

## Interpretation: key models and negative controls

- Negative controls present in this baseline: `baryon` (baryon-only), `ptq` (linear PTQ).
- Among key models (ptq-screen, mond, ptq-nu, nfw1p), the WAIC ordering is: ptq-screen (WAIC ≈ 15194.9, ΔWAIC ≈ 38.3, rank=2); mond (WAIC ≈ 15207.9, ΔWAIC ≈ 51.3, rank=3); ptq-nu (WAIC ≈ 15156.6, ΔWAIC ≈ 0.0, rank=1); nfw1p (WAIC ≈ 35982.9, ΔWAIC ≈ 20826.3, rank=6).
