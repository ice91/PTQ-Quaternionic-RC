# R1B True-MLE Report

**Task ID:** `SPARC-FRONTIERS-SCIHARD-R1B-TRUE-MLE-AND-REPARAMETERIZATION-INVARIANCE`  
**Verdict:** `PASS_WITH_WARN`  
**Date:** 2026-08-07

---

## 1. Authority

| Item | Value |
|------|-------|
| PROJECT_ROOT | `/Users/rocky/ptq-mond` |
| IMPL_ROOT | `/Users/rocky/ptq-mond/PTQ-Quaternionic-RC` |
| Implementation SHA | `9f80b41a284bbbc2f11f147dd0e0580c39c9d917` (matches R1A) |
| Historical baseline | `f561db54363f6d0e5f3c064d3a3876babcef4bfc` |
| `ptquat.__file__` | `$IMPL_ROOT/src/ptquat/__init__.py` |
| `ref/` | **PRESENT** (R0/R1A “MISSING” warning corrected) |
| `main2.tex` SHA256 | `b93bb80ecc0b73c6c6264ed279782fbe0da4c3d4fe738c1e473a49182061473a` (unchanged) |

---

## 2. Method

- Objective: **maximize full-covariance Gaussian `log L` only** (no priors).
- Covariance: production `build_covariance` (meas + D + i + `σ_sys`), full log-det.
- Optimizer: **block-coordinate ascent** on `log L` (globals via L-BFGS-B; per-galaxy `Υ` via 1D bounded search; NFW joint `(Υ, log M200)`).
- Multi-start: historical posterior median + perturbations + mid-scale + mapped paired starts.
- Matched acceleration domain for PTQ-ν/MOND and PTQ-screen/MOND-screen:

```text
a0 ∈ [5e-11, 2e-10] m s^-2
ε = a0/(c H0) ∈ [0.07636, 0.30542]
```

- Linear PTQ uses wide historical `ε ∈ (0,4)` (different physical scale; not the ν-family match).
- BIC convention: `BIC_NPOINTS = -2 max_logL + k ln(1782)` — see `r1b/bic_convention_note.md`.

Outputs: `$IMPL_ROOT/results/scientific_hardening_r1/01_mle/`

---

## 3. Sample / parameter counts

| Quantity | Value |
|----------|------:|
| N_gal | 91 |
| N_points | 1782 |
| baryon k | 92 |
| mond / ptq / ptq-nu k | 93 |
| ptq-screen / mond-screen k | 94 |
| nfw1p k | 183 |

All include learned `σ_sys`.

---

## 4. Core equivalence

| Pair | abs(Δ max_logL) | Gate |
|------|----------------:|------|
| PTQ-ν ↔ MOND | `4.68e-8` | **PASS** (`< 1e-5`) |
| PTQ-screen ↔ MOND-screen (`ν_q`) | `4.55e-12` after shared-family mapped refinement | **PASS** |

Independent per-model screen starts initially differed by ~0.015 in logL (optimizer pathology). Shared-family refinement (map best → both parameterizations) restores exact identity. Deterministic `a0=ε c H0` map verified.

---

## 5. True-MLE table (headline)

| model | max_logL | AIC | BIC_NPOINTS | σ_sys |
|-------|---------:|----:|------------:|------:|
| nfw1p | -6508.170 | 13382.339 | 14386.184 | 6.52 |
| mond | -6663.422 | 13512.844 | 14022.995 | 6.89 |
| ptq-nu | -6663.422 | 13512.844 | 14022.995 | 6.89 |
| mond-screen | -6663.002 | 13514.004 | 14029.640 | 6.88 |
| ptq-screen | -6663.002 | 13514.004 | 14029.640 | 6.88 |
| ptq | -7138.535 | 14463.071 | 14973.221 | 10.31 |
| baryon | -8826.777 | 17837.554 | 18342.219 | 32.63 |

Scale optima interior: MOND `a0≈1.020e-10`; PTQ-ν `ε≈0.1558`; screen `q≈0.984`.

---

## 6. Historical posterior-median vs true MLE

| model | Δ logL (MLE − hist median) | Δ AIC |
|-------|---------------------------:|------:|
| baryon | +659.2 | −1318.4 |
| mond | +127.9 | −255.9 |
| ptq | +189.9 | −379.8 |
| ptq-nu | +136.0 | −271.9 |
| ptq-screen | +132.9* | −265.8* |
| mond-screen | +143.1* | −286.1* |
| nfw1p | +798.5 | −1597.0 |

\*screen values after shared-family refine.  
**Conclusion:** historical AIC/BIC at posterior medians are **not** true MLE criteria (FATAL-GATE-2 confirmed quantitatively).

---

## 7. Warnings (PASS_WITH_WARN)

1. **NFW-1p** cross-start `Δ logL ≈ 29` — optimizer multi-start disagreement; NFW MLE not cross-start stable. Does **not** affect PTQ-ν/MOND identity.
2. Many per-galaxy `Υ` values sit on hard bounds `[0.05,1.5]` (counted in `n_near_boundary`). Global scale parameters are interior.
3. Screen pair required an explicit **shared-family refinement** step after independent multi-starts.

---

## 8. Integrity

- Historical artifact hashes: **0 changed**
- `main2.tex` hash: **unchanged**
- No WAIC/LOO (R1C not started)
- No `main2.tex` edit

## 9. Tests

`47 passed` (identity, inventory, true-MLE, immutability; `test_make_paper_artifacts` ignored for historical safety).

## 10. Remaining FATAL (unchanged by R1B)

| Gate | Status |
|------|--------|
| FATAL-GATE-1 PTQ-screen ms≠code | OPEN — path chosen (rewrite ms at R1K) |
| FATAL-GATE-2 median AIC | **evidence frozen**; manuscript still claims median until R1K |
| FATAL-GATE-3 diag WAIC | OPEN — R1C |
| FATAL-GATE-4 σ_sys inventory in ms | OPEN — R1K |
| FATAL-GATE-5 identity | **CLOSED** (det. + MLE) |

## 11. Next

**STOP.** Await human review. Do not start R1C.
