# MLE equivalence summary (R1B)

## Matched domain

```text
H0 = 67.4 km/s/Mpc  (H0_SI from ptquat.constants)
a0 = ε * c * H0
matched a0 ∈ [5e-11, 2e-10] m s^-2
matched ε ∈ [0.07635543401657195, 0.3054217360662878]
```

Historical MOND/PTQ-ν optima lie in the interior of this window.

## PTQ-ν ↔ MOND

| Quantity | Value |
|----------|------:|
| max_logL (both) | ≈ −6663.422141 |
| abs(Δ logL) | 4.68×10⁻⁸ |
| abs(Δ AIC) | 9.35×10⁻⁸ |
| pass `<1e-5` | **YES** |

## PTQ-screen ↔ MOND-screen (`ν_q`)

Deterministic identity under `a0=ε c H0` confirmed in R1A tests.

Independent multi-start MLEs initially differed by ~0.015 in logL (finite optimizer disagreement on a flat/shared ridge in `(a0,q)` / `(ε,q)`).

After **shared-family mapped refinement**:

| Quantity | Value |
|----------|------:|
| shared max_logL | −6663.002102974754 |
| abs(Δ logL) mapped | 4.55×10⁻¹² |
| pass `<1e-5` | **YES** |

## Interpretation

Likelihood-family identity is established. Residual ranking differences under unmatched priors remain a **prior-measure** question for R1D — not physical radial-law distinction (CF-1, CF-2).
