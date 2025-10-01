# PTQ-Quaternionic-RC — Global Fits & Baseline Comparisons on SPARC

This repository provides a **reproducible pipeline** for
**data → preprocessing → global Bayesian fitting → model comparison**
on the **SPARC** galaxy rotation-curve sample (Lelli+ 2016, VizieR: `J/AJ/152/157`).

The primary model tests a PT-Quaternionic (PTQ) linear term added to baryons:
[
v^2(r) ;=; v_{\rm bar}^2(r);+;(\epsilon,c,H_0),r,
\qquad v_{\rm bar}^2 ;=; \Upsilon,(v_d^2+v_b^2);+;v_g^2.
]
Here (\Upsilon) is a per-galaxy stellar mass-to-light ratio (optionally split
into (\Upsilon_{\rm disk}) and (\Upsilon_{\rm bulge})),
and (\epsilon) is a **global** (shared) parameter.

Built-in comparators:

* **PTQ** (main): global (\epsilon) + per-galaxy (\Upsilon)
* **PTQ-split**: global (\epsilon) + per-galaxy (\Upsilon_{\rm disk}, \Upsilon_{\rm bulge})
* **Baryon-only** (baseline): (\epsilon=0)
* **MOND (simple (\nu))**: global (a_0) (fixed or sampled)
* **NFW-1p**: per-galaxy (M_{200}); (c) from a (c)–(M) relation (c=c_0(M/10^{12}M_\odot)^{\beta})

The likelihood **propagates distance and inclination uncertainties** inside the covariance
and **learns a global velocity floor (\sigma_{\rm sys})** (parameterized as (\ln\sigma) with a weak half-normal prior).
Outputs include full-likelihood (\log\mathcal{L}), AIC/BIC, per-galaxy diagnostics, and plots.

---

## Contents

* [Installation](#installation)
* [Data & Preprocessing](#data--preprocessing)
* [Models & CLI quick reference](#models--cli-quick-reference)
* [Main workflow](#main-workflow)
* [Run in background with `nohup`](#run-in-background-with-nohup)
* [Outputs](#outputs)
* [Interpreting results & model comparison](#interpreting-results--model-comparison)
* [Hardware guidance](#hardware-guidance)
* [Tests](#tests)
* [Troubleshooting](#troubleshooting)
* [Citations](#citations)
* [License](#license)
* [Math appendix](#math-appendix)

---

## Installation

**Requirements**

* Python ≥ 3.10 (3.11/3.12 recommended)
* Linux/macOS
* See `requirements.txt` (`emcee`, `h5py`, `astroquery`, `pandas`, `numpy`, `matplotlib`, `pyyaml`, …)

**Setup**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

This installs the CLI entry point `ptquat`.

---

## Data & Preprocessing

1. **Fetch SPARC** from VizieR (`J/AJ/152/157`)

```bash
ptquat fetch --out dataset/raw
```

Notes:

* We request **all columns** from VizieR and coerce numeric fields (e.g., `Vbulge`) to avoid the “all zeros / 6 names only” pitfall.

2. **Preprocess & quality cuts**

```bash
ptquat preprocess \
  --raw dataset/raw \
  --out dataset/sparc_tidy.csv \
  --i-min 30 --reldmax 0.2 --qual-max 2
```

Quality cuts (configurable):

* (i > 30^\circ), relative distance error (< 0.2), quality flag (Qual \le 2).
* If (e_i) or (e_D) is missing, conservative defaults are injected (see `preprocess.py`).

---

## Models & CLI quick reference

```bash
ptquat fit --help
```

Key options:

* `--model {ptq, ptq-split, baryon, mond, nfw1p}`
* `--prior {galaxies-only, planck-anchored}` (affects PTQ/PTQ-split prior on (\epsilon))
* `--sigma-sys 4` → **initial scale** for the global floor; the code **always learns** (\sigma_{\rm sys}) (no flag needed)
* `--H0-kms-mpc` to override (H_0) (default from Planck 2018 in code)
* MOND: `--a0-si` to fix (a_0) or `--a0-range lo,hi` to sample (a_0)
* NFW-1p: `--logM200-range 9,13`, `--c0 10 --c-slope -0.1`
* Sampler & backend: `--nwalkers 4x`, `--steps 12000`, `--backend-hdf5`, `--resume`, `--thin-by 10`

---

## Main workflow

### 1) Primary PTQ run

```bash
ptquat fit \
  --data dataset/sparc_tidy.csv \
  --outdir results/ptq_main \
  --model ptq \
  --prior galaxies-only \
  --sigma-sys 4 \
  --nwalkers 4x \
  --steps 12000 \
  --thin-by 10 \
  --backend-hdf5 results/ptq_main/chain.h5 \
  --resume
```

### 2) Baselines / Comparators

```bash
# Baryon-only (ε=0)
ptquat fit --data dataset/sparc_tidy.csv --outdir results/baryon \
  --model baryon --sigma-sys 4 --nwalkers 3x --steps 12000 \
  --thin-by 10 --backend-hdf5 results/baryon/chain.h5 --resume

# MOND (sample a0; or use --a0-si to fix)
ptquat fit --data dataset/sparc_tidy.csv --outdir results/mond \
  --model mond --a0-range 5e-11,2e-10 --sigma-sys 4 \
  --nwalkers 3x --steps 12000 \
  --thin-by 10 --backend-hdf5 results/mond/chain.h5 --resume

# NFW-1p (per-galaxy M200; c from c–M relation)
ptquat fit --data dataset/sparc_tidy.csv --outdir results/nfw1p \
  --model nfw1p --logM200-range 9,13 --c0 10 --c-slope -0.1 \
  --sigma-sys 4 --nwalkers 3x --steps 12000 \
  --thin-by 10 --backend-hdf5 results/nfw1p/chain.h5 --resume
```

---

## Run in background with `nohup`

```bash
mkdir -p logs

# PTQ
LOG="logs/ptq_$(date +%F_%H%M%S).log"
nohup bash -lc 'source .venv/bin/activate; ptquat fit \
  --data dataset/sparc_tidy.csv --outdir results/ptq_main \
  --model ptq --prior galaxies-only --sigma-sys 4 \
  --nwalkers 4x --steps 12000 --thin-by 10 \
  --backend-hdf5 results/ptq_main/chain.h5 --resume' \
> "$LOG" 2>&1 & echo "PID=$!  log=$LOG"

# Baryon-only (edit outdir/model)
# MOND (edit outdir/model/a0 options)
# NFW-1p (edit outdir/model/c–M options)
```

Monitor:

```bash
tail -f logs/ptq_*.log
```

---

## Outputs

Each `outdir` contains:

* `global_summary.yaml`
  Key fields:

  * `model`, `n_galaxies`, `N_total`, `prior`, `H0_si`
  * `sigma_sys_median/p16/p84` (always learned)
  * `epsilon_*` (PTQ/PTQ-split) or `a0_*` (MOND, if sampled)
  * `chi2_total`, `logL_total_full`
  * `AIC_quad`, `BIC_quad`, `AIC_full`, `BIC_full` (**prefer full-likelihood for comparison**)
  * `k_parameters`, `steps`, `nwalkers`, `burn_in`, `thin`

* `per_galaxy_summary.csv`
  Per galaxy: (n), (\Upsilon) (or (\Upsilon_{\rm disk/bulge})), (\chi^2), (\chi^2/\nu), (\log\mathcal{L})

* `plot_*.png`
  Rotation curves with data and best model

* `chain.h5` (if `--backend-hdf5`)
  plus a thinned snapshot in `posterior_samples.npz` (for some modes)

---

## Interpreting results & model comparison

Because the covariance includes distance/inclination terms,
**use full-likelihood** criteria for model selection:

[
\mathrm{AIC}*{\rm full} = -2\log\mathcal{L}*{\rm full} + 2k,\quad
\mathrm{BIC}*{\rm full} = -2\log\mathcal{L}*{\rm full} + k\ln N.
]

As a rule of thumb, (\Delta)AIC/BIC (\gtrsim 10) indicates strong preference.

Simple Python helper:

```python
import yaml
def s(p): return yaml.safe_load(open(p))
ptq  = s("results/ptq_main/global_summary.yaml")
bar  = s("results/baryon/global_summary.yaml")
mond = s("results/mond/global_summary.yaml")
nfw  = s("results/nfw1p/global_summary.yaml")

def d(base, other, key="AIC_full"):
    return other[key] - base[key]

print("ΔAIC_full (PTQ vs Baryon):", d(ptq, bar))
print("ΔAIC_full (PTQ vs MOND):  ", d(ptq, mond))
print("ΔAIC_full (PTQ vs NFW1p): ", d(ptq, nfw))
```

Also review `per_galaxy_summary.csv` to identify outliers by reduced (\chi^2).

---

## Hardware guidance

* **CPU**: 8–16 physical cores recommended (the log-likelihood itself is single-process in this codebase; HDF5 helps with I/O).
* **Memory**: ≥ 16 GB is comfortable for ~90 galaxies with `nwalkers ≈ 3–4 × (param_dim)`, `steps ≈ 12k`, `thin ≈ 10`, and HDF5 backend.
* **Disk**: Each model’s `chain.h5` is typically hundreds of MB; several models total a few GB.

If resources are tight: use `--thin-by` (bigger number), reduce `--nwalkers` (e.g. `3x`), and keep the HDF5 backend enabled.

---

## Tests

```bash
pytest -q
```

* `tests/test_models.py` — model outputs sanity checks
* `tests/test_covariance.py` — covariance/loglike stability (Cholesky with jitter)

---

## Troubleshooting

* **`ptquat: error: unrecognized arguments: --sigma-sys-learn`**
  Not needed. This code *always* learns a global (\sigma_{\rm sys}) via (\ln \sigma) with a weak prior.
  Use `--sigma-sys` only as the *initial scale* and prior scale.

* **VizieR returns `Vbulge` all zeros or only a few names**
  The fetcher requests **all columns** and coerces numerics. Re-run:

  ```bash
  ptquat fetch --out dataset/raw
  ptquat preprocess --raw dataset/raw --out dataset/sparc_tidy.csv --i-min 30 --reldmax 0.2 --qual-max 2
  ```

* **Acceptance fractions from HDF5**
  `emcee.backends.HDFBackend` doesn’t expose `get_acceptance_fraction()`. During a live run, inspect `sampler.acceptance_fraction` in Python. (Persisting it is not implemented here.)

* **Pandas `FutureWarning: use_inf_as_na`**
  Harmless; the code already uses explicit replacements in critical paths.

---

## Citations

* **SPARC**
  Lelli, F., McGaugh, S. S., & Schombert, J. M. (2016), *AJ, 152, 157*.
  VizieR catalog: `J/AJ/152/157`. Please cite accordingly.

* **emcee**
  Foreman-Mackey, D., et al. (2013), *PASP, 125, 306*.

If this code underpins your analysis, consider describing the methodology along these lines:
“We used the PTQ-Quaternionic-RC pipeline (this repository, commit …) to perform global posterior inference on a SPARC subsample
((i>30^\circ), (\delta D/D<0.2), (Qual\le 2)); the likelihood propagates distance and inclination uncertainties and learns a global
velocity floor (\sigma_{\rm sys}) (parameterized by (\ln\sigma) with a weak half-normal prior). Model comparison is based on full-likelihood AIC/BIC.”

---

## License

* **Code**: see the repository `LICENSE` (MIT/Apache-2.0 recommended if not yet added).
* **Data**: SPARC per original authors and VizieR terms.

---

## Math appendix

* **PTQ**
  [
  v^2(r)=\Upsilon,(v_d^2+v_b^2)+v_g^2+(\epsilon cH_0) r
  ]
* **PTQ-split**
  [
  v^2(r)=\Upsilon_d,v_d^2+\Upsilon_b,v_b^2+v_g^2+(\epsilon cH_0) r
  ]
* **MOND (simple (\nu))**
  [
  v^2 = v_N^2,\nu!\left(\frac{g_N}{a_0}\right),\quad
  \nu(y)=\tfrac12+\sqrt{\tfrac14+\tfrac1y},\quad
  v_N^2=\Upsilon(v_d^2+v_b^2)+v_g^2,\ g_N=\frac{v_N^2}{r}
  ]
* **NFW-1p**
  [
  v^2(r)=v_{\rm bar}^2(r)+v_{\rm NFW}^2!\big(M_{200},,c(M_{200})\big),\quad
  c(M)=c_0\left(\frac{M}{10^{12}M_\odot}\right)^{\beta}
  ]

---

If you want, I can also generate a minimal `Makefile` to run all four models and output a single comparison table.
