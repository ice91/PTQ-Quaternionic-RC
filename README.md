# SPARC Weak-Field Identifiability and Diagnostic Pipeline

**Public posture:** this repository is a **SPARC weak-field identifiability and diagnostic pipeline**, not a PTQ model-ranking scoreboard.

It accompanies the manuscript *SPARC Diagnostics of MOND-like Weak-Field Dictionaries with a Projected Scalar-Sector Interpretation* (Frontiers / Preprints.org Version‑5 release candidate).

## What this code establishes

- Under \(a_0=\varepsilon\,c\,H_0\) and the same fixed simple-\(\nu\), **PTQ‑\(\nu\) and MOND define the same observable likelihood family**.
- The **same reparameterization identity holds for the shared \(\nu_q\) family** (repository labels `ptq-screen` / `mond-screen` are **historical names** for that implemented family).
- The **unscreened linear dictionary** (`ptq`) is a **comparison / control realization**, not a derived PTQ prediction.
- **NFW‑1p** is a **restricted** one-halo-parameter benchmark, not a full \(\Lambda\)CDM test.
- Headline accounting for the current manuscript uses **full within-galaxy covariance MLE** (see `src/ptquat/mle.py` and `results/scientific_hardening_r1/01_mle/`).
- Archived radius-level **WAIC / PSIS‑LOO** tooling remains for historical reproducibility and is classified as **`LEGACY_DIAGONALIZED_SENSITIVITY`** — **not** the canonical headline ranking layer.
- \(\kappa_{\rm kin}=\sqrt{2}(v/R)\sqrt{1+\beta}\) is an **independent kinematic diagnostic** with \([\kappa_{\rm kin}]=\mathrm{T}^{-1}\). It is **not** the historical dimensionless PTQ geometric efficiency.
- The observed thickness relation involves this independent epicyclic-frequency-like diagnostic; a standard disk-dynamical heuristic may qualitatively motivate its sign pattern, but the manuscript does **not** identify it with a PTQ geometric efficiency.
- **Strict selected cross-scale equality fails** under the adopted \(\Omega_\Lambda(\varepsilon)\) diagnostic map; the numerical reconciliation factor \(\kappa_{\rm req}\equiv\varepsilon_{\rm RC}/\varepsilon_{\rm cos}\) is bookkeeping only, not evidence for closure.
- Open-source value is **auditability of both positive and negative results**, not advertisement of a model winner.

## Immutable release (planned)

| Item | Value |
|------|-------|
| Proposed tag | `sparc-frontiers-v1.0.0` |
| Paper reproduction guide | [`REPRODUCE_PAPER.md`](REPRODUCE_PAPER.md) |
| Paper evidence bundle (companion paper repo) | `paper_release/frontiers_sparc_v1/` |
| Citation metadata | [`CITATION.cff`](CITATION.cff) |

Public push / preprint upload / DOI minting require **separate Human Arbiter authorization**.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest -q
```

## Data

```bash
ptquat fetch --out dataset/raw
ptquat preprocess \
  --raw dataset/raw \
  --out dataset/sparc_tidy.csv \
  --i-min 30.0 --reldmax 0.20 --qual-max 2
```

Quality cuts yield **\(N_{\rm gal}=91\)**, **\(N_{\rm points}=1782\)**.

## Model ↔ repository labels

| Manuscript family | Code / CLI label(s) |
|-------------------|---------------------|
| Baryon | `baryon` |
| MOND (simple‑\(\nu\)) | `mond` |
| PTQ‑\(\nu\) | `ptq-nu` |
| Shared \(\nu_q\) family | `ptq-screen` / `mond-screen` (historical) |
| Unscreened linear dictionary | `ptq` |
| NFW‑1p | `nfw1p` |

## Full-covariance MLE (current manuscript)

Frozen machine-readable summary:

`results/scientific_hardening_r1/01_mle/model_compare_mle.csv`

Runner (local verification; do not treat as a new science campaign unless re-authorized):

`scripts/run_scientific_hardening_mle.py`

Core library: `src/ptquat/mle.py`, `src/ptquat/likelihood.py`.

## Geometry thickness audit (\(\kappa_{\rm kin}\))

Frozen N=19 WLS artifacts:

`results/paper_submission_ptqnu_20260323/03_figures/kappa_h_report_ptqnu.json`
`results/paper_submission_ptqnu_20260323/03_figures/kappa_h_used_ptqnu.csv`

## Legacy WAIC / PSIS‑LOO

Commands such as `ptquat exp compare` / `ptquat exp loo` remain available.

**Classification:** `LEGACY_DIAGONALIZED_SENSITIVITY` — diagonalized per-radius predictive scores under historical Bayesian settings. Multi-seed tables are reproducibility records only. They are **not** the headline theory-ranking layer of the current manuscript.

## Closure diagnostic

`results/paper_submission_ptqnu_20260323/02_ptqnu_main/closure_scan/`
Strict cross-scale equality fails under the selected matching relation + adopted diagnostic map.

## Reproduce the paper

See **[`REPRODUCE_PAPER.md`](REPRODUCE_PAPER.md)** for environment setup, Table/Figure provenance, quick verification against frozen artifacts, and optional full reruns.

## License

See repository license file. SPARC data remain subject to their original terms (Lelli et al. 2016).
