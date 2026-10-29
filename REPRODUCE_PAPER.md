# Reproduce the SPARC / Frontiers manuscript

This guide maps the public repository to the frozen manuscript claims.
It does **not** authorize new scientific campaigns.

**Immutable reproducibility release tag:** `sparc-frontiers-v1.0.0`
**Preprints.org Version 5:** doi:10.20944/preprints202503.2052.v5
**Companion paper evidence bundle:** `../paper_release/frontiers_sparc_v1/` (in the paper workspace) or the mirrored bundle shipped with the release.

---

## 1. Environment

```bash
git clone https://github.com/ice91/PTQ-Quaternionic-RC.git
cd PTQ-Quaternionic-RC
git checkout sparc-frontiers-v1.0.0   # after the tag is published
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest -q
```

Canonical scripts resolve the repository root from their own location (no `~/…` home paths).

---

## 2. SPARC sample

```bash
ptquat fetch --out dataset/raw
ptquat preprocess \
  --raw dataset/raw \
  --out dataset/sparc_tidy.csv \
  --i-min 30.0 --reldmax 0.20 --qual-max 2
```

Expected after cuts: **91 galaxies / 1782 points**.

Sample-flow table: `paper_release/.../tables/sample_flow.csv`.

---

## 3. Model equations ↔ repository labels

| Paper | Label |
|-------|-------|
| MOND simple‑\(\nu\) | `mond` |
| PTQ‑\(\nu\) | `ptq-nu` |
| Shared \(\nu_q\) | `ptq-screen` / `mond-screen` (historical names) |
| Linear dictionary | `ptq` |
| NFW‑1p | `nfw1p` |

Under \(a_0=\varepsilon c H_0\), PTQ‑\(\nu\) ≡ MOND as likelihood families; the same reparameterization identity holds for the shared \(\nu_q\) pair.

---

## 4. Table 3 — full-covariance MLE (quick verify)

**Frozen artifact (preferred):**

```bash
column -s, -t < results/scientific_hardening_r1/01_mle/model_compare_mle.csv | head
shasum -a 256 results/scientific_hardening_r1/01_mle/model_compare_mle.csv
```

Also mirrored in the paper release bundle under `mle/model_compare_mle.csv`.

**Full rerun (expensive; optional):**

```bash
python scripts/run_scientific_hardening_mle.py --help
```

Do not treat a local rerun as license to change frozen paper numbers without a new authorized phase.

---

## 5. Table 4 — thickness regression on \(\kappa_{\rm kin}\) (quick verify)

```bash
python - <<'PY'
import json
p='results/paper_submission_ptqnu_20260323/03_figures/kappa_h_report_ptqnu.json'
d=json.load(open(p))
print('N',d['n_used'],'b',d['b_kappa'],'c',d['c_sigma'],'R2',d['R2'])
print('dAIC',d['dAIC_kappa_only'],d['dAIC_sigma_only'],'boot',d['args']['bootstrap'])
PY
```

WLS weights: inverse-variance from \(h\) uncertainties (\(\sigma_{\log_{10}h}=(\sigma_h/h)/\ln 10\)).

\(\kappa_{\rm kin}\) is kinematic (\(\mathrm{T}^{-1}\)), independent of fitted PTQ/MOND parameters.
A standard disk-dynamical heuristic may qualitatively motivate the sign pattern of the association; the manuscript does not identify \(\kappa_{\rm kin}\) with a PTQ geometric efficiency.

---

## 6. Figure 1 — added-variable plot

Deterministic rendering from frozen `kappa_h_used_ptqnu.csv` + `sparc_with_h` weights:

```bash
python paper_release/frontiers_sparc_v1/scripts/render_fig1_added_variable.py
```

(or the copy under this repo’s release mirror). This **re-renders** the same bivariate WLS result; it does not refit a new hypothesis.

---

## 7. Closure figure

```bash
open results/paper_submission_ptqnu_20260323/02_ptqnu_main/closure_scan/closure_sensitivity_plot.png
# or: xdg-open ...
cat results/paper_submission_ptqnu_20260323/02_ptqnu_main/closure_test.yaml | head
```

Strict cross-scale equality fails under the selected matching relation + adopted diagnostic map.
The ratio \(\kappa_{\rm req}\equiv\varepsilon_{\rm RC}/\varepsilon_{\rm cos}\approx0.10\) is a required bookkeeping factor only; it is not evidence for closure.

---

## 8. Legacy WAIC / PSIS‑LOO

```bash
# historical tooling (example)
ptquat exp compare --help
ptquat exp loo --help
```

**Classification:** `LEGACY_DIAGONALIZED_SENSITIVITY`.
Archived radius-level WAIC/LOO is **not** the canonical headline ranking layer for the current manuscript. Multi-seed tables are reproducibility records only.

---

## 9. What not to do

- Do not advertise a “winning” model from legacy WAIC.
- Do not equate \(\kappa_{\rm kin}\) with a dimensionless PTQ geometric efficiency.
- Do not treat \(\kappa_{\rm req}\) or historical OOM language as closure success.
- Do not require author home-directory paths.
