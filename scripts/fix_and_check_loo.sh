#!/usr/bin/env bash
set -euo pipefail

cd ~/PTQ-Quaternionic-RC
source .venv/bin/activate

export RUNROOT=results/revision_prod_20260306_core3000
mkdir -p "$RUNROOT/logs"

echo "[INFO] checking python / package environment"
python --version | tee "$RUNROOT/logs/loo_env_check.log"

python - <<'PY' | tee -a "results/revision_prod_20260306_core3000/logs/loo_env_check.log"
import sys
mods = ["arviz", "numpy", "scipy", "xarray", "pandas"]
for m in mods:
    try:
        mod = __import__(m)
        print(f"{m}: {getattr(mod, '__version__', 'unknown')}")
    except Exception as e:
        print(f"{m}: NOT IMPORTABLE ({e})")
PY

echo "[INFO] trying ArviZ upgrade inside current venv..."
pip install -U arviz 2>&1 | tee "$RUNROOT/logs/pip_upgrade_arviz.log"

echo "[INFO] re-checking versions after upgrade"
python - <<'PY' | tee -a "results/revision_prod_20260306_core3000/logs/loo_env_check.log"
import sys
mods = ["arviz", "numpy", "scipy", "xarray", "pandas"]
for m in mods:
    try:
        mod = __import__(m)
        print(f"{m}: {getattr(mod, '__version__', 'unknown')}")
    except Exception as e:
        print(f"{m}: NOT IMPORTABLE ({e})")
PY

echo "[INFO] smoke-testing az.from_dict(log_likelihood=...)"
python - <<'PY' | tee "$RUNROOT/logs/loo_api_smoke.log"
import numpy as np
import arviz as az

log_lik = np.random.randn(1, 10, 20)   # chains, draws, observations
idata = az.from_dict(log_likelihood={"y": log_lik})
print("SUCCESS: az.from_dict accepted log_likelihood")
print(idata)
PY

echo "[INFO] running actual LOO command (core three models)"
mkdir -p "$RUNROOT/paper_extra/loo_compare_core"

ptquat exp loo \
  --models ptq-screen mond-screen mond \
  --data dataset/sparc_tidy.csv \
  --fit-root "$RUNROOT" \
  --outdir "$RUNROOT/paper_extra/loo_compare_core" \
  --seed 0 \
  2>&1 | tee "$RUNROOT/logs/loo_core_retry.log"

echo "[INFO] LOO check finished successfully"
