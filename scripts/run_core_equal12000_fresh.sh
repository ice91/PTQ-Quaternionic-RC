#!/usr/bin/env bash
set -euo pipefail

cd ~/PTQ-Quaternionic-RC
source .venv/bin/activate

export RUNROOT=results/revision_prod_equal12000_fresh
mkdir -p "$RUNROOT/logs"
mkdir -p "$RUNROOT/paper_extra"

echo "[INFO] run_core_equal12000_fresh start"
echo "[INFO] RUNROOT=$RUNROOT"
echo "[INFO] git commit: $(git rev-parse HEAD)" | tee "$RUNROOT/logs/git_commit.txt"
python --version | tee "$RUNROOT/logs/python_version.txt"

###############################################################################
# 0. Clean start check
###############################################################################
if [ -d "$RUNROOT/ptq-screen_gauss" ] || [ -d "$RUNROOT/mond_gauss" ] || [ -d "$RUNROOT/mond-screen_gauss" ]; then
  echo "[ERROR] RUNROOT already contains model output directories."
  echo "[ERROR] Please remove them or choose a new RUNROOT."
  exit 1
fi

###############################################################################
# 1. Fresh production fits: equal-chain-length audit
###############################################################################

ptquat fit \
  --model ptq-screen \
  --data dataset/sparc_tidy.csv \
  --outdir "$RUNROOT/ptq-screen_gauss" \
  --nwalkers 192 \
  --steps 12000 \
  --seed 0 \
  --backend-hdf5 "$RUNROOT/ptq-screen_gauss/chain.h5" \
  --thin-by 1 \
  2>&1 | tee "$RUNROOT/logs/ptq-screen_gauss_12000.log"

ptquat fit \
  --model mond \
  --data dataset/sparc_tidy.csv \
  --outdir "$RUNROOT/mond_gauss" \
  --nwalkers 192 \
  --steps 12000 \
  --seed 0 \
  --backend-hdf5 "$RUNROOT/mond_gauss/chain.h5" \
  --thin-by 1 \
  2>&1 | tee "$RUNROOT/logs/mond_gauss_12000.log"

ptquat fit \
  --model mond-screen \
  --data dataset/sparc_tidy.csv \
  --outdir "$RUNROOT/mond-screen_gauss" \
  --nwalkers 192 \
  --steps 12000 \
  --seed 0 \
  --backend-hdf5 "$RUNROOT/mond-screen_gauss/chain.h5" \
  --thin-by 1 \
  2>&1 | tee "$RUNROOT/logs/mond-screen_gauss_12000.log"

###############################################################################
# 2. WAIC compare
###############################################################################

mkdir -p "$RUNROOT/paper_extra/model_compare_core"

ptquat exp compare \
  --models ptq-screen mond mond-screen \
  --data dataset/sparc_tidy.csv \
  --fit-root "$RUNROOT" \
  --outdir "$RUNROOT/paper_extra/model_compare_core" \
  --seed 0 \
  2>&1 | tee "$RUNROOT/logs/compare_core_12000.log"

###############################################################################
# 3. PSIS-LOO
###############################################################################

mkdir -p "$RUNROOT/paper_extra/loo_compare_core"

ptquat exp loo \
  --models ptq-screen mond mond-screen \
  --data dataset/sparc_tidy.csv \
  --fit-root "$RUNROOT" \
  --outdir "$RUNROOT/paper_extra/loo_compare_core" \
  --seed 0 \
  2>&1 | tee "$RUNROOT/logs/loo_core_12000.log"

###############################################################################
# 4. Optional single-model compare runs for per-model pWAIC diagnostics
###############################################################################

mkdir -p "$RUNROOT/paper_extra/model_compare_ptqscreen_diag"
ptquat exp compare \
  --models ptq-screen \
  --data dataset/sparc_tidy.csv \
  --fit-root "$RUNROOT" \
  --outdir "$RUNROOT/paper_extra/model_compare_ptqscreen_diag" \
  --seed 0 \
  2>&1 | tee "$RUNROOT/logs/compare_ptqscreen_diag_12000.log"

mkdir -p "$RUNROOT/paper_extra/model_compare_mond_diag"
ptquat exp compare \
  --models mond \
  --data dataset/sparc_tidy.csv \
  --fit-root "$RUNROOT" \
  --outdir "$RUNROOT/paper_extra/model_compare_mond_diag" \
  --seed 0 \
  2>&1 | tee "$RUNROOT/logs/compare_mond_diag_12000.log"

mkdir -p "$RUNROOT/paper_extra/model_compare_mondscreen_diag"
ptquat exp compare \
  --models mond-screen \
  --data dataset/sparc_tidy.csv \
  --fit-root "$RUNROOT" \
  --outdir "$RUNROOT/paper_extra/model_compare_mondscreen_diag" \
  --seed 0 \
  2>&1 | tee "$RUNROOT/logs/compare_mondscreen_diag_12000.log"

###############################################################################
# 5. Closure + closure-scan
###############################################################################

# NOTE:
# - The closure / closure-scan pipeline is defined in terms of epsilon_RC vs epsilon_cos
#   and is therefore **PTQ-family only** (it relies on `epsilon_median` in global_summary.yaml).
# - `mond` and `mond-screen` do not have epsilon; they are benchmark models and must
#   NOT be passed into the current epsilon-based closure code.
# - We therefore only run closure / closure-scan for `ptq-screen` here.

ptquat exp closure \
  --results "$RUNROOT/ptq-screen_gauss" \
  --omega-lambda 0.69 \
  2>&1 | tee "$RUNROOT/logs/closure_ptqscreen_12000.log"

ptquat exp closure-scan \
  --results "$RUNROOT/ptq-screen_gauss" \
  --outdir "$RUNROOT/ptq-screen_gauss/closure_scan" \
  --omega-lambda-min 0.65 \
  --omega-lambda-max 0.75 \
  --n 11 \
  2>&1 | tee "$RUNROOT/logs/closure_scan_ptqscreen_12000.log"

###############################################################################
# 6. Paper artifacts
###############################################################################

python scripts/make_paper_artifacts.py \
  --data dataset/sparc_tidy.csv \
  --results-dir "$RUNROOT" \
  --out "$RUNROOT/paper_run" \
  2>&1 | tee "$RUNROOT/logs/make_paper_artifacts_12000.log"

###############################################################################
# 7. Quick summary table
###############################################################################

python - <<'PY'
import yaml, pandas as pd
from pathlib import Path

runroot = Path("results/revision_prod_equal12000_fresh")
runs = {
    "PTQ-screen": runroot / "ptq-screen_gauss" / "global_summary.yaml",
    "MOND": runroot / "mond_gauss" / "global_summary.yaml",
    "MOND-screen": runroot / "mond-screen_gauss" / "global_summary.yaml",
}

rows = []
for name, yml in runs.items():
    s = yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
    rows.append({
        "model": name,
        "chi2_total": s.get("chi2_total"),
        "AIC_full": s.get("AIC_full"),
        "BIC_full": s.get("BIC_full"),
        "k_parameters": s.get("k_parameters"),
        "epsilon_median": s.get("epsilon_median"),
        "a0_median": s.get("a0_median"),
        "q_median": s.get("q_median"),
        "steps": s.get("steps"),
        "nwalkers": s.get("nwalkers"),
        "burn_in": s.get("burn_in"),
        "resumed": s.get("resumed"),
    })

df = pd.DataFrame(rows).sort_values("BIC_full", na_position="last")
out = runroot / "quicklook_core_compare.csv"
df.to_csv(out, index=False)
print(df)
print(f"[INFO] saved: {out}")
PY

echo "[INFO] run_core_equal12000_fresh finished"
