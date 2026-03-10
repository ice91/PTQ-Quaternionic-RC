#!/usr/bin/env bash
set -euo pipefail

cd ~/PTQ-Quaternionic-RC
source .venv/bin/activate

export RUNROOT=results/revision_prod_20260306_core3000
mkdir -p "$RUNROOT/logs"

echo "[INFO] run_ptqscreen_resume12000 start"
echo "[INFO] RUNROOT=$RUNROOT"
echo "[INFO] git commit: $(git rev-parse HEAD)" | tee -a "$RUNROOT/logs/git_commit_resume12000.txt"

###############################################################################
# 1. Resume ptq-screen from existing chain.h5
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
  --resume \
  2>&1 | tee "$RUNROOT/logs/ptq-screen_gauss_resume12000.log"

###############################################################################
# 2. Re-run WAIC compare (using updated ptq-screen vs existing mond / mond-screen)
###############################################################################

mkdir -p "$RUNROOT/paper_extra/model_compare_core_after12000"

ptquat exp compare \
  --models ptq-screen mond-screen mond \
  --data dataset/sparc_tidy.csv \
  --fit-root "$RUNROOT" \
  --outdir "$RUNROOT/paper_extra/model_compare_core_after12000" \
  --seed 0 \
  2>&1 | tee "$RUNROOT/logs/compare_core_after12000.log"

###############################################################################
# 3. Re-run PSIS-LOO
###############################################################################

mkdir -p "$RUNROOT/paper_extra/loo_compare_core_after12000"

ptquat exp loo \
  --models ptq-screen mond-screen mond \
  --data dataset/sparc_tidy.csv \
  --fit-root "$RUNROOT" \
  --outdir "$RUNROOT/paper_extra/loo_compare_core_after12000" \
  --seed 0 \
  2>&1 | tee "$RUNROOT/logs/loo_core_after12000.log"

###############################################################################
# 4. Re-run closure / closure-scan for updated ptq-screen
###############################################################################

ptquat exp closure \
  --results "$RUNROOT/ptq-screen_gauss" \
  --omega-lambda 0.69 \
  2>&1 | tee "$RUNROOT/logs/closure_ptqscreen_after12000.log"

ptquat exp closure-scan \
  --results "$RUNROOT/ptq-screen_gauss" \
  --omega-lambda-min 0.65 \
  --omega-lambda-max 0.75 \
  --n 11 \
  2>&1 | tee "$RUNROOT/logs/closure_scan_ptqscreen_after12000.log"

###############################################################################
# 5. Rebuild paper artifacts
###############################################################################

python scripts/make_paper_artifacts.py \
  --data dataset/sparc_tidy.csv \
  --results-dir "$RUNROOT" \
  --out "$RUNROOT/paper_run_after12000" \
  2>&1 | tee "$RUNROOT/logs/make_paper_artifacts_after12000.log"

###############################################################################
# 6. Quick summary
###############################################################################

python - <<'PY'
import yaml, pandas as pd
from pathlib import Path

runroot = Path("results/revision_prod_20260306_core3000")
runs = {
    "PTQ-screen": runroot / "ptq-screen_gauss" / "global_summary.yaml",
    "MOND-screen": runroot / "mond-screen_gauss" / "global_summary.yaml",
    "MOND": runroot / "mond_gauss" / "global_summary.yaml",
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
out = runroot / "quicklook_core_compare_after12000.csv"
df.to_csv(out, index=False)
print(df)
print(f"[INFO] saved: {out}")
PY

echo "[INFO] run_ptqscreen_resume12000 finished"
