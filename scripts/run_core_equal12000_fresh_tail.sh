#!/usr/bin/env bash
set -euo pipefail

# Tail script for the equal12000 fresh production run.
# - Assumes the three fits (ptq-screen, mond, mond-screen),
#   core compare / loo, and ptq-screen closure have already been run.
# - Only:
#   1) Ensures ptq-screen closure-scan exists (PTQ-family only).
#   2) Rebuilds paper artifacts.
#   3) Writes a quick summary CSV.

cd ~/PTQ-Quaternionic-RC
source .venv/bin/activate

export RUNROOT=results/revision_prod_equal12000_fresh
mkdir -p "$RUNROOT/logs"

echo "[INFO] run_core_equal12000_fresh_tail start"
echo "[INFO] RUNROOT=$RUNROOT"
echo "[INFO] git commit: $(git rev-parse HEAD)" | tee -a "$RUNROOT/logs/git_commit_tail.txt"

###############################################################################
# 1. Ensure ptq-screen closure-scan exists (PTQ-family only)
###############################################################################

PTQ_CLOSURE_SCAN_DIR="$RUNROOT/ptq-screen_gauss/closure_scan"
PTQ_CLOSURE_SCAN_CSV="$PTQ_CLOSURE_SCAN_DIR/closure_sensitivity.csv"

if [ -f "$PTQ_CLOSURE_SCAN_CSV" ]; then
  echo "[INFO] ptq-screen closure-scan already present at $PTQ_CLOSURE_SCAN_CSV"
else
  echo "[INFO] ptq-screen closure-scan not found, running closure-scan..."
  mkdir -p "$PTQ_CLOSURE_SCAN_DIR"
  ptquat exp closure-scan \
    --results "$RUNROOT/ptq-screen_gauss" \
    --outdir "$PTQ_CLOSURE_SCAN_DIR" \
    --omega-lambda-min 0.65 \
    --omega-lambda-max 0.75 \
    --n 11 \
    2>&1 | tee "$RUNROOT/logs/closure_scan_ptqscreen_tail_12000.log"
fi

###############################################################################
# 2. Rebuild paper artifacts for this RUNROOT
###############################################################################

python scripts/make_paper_artifacts.py \
  --data dataset/sparc_tidy.csv \
  --results-dir "$RUNROOT" \
  --out "$RUNROOT/paper_run" \
  2>&1 | tee "$RUNROOT/logs/make_paper_artifacts_tail_12000.log"

###############################################################################
# 3. Quick summary table (same shape as main script)
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
    if not yml.exists():
        continue
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

if rows:
    df = pd.DataFrame(rows).sort_values("BIC_full", na_position="last")
    out = runroot / "quicklook_core_compare_tail.csv"
    df.to_csv(out, index=False)
    print(df)
    print(f"[INFO] saved: {out}")
else:
    print("[WARN] No global_summary.yaml files found under", runroot)
PY

echo "[INFO] run_core_equal12000_fresh_tail finished"

