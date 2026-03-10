#!/usr/bin/env bash
set -euo pipefail

cd ~/PTQ-Quaternionic-RC
source .venv/bin/activate

export RUNROOT=results/revision_prod_20260306_core3000
mkdir -p "$RUNROOT/logs"
mkdir -p "$RUNROOT/paper_extra"

echo "[INFO] RUNROOT=$RUNROOT"
echo "[INFO] git commit: $(git rev-parse HEAD)" | tee "$RUNROOT/logs/git_commit.txt"

###############################################################################
# 1. Production fits: core three models
###############################################################################

ptquat fit \
  --model ptq-screen \
  --data dataset/sparc_tidy.csv \
  --outdir "$RUNROOT/ptq-screen_gauss" \
  --nwalkers 192 \
  --steps 3000 \
  --seed 0 \
  --backend-hdf5 "$RUNROOT/ptq-screen_gauss/chain.h5" \
  --thin-by 1 \
  2>&1 | tee "$RUNROOT/logs/ptq-screen_gauss_prod.log"

ptquat fit \
  --model mond-screen \
  --data dataset/sparc_tidy.csv \
  --outdir "$RUNROOT/mond-screen_gauss" \
  --nwalkers 192 \
  --steps 3000 \
  --seed 0 \
  --backend-hdf5 "$RUNROOT/mond-screen_gauss/chain.h5" \
  --thin-by 1 \
  2>&1 | tee "$RUNROOT/logs/mond-screen_gauss_prod.log"

ptquat fit \
  --model mond \
  --data dataset/sparc_tidy.csv \
  --outdir "$RUNROOT/mond_gauss" \
  --nwalkers 192 \
  --steps 3000 \
  --seed 0 \
  --backend-hdf5 "$RUNROOT/mond_gauss/chain.h5" \
  --thin-by 1 \
  2>&1 | tee "$RUNROOT/logs/mond_gauss_prod.log"

###############################################################################
# 2. Multi-model WAIC compare (core three)
###############################################################################

mkdir -p "$RUNROOT/paper_extra/model_compare_core"

ptquat exp compare \
  --models ptq-screen mond-screen mond \
  --data dataset/sparc_tidy.csv \
  --fit-root "$RUNROOT" \
  --outdir "$RUNROOT/paper_extra/model_compare_core" \
  --seed 0 \
  2>&1 | tee "$RUNROOT/logs/compare_core.log"

###############################################################################
# 3. PSIS-LOO compare (core three)
###############################################################################

mkdir -p "$RUNROOT/paper_extra/loo_compare_core"

ptquat exp loo \
  --models ptq-screen mond-screen mond \
  --data dataset/sparc_tidy.csv \
  --fit-root "$RUNROOT" \
  --outdir "$RUNROOT/paper_extra/loo_compare_core" \
  --seed 0 \
  2>&1 | tee "$RUNROOT/logs/loo_core.log"

###############################################################################
# 4. Single-model WAIC runs to force pWAIC diagnostics per model
#    (useful if compare only writes diagnostics for best model)
###############################################################################

mkdir -p "$RUNROOT/paper_extra/model_compare_ptqscreen_diag"
ptquat exp compare \
  --models ptq-screen \
  --data dataset/sparc_tidy.csv \
  --fit-root "$RUNROOT" \
  --outdir "$RUNROOT/paper_extra/model_compare_ptqscreen_diag" \
  --seed 0 \
  2>&1 | tee "$RUNROOT/logs/compare_ptqscreen_diag.log"

mkdir -p "$RUNROOT/paper_extra/model_compare_mondscreen_diag"
ptquat exp compare \
  --models mond-screen \
  --data dataset/sparc_tidy.csv \
  --fit-root "$RUNROOT" \
  --outdir "$RUNROOT/paper_extra/model_compare_mondscreen_diag" \
  --seed 0 \
  2>&1 | tee "$RUNROOT/logs/compare_mondscreen_diag.log"

mkdir -p "$RUNROOT/paper_extra/model_compare_mond_diag"
ptquat exp compare \
  --models mond \
  --data dataset/sparc_tidy.csv \
  --fit-root "$RUNROOT" \
  --outdir "$RUNROOT/paper_extra/model_compare_mond_diag" \
  --seed 0 \
  2>&1 | tee "$RUNROOT/logs/compare_mond_diag.log"

###############################################################################
# 5. Closure tests
###############################################################################

ptquat exp closure \
  --results "$RUNROOT/ptq-screen_gauss" \
  --omega-lambda 0.69 \
  2>&1 | tee "$RUNROOT/logs/closure_ptqscreen.log"

# 如果您的 closure-scan 支援 omega-lambda 掃描，用這個：
ptquat exp closure-scan \
  --results "$RUNROOT/ptq-screen_gauss" \
  --omega-lambda-min 0.65 \
  --omega-lambda-max 0.75 \
  --n 11 \
  2>&1 | tee "$RUNROOT/logs/closure_scan_ptqscreen.log"

# 若您也想看 mond-screen 的對照 closure-scan，可保留；不需要可註解掉
ptquat exp closure-scan \
  --results "$RUNROOT/mond-screen_gauss" \
  --omega-lambda-min 0.65 \
  --omega-lambda-max 0.75 \
  --n 11 \
  2>&1 | tee "$RUNROOT/logs/closure_scan_mondscreen.log"

###############################################################################
# 6. Paper artifacts
###############################################################################

python scripts/make_paper_artifacts.py \
  --data dataset/sparc_tidy.csv \
  --results-dir "$RUNROOT" \
  --out "$RUNROOT/paper_run" \
  2>&1 | tee "$RUNROOT/logs/make_paper_artifacts.log"

###############################################################################
# 7. Quick summary table
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
    })

df = pd.DataFrame(rows).sort_values("BIC_full", na_position="last")
out = runroot / "quicklook_core_compare.csv"
df.to_csv(out, index=False)
print(df)
print(f"[INFO] saved: {out}")
PY

echo "[INFO] run_revision_prod_core.sh finished."
