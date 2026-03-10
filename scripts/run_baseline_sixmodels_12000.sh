#!/usr/bin/env bash
set -euo pipefail

# Layer A: six-model single-seed long-chain baseline (Table-style landscape)
# - Models: ptq-screen, mond, ptq-nu, nfw1p, ptq, baryon
# - Purpose: main baseline landscape for the paper (Table 5.1 / Table 4 style),
#   not a robustness audit.

cd ~/PTQ-Quaternionic-RC
source .venv/bin/activate

export RUNROOT=results/revision_baseline_sixmodels_12000
mkdir -p "$RUNROOT/logs"
mkdir -p "$RUNROOT/paper_extra"

echo "[INFO] run_baseline_sixmodels_12000 start"
echo "[INFO] RUNROOT=$RUNROOT"
echo "[INFO] git commit: $(git rev-parse HEAD)" | tee "$RUNROOT/logs/git_commit.txt"
python --version | tee "$RUNROOT/logs/python_version.txt"

###############################################################################
# 0. Clean start check
###############################################################################

for m in ptq-screen mond ptq-nu nfw1p ptq baryon; do
  if [ -d "$RUNROOT/${m}_gauss" ]; then
    echo "[ERROR] RUNROOT already contains $m_gauss output directory: $RUNROOT/${m}_gauss"
    echo "[ERROR] Please remove it or choose a new RUNROOT."
    exit 1
  fi
done

###############################################################################
# 1. Fresh production fits: six-model baseline (single seed)
###############################################################################

MODELS=(ptq-screen mond ptq-nu nfw1p ptq baryon)

for model in "${MODELS[@]}"; do
  echo "[INFO] Fitting baseline model: $model (steps=12000, nwalkers=192, seed=0)"
  ptquat fit \
    --model "$model" \
    --data dataset/sparc_tidy.csv \
    --outdir "$RUNROOT/${model}_gauss" \
    --nwalkers 192 \
    --steps 12000 \
    --seed 0 \
    --backend-hdf5 "$RUNROOT/${model}_gauss/chain.h5" \
    --thin-by 1 \
    2>&1 | tee "$RUNROOT/logs/${model}_gauss_12000.log"
done

###############################################################################
# 2. WAIC compare (six-model baseline)
###############################################################################

mkdir -p "$RUNROOT/paper_extra/model_compare_baseline"

ptquat exp compare \
  --models ptq-screen mond ptq-nu nfw1p ptq baryon \
  --data dataset/sparc_tidy.csv \
  --fit-root "$RUNROOT" \
  --outdir "$RUNROOT/paper_extra/model_compare_baseline" \
  --seed 0 \
  2>&1 | tee "$RUNROOT/logs/compare_baseline_12000.log"

###############################################################################
# 3. PSIS-LOO compare (six-model baseline)
###############################################################################

mkdir -p "$RUNROOT/paper_extra/loo_compare_baseline"

ptquat exp loo \
  --models ptq-screen mond ptq-nu nfw1p ptq baryon \
  --data dataset/sparc_tidy.csv \
  --fit-root "$RUNROOT" \
  --outdir "$RUNROOT/paper_extra/loo_compare_baseline" \
  --seed 0 \
  2>&1 | tee "$RUNROOT/logs/loo_baseline_12000.log"

###############################################################################
# 4. Paper artifacts for baseline (WAIC/LOO/pWAIC summary)
###############################################################################

python scripts/make_paper_artifacts.py \
  --data dataset/sparc_tidy.csv \
  --results-dir "$RUNROOT" \
  --out "$RUNROOT/paper_run" \
  --models ptq-screen mond ptq-nu nfw1p ptq baryon \
  2>&1 | tee "$RUNROOT/logs/make_paper_artifacts_baseline_12000.log"

echo "[INFO] run_baseline_sixmodels_12000 finished"

