#!/usr/bin/env bash
set -euo pipefail

# Layer B: four-model multi-seed robustness audit
# - Models: ptq-nu, ptq-screen, mond, mond-screen
# - Seeds: 0, 7, 13
# - Purpose:
#   (i) check seed / chain-path sensitivity of the leading PTQ-family models
#   (ii) compare against MOND and the matched-kernel control mond-screen
# - This is a robustness layer, NOT a replacement for the six-model baseline
#   landscape in Layer A.

cd ~/PTQ-Quaternionic-RC
source .venv/bin/activate

export RUNROOT=results/revision_robustness_core_multiseed12000
mkdir -p "$RUNROOT"
mkdir -p "$RUNROOT/logs"

echo "[INFO] run_robustness_core_multiseed12000 start"
echo "[INFO] RUNROOT=$RUNROOT"
echo "[INFO] git commit: $(git rev-parse HEAD)" | tee "$RUNROOT/git_commit.txt"
python --version | tee "$RUNROOT/python_version.txt"

###############################################################################
# 0. Configuration
###############################################################################

SEEDS=(0 7 13)
MODELS=(ptq-nu ptq-screen mond mond-screen)

DATA_PATH="dataset/sparc_tidy.csv"
STEPS=12000
NWALKERS=192
THIN_BY=1

is_model_complete() {
  local seedroot="$1"
  local model="$2"
  local outdir="$seedroot/${model}_gauss"
  [[ -f "$outdir/global_summary.yaml" && -f "$outdir/per_galaxy_summary.csv" ]]
}

###############################################################################
# 1. Fresh fits + compare/loo block for each seed
###############################################################################

for seed in "${SEEDS[@]}"; do
  SEEDROOT="$RUNROOT/seed_${seed}"
  mkdir -p "$SEEDROOT/logs"
  mkdir -p "$SEEDROOT/paper_extra"

  echo "[INFO] ============================================================="
  echo "[INFO] Seed ${seed} robustness block"
  echo "[INFO] SEEDROOT=$SEEDROOT"
  echo "[INFO] ============================================================="

  ###########################################################################
  # 1A. Fresh fits for this seed (skip completed models)
  ###########################################################################

  for model in "${MODELS[@]}"; do
    outdir="$SEEDROOT/${model}_gauss"
    chain_path="$outdir/chain.h5"

    if is_model_complete "$SEEDROOT" "$model"; then
      echo "[INFO] [seed=${seed}] Skip completed model: $model"
      continue
    fi

    echo "[INFO] [seed=${seed}] Fitting model: $model (steps=$STEPS, nwalkers=$NWALKERS)"
    mkdir -p "$outdir"

    ptquat fit \
      --model "$model" \
      --data "$DATA_PATH" \
      --outdir "$outdir" \
      --nwalkers "$NWALKERS" \
      --steps "$STEPS" \
      --seed "$seed" \
      --backend-hdf5 "$chain_path" \
      --thin-by "$THIN_BY" \
      2>&1 | tee "$SEEDROOT/logs/${model}_gauss_seed${seed}_${STEPS}.log"
  done

  ###########################################################################
  # 1B. Verify completeness for this seed before compare / loo
  ###########################################################################

  missing_models=()
  for model in "${MODELS[@]}"; do
    if ! is_model_complete "$SEEDROOT" "$model"; then
      missing_models+=("$model")
    fi
  done

  if (( ${#missing_models[@]} > 0 )); then
    echo "[WARN] [seed=${seed}] Missing completed outputs for: ${missing_models[*]}"
    echo "[WARN] [seed=${seed}] compare / loo skipped for now."
    continue
  fi

  ###########################################################################
  # 1C. WAIC compare for this seed
  ###########################################################################

  mkdir -p "$SEEDROOT/paper_extra/model_compare_core"
  ptquat exp compare \
    --models ptq-nu ptq-screen mond mond-screen \
    --data "$DATA_PATH" \
    --fit-root "$SEEDROOT" \
    --outdir "$SEEDROOT/paper_extra/model_compare_core" \
    --seed "$seed" \
    2>&1 | tee "$SEEDROOT/logs/compare_core_seed${seed}_${STEPS}.log"

  ###########################################################################
  # 1D. PSIS-LOO compare for this seed
  ###########################################################################

  mkdir -p "$SEEDROOT/paper_extra/loo_compare_core"
  ptquat exp loo \
    --models ptq-nu ptq-screen mond mond-screen \
    --data "$DATA_PATH" \
    --fit-root "$SEEDROOT" \
    --outdir "$SEEDROOT/paper_extra/loo_compare_core" \
    --seed "$seed" \
    2>&1 | tee "$SEEDROOT/logs/loo_core_seed${seed}_${STEPS}.log"
done

echo "[INFO] run_robustness_core_multiseed12000 finished"

