#!/usr/bin/env bash
set -euo pipefail

# Layer B: three-model multi-seed robustness audit (core equal-chain-length)
# - Models: ptq-screen, mond, mond-screen
# - Seeds: 0, 7, 13
# - Purpose: robustness / ranking stability audit, NOT a replacement for the
#   six-model baseline landscape (Layer A).

cd ~/PTQ-Quaternionic-RC
source .venv/bin/activate

export RUNROOT=results/revision_robustness_core_multiseed12000
mkdir -p "$RUNROOT"

echo "[INFO] run_robustness_core_multiseed12000 start"
echo "[INFO] RUNROOT=$RUNROOT"
echo "[INFO] git commit: $(git rev-parse HEAD)" | tee "$RUNROOT/git_commit.txt"
python --version | tee "$RUNROOT/python_version.txt"

SEEDS=(0 7 13)
MODELS=(ptq-screen mond mond-screen)

for seed in "${SEEDS[@]}"; do
  SEEDROOT="$RUNROOT/seed_${seed}"
  mkdir -p "$SEEDROOT/logs"
  mkdir -p "$SEEDROOT/paper_extra"

  echo "[INFO] === Seed ${seed} robustness block ==="

  # Clean-start check per seed/model (avoid accidental overwrite)
  for model in "${MODELS[@]}"; do
    if [ -d "$SEEDROOT/${model}_gauss" ]; then
      echo "[ERROR] Seed ${seed} already has ${model}_gauss under $SEEDROOT"
      echo "[ERROR] Please remove it or choose a new RUNROOT/seed."
      exit 1
    fi
  done

  #############################################################################
  # 1. Fresh fits for this seed (three-model block)
  #############################################################################

  for model in "${MODELS[@]}"; do
    echo "[INFO] [seed=${seed}] Fitting model: $model (steps=12000, nwalkers=192)"
    ptquat fit \
      --model "$model" \
      --data dataset/sparc_tidy.csv \
      --outdir "$SEEDROOT/${model}_gauss" \
      --nwalkers 192 \
      --steps 12000 \
      --seed "${seed}" \
      --backend-hdf5 "$SEEDROOT/${model}_gauss/chain.h5" \
      --thin-by 1 \
      2>&1 | tee "$SEEDROOT/logs/${model}_gauss_seed${seed}_12000.log"
  done

  #############################################################################
  # 2. WAIC compare for this seed (three-model core)
  #############################################################################

  mkdir -p "$SEEDROOT/paper_extra/model_compare_core"
  ptquat exp compare \
    --models ptq-screen mond mond-screen \
    --data dataset/sparc_tidy.csv \
    --fit-root "$SEEDROOT" \
    --outdir "$SEEDROOT/paper_extra/model_compare_core" \
    --seed "${seed}" \
    2>&1 | tee "$SEEDROOT/logs/compare_core_seed${seed}_12000.log"

  #############################################################################
  # 3. PSIS-LOO compare for this seed (three-model core)
  #############################################################################

  mkdir -p "$SEEDROOT/paper_extra/loo_compare_core"
  ptquat exp loo \
    --models ptq-screen mond mond-screen \
    --data dataset/sparc_tidy.csv \
    --fit-root "$SEEDROOT" \
    --outdir "$SEEDROOT/paper_extra/loo_compare_core" \
    --seed "${seed}" \
    2>&1 | tee "$SEEDROOT/logs/loo_core_seed${seed}_12000.log"

done

echo "[INFO] run_robustness_core_multiseed12000 finished"

