#!/usr/bin/env bash
set -euo pipefail

# Layer A: six-model single-seed long-chain baseline (Table-style landscape)
# - Models: ptq-screen, mond, ptq-nu, nfw1p, ptq, baryon
# - Purpose: main baseline landscape for the paper (Table 5.1 / Table 4 style),
#   not a robustness audit.
#
# Design choices:
# - Keep STEPS uniform (=12000) across all six models for a clean paper baseline.
# - Use model-specific nwalkers only where required by sampler geometry:
#     * nfw1p uses 384 walkers because its dimensionality is high (k~183),
#       and emcee red-blue move requires nwalkers >= 2 * ndim.
# - Skip already-finished models instead of failing the whole run.
# - Only run compare / loo / artifact aggregation after confirming that all
#   six model outputs exist.

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
# 0. Configuration
###############################################################################

SEED=0
STEPS=12000
THIN_BY=1
DATA_PATH="dataset/sparc_tidy.csv"

MODELS=(ptq-screen mond ptq-nu nfw1p ptq baryon)

# Default walkers for most models
default_nwalkers=192

# Model-specific override
get_nwalkers() {
  local model="$1"
  case "$model" in
    nfw1p)
      echo 384
      ;;
    *)
      echo "$default_nwalkers"
      ;;
  esac
}

is_model_complete() {
  local model="$1"
  local outdir="$RUNROOT/${model}_gauss"
  [[ -f "$outdir/global_summary.yaml" && -f "$outdir/per_galaxy_summary.csv" ]]
}

###############################################################################
# 1. Fresh production fits: six-model baseline (single seed)
###############################################################################

for model in "${MODELS[@]}"; do
  outdir="$RUNROOT/${model}_gauss"
  chain_path="$outdir/chain.h5"
  nwalkers="$(get_nwalkers "$model")"

  if is_model_complete "$model"; then
    echo "[INFO] Skip completed baseline model: $model"
    continue
  fi

  echo "[INFO] Fitting baseline model: $model (steps=$STEPS, nwalkers=$nwalkers, seed=$SEED)"
  mkdir -p "$outdir"

  ptquat fit \
    --model "$model" \
    --data "$DATA_PATH" \
    --outdir "$outdir" \
    --nwalkers "$nwalkers" \
    --steps "$STEPS" \
    --seed "$SEED" \
    --backend-hdf5 "$chain_path" \
    --thin-by "$THIN_BY" \
    2>&1 | tee "$RUNROOT/logs/${model}_gauss_${STEPS}.log"
done

###############################################################################
# 2. Verify all six model outputs exist before compare / loo / artifacts
###############################################################################

missing_models=()
for model in "${MODELS[@]}"; do
  if ! is_model_complete "$model"; then
    missing_models+=("$model")
  fi
done

if (( ${#missing_models[@]} > 0 )); then
  echo "[WARN] Baseline fit stage incomplete."
  echo "[WARN] Missing completed outputs for: ${missing_models[*]}"
  echo "[WARN] compare / loo / artifact aggregation are skipped for now."
  exit 0
fi

###############################################################################
# 3. WAIC compare (six-model baseline)
###############################################################################

mkdir -p "$RUNROOT/paper_extra/model_compare_baseline"

ptquat exp compare \
  --models ptq-screen mond ptq-nu nfw1p ptq baryon \
  --data "$DATA_PATH" \
  --fit-root "$RUNROOT" \
  --outdir "$RUNROOT/paper_extra/model_compare_baseline" \
  --seed "$SEED" \
  2>&1 | tee "$RUNROOT/logs/compare_baseline_${STEPS}.log"

###############################################################################
# 4. PSIS-LOO compare (six-model baseline)
###############################################################################

mkdir -p "$RUNROOT/paper_extra/loo_compare_baseline"

ptquat exp loo \
  --models ptq-screen mond ptq-nu nfw1p ptq baryon \
  --data "$DATA_PATH" \
  --fit-root "$RUNROOT" \
  --outdir "$RUNROOT/paper_extra/loo_compare_baseline" \
  --seed "$SEED" \
  2>&1 | tee "$RUNROOT/logs/loo_baseline_${STEPS}.log"

###############################################################################
# 5. Paper artifacts for baseline (WAIC/LOO/pWAIC summary)
###############################################################################

mkdir -p "$RUNROOT/paper_artifacts"
mkdir -p "$RUNROOT/paper_run"

python scripts/make_paper_artifacts.py \
  --data "$DATA_PATH" \
  --results-dir "$RUNROOT" \
  --out "$RUNROOT/paper_run" \
  --models ptq-screen mond ptq-nu nfw1p ptq baryon \
  2>&1 | tee "$RUNROOT/logs/make_paper_artifacts_baseline_${STEPS}.log"

echo "[INFO] run_baseline_sixmodels_12000 finished"