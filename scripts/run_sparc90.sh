#!/usr/bin/env bash
# scripts/run_sparc90.sh
# Usage:
#   bash scripts/run_sparc90.sh [--bg fg|tmux|nohup] [--steps 12000] [--sigma 4] [--prior galaxies-only] [--nwalkers 4x]
# Examples:
#   bash scripts/run_sparc90.sh
#   bash scripts/run_sparc90.sh --bg tmux --steps 16000 --sigma 6 --prior planck-anchored

set -Eeuo pipefail

### ---- Config (defaults) ----
BG_MODE="auto"                 # auto | fg | tmux | nohup
STEPS_FULL=12000
SIGMA_SYS=4
PRIOR="galaxies-only"          # galaxies-only | planck-anchored
NWALKERS="4x"
H0_KMS_MPC=""                  # e.g. 67.4 ; leave empty to use default
I_MIN=30
RELD_MAX=0.2
QUAL_MAX=2

### ---- Parse args ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --bg)         BG_MODE="${2:-auto}"; shift 2;;
    --steps)      STEPS_FULL="${2}"; shift 2;;
    --sigma)      SIGMA_SYS="${2}"; shift 2;;
    --prior)      PRIOR="${2}"; shift 2;;
    --nwalkers)   NWALKERS="${2}"; shift 2;;
    --H0-kms-mpc) H0_KMS_MPC="${2}"; shift 2;;
    --i-min)      I_MIN="${2}"; shift 2;;
    --reldmax)    RELD_MAX="${2}"; shift 2;;
    --qual-max)   QUAL_MAX="${2}"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

### ---- Paths ----
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
RAW_DIR="dataset/raw"
TIDY_CSV="dataset/sparc_tidy.csv"
OUT_SMOKE="results/sparc90_smoke"
OUT_FULL="results/sparc90"
LOG_DIR="logs"
mkdir -p "$RAW_DIR" "$(dirname "$TIDY_CSV")" "$OUT_SMOKE" "$OUT_FULL" "$LOG_DIR"

log() { echo "[$(date +'%F %T')] $*"; }

### ---- Activate venv & sanity checks ----
if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  log "WARNING: .venv not found; proceeding with current Python environment."
fi

if ! command -v ptquat >/dev/null 2>&1; then
  log "ERROR: 'ptquat' not found. Install the package first, e.g.:"
  log "  pip install -e ."
  exit 1
fi

### ---- Stage 1: Fetch ----
log "[1/4] Fetching SPARC from VizieR..."
ptquat fetch --out "$RAW_DIR"
log "Fetch done."

### ---- Stage 2: Preprocess ----
log "[2/4] Preprocessing (quality cuts: i>${I_MIN}, relD<${RELD_MAX}, Qual<=${QUAL_MAX})..."
ptquat preprocess --raw "$RAW_DIR" --out "$TIDY_CSV" --i-min "$I_MIN" --reldmax "$RELD_MAX" --qual-max "$QUAL_MAX"
log "Preprocess done -> $TIDY_CSV"

### ---- Stage 3: Smoke test (short MCMC) ----
log "[3/4] Smoke test fit (short chain) -> $OUT_SMOKE"
ptquat fit \
  --data "$TIDY_CSV" \
  --outdir "$OUT_SMOKE" \
  --prior "$PRIOR" \
  --sigma-sys "$SIGMA_SYS" \
  --nwalkers "$NWALKERS" \
  --steps 2000 \
  --seed 42 \
  ${H0_KMS_MPC:+--H0-kms-mpc "$H0_KMS_MPC"}
log "Smoke test finished."

### ---- Stage 4: Full run (long MCMC) ----
CMD_FULL=(ptquat fit
  --data "$TIDY_CSV"
  --outdir "$OUT_FULL"
  --prior "$PRIOR"
  --sigma-sys "$SIGMA_SYS"
  --nwalkers "$NWALKERS"
  --steps "$STEPS_FULL"
  --seed 42
)
if [[ -n "$H0_KMS_MPC" ]]; then
  CMD_FULL+=(--H0-kms-mpc "$H0_KMS_MPC")
fi

# Decide background mode
if [[ "$BG_MODE" == "auto" ]]; then
  if command -v tmux >/dev/null 2>&1; then
    BG_MODE="tmux"
  else
    BG_MODE="nohup"
  fi
fi

log "[4/4] Full run -> $OUT_FULL (mode: $BG_MODE)"

case "$BG_MODE" in
  fg)
    "${CMD_FULL[@]}"
    log "Full run completed in foreground."
    ;;
  tmux)
    SESSION="sparcfit_$(date +%Y%m%d_%H%M%S)"
    tmux new-session -d -s "$SESSION" "source \"$REPO_ROOT/.venv/bin/activate\" 2>/dev/null || true; cd \"$REPO_ROOT\"; ${CMD_FULL[*]}; echo DONE; sleep 10"
    log "Started in tmux session: $SESSION"
    log "Attach with: tmux attach -t $SESSION"
    ;;
  nohup)
    LOGFILE="$LOG_DIR/fit_full_$(date +%F_%H%M%S).log"
    # Use login shell to ensure venv activation works in subshells
    nohup bash -lc "source \"$REPO_ROOT/.venv/bin/activate\" 2>/dev/null || true; cd \"$REPO_ROOT\"; ${CMD_FULL[*]}" \
      > "$LOGFILE" 2>&1 &
    PID=$!
    log "Started with nohup (pid=$PID). Tail logs with:"
    log "  tail -f \"$LOGFILE\""
    ;;
  *)
    log "ERROR: Unknown BG mode: $BG_MODE"
    exit 1
    ;;
esac

log "Pipeline launched. Outputs:"
log "  Smoke: $OUT_SMOKE/{global_summary.yaml, per_galaxy_summary.csv, plot_*.png}"
log "  Full : $OUT_FULL/{global_summary.yaml, per_galaxy_summary.csv, plot_*.png} (after completion)"
