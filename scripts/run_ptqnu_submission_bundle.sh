#!/usr/bin/env bash
set -euo pipefail

# --------------------------------------------------------------------
# run_ptqnu_submission_bundle.sh
#
# Purpose:
#   1. Rebuild summary tables for:
#      - six-model baseline
#      - four-model multi-seed robustness
#   2. Run the additional ptq-nu-focused experiments needed to make ptq-nu
#      the "hero" model in the manuscript:
#      - closure / closure-scan
#      - plateau
#      - kappa-gal (obs + model variants)
#      - kappa-profile + bootstrap fit
#      - z-profile
#      - optional kappa_h thickness regression
#   3. Assemble a clean submission bundle directory.
#
# Notes:
#   - Assumes baseline and robustness core runs already exist.
#   - Does NOT rerun the expensive six-model baseline or multi-seed fits.
#   - Stops on first error (set -euo pipefail).
# --------------------------------------------------------------------

cd ~/PTQ-Quaternionic-RC
source .venv/bin/activate

# -----------------------------
# Core paths
# -----------------------------
export BASE=results/revision_baseline_sixmodels_12000
export HERO="$BASE/ptq-nu_gauss"
export ROB=results/revision_robustness_core_multiseed12000
export SUB=results/paper_submission_ptqnu_20260323

mkdir -p "$BASE/logs"
mkdir -p "$ROB/logs"
mkdir -p "$SUB"

echo "[INFO] run_ptqnu_submission_bundle start"
echo "[INFO] BASE=$BASE"
echo "[INFO] HERO=$HERO"
echo "[INFO] ROB=$ROB"
echo "[INFO] SUB=$SUB"
echo "[INFO] git commit: $(git rev-parse HEAD)" | tee "$SUB/git_commit.txt"
python --version | tee "$SUB/python_version.txt"

# -----------------------------
# Sanity checks
# -----------------------------
need_file() {
  local f="$1"
  if [[ ! -e "$f" ]]; then
    echo "[ERROR] missing required file: $f"
    exit 1
  fi
}

need_file "$HERO/global_summary.yaml"
need_file "$BASE/paper_extra/model_compare_baseline/compare_table.csv"
need_file "$BASE/paper_extra/loo_compare_baseline/loo_table.csv"
need_file "$ROB/seed_0/paper_extra/model_compare_core/compare_table.csv"
need_file dataset/sparc_tidy.csv

# -----------------------------
# Helper: run only if output missing
# -----------------------------
run_if_missing() {
  local out="$1"
  shift
  if [[ -e "$out" ]]; then
    echo "[INFO] Skip existing: $out"
  else
    echo "[INFO] Running: $*"
    "$@"
  fi
}

# -----------------------------
# 0. Refresh aggregate summaries
# -----------------------------
echo "[INFO] Step 0: summarize baseline and robustness"
python scripts/summarize_baseline_sixmodels.py \
  2>&1 | tee "$BASE/logs/summarize_baseline_sixmodels.log"

python scripts/summarize_robustness_core_multiseed.py \
  2>&1 | tee "$ROB/logs/summarize_robustness_core_multiseed.log"

# -----------------------------
# 1. ptq-nu closure / closure scan
# -----------------------------
echo "[INFO] Step 1: ptq-nu closure"

run_if_missing \
  "$HERO/closure_test.yaml" \
  ptquat exp closure \
    --results "$HERO" \
    --omega-lambda 0.69 \
  2>&1 | tee "$BASE/logs/ptqnu_closure.log"

mkdir -p "$HERO/closure_scan"
run_if_missing \
  "$HERO/closure_scan/closure_sensitivity.csv" \
  ptquat exp closure-scan \
    --results "$HERO" \
    --outdir "$HERO/closure_scan" \
    --omega-lambda-min 0.65 \
    --omega-lambda-max 0.75 \
    --n 11 \
  2>&1 | tee "$BASE/logs/ptqnu_closure_scan.log"

# -----------------------------
# 2. ptq-nu plateau
# -----------------------------
echo "[INFO] Step 2: ptq-nu plateau"

run_if_missing \
  "$HERO/plateau.png" \
  ptquat exp plateau \
    --results "$HERO" \
    --data dataset/sparc_tidy.csv \
  2>&1 | tee "$BASE/logs/ptqnu_plateau.log"

# -----------------------------
# 3. ptq-nu kappa-gal (obs-debiased, r* from obs)
# -----------------------------
echo "[INFO] Step 3: ptq-nu kappa-gal (obs)"

run_if_missing \
  "$HERO/kappa_gal_obsdebiased_cos_f095_summary.json" \
  ptquat exp kappa-gal \
    --results "$HERO" \
    --data dataset/sparc_tidy.csv \
    --y-source obs-debias \
    --eps-norm cos \
    --rstar-from obs \
    --frac-vmax 0.95 \
    --regression deming \
    --deming-lambda 1.0 \
    --prefix kappa_gal_obsdebiased_cos_f095 \
  2>&1 | tee "$BASE/logs/ptqnu_kappa_gal_obs.log"

# -----------------------------
# 4. ptq-nu kappa-gal (obs-debiased, r* from model)
# -----------------------------
echo "[INFO] Step 4: ptq-nu kappa-gal (model-rstar)"

run_if_missing \
  "$HERO/kappa_gal_obsdebiased_cos_rstar_model_summary.json" \
  ptquat exp kappa-gal \
    --results "$HERO" \
    --data dataset/sparc_tidy.csv \
    --y-source obs-debias \
    --eps-norm cos \
    --rstar-from model \
    --frac-vmax 0.90 \
    --regression deming \
    --deming-lambda 1.0 \
    --prefix kappa_gal_obsdebiased_cos_rstar_model \
  2>&1 | tee "$BASE/logs/ptqnu_kappa_gal_model.log"

# -----------------------------
# 5. ptq-nu kappa-profile
# -----------------------------
echo "[INFO] Step 5: ptq-nu kappa-profile"

run_if_missing \
  "$HERO/kappa_profile_cos.png" \
  ptquat exp kappa-prof \
    --results "$HERO" \
    --data dataset/sparc_tidy.csv \
    --eta 0.15 \
    --nbins 24 \
    --min-per-bin 20 \
    --x-kind r_over_Rd \
    --eps-norm cos \
    --prefix kappa_profile_cos \
  2>&1 | tee "$BASE/logs/ptqnu_kappa_prof.log"

run_if_missing \
  "$HERO/kappa_profile_cos_fit_summary.json" \
  ptquat exp kappa-fit \
    --results "$HERO" \
    --prefix kappa_profile_cos \
    --eps-norm cos \
    --bootstrap 2000 \
    --seed 1234 \
  2>&1 | tee "$BASE/logs/ptqnu_kappa_fit.log"

# -----------------------------
# 6. ptq-nu z-profile
# -----------------------------
echo "[INFO] Step 6: ptq-nu z-profile"

run_if_missing \
  "$HERO/z_profile.png" \
  ptquat exp zprof \
    --results "$HERO" \
    --data dataset/sparc_tidy.csv \
    --nbins 24 \
    --min-per-bin 20 \
    --eps-norm cos \
    --prefix z_profile \
  2>&1 | tee "$BASE/logs/ptqnu_zprof.log"

# -----------------------------
# 7. Optional: thickness regression (kappa_h)
# -----------------------------
# Set RUN_KAPPA_H=0 to skip this part.
# Default: run it.
# -----------------------------
RUN_KAPPA_H="${RUN_KAPPA_H:-1}"

if [[ "$RUN_KAPPA_H" == "1" ]]; then
  echo "[INFO] Step 7: ptq-nu kappa_h thickness regression"

  need_file dataset/geometry/sparc_with_h.csv

  KAPPA_H_DONE="$HERO/kappa_h_report_ptqnu.json"
  if [[ -e "$KAPPA_H_DONE" ]]; then
    echo "[INFO] Skip existing: $KAPPA_H_DONE"
  else
    if python -m ptq.experiments.kappa_h --help 2>/dev/null | grep -q -- '--plot-out'; then
      python -m ptq.experiments.kappa_h \
        --sparc-with-h dataset/geometry/sparc_with_h.csv \
        --per-galaxy \
        --rstar vdisk-peak \
        --wls \
        --ml36 0.5 \
        --rgas-mult 1.7 \
        --gas-helium 1.33 \
        --loo \
        --bootstrap 5000 \
        --cv-by-galaxy \
        --out-csv "$HERO/kappa_h_used_ptqnu.csv" \
        --report-json "$HERO/kappa_h_report_ptqnu.json" \
        --plot-out "$HERO" \
      2>&1 | tee "$BASE/logs/ptqnu_kappa_h.log"
    elif python -m ptq.experiments.kappa_h --help 2>/dev/null | grep -q -- '--out-plot'; then
      python -m ptq.experiments.kappa_h \
        --sparc-with-h dataset/geometry/sparc_with_h.csv \
        --per-galaxy \
        --rstar vdisk-peak \
        --wls \
        --ml36 0.5 \
        --rgas-mult 1.7 \
        --gas-helium 1.33 \
        --loo \
        --bootstrap 5000 \
        --cv-by-galaxy \
        --out-csv "$HERO/kappa_h_used_ptqnu.csv" \
        --report-json "$HERO/kappa_h_report_ptqnu.json" \
        --out-plot "$HERO/kappa_h_scatter.png" \
      2>&1 | tee "$BASE/logs/ptqnu_kappa_h.log"
    else
      echo "[WARN] Could not detect kappa_h plotting flag; skip kappa_h."
    fi
  fi
else
  echo "[INFO] Step 7 skipped: RUN_KAPPA_H=$RUN_KAPPA_H"
fi

# -----------------------------
# 8. Build submission bundle
# -----------------------------
echo "[INFO] Step 8: build submission bundle"

mkdir -p "$SUB"/{00_inputs,01_main_baseline,02_ptqnu_main,03_figures,04_robustness,05_provenance,06_optional}

# Inputs
cp dataset/sparc_tidy.csv "$SUB/00_inputs/"
cp dataset/geometry/sparc_with_h.csv "$SUB/00_inputs/" 2>/dev/null || true
cp dataset/geometry/h_catalog.csv "$SUB/00_inputs/" 2>/dev/null || true

# Main baseline
cp "$BASE/paper_extra/model_compare_baseline/compare_table.csv" "$SUB/01_main_baseline/baseline_compare_table.csv"
cp "$BASE/paper_extra/model_compare_baseline/compare_table.tex" "$SUB/01_main_baseline/baseline_compare_table.tex" 2>/dev/null || true
cp "$BASE/paper_extra/model_compare_baseline/breakdown.csv" "$SUB/01_main_baseline/baseline_breakdown.csv"
cp "$BASE/paper_extra/loo_compare_baseline/loo_table.csv" "$SUB/01_main_baseline/baseline_loo_table.csv"
cp "$BASE/paper_run/ejpc_model_compare.csv" "$SUB/01_main_baseline/baseline_model_compare.csv" 2>/dev/null || true
cp "$BASE/aggregate/baseline_sixmodels_summary.md" "$SUB/01_main_baseline/" 2>/dev/null || true
cp "$BASE/aggregate/baseline_sixmodels_table.tex" "$SUB/01_main_baseline/" 2>/dev/null || true

# Hero model
cp "$HERO/global_summary.yaml" "$SUB/02_ptqnu_main/ptqnu_global_summary.yaml"
cp "$HERO/per_galaxy_summary.csv" "$SUB/02_ptqnu_main/ptqnu_per_galaxy_summary.csv"

cp "$HERO/closure_test.yaml" "$SUB/02_ptqnu_main/" 2>/dev/null || true
cp -r "$HERO/closure_scan" "$SUB/02_ptqnu_main/" 2>/dev/null || true

# Figures / diagnostics
cp "$HERO/plateau.png" "$SUB/03_figures/" 2>/dev/null || true
cp "$HERO/plateau_binned.csv" "$SUB/03_figures/" 2>/dev/null || true
cp "$HERO/plateau_per_point.csv" "$SUB/03_figures/" 2>/dev/null || true

cp "$HERO"/kappa_gal_obsdebiased_cos_f095* "$SUB/03_figures/" 2>/dev/null || true
cp "$HERO"/kappa_gal_obsdebiased_cos_rstar_model* "$SUB/03_figures/" 2>/dev/null || true
cp "$HERO"/kappa_profile_cos* "$SUB/03_figures/" 2>/dev/null || true
cp "$HERO"/z_profile* "$SUB/03_figures/" 2>/dev/null || true

cp "$HERO"/kappa_h_used_ptqnu.csv "$SUB/03_figures/" 2>/dev/null || true
cp "$HERO"/kappa_h_report_ptqnu.json "$SUB/03_figures/" 2>/dev/null || true
cp "$HERO"/kappa_h_scatter*.png "$SUB/03_figures/" 2>/dev/null || true

# Robustness
cp "$ROB/aggregate/multiseed_compare_metrics_long.csv" "$SUB/04_robustness/" 2>/dev/null || true
cp "$ROB/aggregate/multiseed_fit_metrics_long.csv" "$SUB/04_robustness/" 2>/dev/null || true
cp "$ROB/aggregate/multiseed_model_summary.csv" "$SUB/04_robustness/" 2>/dev/null || true
cp "$ROB/aggregate/multiseed_rank_stability.csv" "$SUB/04_robustness/" 2>/dev/null || true
cp "$ROB/aggregate/multiseed_robustness_table.tex" "$SUB/04_robustness/" 2>/dev/null || true
cp "$ROB/aggregate/multiseed_summary.md" "$SUB/04_robustness/" 2>/dev/null || true

cp "$ROB/seed_0/paper_extra/model_compare_core/compare_table.csv" "$SUB/04_robustness/seed0_compare_table.csv" 2>/dev/null || true
cp "$ROB/seed_0/paper_extra/loo_compare_core/loo_table.csv" "$SUB/04_robustness/seed0_loo_table.csv" 2>/dev/null || true

cp "$ROB/seed_7/paper_extra/model_compare_core/compare_table.csv" "$SUB/04_robustness/seed7_compare_table.csv" 2>/dev/null || true
cp "$ROB/seed_7/paper_extra/loo_compare_core/loo_table.csv" "$SUB/04_robustness/seed7_loo_table.csv" 2>/dev/null || true

cp "$ROB/seed_13/paper_extra/model_compare_core/compare_table.csv" "$SUB/04_robustness/seed13_compare_table.csv" 2>/dev/null || true
cp "$ROB/seed_13/paper_extra/loo_compare_core/loo_table.csv" "$SUB/04_robustness/seed13_loo_table.csv" 2>/dev/null || true

mkdir -p "$SUB/04_robustness/pwaic_diagnostics_seed0"
mkdir -p "$SUB/04_robustness/pwaic_diagnostics_seed7"
mkdir -p "$SUB/04_robustness/pwaic_diagnostics_seed13"

cp -r "$ROB/seed_0/paper_extra/model_compare_core/pwaic_diagnostics/"* "$SUB/04_robustness/pwaic_diagnostics_seed0/" 2>/dev/null || true
cp -r "$ROB/seed_7/paper_extra/model_compare_core/pwaic_diagnostics/"* "$SUB/04_robustness/pwaic_diagnostics_seed7/" 2>/dev/null || true
cp -r "$ROB/seed_13/paper_extra/model_compare_core/pwaic_diagnostics/"* "$SUB/04_robustness/pwaic_diagnostics_seed13/" 2>/dev/null || true

# Provenance
cp "$BASE/logs/git_commit.txt" "$SUB/05_provenance/baseline_git_commit.txt" 2>/dev/null || true
cp "$BASE/logs/python_version.txt" "$SUB/05_provenance/baseline_python_version.txt" 2>/dev/null || true
cp "$ROB/git_commit.txt" "$SUB/05_provenance/robustness_git_commit.txt" 2>/dev/null || true
cp "$ROB/python_version.txt" "$SUB/05_provenance/robustness_python_version.txt" 2>/dev/null || true
cp "$ROB/run_robustness_core_multiseed12000.nohup.out" "$SUB/05_provenance/" 2>/dev/null || true
cp "$SUB/git_commit.txt" "$SUB/05_provenance/submission_git_commit.txt"
cp "$SUB/python_version.txt" "$SUB/05_provenance/submission_python_version.txt"

cat > "$SUB/05_provenance/MANIFEST.md" <<'EOF'
# Submission bundle manifest

## Main character
ptq-nu

## Main baseline source
results/revision_baseline_sixmodels_12000

## Robustness source
results/revision_robustness_core_multiseed12000

## Included sections
- 00_inputs: core datasets
- 01_main_baseline: six-model baseline comparison tables
- 02_ptqnu_main: ptq-nu global summary and closure outputs
- 03_figures: plateau / kappa / z-profile / thickness outputs
- 04_robustness: multi-seed robustness summaries and per-seed compare/loo tables
- 05_provenance: git/python/log provenance

## Notes
This bundle is organized for manuscript drafting and submission support.
Large chain.h5 files and all per-galaxy plot PNGs are intentionally not duplicated here.
EOF

# Optional archive
tar -czf "${SUB}.tar.gz" -C "$(dirname "$SUB")" "$(basename "$SUB")"

echo "[INFO] Submission bundle created:"
echo "[INFO]   $SUB"
echo "[INFO]   ${SUB}.tar.gz"
echo "[INFO] run_ptqnu_submission_bundle finished"