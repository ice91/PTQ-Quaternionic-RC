#!/usr/bin/env bash
set -euo pipefail

SPARC="dataset/geometry/sparc_with_h.csv"
OUTDIR="out/robustness"
mkdir -p "$OUTDIR"

# 掃描參數組合
MLS=("0.45" "0.50" "0.55" "0.60")
RGAS=("1.5" "1.7" "2.0")
EXPFlag="--use-exp"     # 拿掉就改成在 r_char 不衰減

for ml in "${MLS[@]}"; do
  for rg in "${RGAS[@]}"; do
    tag="ml${ml}_rg${rg}"
    python -m ptq.experiments.kappa_h \
      --sparc-with-h "$SPARC" \
      --use-total-sigma ${EXPFlag} \
      --ml36 "${ml}" \
      --rgas-mult "${rg}" \
      --gas-helium 1.33 \
      --wls --loo --bootstrap 2000 \
      --out-csv  "${OUTDIR}/kappa_sigma_${tag}.csv" \
      --out-plot "${OUTDIR}/kappa_sigma_${tag}.png" \
      --report-json "${OUTDIR}/report_${tag}.json"
  done
done
