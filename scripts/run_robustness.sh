#!/usr/bin/env bash
set -euo pipefail

DATA="${1:-dataset/sparc_tidy.csv}"
BASE_RES="${2:-results/ejpc_run_20251010_122221/ptq-screen_gauss}"
OL="${3:-0.685}"
ETA="${4:-0.15}"

echo "[robust] DATA=$DATA"
echo "[robust] BASE_RES=$BASE_RES  OmegaLambda=$OL  eta=$ETA"

# 1) 內半徑遮罩 rmin sensitivity
OUTROOT="results/mask_scan"
for RMIN in 0 1 2 3; do
  echo "[mask] rmin=$RMIN kpc"
  ptquat exp mask --data "$DATA" --outroot "$OUTROOT" --model ptq-screen \
    --rmin-kpc "$RMIN" --likelihood gauss --t-dof 8
  RESDIR="$OUTROOT/ptq-screen_rmin_$(printf %.2f "$RMIN")kpc"

  ptquat exp kappa-prof --results "$RESDIR" --data "$DATA" \
    --eta "$ETA" --eps-norm cos --omega-lambda "$OL" \
    --x-kind r_over_Rd --prefix "kappa_profile_rmin_${RMIN}"

  ptquat exp kappa-fit --results "$RESDIR" --prefix "kappa_profile_rmin_${RMIN}" \
    --eps-norm cos --omega-lambda "$OL" --bootstrap 1000 --seed 1234
done

# 彙整成一個 CSV
python - <<'PY'
import json, glob, os, pandas as pd
rows=[]
for js in glob.glob("results/mask_scan/ptq-screen_rmin_*kpc/kappa_profile_rmin_*_abfit.json"):
    with open(js) as f: d=json.load(f)
    rmin = float(js.split("rmin_")[1].split("_")[0])
    rows.append(dict(rmin_kpc=rmin, eta_hat=d["eta_hat"], B_cos=d["B_cos"]))
pd.DataFrame(rows).sort_values("rmin_kpc").to_csv("results/mask_scan/kappa_ab_vs_rmin.csv", index=False)
print("Saved results/mask_scan/kappa_ab_vs_rmin.csv")
PY

# 2) OmegaLambda（epsilon_cos）敏感度（不需重跑全域擬合）
for OLv in 0.665 0.685 0.705; do
  PREF="kappa_profile_ol${OLv}"
  ptquat exp kappa-prof --results "$BASE_RES" --data "$DATA" \
    --eta "$ETA" --eps-norm cos --omega-lambda "$OLv" \
    --x-kind r_over_Rd --prefix "$PREF"
  ptquat exp kappa-fit --results "$BASE_RES" --prefix "$PREF" \
    --eps-norm cos --omega-lambda "$OLv" --bootstrap 800 --seed 66
done

python - <<'PY'
import json, glob, pandas as pd
rows=[]
for js in glob.glob("results/ejpc_run_*/ptq-screen_gauss/kappa_profile_ol*_abfit.json"):
    with open(js) as f: d=json.load(f)
    ol = float(d["epsilon_cos"]**2/(1+d["epsilon_cos"]**2))
    rows.append(dict(omega_lambda=ol, eta_hat=d["eta_hat"], B_cos=d["B_cos"]))
pd.DataFrame(rows).sort_values("omega_lambda").to_csv("results/omega_scan/kappa_ab_vs_omega.csv", index=False)
print("Saved results/omega_scan/kappa_ab_vs_omega.csv")
PY
