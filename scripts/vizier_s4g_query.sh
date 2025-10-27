#!/usr/bin/env bash
# scripts/vizier_s4g_query.sh
# One-shot, auditable downloader for S4G Table A1 via VizieR (TSV).
# Produces: dataset/geometry/s4g_tablea1.tsv and a .sha256 checksum.

set -euo pipefail

MIRROR="${VIZIER_MIRROR:-https://vizier.cds.unistra.fr}"
CAT="J/A+A/587/A160"
TABLE="tablea1"
OUT_DIR="dataset/geometry"
OUT_TSV="${OUT_DIR}/s4g_tablea1.tsv"

mkdir -p "${OUT_DIR}"

URL="${MIRROR}/viz-bin/asu-tsv?-source=${CAT}/${TABLE}&-out=Galaxy,Dist,hz,e_hz1,e_hz2&-out.max=unlimited"

echo "[S4G] Downloading: ${URL}"
curl -fsSL "${URL}" -o "${OUT_TSV}"

# Write SHA-256 (portable)
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "${OUT_TSV}" > "${OUT_TSV}.sha256"
elif command -v shasum >/dev/null 2>&1; then
  shasum -a 256 "${OUT_TSV}" > "${OUT_TSV}.sha256"
else
  python - <<'PY'
import hashlib,sys
p="dataset/geometry/s4g_tablea1.tsv"
h=hashlib.sha256(open(p,'rb').read()).hexdigest()
open(p+".sha256","w").write(f"{h}  {p}\n")
print("[S4G] Wrote fallback SHA-256")
PY
fi

echo "[S4G] Saved -> ${OUT_TSV}"
echo "[S4G] SHA256 -> ${OUT_TSV}.sha256"
