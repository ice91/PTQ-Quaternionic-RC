#!/usr/bin/env python3
# scripts/etl_s4g_h.py
# Convert S4G Table A1 (TSV) to h-catalog (kpc) and merge into SPARC tidy.

from __future__ import annotations
import argparse, re, hashlib
from pathlib import Path
import pandas as pd
import numpy as np

ARCS2RAD = 1.0 / 206265.0  # arcsec -> rad

def sha256_write(path: Path):
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    (path.with_suffix(path.suffix + ".sha256")).write_text(f"{h}  {path}\n")
    return h

def canon(s: str) -> str:
    if pd.isna(s): return ""
    s = str(s).upper().strip()
    s = re.sub(r"[ \-_/\.]+", "", s)
    # Cyrillic lookalikes (rare but robust)
    s = s.replace("О","O").replace("А","A").replace("В","B").replace("С","C")
    return s

def load_aliases(p: Path|None):
    if not p or not p.exists(): return {}
    df = pd.read_csv(p, sep=None, engine="python")
    if df.shape[1] < 2:
        raise ValueError("aliases file must have >=2 columns (alias, canonical)")
    a,b = df.columns[:2]
    d={}
    for _,r in df.iterrows():
        k = canon(r[a]); v = canon(r[b])
        if k and v: d[k]=v
    return d

def build_h_catalog(tsv: Path, out_csv: Path, outliers_csv: Path|None=None):
    df = pd.read_csv(tsv, sep=r"\s+", comment="#", engine="python")
    need = ["Galaxy","Dist","hz","e_hz1","e_hz2"]
    for c in need:
        if c not in df.columns:
            raise RuntimeError(f"[S4G] missing column: {c}")

    df = df.rename(columns={
        "Dist":"D_Mpc_h", "hz":"h_arcsec",
        "e_hz1":"hz_e_hi_arcsec", "e_hz2":"hz_e_lo_arcsec"
    })
    for c in ["D_Mpc_h","h_arcsec","hz_e_hi_arcsec","hz_e_lo_arcsec"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["hz_e_lo_arcsec"] = df["hz_e_lo_arcsec"].abs()

    df = df[np.isfinite(df["D_Mpc_h"]) & np.isfinite(df["h_arcsec"])].copy()
    D_kpc = df["D_Mpc_h"] * 1e3
    kpc_per_arcsec = D_kpc * ARCS2RAD
    df["h_kpc"]        = df["h_arcsec"]       * kpc_per_arcsec
    df["h_kpc_err_hi"] = df["hz_e_hi_arcsec"] * kpc_per_arcsec
    df["h_kpc_err_lo"] = df["hz_e_lo_arcsec"] * kpc_per_arcsec

    df["galaxy_h"] = df["Galaxy"].astype(str)
    df["gkey"]     = df["galaxy_h"].map(canon)

    # Conservative outlier tag (for audit; not filtering)
    reasons=[]
    for _,r in df.iterrows():
        reason=""
        if r["D_Mpc_h"]<=0: reason="bad_dist"
        elif r["h_arcsec"]<=0: reason="bad_hz"
        else:
            if pd.notna(r["hz_e_hi_arcsec"]) and r["h_arcsec"]>0 and (r["hz_e_hi_arcsec"]/r["h_arcsec"]>1.25):
                reason="huge_err_hi"
            elif pd.notna(r["hz_e_lo_arcsec"]) and r["h_arcsec"]>0 and (r["hz_e_lo_arcsec"]/r["h_arcsec"]>1.25):
                reason="huge_err_lo"
        reasons.append(reason)
    df["h_outlier_reason"]=reasons

    keep = ["gkey","galaxy_h","D_Mpc_h","h_arcsec","hz_e_hi_arcsec","hz_e_lo_arcsec",
            "h_kpc","h_kpc_err_hi","h_kpc_err_lo","h_outlier_reason"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df[keep].to_csv(out_csv, index=False)
    print(f"[ETL] h-catalog -> {out_csv} (N={len(df)})")
    sha256_write(out_csv)

    if outliers_csv:
        df[df["h_outlier_reason"]!=""][keep].to_csv(outliers_csv, index=False)
        print(f"[ETL] outliers -> {outliers_csv}")

def merge_into_sparc(sparc_csv: Path, h_csv: Path, out_csv: Path,
                     alias_csv: Path|None=None, unmatched_csv: Path|None=None):
    sp = pd.read_csv(sparc_csv)
    # detect key column
    for cand in ["galaxy","galaxy_key","Galaxy","name","Name"]:
        if cand in sp.columns:
            key_col=cand; break
    else:
        key_col = sp.columns[0]
    sp["galaxy_s"]=sp[key_col].astype(str)
    sp["gkey"]=sp["galaxy_s"].map(canon)

    amap = load_aliases(alias_csv) if alias_csv else {}
    if amap:
        sp["gkey"]=sp["gkey"].map(lambda x: amap.get(x,x))

    h = pd.read_csv(h_csv)
    if "gkey" not in h.columns:
        raise RuntimeError("[ETL] h-catalog missing 'gkey'")
    carry = ["gkey","galaxy_h","D_Mpc_h","h_kpc","h_kpc_err_hi","h_kpc_err_lo",
             "h_arcsec","hz_e_hi_arcsec","hz_e_lo_arcsec","h_outlier_reason"]
    h = h[carry].copy()

    m = sp.merge(h, how="left", on="gkey", suffixes=("", "_h"))
    m.to_csv(out_csv, index=False)
    print(f"[ETL] merged -> {out_csv} (matched rows with h: {(~m['h_kpc'].isna()).sum()})")
    sha256_write(out_csv)

    if unmatched_csv:
        um = m[m["h_kpc"].isna()][["gkey","galaxy_s"]].drop_duplicates().sort_values("galaxy_s")
        um.to_csv(unmatched_csv, index=False)
        print(f"[ETL] unmatched -> {unmatched_csv} (unique={len(um)})")

def main():
    ap = argparse.ArgumentParser(description="S4G ETL (TSV->h-catalog->merge SPARC)")
    ap.add_argument("--tsv", default="dataset/geometry/s4g_tablea1.tsv")
    ap.add_argument("--sparc", default="dataset/sparc_tidy.csv")
    ap.add_argument("--aliases", default="dataset/geometry/aliases.csv")
    ap.add_argument("--hcat", default="dataset/geometry/h_catalog.csv")
    ap.add_argument("--out", default="dataset/geometry/sparc_with_h.csv")
    ap.add_argument("--outliers", default="dataset/geometry/h_z_outliers.csv")
    args = ap.parse_args()

    build_h_catalog(Path(args.tsv), Path(args.hcat),
                    Path(args.outliers) if args.outliers else None)
    merge_into_sparc(Path(args.sparc), Path(args.hcat), Path(args.out),
                     Path(args.aliases) if args.aliases else None,
                     Path("dataset/geometry/unmatched_galaxies.csv"))

if __name__ == "__main__":
    main()
