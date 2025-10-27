# -*- coding: utf-8 -*-
"""
Pipeline for building S4G disk scale-height catalog (with distances) and
merging it into SPARC rows.

This module provides two CLI subcommands:

1) build-h
   - Input:  S4G Table A1 (Vizier export; TSV-like, whitespace separated)
   - Output: h-catalog with columns:
       gkey, galaxy_h, D_Mpc_h,
       h_arcsec, hz_e_hi_arcsec, hz_e_lo_arcsec,
       h_kpc,  h_kpc_err_hi,   h_kpc_err_lo,
       h_outlier_reason
     and an optional outliers CSV (rows with flagged reasons).

   Notes:
   - Distances are assumed in Mpc; vertical scale-heights (hz) in arcsec.
   - Conversion: kpc_per_arcsec = D_Mpc * 1e3 / 206265.
   - e_hz2 in S4G is typically the "lower" (often negative) deviation; we store its magnitude.

2) merge-sparc-h
   - Input:  SPARC tidy CSV (per-radius rows), h-catalog from build-h,
             optional alias mapping (alias -> canonical, 2 columns).
   - Output: merged CSV that keeps SPARC columns and appends the h-catalog columns,
             preserving D_Mpc_h for downstream DIST-INV tests.
   - Also writes an optional list of unmatched galaxy keys.

Design goals / publication-grade notes:
- Canonicalized galaxy keys (case-insensitive, punctuation-insensitive,
  with common Unicode lookalikes normalized) to maximize join robustness.
- Explicit unit conversions; numerical validation; informative outlier tags.
- Deterministic behavior; clear logging of row counts and dropped noise lines.

Example:
    python -m ptq.data.s4g_h_pipeline build-h \
        --src dataset/geometry/s4g_tablea1.tsv \
        --out dataset/geometry/h_catalog.csv \
        --outliers dataset/geometry/h_z_outliers.csv

    python -m ptq.data.s4g_h_pipeline merge-sparc-h \
        --sparc dataset/sparc_tidy.csv \
        --h dataset/geometry/h_catalog.csv \
        --out dataset/geometry/sparc_with_h.csv \
        --alias dataset/geometry/aliases.csv \
        --unmatched dataset/geometry/unmatched_galaxies.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

# ---------------------------
# Constants
# ---------------------------
ARCS2RAD = 1.0 / 206265.0  # arcsec -> rad
EPS = 1e-12


# ---------------------------
# Helpers
# ---------------------------
def _canon_name(s: str) -> str:
    """
    Canonicalize galaxy names for joining across catalogs.
    - Uppercase, trim whitespace
    - Remove spaces, hyphens, underscores, slashes, dots
    - Normalize common Cyrillic/Unicode lookalikes that appear in catalogs
    """
    if pd.isna(s):
        return ""
    s = str(s).upper().strip()
    s = re.sub(r"[ \-_/\.]+", "", s)

    # Normalize common prefixes (kept as-is but ensures uniformity)
    # (Explicit lines left for readability and future extension)
    s = s.replace("ESO", "ESO").replace("NGC", "NGC").replace("UGC", "UGC") \
         .replace("IC", "IC").replace("MRK", "MRK").replace("PGC", "PGC")

    # Cyrillic lookalikes occasionally leak in OCR/exports
    s = s.replace("О", "O").replace("А", "A").replace("В", "B").replace("С", "C")
    return s


def _read_aliases(path: Path | None) -> Dict[str, str]:
    """
    Read two-column alias mapping (alias -> canonical).
    The file may be CSV/TSV/whitespace separated; we let pandas infer the delimiter.
    """
    if path is None or not Path(path).exists():
        return {}
    try:
        df = pd.read_csv(path, sep=None, engine="python")
    except Exception:
        # fallback: try comma
        df = pd.read_csv(path)

    if df.shape[1] < 2:
        raise ValueError("aliases file must have at least 2 columns (alias, canonical)")
    a, b = df.columns[:2]
    m: Dict[str, str] = {}
    for _, r in df.iterrows():
        key = _canon_name(r[a])
        val = _canon_name(r[b])
        if key and val:
            m[key] = val
    return m


def _mark_outliers(
    df: pd.DataFrame,
    dist_min_mpc: float = 1.0,
    dist_max_mpc: float = 90.0,
    rel_err_hi_max: float = 1.25,
    rel_err_lo_max: float = 1.25,
) -> Tuple[pd.Series, List[str]]:
    """
    Flag suspicious rows; return boolean mask and reasons (first reason per row).
    Criteria (conservative; meant for flagging, not hard filtering):
    - Non-finite or non-positive distance/scale height
    - Very near/far distances (outside [1,90] Mpc)
    - Excessive relative error bars on hz (arcsec)
    """
    reasons: List[str] = []
    flags: List[bool] = []
    for _, r in df.iterrows():
        reason = None
        D = r.get("D_Mpc_h", np.nan)
        hz = r.get("h_arcsec", np.nan)
        e_hi = r.get("hz_e_hi_arcsec", np.nan)
        e_lo = r.get("hz_e_lo_arcsec", np.nan)

        if not np.isfinite(D) or D <= 0:
            reason = "bad_dist"
        elif not np.isfinite(hz) or hz <= 0:
            reason = "bad_hz"
        elif D < dist_min_mpc:
            reason = f"very_near(<{dist_min_mpc}Mpc)"
        elif D > dist_max_mpc:
            reason = f"very_far(>{dist_max_mpc}Mpc)"
        else:
            if np.isfinite(e_hi) and hz > 0 and (e_hi / hz) > rel_err_hi_max:
                reason = "huge_err_hi"
            elif np.isfinite(e_lo) and hz > 0 and (e_lo / hz) > rel_err_lo_max:
                reason = "huge_err_lo"

        reasons.append(reason if reason else "")
        flags.append(True if reason else False)

    return pd.Series(flags, index=df.index), reasons


# ---------------------------
# build-h
# ---------------------------
def cmd_build_h(args: argparse.Namespace) -> None:
    """
    Build h-catalog (with D_Mpc_h and h_kpc, error bars, and outlier flags)
    from an S4G Table A1 TSV-like export.
    """
    src = Path(args.src)
    out = Path(args.out)
    outliers = Path(args.outliers) if args.outliers else None

    # Read TSV-like table (Vizier exports are often whitespace-separated)
    # We ignore commented lines and keep robustness against irregular spacing.
    df = pd.read_csv(src, sep=r"\s+", comment="#", engine="python")

    # Required incoming column names from S4G A1
    needed = ["Galaxy", "Dist", "hz", "e_hz1", "e_hz2"]
    for col in needed:
        if col not in df.columns:
            raise RuntimeError(f"Missing column '{col}' in {src}")

    # Normalize column names to our internal schema
    df = df.rename(
        columns={
            "Dist": "D_Mpc_h",
            "hz": "h_arcsec",
            "e_hz1": "hz_e_hi_arcsec",
            "e_hz2": "hz_e_lo_arcsec",
        }
    )

    # Convert to numeric (non-numeric -> NaN) and fix the lower error sign
    for c in ["D_Mpc_h", "h_arcsec", "hz_e_hi_arcsec", "hz_e_lo_arcsec"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["hz_e_lo_arcsec"] = df["hz_e_lo_arcsec"].abs()

    # Drop rows with NaNs in key fields (often unit headers or broken lines)
    before = len(df)
    df = df[np.isfinite(df["D_Mpc_h"]) & np.isfinite(df["h_arcsec"])]
    dropped = before - len(df)
    if dropped > 0:
        print(f"Filtered non-numeric/invalid rows from TSV: dropped {dropped}")

    # Geometry conversions
    # kpc per arcsec = (D in kpc) * ARCS2RAD; D_kpc = D_Mpc * 1e3
    D_kpc = df["D_Mpc_h"] * 1e3
    kpc_per_arcsec = D_kpc * ARCS2RAD
    df["h_kpc"] = df["h_arcsec"] * kpc_per_arcsec
    df["h_kpc_err_hi"] = df["hz_e_hi_arcsec"] * kpc_per_arcsec
    df["h_kpc_err_lo"] = df["hz_e_lo_arcsec"] * kpc_per_arcsec  # <-- fixed typo

    # Keys
    df["galaxy_h"] = df["Galaxy"].astype(str)
    df["gkey"] = df["galaxy_h"].map(_canon_name)

    # Outlier tagging
    _, reasons = _mark_outliers(df)
    df["h_outlier_reason"] = reasons

    # Output
    keep = [
        "gkey",
        "galaxy_h",
        "D_Mpc_h",
        "h_arcsec",
        "hz_e_hi_arcsec",
        "hz_e_lo_arcsec",
        "h_kpc",
        "h_kpc_err_hi",
        "h_kpc_err_lo",
        "h_outlier_reason",
    ]
    df_out = df[keep].copy()
    out.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out, index=False)
    print(f"Saved -> {out} (rows={len(df_out)})")

    if outliers is not None:
        df_ol = df_out[df_out["h_outlier_reason"] != ""].copy()
        df_ol.to_csv(outliers, index=False)
        print(f"Outliers -> {outliers} (rows={len(df_ol)})")


# ---------------------------
# merge-sparc-h
# ---------------------------
def _detect_sparc_key_column(df: pd.DataFrame) -> str:
    """
    Best-effort detection of the SPARC galaxy name/key column.
    Returns the first found among common candidates; falls back to the first column.
    """
    for cand in ["galaxy", "galaxy_key", "Galaxy", "name", "Name"]:
        if cand in df.columns:
            return cand
    return df.columns[0]


def cmd_merge_sparc_h(args: argparse.Namespace) -> None:
    """
    Merge SPARC tidy rows with the h-catalog; propagate D_Mpc_h and scale-height
    columns to every SPARC per-radius row in matched galaxies.
    """
    sparc = Path(args.sparc)
    hcat = Path(args.h)
    out = Path(args.out)
    alias = Path(args.alias) if args.alias else None
    unmatched = Path(args.unmatched) if args.unmatched else None

    # Load SPARC tidy
    df_s = pd.read_csv(sparc)
    key_col = _detect_sparc_key_column(df_s)
    df_s["galaxy_s"] = df_s[key_col].astype(str)
    df_s["gkey"] = df_s["galaxy_s"].map(_canon_name)

    # Apply alias mapping (if provided)
    amap = _read_aliases(alias)
    if amap:
        df_s["gkey_alias"] = df_s["gkey"].map(lambda x: amap.get(x, x))
        df_s["gkey"] = df_s["gkey_alias"]

    # Load h-catalog
    df_h = pd.read_csv(hcat)
    if "gkey" not in df_h.columns:
        raise RuntimeError(f"{hcat} missing 'gkey' (did you run build-h from this version?)")

    # Minimal set to carry forward (crucially includes D_Mpc_h + h values)
    use_h_cols = [
        "gkey",
        "galaxy_h",
        "D_Mpc_h",
        "h_kpc",
        "h_kpc_err_hi",
        "h_kpc_err_lo",
        "h_arcsec",
        "hz_e_hi_arcsec",
        "hz_e_lo_arcsec",
        "h_outlier_reason",
    ]
    df_h = df_h[use_h_cols].copy()

    # Left-join SPARC rows to H catalog
    df_m = df_s.merge(df_h, how="left", on="gkey", suffixes=("", "_hcat"))

    # Basic report
    matched_rows = df_m["h_kpc"].notna().sum()
    total_rows = len(df_m)
    print(f"Matched {matched_rows} / {total_rows} rows")

    if unmatched is not None:
        not_match = df_m[df_m["h_kpc"].isna()][["gkey", "galaxy_s"]].drop_duplicates()
        not_match.to_csv(unmatched, index=False)
        print(f"Unmatched galaxy keys -> {unmatched} (unique={len(not_match)})")

    # Save merged
    df_m.to_csv(out, index=False)
    print(f"Saved -> {out}")


# ---------------------------
# CLI
# ---------------------------
def main():
    p = argparse.ArgumentParser(
        prog="ptq.data.s4g_h_pipeline",
        description="S4G h-catalog build & merge (publication-grade, unit-aware, robust I/O)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # build-h
    p_build = sub.add_parser("build-h", help="Build h-catalog (with D_Mpc_h) from S4G Table A1 TSV")
    p_build.add_argument("--src", type=str, required=True, help="S4G Table A1 TSV (Vizier export)")
    p_build.add_argument("--out", type=str, required=True, help="Output CSV for h-catalog")
    p_build.add_argument("--outliers", type=str, default=None, help="Optional CSV for flagged outliers")
    p_build.set_defaults(func=cmd_build_h)

    # merge-sparc-h
    p_merge = sub.add_parser("merge-sparc-h", help="Merge SPARC rows with S4G h-catalog (carry D_Mpc_h)")
    p_merge.add_argument("--sparc", type=str, required=True, help="SPARC tidy CSV")
    p_merge.add_argument("--h", type=str, required=True, help="h-catalog CSV (built by build-h)")
    p_merge.add_argument("--out", type=str, required=True, help="Output merged CSV")
    p_merge.add_argument("--alias", type=str, default=None, help="Optional two-col alias CSV (alias,canonical)")
    p_merge.add_argument("--unmatched", type=str, default=None, help="Optional CSV to list unmatched galaxy keys")
    p_merge.set_defaults(func=cmd_merge_sparc_h)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
