# -*- coding: utf-8 -*-
"""
Pipeline for building S4G disk scale-height catalog (with distances) and
merging it into SPARC rows.

新增要點：
- build-h：從 S4G Table A1 讀入 Dist (Mpc) 與 hz (arcsec)，輸出含 D_Mpc_h、h_kpc 等欄位。
- merge-sparc-h：保留 D_Mpc_h 到合併檔，供後續 DIST-INV 使用。
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

ARCS2RAD = 1.0 / 206265.0  # arcsec -> rad
EPS = 1e-12


# ---------------------------
# helpers
# ---------------------------
def _canon_name(s: str) -> str:
    """Canonicalize galaxy names for joining across catalogs."""
    if pd.isna(s):
        return ""
    s = str(s).upper().strip()
    # normalize common prefixes/spaces/dashes
    s = re.sub(r"[\s\-_]+", "", s)
    s = s.replace("ESO", "ESO").replace("NGC", "NGC").replace("UGC", "UGC").replace("IC", "IC")
    s = s.replace("MRK", "MRK").replace("PGC", "PGC")
    # common Cyrillic/Unicode lookalikes
    s = s.replace("О", "O").replace("А", "A")
    return s


def _read_aliases(path: Path | None) -> Dict[str, str]:
    """Read two-column alias csv/tsv: alias -> canonical."""
    if path is None or not Path(path).exists():
        return {}
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        raise ValueError("aliases file must have at least 2 columns (alias, canonical)")
    a, b = df.columns[:2]
    m = {}
    for _, r in df.iterrows():
        key = _canon_name(r[a])
        val = _canon_name(r[b])
        if key and val:
            m[key] = val
    return m


def _mark_outliers(df: pd.DataFrame) -> Tuple[pd.Series, List[str]]:
    """Flag suspicious rows; return boolean mask and reasons (first reason per row)."""
    reasons = []
    flags = []
    for _, r in df.iterrows():
        reason = None
        if not np.isfinite(r["D_Mpc_h"]) or r["D_Mpc_h"] <= 0:
            reason = "bad_dist"
        elif not np.isfinite(r["h_arcsec"]) or r["h_arcsec"] <= 0:
            reason = "bad_hz"
        elif r["D_Mpc_h"] < 1.0:
            reason = "very_near(<1Mpc)"
        elif r["D_Mpc_h"] > 90.0:
            reason = "very_far(>90Mpc)"
        else:
            # relative err too large (use e_hz1/~upper and |e_hz2|/~lower)
            e_hi = r.get("hz_e_hi_arcsec", np.nan)
            e_lo = r.get("hz_e_lo_arcsec", np.nan)
            if np.isfinite(e_hi) and r["h_arcsec"] > 0 and e_hi / r["h_arcsec"] > 1.25:
                reason = "huge_err_hi"
            elif np.isfinite(e_lo) and r["h_arcsec"] > 0 and e_lo / r["h_arcsec"] > 1.25:
                reason = "huge_err_lo"
        reasons.append(reason if reason else "")
        flags.append(True if reason else False)
    return pd.Series(flags, index=df.index), reasons


# ---------------------------
# build-h
# ---------------------------
def cmd_build_h(args: argparse.Namespace) -> None:
    src = Path(args.src)
    out = Path(args.out)
    outliers = Path(args.outliers) if args.outliers else None

    # read TSV (Vizier export) with commented header lines
    df = pd.read_csv(src, sep=r"\s+", comment="#", engine="python")
    # Ensure required columns exist
    for col in ["Galaxy", "Dist", "hz", "e_hz1", "e_hz2"]:
        if col not in df.columns:
            raise RuntimeError(f"Missing column '{col}' in {src}")

    # rename and compute geometry
    df = df.rename(columns={"Dist": "D_Mpc_h", "hz": "h_arcsec", "e_hz1": "hz_e_hi_arcsec", "e_hz2": "hz_e_lo_arcsec"})
    # Note: e_hz2 在原表用負號表「向下偏差」，我們在數值上取絕對值紀錄。
    df["hz_e_lo_arcsec"] = df["hz_e_lo_arcsec"].abs()

    # linear scale per arcsec at distance D (kpc per arcsec) ~ D_kpc * arcs2rad
    D_kpc = df["D_Mpc_h"] * 1e3
    kpc_per_arcsec = D_kpc * ARCS2RAD
    df["h_kpc"] = df["h_arcsec"] * kpc_per_arcsec
    df["h_kpc_err_hi"] = df["hz_e_hi_arcsec"] * kpc_per_arcsec
    df["h_kpc_err_lo"] = df["hz_e_lo_arcsec"] * kpc_per_arcsec

    # canonical key
    df["galaxy_h"] = df["Galaxy"].astype(str)
    df["gkey"] = df["galaxy_h"].map(_canon_name)

    # outliers
    with pd.option_context("mode.use_inf_as_na", True):
        mask, reasons = _mark_outliers(df)
    df["h_outlier_reason"] = reasons

    # Save main catalog (keep useful columns)
    keep = [
        "gkey", "galaxy_h", "D_Mpc_h",
        "h_arcsec", "hz_e_hi_arcsec", "hz_e_lo_arcsec",
        "h_kpc", "h_kpc_err_hi", "h_kpc_err_lo",
        "h_outlier_reason",
    ]
    df_out = df[keep].copy()
    df_out.to_csv(out, index=False)
    print(f"Saved -> {out} (rows={len(df_out)})")

    # Save outliers file if requested
    if outliers is not None:
        df_ol = df_out[df_out["h_outlier_reason"] != ""].copy()
        df_ol.to_csv(outliers, index=False)
        print(f"Outliers -> {outliers} (rows={len(df_ol)})")


# ---------------------------
# merge-sparc-h
# ---------------------------
def _detect_sparc_key_column(df: pd.DataFrame) -> str:
    """Best-effort detect the SPARC galaxy key column."""
    for cand in ["galaxy", "galaxy_key", "Galaxy", "name"]:
        if cand in df.columns:
            return cand
    # fallback: try first column
    return df.columns[0]


def cmd_merge_sparc_h(args: argparse.Namespace) -> None:
    sparc = Path(args.sparc)
    hcat = Path(args.h)
    out = Path(args.out)
    alias = Path(args.alias) if args.alias else None
    unmatched = Path(args.unmatched) if args.unmatched else None

    df_s = pd.read_csv(sparc)
    key_col = _detect_sparc_key_column(df_s)
    df_s["galaxy_s"] = df_s[key_col].astype(str)
    df_s["gkey"] = df_s["galaxy_s"].map(_canon_name)

    # apply alias mapping if provided
    amap = _read_aliases(alias)
    if amap:
        df_s["gkey_alias"] = df_s["gkey"].map(lambda x: amap.get(x, x))
        df_s["gkey"] = df_s["gkey_alias"]

    df_h = pd.read_csv(hcat)
    if "gkey" not in df_h.columns:
        raise RuntimeError(f"{hcat} missing 'gkey' (did you run build-h from this version?)")

    # keep minimal set from H catalog (crucially D_Mpc_h + h values)
    use_h_cols = ["gkey", "galaxy_h", "D_Mpc_h", "h_kpc", "h_kpc_err_hi", "h_kpc_err_lo", "h_arcsec",
                  "hz_e_hi_arcsec", "hz_e_lo_arcsec", "h_outlier_reason"]
    df_h = df_h[use_h_cols].copy()

    # left-join SPARC rows to H catalog
    df_m = df_s.merge(df_h, how="left", on="gkey", suffixes=("", "_hcat"))

    # basic report
    matched_rows = df_m["h_kpc"].notna().sum()
    total_rows = len(df_m)
    print(f"Matched {matched_rows} / {total_rows} rows")

    if unmatched is not None:
        not_match = df_m[df_m["h_kpc"].isna()][["gkey", "galaxy_s"]].drop_duplicates()
        not_match.to_csv(unmatched, index=False)
        print(f"Unmatched galaxy keys -> {unmatched} (unique={len(not_match)})")

    df_m.to_csv(out, index=False)
    print(f"Saved -> {out}")


# ---------------------------
# CLI
# ---------------------------
def main():
    p = argparse.ArgumentParser(prog="ptq.data.s4g_h_pipeline", description="S4G h-catalog build & merge")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build-h", help="Build h-catalog (with D_Mpc_h) from S4G Table A1 TSV")
    p_build.add_argument("--src", type=str, required=True, help="S4G Table A1 TSV (Vizier export)")
    p_build.add_argument("--out", type=str, required=True, help="Output CSV for h-catalog")
    p_build.add_argument("--outliers", type=str, default=None, help="Optional CSV for flagged outliers")
    p_build.set_defaults(func=cmd_build_h)

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
