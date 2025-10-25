# src/ptq/data/s4g_h_pipeline.py
import argparse
import math
import os
import re
from typing import Optional, Tuple
import pandas as pd

# arcsec -> kpc 的常數（每 1 arcsec 在 1 Mpc 的 kpc）
K = math.pi / 648000.0 * 1000.0  # 0.00484813681109536 kpc / (arcsec·Mpc)

def _strip_strings(df: pd.DataFrame) -> pd.DataFrame:
    obj_cols = df.select_dtypes(include="object").columns
    if len(obj_cols) > 0:
        df[obj_cols] = df[obj_cols].apply(lambda s: s.str.strip())
    return df

_norm_keep = re.compile(r"[A-Z0-9]+")
def _norm_name(s: str) -> str:
    s = re.sub(r"\s+", " ", str(s)).upper().strip()
    return "".join(_norm_keep.findall(s))

def _maybe_write_parquet(df: pd.DataFrame, path: Optional[str]) -> None:
    if not path:
        return
    try:
        df.to_parquet(path, index=False)
    except Exception:
        pass

def build_h_catalog(
    src_tsv: str,
    out_csv: str,
    out_parquet: Optional[str] = None,
    outliers_csv: Optional[str] = None,
    outliers_ratio: float = 3.0,
) -> Tuple[int, int]:
    df = (
        pd.read_csv(src_tsv, sep="\t", comment="#", dtype=str, encoding="utf-8")
        .rename(columns=str.strip).pipe(_strip_strings)
    )
    for c in ["Dist", "hz", "e_hz1", "e_hz2"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = pd.Series(dtype="float64")
    df = df.dropna(subset=["Galaxy", "Dist", "hz"]).copy()

    df["h_kpc"] = df["hz"] * df["Dist"] * K
    df["h_err_plus_kpc"] = df["e_hz1"].abs() * df["Dist"] * K
    df["h_err_minus_kpc"] = df["e_hz2"].abs() * df["Dist"] * K

    out = df[["Galaxy", "Dist", "hz", "e_hz1", "e_hz2",
              "h_kpc", "h_err_plus_kpc", "h_err_minus_kpc"]].copy()
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    out.to_csv(out_csv, index=False, encoding="utf-8")
    _maybe_write_parquet(out, out_parquet)
    n_rows = len(out)

    if outliers_csv:
        with pd.option_context("mode.use_inf_as_na", True):
            cond = (
                (out["hz"] > 0)
                & (
                    (out["e_hz1"].abs() / out["hz"] > outliers_ratio)
                    | (out["e_hz2"].abs() / out["hz"] > outliers_ratio)
                )
            )
        outliers = out.loc[cond].copy()
        outliers.to_csv(outliers_csv, index=False, encoding="utf-8")
        n_out = len(outliers)
    else:
        n_out = 0

    return n_rows, n_out

def merge_sparc_with_h(
    sparc_csv: str,
    h_csv: str,
    out_csv: str,
    alias_csv: Optional[str] = None,
    unmatched_csv: Optional[str] = None,
    out_parquet: Optional[str] = None,
) -> Tuple[int, int, int]:
    """
    合併 SPARC 與 h 表；回傳：(配對成功列數, 總列數, 未配對 galaxy 數)
    並在輸出中加掛 galaxy-level 欄位：
      D_Mpc_gal (median by gkey)
      L36_tot_gal / L36_bulge_gal / L36_disk_gal / M_HI_gal
    """
    h = pd.read_csv(h_csv, encoding="utf-8")
    if "Galaxy" not in h.columns:
        raise ValueError(f"{h_csv} 缺少 'Galaxy' 欄位")
    h["gkey"] = h["Galaxy"].map(_norm_name)

    sp = pd.read_csv(sparc_csv, encoding="utf-8")
    if "galaxy" not in sp.columns:
        raise ValueError(f"{sparc_csv} 缺少 'galaxy' 欄位")
    sp["gkey"] = sp["galaxy"].map(_norm_name)

    if alias_csv and os.path.exists(alias_csv):
        alias = pd.read_csv(alias_csv, encoding="utf-8")
        if not {"galaxy", "h_galaxy"}.issubset(alias.columns):
            raise ValueError(f"{alias_csv} 需包含 'galaxy','h_galaxy'")
        alias["s_gkey"] = alias["galaxy"].map(_norm_name)
        alias["h_gkey"] = alias["h_galaxy"].map(_norm_name)
        sp = sp.merge(alias[["s_gkey", "h_gkey"]],
                      left_on="gkey", right_on="s_gkey", how="left")
        sp["gkey"] = sp["h_gkey"].fillna(sp["gkey"])
        sp = sp.drop(columns=["s_gkey", "h_gkey"])

    m = sp.merge(
        h[["gkey", "h_kpc", "h_err_plus_kpc", "h_err_minus_kpc"]],
        on="gkey", how="left",
    )

    # galaxy-level 欄位
    g = (
        sp.groupby("gkey", as_index=False)
          .agg(
              D_Mpc_gal=("D_Mpc", "median"),
              L36_disk_first=("L36_disk", lambda s: s.dropna().iloc[0] if s.notna().any() else float("nan")),
              L36_tot_gal=("L36_tot", "max"),
              L36_bulge_gal=("L36_bulge", "max"),
              M_HI_gal=("M_HI", "max"),
          )
    )
    g["L36_bulge_gal"] = g["L36_bulge_gal"].fillna(0.0)
    g["L36_disk_gal"] = g["L36_disk_first"].fillna(g["L36_tot_gal"] - g["L36_bulge_gal"])
    g = g.drop(columns=["L36_disk_first"])
    m = m.merge(g, on="gkey", how="left")

    matched = int(m["h_kpc"].notna().sum())
    total = len(m)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    m.to_csv(out_csv, index=False, encoding="utf-8")
    _maybe_write_parquet(m, out_parquet)

    if unmatched_csv:
        missing = m.loc[m["h_kpc"].isna(), "galaxy"].dropna().map(_norm_name).drop_duplicates()
        pd.DataFrame({"gkey": missing}).to_csv(unmatched_csv, index=False, encoding="utf-8")
        n_unmatched_galaxies = int(len(missing))
    else:
        n_unmatched_galaxies = 0

    return matched, total, n_unmatched_galaxies

def main() -> None:
    ap = argparse.ArgumentParser(prog="ptq-hz", description="S4G h 轉換與 SPARC 合併工具")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-h", help="從 S4G Table A1 產生 h_catalog")
    b.add_argument("--src", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--parquet", required=False)
    b.add_argument("--outliers", required=False)
    b.add_argument("--outliers-ratio", type=float, default=3.0)

    m = sub.add_parser("merge-sparc-h", help="SPARC 與 h_catalog 合併")
    m.add_argument("--sparc", required=True)
    m.add_argument("--h", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--alias", required=False)
    m.add_argument("--unmatched", required=False)
    m.add_argument("--parquet", required=False)

    args = ap.parse_args()

    if args.cmd == "build-h":
        n_rows, n_out = build_h_catalog(
            src_tsv=args.src, out_csv=args.out, out_parquet=args.parquet,
            outliers_csv=args.outliers, outliers_ratio=args.outliers_ratio,
        )
        print(f"Saved -> {args.out} (rows={n_rows})")
        if args.parquet:
            print(f"Parquet -> {args.parquet}")
        if args.outliers:
            print(f"Outliers -> {args.outliers} (rows={n_out})")
    else:
        matched, total, n_unmatched_g = merge_sparc_with_h(
            sparc_csv=args.sparc, h_csv=args.h, out_csv=args.out,
            alias_csv=args.alias, unmatched_csv=args.unmatched, out_parquet=args.parquet,
        )
        print(f"Matched {matched} / {total} rows")
        if args.unmatched:
            print(f"Unmatched galaxy keys -> {args.unmatched} (unique={n_unmatched_g})")
        print(f"Saved -> {args.out}")
        if args.parquet:
            print(f"Parquet -> {args.parquet}")

if __name__ == "__main__":
    main()
