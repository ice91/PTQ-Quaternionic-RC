# src/ptquat/preprocess.py
from __future__ import annotations
import pandas as pd
from pathlib import Path
import numpy as np
import re

_MAP2 = {
    "Name":   "galaxy",
    "Rad":    "r_kpc",
    "Vobs":   "v_obs_kms",
    "e_Vobs": "v_err_kms",
    "Vgas":   "v_gas_kms",
    "Vdisk":  "v_disk_kms",
    "Vbulge": "v_bulge_kms",
    "Vbul":   "v_bulge_kms",
    "Vblg":   "v_bulge_kms",
}

_MAP1 = {
    "Name": "galaxy",
    "Dist": "D_Mpc",
    "e_Dist": "D_err_Mpc",
    "f_Dist": "f_Dist",
    "i": "i_deg",
    "e_i": "i_err_deg",
    "Qual": "Qual",
}

_DEFAULT_REL_D_BY_FLAG = {1: 0.20, 2: 0.10}
_DEFAULT_I_ERR_DEG = 3.0

def _rename_safe(df, mapping):
    cols = {k: v for k, v in mapping.items() if k in df.columns}
    return df.rename(columns=cols)

def _find_col_regex(cols, must_all: list[str], any_of: list[str] | None = None):
    any_of = any_of or []
    def norm(s): return re.sub(r'[^a-z0-9]+', '', s.lower())
    ncols = {c: norm(c) for c in cols}
    for c, n in ncols.items():
        if all(k in n for k in must_all) and (not any_of or any(k in n for k in any_of)):
            return c
    return None

def build_tidy_csv(
    table1_csv: str | Path,
    table2_csv: str | Path,
    out_csv: str | Path,
    i_min_deg: float = 30.0,
    relD_max: float = 0.2,
    qual_max: int = 2,
) -> Path:
    t1_raw = pd.read_csv(table1_csv)
    t2_raw = pd.read_csv(table2_csv)

    t1 = _rename_safe(t1_raw, _MAP1)
    t2 = _rename_safe(t2_raw, _MAP2)

    need2 = ["galaxy", "r_kpc", "v_obs_kms", "v_err_kms"]
    miss2 = sorted(set(need2) - set(t2.columns))
    if miss2:
        raise ValueError(f"table2 缺少欄位: {miss2}")

    # 缺的重子分量欄位補 0
    for b in ["v_gas_kms","v_disk_kms","v_bulge_kms"]:
        if b not in t2.columns:
            t2[b] = 0.0

    # 距離與誤差
    if "D_Mpc" not in t1.columns:
        if {"Name","Dist"} <= set(t2_raw.columns):
            t1 = t1.merge(
                t2_raw[["Name","Dist"]].rename(columns={"Name":"galaxy","Dist":"D_Mpc"}).drop_duplicates("galaxy"),
                on="galaxy", how="outer"
            )
        else:
            raise ValueError("找不到 D_Mpc（距離）；table1 無 Dist 且 table2 也無 Dist。")

    if "D_err_Mpc" not in t1.columns:
        rel = t1.get("f_Dist", pd.Series(index=t1.index, dtype=float)).map(_DEFAULT_REL_D_BY_FLAG).fillna(0.20)
        t1["D_err_Mpc"] = rel * t1["D_Mpc"]
    if "i_err_deg" not in t1.columns:
        t1["i_err_deg"] = _DEFAULT_I_ERR_DEG

    # 合併並做品質切
    cols1 = ["galaxy","D_Mpc","D_err_Mpc","i_deg","i_err_deg","Qual"]
    for c in cols1:
        if c not in t1.columns:
            t1[c] = 9 if c == "Qual" else np.nan

    tidy = (t2.merge(t1[cols1], on="galaxy", how="left")
              .dropna(subset=["D_Mpc","D_err_Mpc","i_deg","i_err_deg"])
              .assign(relD=lambda d: d["D_err_Mpc"]/d["D_Mpc"])
              .query("i_deg > @i_min_deg and relD <= @relD_max and Qual <= @qual_max")
              .drop(columns=["relD"]))

    # ===== 把 table1 的 3.6µm 總亮度與氫氣質量併入 tidy =====
    t1_aux = t1_raw.rename(columns={"Name":"galaxy"}).copy()
    extra = pd.DataFrame({"galaxy": t1_aux["galaxy"]})

    # L3.6（SPARC/ReadMe 的欄名；單位常為 10^9 Lsun）
    if "L3.6" in t1_aux.columns:
        extra["L36_tot"] = pd.to_numeric(t1_aux["L3.6"], errors="coerce")
    else:
        # 保險：找像 L3.6 的變體
        c = _find_col_regex(t1_aux.columns, must_all=["l","36"])
        if c: extra["L36_tot"] = pd.to_numeric(t1_aux[c], errors="coerce")

    # MHI（單位常為 10^9 Msun）
    if "MHI" in t1_aux.columns:
        extra["M_HI"] = pd.to_numeric(t1_aux["MHI"], errors="coerce")
    else:
        c = _find_col_regex(t1_aux.columns, must_all=["m","hi"])
        if c: extra["M_HI"] = pd.to_numeric(t1_aux[c], errors="coerce")

    # 若 table1 真有分量欄位（少見），也併入
    for cand, outc in [
        (["L36_disk","L3.6_disk","L36,disc","L36_disc"], "L36_disk"),
        (["L36_bulge","L3.6_bulge"], "L36_bulge"),
    ]:
        for col in cand:
            if col in t1_aux.columns:
                extra[outc] = pd.to_numeric(t1_aux[col], errors="coerce")
                break

    tidy = tidy.merge(extra, on="galaxy", how="left")

    # 欄位順序與型別
    base_cols = ["galaxy","r_kpc","v_obs_kms","v_err_kms",
                 "v_disk_kms","v_bulge_kms","v_gas_kms",
                 "D_Mpc","D_err_Mpc","i_deg","i_err_deg"]
    opt_cols = ["L36_tot","M_HI","L36_disk","L36_bulge"]
    for c in opt_cols:
        if c not in tidy.columns:
            tidy[c] = np.nan

    tidy = tidy[base_cols + opt_cols].copy()
    for c in base_cols + opt_cols:
        if c != "galaxy":
            tidy[c] = pd.to_numeric(tidy[c], errors="coerce")

    tidy = tidy.dropna(subset=["r_kpc","v_obs_kms","v_err_kms","D_Mpc","D_err_Mpc","i_deg","i_err_deg"])

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(out_csv, index=False)
    return out_csv