# src/ptquat/preprocess.py
from __future__ import annotations
import pandas as pd
from pathlib import Path
import numpy as np

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

    # 填補缺失的重子貢獻欄位
    for b in ["v_gas_kms","v_disk_kms","v_bulge_kms"]:
        if b not in t2.columns:
            t2[b] = 0.0

    # 處理距離與其不確定度
    if "D_Mpc" not in t1.columns:
        # 試著從 table2 搬運
        if "Dist" in t2_raw.columns and "Name" in t2_raw.columns:
            t1_fallback = t2_raw[["Name","Dist"]].rename(columns={"Name":"galaxy","Dist":"D_Mpc"}).drop_duplicates("galaxy")
            t1 = t1.merge(t1_fallback, on="galaxy", how="outer")
        else:
            raise ValueError("找不到 D_Mpc（距離）；table1 無 Dist 且 table2 也無 Dist。")

    if "D_err_Mpc" not in t1.columns:
        rel = t1.get("f_Dist", pd.Series(index=t1.index, dtype=float)).map(_DEFAULT_REL_D_BY_FLAG).fillna(0.20)
        t1["D_err_Mpc"] = rel * t1["D_Mpc"]

    if "i_err_deg" not in t1.columns:
        t1["i_err_deg"] = _DEFAULT_I_ERR_DEG

    # 合併，套品質切
    cols1 = ["galaxy","D_Mpc","D_err_Mpc","i_deg","i_err_deg","Qual"]
    for c in cols1:
        if c not in t1.columns:
            # 缺的欄給保守值（Qual 給 9 以便被切掉；其他給 NaN）
            t1[c] = 9 if c == "Qual" else np.nan

    tidy = (t2.merge(t1[cols1], on="galaxy", how="left")
              .dropna(subset=["D_Mpc","D_err_Mpc","i_deg","i_err_deg"])
              .assign(relD=lambda d: d["D_err_Mpc"]/d["D_Mpc"])
              .query("i_deg > @i_min_deg and relD <= @relD_max and Qual <= @qual_max")
              .drop(columns=["relD"]))

    # 最終欄位順序與型別
    wanted = ["galaxy","r_kpc","v_obs_kms","v_err_kms",
              "v_disk_kms","v_bulge_kms","v_gas_kms",
              "D_Mpc","D_err_Mpc","i_deg","i_err_deg"]
    tidy = tidy[wanted].copy()
    for c in ["r_kpc","v_obs_kms","v_err_kms","v_disk_kms","v_bulge_kms","v_gas_kms","D_Mpc","D_err_Mpc","i_deg","i_err_deg"]:
        tidy[c] = pd.to_numeric(tidy[c], errors="coerce")
    tidy = tidy.dropna()

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(out_csv, index=False)
    return out_csv
