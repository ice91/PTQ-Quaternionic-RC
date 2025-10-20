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

def _norm(s: str) -> str:
    """欄名正規化：小寫、只留英數"""
    return re.sub(r'[^a-z0-9]+', '', str(s).lower())

def _find_col_regex(cols, must_all: list[str], any_of: list[str] | None = None):
    """在 columns 中找同時包含 must_all（皆需出現）、且（可選）包含 any_of 任一的欄。
       使用正規化欄名判斷；回傳第一個命中的『原始欄名』或 None。"""
    any_of = any_of or []
    ncols = {c: _norm(c) for c in cols}
    for c, n in ncols.items():
        if all(k in n for k in must_all) and (not any_of or any(k in n for k in any_of)):
            return c
    return None

def _extract_btfr_aux_from_table1(t1_raw: pd.DataFrame) -> pd.DataFrame | None:
    """
    從 VizieR table1 擷取 BTFR 需要的欄位：
    - L36_total（由 L3.6 或其變體取得）
    - L36_disk（若無分量，fallback= L36_total）
    - L36_bulge（若無分量，fallback= 0）
    - M_HI（由 MHI/M_HI 或其 log 欄取得；log 會轉回線性）
    回傳含 [galaxy, L36_total, L36_disk, L36_bulge, M_HI] 的 dataframe；若皆拿不到則回傳 None。
    """
    if "Name" not in t1_raw.columns:
        return None
    t1 = t1_raw.rename(columns={"Name": "galaxy"}).copy()
    cols = list(t1.columns)

    # 先找「線性」欄；若找不到再找 log 欄
    c_Ltot   = _find_col_regex(cols, must_all=["l","36"])  # e.g., "L3.6"
    c_logL   = _find_col_regex(cols, must_all=["log","l","36"])

    c_MHI    = _find_col_regex(cols, must_all=["m","hi"]) or _find_col_regex(cols, must_all=["m","gas"])
    c_logMHI = _find_col_regex(cols, must_all=["log","m","hi"]) or _find_col_regex(cols, must_all=["log","m","gas"])

    extra = pd.DataFrame({"galaxy": t1["galaxy"]})

    # 亮度：總量
    if c_Ltot is not None:
        extra["L36_total"] = pd.to_numeric(t1[c_Ltot], errors="coerce")
    elif c_logL is not None:
        extra["L36_total"] = 10.0 ** pd.to_numeric(t1[c_logL], errors="coerce")

    # 拆成 disk/bulge：若沒有分量，採用 fallback
    if "L36_total" in extra.columns:
        extra["L36_disk"]  = extra["L36_total"]
        extra["L36_bulge"] = 0.0

    # 氣體：M_HI
    if c_MHI is not None:
        extra["M_HI"] = pd.to_numeric(t1[c_MHI], errors="coerce")
    elif c_logMHI is not None:
        extra["M_HI"] = 10.0 ** pd.to_numeric(t1[c_logMHI], errors="coerce")

    useful = set(extra.columns) - {"galaxy"}
    if not useful:
        return None
    return extra

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
            t1[c] = 9 if c == "Qual" else np.nan

    tidy = (t2.merge(t1[cols1], on="galaxy", how="left")
              .dropna(subset=["D_Mpc","D_err_Mpc","i_deg","i_err_deg"])
              .assign(relD=lambda d: d["D_err_Mpc"]/d["D_Mpc"])
              .query("i_deg > @i_min_deg and relD <= @relD_max and Qual <= @qual_max")
              .drop(columns=["relD"]))

    # 追加 BTFR 輔助欄位（從 table1 擷取）
    extra = _extract_btfr_aux_from_table1(t1_raw)
    if extra is not None:
        tidy = tidy.merge(extra, on="galaxy", how="left")

    # 最終欄位順序與型別
    wanted = ["galaxy","r_kpc","v_obs_kms","v_err_kms",
              "v_disk_kms","v_bulge_kms","v_gas_kms",
              "D_Mpc","D_err_Mpc","i_deg","i_err_deg",
              "L36_total","L36_disk","L36_bulge","M_HI"]
    for c in wanted:
        if c not in tidy.columns:
            tidy[c] = np.nan

    tidy = tidy[wanted].copy()
    for c in ["r_kpc","v_obs_kms","v_err_kms","v_disk_kms","v_bulge_kms","v_gas_kms",
              "D_Mpc","D_err_Mpc","i_deg","i_err_deg","L36_total","L36_disk","L36_bulge","M_HI"]:
        tidy[c] = pd.to_numeric(tidy[c], errors="coerce")
    tidy = tidy.dropna(subset=["galaxy","r_kpc","v_obs_kms","v_err_kms","D_Mpc","D_err_Mpc","i_deg","i_err_deg"])

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(out_csv, index=False)
    return out_csv
