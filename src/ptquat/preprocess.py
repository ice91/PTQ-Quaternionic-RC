# src/ptquat/preprocess.py
from __future__ import annotations
import pandas as pd
from pathlib import Path

# 欄位映射：不同 VizieR 鏡像偶爾用簡寫，這裡做容錯
# 欄位映射：不同 VizieR 鏡像偶爾用簡寫，這裡做容錯
_MAP2 = {
    "Name":   "galaxy",
    "Rad":    "r_kpc",
    "Vobs":   "v_obs_kms",
    "e_Vobs": "v_err_kms",
    "Vgas":   "v_gas_kms",
    "Vdisk":  "v_disk_kms",
    "Vbulge": "v_bulge_kms",
    # 可能遇到的短名別名（防呆）
    "Vbul":   "v_bulge_kms",
    "Vblg":   "v_bulge_kms",
}

_MAP1 = {
    "Name": "galaxy",
    "Dist": "D_Mpc",
    # 有些鏡像沒有 e_Dist，只有 f_Dist（方法旗標）；一起映射進來
    "e_Dist": "D_err_Mpc",
    "f_Dist": "f_Dist",
    "i": "i_deg",
    "e_i": "i_err_deg",
    "Qual": "Qual",
}

# 預設：用 f_Dist → 相對距離誤差 的保守映射（可依需要調整）
# 這裡假設：1 ≈ 次級/TF/哈柏流 ~20%；2 ≈ 一級(Cepheid/TRGB) ~10%；其他→20%
_DEFAULT_REL_D_BY_FLAG = {1: 0.20, 2: 0.10}

# 若缺 i_err，給個保守常數（度）
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
    """
    將 VizieR table1+table2 合併成 tidy long-format（逐半徑）CSV，
    並套用品質切（i>i_min、e_D/D<relD_max、Qual<=qual_max）。
    若 table1 無 D_err_Mpc，則用 f_Dist 合成；若無 i_err_deg，補上常數。
    """
    t1_raw = pd.read_csv(table1_csv)
    t2_raw = pd.read_csv(table2_csv)

    t1 = _rename_safe(t1_raw, _MAP1)
    t2 = _rename_safe(t2_raw, _MAP2)

    # 缺少任一核心欄位則報錯（table2 的徑向資料與觀測誤差必須在）
    need2 = ["galaxy", "r_kpc", "v_obs_kms", "v_err_kms"]
    if not set(need2).issubset(t2.columns):
        missing = sorted(set(need2) - set(t2.columns))
        raise ValueError(f"table2 缺少欄位: {missing}")

    # 沒提供的 baryon 欄位補 0（有些星系沒 bulge）
    for b in ["v_gas_kms", "v_disk_kms", "v_bulge_kms"]:
        if b not in t2.columns:
            t2[b] = 0.0

    # --- 處理 table1 的距離與其不確定度 ---
    # 確保最少有 D_Mpc
    if "D_Mpc" not in t1.columns:
        # 部分鏡像可能放在 table2；如果 table2 有 Dist，先搬到 t1 再去重
        if "Dist" in t2_raw.columns and "Name" in t2_raw.columns:
            t1_fallback = t2_raw[["Name", "Dist"]].rename(
                columns={"Name": "galaxy", "Dist": "D_Mpc"}
            ).drop_duplicates("galaxy")
            t1 = t1.merge(t1_fallback, on="galaxy", how="outer")
        else:
            raise ValueError("找不到 D_Mpc（距離）；table1 無 Dist 且 table2 也無 Dist。")

    # 若沒有 D_err_Mpc → 由 f_Dist 合成
    if "D_err_Mpc" not in t1.columns:
        if "f_Dist" in t1.columns:
            rel = t1["f_Dist"].map(_DEFAULT_REL_D_BY_FLAG).fillna(0.20)
            t1["D_err_Mpc"] = rel * t1["D_Mpc"]
        else:
            # 完全沒有 f_Dist，就用保守 20%
            t1["D_err_Mpc"] = 0.20 * t1["D_Mpc"]

    # 若沒有 i_err_deg → 補上常數
    if "i_err_deg" not in t1.columns:
        t1["i_err_deg"] = _DEFAULT_I_ERR_DEG

    # 若沒有 Qual → 全部給個中等品質(=2) 以便通過/不通過切線
    if "Qual" not in t1.columns:
        t1["Qual"] = 2

    # 合併（逐星系 meta）
    keep1 = ["galaxy", "D_Mpc", "D_err_Mpc", "i_deg", "i_err_deg", "Qual"]
    for k in keep1:
        if k not in t1.columns:
            raise ValueError(f"table1 缺少欄位: {k}")
    meta = t1[keep1].drop_duplicates("galaxy")
    df = t2.merge(meta, on="galaxy", how="left")

    # 基本清理
    df = df.dropna(subset=["r_kpc", "v_obs_kms", "v_err_kms", "D_Mpc", "i_deg", "Qual"])
    df = df[df["r_kpc"] > 0].copy()
    df = df[df["v_err_kms"] >= 0].copy()

    # 品質切
    with pd.option_context("mode.use_inf_as_na", True):
        relD = (df["D_err_Mpc"] / df["D_Mpc"]).abs()
    mask = (
        (df["i_deg"] > i_min_deg) &
        (relD < relD_max) &
        (df["Qual"] <= qual_max)
    )
    df = df[mask].copy()

    # 排序後輸出
    df = df.sort_values(["galaxy", "r_kpc"])
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv
