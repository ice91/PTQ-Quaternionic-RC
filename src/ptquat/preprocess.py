# src/ptquat/preprocess.py
from __future__ import annotations
import pandas as pd
from pathlib import Path

# 欄位映射：不同 VizieR 鏡像偶爾用簡寫，這裡做容錯
_MAP2 = {
    "Name": "galaxy",
    "Rad": "r_kpc",
    "Vobs": "v_obs_kms",
    "e_Vobs": "v_err_kms",
    "Vgas": "v_gas_kms",
    "Vdisk": "v_disk_kms",
    "Vbulge": "v_bulge_kms",
}
_MAP1 = {
    "Name": "galaxy",
    "Dist": "D_Mpc",
    "e_Dist": "D_err_Mpc",
    "i": "i_deg",
    "e_i": "i_err_deg",
    "Qual": "Qual",
}

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
    """
    t1 = pd.read_csv(table1_csv)
    t2 = pd.read_csv(table2_csv)

    t1 = _rename_safe(t1, _MAP1)
    t2 = _rename_safe(t2, _MAP2)

    # 缺少任一核心欄位則報錯
    need2 = ["galaxy","r_kpc","v_obs_kms","v_err_kms"]
    if not set(need2).issubset(t2.columns):
        missing = sorted(set(need2) - set(t2.columns))
        raise ValueError(f"table2 缺少欄位: {missing}")

    # 沒提供的 baryon 欄位補 0（有些星系沒 bulge）
    for b in ["v_gas_kms","v_disk_kms","v_bulge_kms"]:
        if b not in t2.columns:
            t2[b] = 0.0

    # 合併
    keep1 = ["galaxy","D_Mpc","D_err_Mpc","i_deg","i_err_deg","Qual"]
    for k in keep1:
        if k not in t1.columns:
            raise ValueError(f"table1 缺少欄位: {k}")
    meta = t1[keep1].drop_duplicates("galaxy")
    df = t2.merge(meta, on="galaxy", how="left")

    # 基本清理
    df = df.dropna(subset=["r_kpc","v_obs_kms","v_err_kms","D_Mpc","i_deg","Qual"])
    df = df[df["r_kpc"] > 0].copy()
    df = df[df["v_err_kms"] >= 0].copy()

    # 品質切
    mask = (
        (df["i_deg"] > i_min_deg) &
        ((df["D_err_Mpc"] / df["D_Mpc"]) < relD_max) &
        (df["Qual"] <= qual_max)
    )
    df = df[mask].copy()

    # 排序後輸出
    df = df.sort_values(["galaxy","r_kpc"])
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return out_csv
