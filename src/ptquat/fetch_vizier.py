# src/ptquat/fetch_vizier.py
from __future__ import annotations
from pathlib import Path
from astroquery.vizier import Vizier
import pandas as pd

_NUMERIC_COLS_T2 = ["Dist","Rad","Vobs","e_Vobs","Vgas","Vdisk","Vbulge"]

def fetch_sparc_to_csv(outdir: str | Path) -> dict[str, Path]:
    """
    下載 Lelli+ (2016) SPARC 的 table1/ table2（VizieR: J/AJ/152/157）
    並存成 CSV。回傳檔案路徑 dict。
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 重要：取完整欄位 + 取消列數上限
    Vizier.ROW_LIMIT = -1
    Vizier.columns   = ["**"]

    cats = Vizier.get_catalogs("J/AJ/152/157")

    t1 = cats["J/AJ/152/157/table1"].to_pandas()
    t2 = cats["J/AJ/152/157/table2"].to_pandas()

    # 型別保險：把數值欄強制轉 numeric（字串/遮罩→NaN）
    for col in _NUMERIC_COLS_T2:
        if col in t2.columns:
            t2[col] = pd.to_numeric(t2[col], errors="coerce")

    # 基本健檢：確保關鍵欄位存在
    must2 = {"Name","Rad","Vobs","e_Vobs","Vdisk","Vgas","Vbulge"}
    missing = sorted(must2 - set(t2.columns))
    if missing:
        raise RuntimeError(f"VizieR table2 is missing columns: {missing}")

    p1 = outdir / "vizier_table1.csv"
    p2 = outdir / "vizier_table2.csv"
    t1.to_csv(p1, index=False)
    t2.to_csv(p2, index=False)

    # 可選：列印統計協助除錯
    print(f"[fetch] table1 names={t1['Name'].nunique()} rows={len(t1)}")
    print(f"[fetch] table2 names={t2['Name'].nunique()} rows={len(t2)} "
          f"Vbulge_nonzero={(t2['Vbulge'].fillna(0)!=0).sum()}")

    return {"table1": p1, "table2": p2}
