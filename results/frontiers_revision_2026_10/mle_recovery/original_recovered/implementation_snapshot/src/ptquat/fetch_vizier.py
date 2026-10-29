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

    try:
        Vizier.ROW_LIMIT = -1
        Vizier.columns   = ["**"]
        cats = Vizier.get_catalogs("J/AJ/152/157")
    except Exception as e:
        raise RuntimeError(
            "[fetch] 無法連線到 VizieR 或取得 catalog 'J/AJ/152/157'；"
            "請稍後重試或檢查網路環境。原始錯誤: " + repr(e)
        )

    if "J/AJ/152/157/table1" not in cats.keys() or "J/AJ/152/157/table2" not in cats.keys():
        raise RuntimeError("[fetch] 取得的 catalogs 缺少 table1 或 table2。")

    t1 = cats["J/AJ/152/157/table1"].to_pandas()
    t2 = cats["J/AJ/152/157/table2"].to_pandas()

    for col in _NUMERIC_COLS_T2:
        if col in t2.columns:
            t2[col] = pd.to_numeric(t2[col], errors="coerce")

    must2 = {"Name","Rad","Vobs","e_Vobs","Vdisk","Vgas","Vbulge"}
    missing = sorted(must2 - set(t2.columns))
    if missing:
        raise RuntimeError(f"[fetch] VizieR table2 缺少欄位: {missing}")

    p1 = outdir / "vizier_table1.csv"
    p2 = outdir / "vizier_table2.csv"
    t1.to_csv(p1, index=False)
    t2.to_csv(p2, index=False)

    print(f"[fetch] table1 names={t1['Name'].nunique()} rows={len(t1)}")
    print(f"[fetch] table2 names={t2['Name'].nunique()} rows={len(t2)} "
          f"Vbulge_nonzero={(t2['Vbulge'].fillna(0)!=0).sum()}")

    return {"table1": p1, "table2": p2}
