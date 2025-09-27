# src/ptquat/fetch_vizier.py
from __future__ import annotations
from pathlib import Path
from typing import Dict
import pandas as pd
from astroquery.vizier import Vizier

CAT_BASE = "J/AJ/152/157"

def _to_csv_safely(table, path: Path):
    """Convert astropy Table -> pandas -> CSV，並盡量轉成乾淨 dtype。"""
    df = table.to_pandas()
    # 嘗試把 object 轉數值（失敗就保留）
    for c in df.columns:
        if df[c].dtype == object:
            try:
                df[c] = pd.to_numeric(df[c])
            except Exception:
                pass
    df.to_csv(path, index=False)

def fetch_sparc_to_csv(outdir: str | Path) -> Dict[str, Path]:
    """
    下載 Lelli+ (2016) SPARC 的 table1 / table2（VizieR: J/AJ/152/157）
    並存成 CSV。回傳 {'table1': Path, 'table2': Path}。
    - table1 欄位：Name, Dist, e_Dist, i, e_i, Qual
    - table2 欄位：Name, Dist, Rad, Vobs, e_Vobs, Vgas, Vdisk, Vbulge
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Vizier 設定
    Vizier.ROW_LIMIT = -1
    Vizier.TIMEOUT = 120  # 秒

    # 只抓需要的欄位，確保有 Vbulge
    v1 = Vizier(columns=["Name", "Dist", "e_Dist", "i", "e_i", "Qual"])
    v2 = Vizier(columns=["Name", "Dist", "Rad", "Vobs", "e_Vobs", "Vgas", "Vdisk", "Vbulge"])

    t1 = v1.get_catalogs(f"{CAT_BASE}/table1")[0]
    t2 = v2.get_catalogs(f"{CAT_BASE}/table2")[0]

    p1 = outdir / "vizier_table1.csv"
    p2 = outdir / "vizier_table2.csv"
    _to_csv_safely(t1, p1)
    _to_csv_safely(t2, p2)

    # 簡單健檢
    cols2 = set(pd.read_csv(p2, nrows=0).columns)
    need2 = {"Name", "Rad", "Vobs", "e_Vobs"}
    missing = need2 - cols2
    if missing:
        raise RuntimeError(f"Downloaded table2 missing required columns: {sorted(missing)}")

    if "Vbulge" not in cols2:
        # 正常不該發生（上面已指定 columns），但還是給個提示
        print("WARNING: 'Vbulge' not present in table2; bulge will be treated as 0 in preprocessing.")

    return {"table1": p1, "table2": p2}
