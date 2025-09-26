# src/ptquat/fetch_vizier.py
from __future__ import annotations
from pathlib import Path
from astroquery.vizier import Vizier

def fetch_sparc_to_csv(outdir: str | Path) -> dict[str, Path]:
    """
    下載 Lelli+ (2016) SPARC 的 table1/ table2（VizieR: J/AJ/152/157）
    並存成 CSV。回傳檔案路徑 dict。
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    Vizier.ROW_LIMIT = -1
    cats = Vizier.get_catalogs('J/AJ/152/157')

    t1 = cats['J/AJ/152/157/table1']   # Galaxy meta
    t2 = cats['J/AJ/152/157/table2']   # Mass models (Rad, Vobs, e_Vobs, Vgas, Vdisk, Vbulge, ...)

    p1 = outdir / "vizier_table1.csv"
    p2 = outdir / "vizier_table2.csv"
    t1.to_pandas().to_csv(p1, index=False)
    t2.to_pandas().to_csv(p2, index=False)

    return {"table1": p1, "table2": p2}
