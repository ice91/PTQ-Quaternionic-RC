# src/ptquat/geometry.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import numpy as np
import pandas as pd

try:
    from astroquery.vizier import Vizier
except Exception as e:
    raise RuntimeError(
        "astroquery 未安裝或不可用；請先 pip install astroquery 再重試。"
    ) from e


_ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)

# 可能出現在文獻/表格的厚度欄位名稱關鍵字（大小寫不敏感）
_THICKNESS_PATTERNS = [
    r"^hz$", r"^h_z$", r"^z0$", r"^z_?0$", r"^z0[_-]?thin$", r"^z0[_-]?disk$",
    r"^hz[_-]?thin$", r"^sthick$", r"^sth$", r"^z0thin$", r"^hzt$", r"^hz1$",
]
_ERR_PREFIX = ("e_", "err_", "e", "err")  # 常見誤差欄位前綴

_NAME_COLS = ["Name", "name", "Galaxy", "GALAXY", "ID", "ID2"]
_RA_COLS   = ["RAJ2000", "RAdeg", "RA_ICRS", "RA"]
_DEC_COLS  = ["DEJ2000", "DEdeg", "DE_ICRS", "DEC", "Dec"]

def _norm_name(s: str) -> str:
    """正規化星系名稱（去空白與短橫，大小寫、一致化常見前綴）。"""
    if not isinstance(s, str):
        return ""
    s = s.strip().upper()
    s = s.replace("-", " ").replace("_", " ")
    s = re.sub(r"\s+", " ", s)
    # 常見別名規則：'UGC'、'NGC'、'IC'、'ESO'、'PGC' 等，統一一個空白
    s = re.sub(r"\b(UGC|NGC|IC|ESO|PGC|DDO|M|MRK|UGCA|KUG)\s*", r"\1 ", s)
    return s

def _find_first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    # 大小寫不敏感的兜底
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None

def _guess_thickness_cols(df: pd.DataFrame) -> List[Tuple[str, Optional[str], str]]:
    """
    從 df 中找可能的厚度欄位。
    回傳 [(value_col, err_col_or_None, flavor)]，
    其中 flavor ∈ {"thin","thick","unknown"}（從欄名猜測）。
    """
    cols = list(df.columns)
    results = []
    for c in cols:
        lc = c.lower()
        if any(re.match(pat, lc) for pat in _THICKNESS_PATTERNS):
            flavor = "thin" if ("thin" in lc or lc.endswith("1") or lc.endswith("t")) else "unknown"
            # 嘗試找對應的誤差欄位
            err = None
            for pref in _ERR_PREFIX:
                cand = f"{pref}{c}"
                if cand in df.columns:
                    err = cand
                    break
                # 大小寫容忍
                cand_low = f"{pref}{c}".lower()
                for cc in df.columns:
                    if cc.lower() == cand_low:
                        err = cc
                        break
            results.append((c, err, flavor))
        # 再廣義一點：包含 'hz' 或 'z0' 的欄名
        elif ("hz" in lc or re.search(r"\bz[\W_]*0\b", lc)) and re.search(r"(thin|disk|1|t|z0|hz)", lc):
            flavor = "thin" if ("thin" in lc or "disk" in lc or lc.endswith("1") or lc.endswith("t")) else "unknown"
            err = None
            for pref in _ERR_PREFIX:
                cand_low = f"{pref}{c}".lower()
                for cc in df.columns:
                    if cc.lower() == cand_low:
                        err = cc
                        break
            results.append((c, err, flavor))
    return results

def _is_arcsec_series(s: pd.Series) -> bool:
    """
    簡單啟發式：若數值多在 [0.1, 50] 內，且中位數 < 10，傾向 arcsec。
    S4G 的 edge-on 厚度常見數弧秒到十幾弧秒。
    """
    try:
        v = pd.to_numeric(s, errors="coerce").astype(float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return True
        med = float(np.median(v))
        return (0.01 <= med <= 50.0)
    except Exception:
        return True

def _to_kpc(val: float, D_Mpc: float, is_arcsec: bool) -> float:
    if not np.isfinite(val):
        return np.nan
    if is_arcsec:
        return float(val) * (D_Mpc * 1000.0) * _ARCSEC_TO_RAD
    return float(val)  # 已是 kpc

def _load_sparc_distance_map(sparc_tidy_csv: str) -> Tuple[Dict[str,float], Dict[str,str]]:
    df = pd.read_csv(sparc_tidy_csv)
    need = {"galaxy","D_Mpc"}
    miss = sorted(need - set(df.columns))
    if miss:
        raise ValueError(f"{sparc_tidy_csv} 缺少欄位 {miss}")
    # 取每星系第一筆距離
    g2D = {}
    g2orig = {}
    for name, sub in df.groupby("galaxy"):
        g2D[_norm_name(name)] = float(sub["D_Mpc"].iloc[0])
        g2orig[_norm_name(name)] = str(name)
    return g2D, g2orig

def _gather_s4g_edgeon_tables() -> Dict[str, pd.DataFrame]:
    """
    從 VizieR 搜尋 S4G / edge-on / Comerón 相關 catalog，回傳 {catalog_id: DataFrame}
    """
    Vizier.ROW_LIMIT = -1
    Vizier.columns = ["**"]

    tables: Dict[str, pd.DataFrame] = {}
    # 關鍵詞盡量涵蓋（Comerón S4G edge-on vertical）
    queries = [
        "S4G edge-on", "Comeron S4G edge-on", "Spitzer S4G edge-on vertical",
        "S4G thick thin disk", "Comeron vertical structure S4G"
    ]
    seen = set()
    for q in queries:
        cats = Vizier.find_catalogs(q)
        for catid in cats.keys():
            if catid in seen:
                continue
            seen.add(catid)
            try:
                cat = Vizier.get_catalogs(catid)
            except Exception:
                continue
            for key, table in cat.items():
                try:
                    df = table.to_pandas()
                except Exception:
                    continue
                # 需要至少有名稱與某種厚度欄位跡象
                name_col = _find_first_col(df, _NAME_COLS)
                th_cols = _guess_thickness_cols(df)
                if name_col and th_cols:
                    # 避免過大/無意義表格
                    if len(df) > 0 and len(df.columns) <= 200:
                        tables[f"{catid}/{key}"] = df
    return tables

def build_s4g_h_catalog(
    sparc_tidy_csv: str = "dataset/sparc_tidy.csv",
    out_csv: str = "dataset/geometry/h_catalog.csv",
    prefer: str = "thin",                 # "thin" or "thick"
    default_rel_err: float = 0.25,
) -> pd.DataFrame:
    """
    搜尋並彙整 S4G edge-on 厚度為 h_catalog.csv：
      galaxy,h_kpc,h_err_kpc,RA_deg,DEC_deg,band,source,profile
    只輸出能與 SPARC 名稱成功對齊者，厚度轉 kpc（距離以 SPARC 距離）。
    """
    prefer = prefer.lower().strip()
    if prefer not in ("thin","thick"):
        raise ValueError("prefer 只能是 'thin' 或 'thick'")

    g2D, g2orig = _load_sparc_distance_map(sparc_tidy_csv)
    sparc_set = set(g2D.keys())

    tables = _gather_s4g_edgeon_tables()
    if not tables:
        raise RuntimeError("在 VizieR 找不到 S4G edge-on 相關可用表格；請稍後再試或檢查網路。")

    # 收集候選
    cand_rows = []
    for catkey, df in tables.items():
        name_col = _find_first_col(df, _NAME_COLS)
        ra_col   = _find_first_col(df, _RA_COLS)
        de_col   = _find_first_col(df, _DEC_COLS)
        th_cols  = _guess_thickness_cols(df)
        if not (name_col and th_cols):
            continue

        for _, row in df.iterrows():
            raw_name = str(row.get(name_col, "")).strip()
            nn = _norm_name(raw_name)
            if nn not in sparc_set:
                continue

            # 每個表可能有多個厚度欄；挑選 prefer 對應或 fallback
            # 先把所有候選加進來，後面再 per-galaxy 做決策
            for (val_col, err_col, flavor) in th_cols:
                try:
                    val = float(row.get(val_col, np.nan))
                except Exception:
                    val = np.nan
                if not np.isfinite(val):
                    continue
                # 決定 arcsec vs kpc（啟發式）
                is_arcsec = _is_arcsec_series(df[val_col])
                D_Mpc = g2D[nn]
                h_kpc = _to_kpc(val, D_Mpc, is_arcsec=is_arcsec)

                if np.isnan(h_kpc) or h_kpc <= 0:
                    continue

                # 誤差
                h_err_kpc = np.nan
                if err_col and err_col in df.columns:
                    try:
                        ev = float(row.get(err_col, np.nan))
                        if np.isfinite(ev) and ev >= 0:
                            h_err_kpc = _to_kpc(ev, D_Mpc, is_arcsec=is_arcsec)
                    except Exception:
                        h_err_kpc = np.nan
                if not np.isfinite(h_err_kpc) or h_err_kpc <= 0:
                    h_err_kpc = float(default_rel_err) * h_kpc

                RA = float(row.get(ra_col)) if (ra_col and str(row.get(ra_col)) not in ("", "nan", "None")) else np.nan
                DE = float(row.get(de_col)) if (de_col and str(row.get(de_col)) not in ("", "nan", "None")) else np.nan

                # flavor/score：偏好 thin；欄名包含 'thin' / '1' / 't' 加分
                score = 0.0
                if flavor == "thin":
                    score += 2.0
                if prefer == "thin" and flavor == "thin":
                    score += 2.0
                if prefer == "thick" and ("thick" in val_col.lower()):
                    score += 2.0
                # 小一點通常更像薄盤
                score += float(-h_kpc)

                cand_rows.append(dict(
                    nn=nn,
                    galaxy=g2orig[nn],
                    h_kpc=h_kpc,
                    h_err_kpc=float(h_err_kpc),
                    RA_deg=(float(RA) if np.isfinite(RA) else np.nan),
                    DEC_deg=(float(DE) if np.isfinite(DE) else np.nan),
                    profile=("thin" if flavor=="thin" else ("thick" if "thick" in val_col.lower() else "unknown")),
                    band="3.6um",
                    source=str(catkey),
                    score=float(score),
                    _val_col=val_col,
                    _err_col=(err_col or "")
                ))

    if not cand_rows:
        raise RuntimeError("雖找到 S4G 相關表格，但無法對齊到任何 SPARC 星系或缺少厚度欄。")

    cand = pd.DataFrame(cand_rows)

    # 每星系留最佳一筆（按 score 最大；若 tie 取 h_kpc 較小者）
    cand.sort_values(["nn", "score", "h_kpc"], ascending=[True, False, True], inplace=True)
    best = cand.groupby("nn", as_index=False).first()

    # 僅輸出必要欄位
    out = best[["galaxy","h_kpc","h_err_kpc","RA_deg","DEC_deg","band","source","profile"]].copy()
    out = out.sort_values("galaxy").reset_index(drop=True)

    # 寫檔
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out
