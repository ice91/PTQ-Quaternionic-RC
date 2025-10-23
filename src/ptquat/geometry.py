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
        "astroquery 未安裝或不可用；請先 `pip install astroquery` 後重試。"
    ) from e


_ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)

# 可能出現在文獻/表格的厚度欄位名稱關鍵字（大小寫不敏感）
_THICKNESS_PATTERNS = [
    r"^hz$", r"^h_z$", r"^z0$", r"^z_?0$",
    r"^z0[_-]?thin$", r"^z0[_-]?disk$",
    r"^hz[_-]?thin$", r"^z0thin$", r"^hzt$", r"^hz1$",
    r"^sthick$", r"^sth$",
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
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def _guess_thickness_cols(df: pd.DataFrame) -> List[Tuple[str, Optional[str], str]]:
    """
    從 df 中找可能的厚度欄位。
    回傳 [(value_col, err_col_or_None, flavor)]，
    其中 flavor ∈ {"thin","thick","unknown"}（從欄名猜測）。
    """
    cols = list(df.columns)
    results: List[Tuple[str, Optional[str], str]] = []
    for c in cols:
        lc = c.lower()
        is_candidate = any(re.match(pat, lc) for pat in _THICKNESS_PATTERNS)
        wide_candidate = (("hz" in lc) or re.search(r"\bz[\W_]*0\b", lc)) and re.search(r"(thin|disk|1|t|z0|hz|thick)", lc)
        if is_candidate or wide_candidate:
            flavor = "thin" if ("thin" in lc or "disk" in lc or lc.endswith("1") or lc.endswith("t")) else \
                     ("thick" if "thick" in lc else "unknown")
            # 嘗試找對應的誤差欄位
            err = None
            for pref in _ERR_PREFIX:
                cand = f"{pref}{c}"
                if cand in df.columns:
                    err = cand
                    break
                cand_low = f"{pref}{c}".lower()
                for cc in df.columns:
                    if cc.lower() == cand_low:
                        err = cc
                        break
            results.append((c, err, flavor))
    return results


def _is_arcsec_series(s: pd.Series) -> bool:
    """
    簡單啟發式：若數值多在 [0.01, 50] 內，且中位數 < 10，傾向 arcsec。
    S4G 的 edge-on 厚度常見數弧秒到十幾弧秒。
    """
    try:
        v = pd.to_numeric(s, errors="coerce").astype(float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return True
        med = float(np.nanmedian(v))
        return (0.01 <= med <= 50.0)
    except Exception:
        return True


def _to_kpc(val: float, D_Mpc: float, is_arcsec: bool) -> float:
    if not np.isfinite(val):
        return np.nan
    if is_arcsec:
        return float(val) * (D_Mpc * 1000.0) * _ARCSEC_TO_RAD
    return float(val)  # 已是 kpc


def _load_sparc_distance_map(sparc_tidy_csv: str) -> Tuple[Dict[str, float], Dict[str, str]]:
    df = pd.read_csv(sparc_tidy_csv)
    need = {"galaxy", "D_Mpc"}
    miss = sorted(need - set(df.columns))
    if miss:
        raise ValueError(f"{sparc_tidy_csv} 缺少欄位 {miss}")
    # 取每星系第一筆距離
    g2D: Dict[str, float] = {}
    g2orig: Dict[str, str] = {}
    for name, sub in df.groupby("galaxy"):
        g2D[_norm_name(name)] = float(sub["D_Mpc"].iloc[0])
        g2orig[_norm_name(name)] = str(name)
    return g2D, g2orig


def _iter_vizier_tables(cat, catid: str):
    """
    將 Vizier.get_catalogs 的回傳統一成 (key, Table) 的 iterable。
    可能型別：
      - OrderedDict-like: 有 .items()
      - TableList (list-like): 只能 enumerate
    """
    # dict-like
    if hasattr(cat, "items"):
        for k, tbl in cat.items():
            yield str(k), tbl
        return
    # list-like（TableList）
    if isinstance(cat, (list, tuple)):
        for i, tbl in enumerate(cat):
            yield f"{catid}/{i}", tbl
        return
    # 兜底：嘗試當作可迭代
    try:
        for i, tbl in enumerate(cat):
            yield f"{catid}/{i}", tbl
    except Exception:
        return


def _gather_s4g_edgeon_tables(
    vizier_ids: list[str] | None = None,
    verbose: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    從 VizieR 搜尋/下載 S4G edge-on / Comerón 相關 catalog，回傳 {catalog_id: DataFrame}
    若提供 vizier_ids，就僅抓這些 ID（最穩）；否則用關鍵詞搜一輪。
    """
    Vizier.ROW_LIMIT = -1
    Vizier.columns = ["**"]

    tables: Dict[str, pd.DataFrame] = {}

    # 準備 catalog ID 清單
    if vizier_ids and len(vizier_ids) > 0:
        id_list = list(dict.fromkeys([str(x) for x in vizier_ids]))  # 去重保序
        if verbose:
            print(f"[geom] Using explicit VizieR IDs: {id_list}")
    else:
        # 關鍵詞搜尋（可能因網路/代理/SSL 環境而搜不到）
        queries = [
            "S4G edge-on",
            "Comeron S4G edge-on",
            "Spitzer S4G edge-on vertical",
            "S4G thick thin disk",
            "Comeron vertical structure S4G",
        ]
        id_list: List[str] = []
        seen = set()
        for q in queries:
            try:
                found = Vizier.find_catalogs(q)
            except Exception as e:
                if verbose:
                    print(f"[geom] Vizier.find_catalogs({q!r}) failed: {e}")
                continue
            keys = list(getattr(found, "keys", lambda: [])())
            if verbose:
                print(f"[geom] Query {q!r} → {len(keys)} candidates")
            for catid in keys:
                if catid not in seen:
                    seen.add(catid)
                    id_list.append(catid)

    # 抓各 catalog 的表
    for catid in id_list:
        try:
            cat = Vizier.get_catalogs(catid)
        except Exception as e:
            if verbose:
                print(f"[geom] Vizier.get_catalogs({catid}) failed: {e}")
            continue
        hit_any = False
        for key, table in _iter_vizier_tables(cat, catid):
            try:
                df = table.to_pandas()
            except Exception:
                continue
            name_col = _find_first_col(df, _NAME_COLS)
            th_cols  = _guess_thickness_cols(df)
            if name_col and th_cols and len(df) > 0 and len(df.columns) <= 300:
                tables[f"{catid}/{key}"] = df
                hit_any = True
        if verbose:
            if hit_any:
                print(f"[geom] {catid}: accepted")
            else:
                print(f"[geom] {catid}: no thickness-like columns, skipped")

    if verbose:
        print(f"[geom] Total candidate tables: {len(tables)}")
        for k, df in list(tables.items())[:12]:
            print(f"  - {k}: nrows={len(df)}, ncols={len(df.columns)}, head_cols={list(df.columns)[:8]}")

    return tables


def build_s4g_h_catalog(
    sparc_tidy_csv: str = "dataset/sparc_tidy.csv",
    out_csv: str = "dataset/geometry/h_catalog.csv",
    prefer: str = "thin",                 # "thin" or "thick"
    default_rel_err: float = 0.25,
    vizier_ids: list[str] | None = None,  # 顯式指定 VizieR IDs（最穩）
    verbose: bool = False,
) -> pd.DataFrame:
    """
    搜尋並彙整 S4G edge-on 厚度為 h_catalog.csv：
      欄位： galaxy,h_kpc,h_err_kpc,RA_deg,DEC_deg,band,source,profile
    僅輸出能與 SPARC 名稱成功對齊者；厚度轉 kpc（距離以 SPARC tidy CSV 內的 D_Mpc）。
    """
    prefer = prefer.lower().strip()
    if prefer not in ("thin", "thick"):
        raise ValueError("prefer 只能是 'thin' 或 'thick'")

    g2D, g2orig = _load_sparc_distance_map(sparc_tidy_csv)
    sparc_set = set(g2D.keys())

    tables = _gather_s4g_edgeon_tables(vizier_ids=vizier_ids, verbose=verbose)
    if not tables:
        raise RuntimeError(
            "在 VizieR 找不到 S4G edge-on 相關可用表格；"
            "可嘗試：1) 加 `--ids` 指定目錄，2) 檢查網路/代理/SSL，3) 稍後重試。"
        )

    cand_rows: List[dict] = []
    for catkey, df in tables.items():
        name_col = _find_first_col(df, _NAME_COLS)
        ra_col   = _find_first_col(df, _RA_COLS)
        de_col   = _find_first_col(df, _DEC_COLS)
        th_cols  = _guess_thickness_cols(df)
        if not (name_col and th_cols):
            if verbose:
                print(f"[geom] Skip {catkey} (no name/thickness columns)")
            continue

        for _, row in df.iterrows():
            raw_name = str(row.get(name_col, "")).strip()
            nn = _norm_name(raw_name)
            if nn not in sparc_set:
                continue

            # 對該 row，檢視所有偵測到的厚度欄
            for (val_col, err_col, flavor) in th_cols:
                try:
                    val = float(row.get(val_col, np.nan))
                except Exception:
                    val = np.nan
                if not np.isfinite(val):
                    continue

                # arcsec vs kpc（啟發式判定）
                is_arcsec = _is_arcsec_series(df[val_col])
                D_Mpc = g2D[nn]
                h_kpc = _to_kpc(val, D_Mpc, is_arcsec=is_arcsec)
                if not (np.isfinite(h_kpc) and h_kpc > 0):
                    continue

                # 誤差處理
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

                # 座標（可為 NaN）
                RA = row.get(ra_col, np.nan) if ra_col else np.nan
                DE = row.get(de_col, np.nan) if de_col else np.nan
                try:
                    RA = float(RA)
                except Exception:
                    RA = np.nan
                try:
                    DE = float(DE)
                except Exception:
                    DE = np.nan

                # flavor/score：偏好 thin；欄名包含 'thin' / '1' / 't' 加分；小 h 傾向薄盤
                score = 0.0
                if flavor == "thin": score += 2.0
                if prefer == "thin" and flavor == "thin": score += 2.0
                if prefer == "thick" and "thick" in val_col.lower(): score += 2.0
                score += float(-h_kpc)

                cand_rows.append(dict(
                    nn=nn,
                    galaxy=g2orig[nn],
                    h_kpc=float(h_kpc),
                    h_err_kpc=float(h_err_kpc),
                    RA_deg=(float(RA) if np.isfinite(RA) else np.nan),
                    DEC_deg=(float(DE) if np.isfinite(DE) else np.nan),
                    profile=("thin" if flavor == "thin" else ("thick" if "thick" in val_col.lower() else "unknown")),
                    band="3.6um",
                    source=str(catkey),
                    score=float(score),
                    _val_col=str(val_col),
                    _err_col=str(err_col or ""),
                ))

    if not cand_rows:
        raise RuntimeError(
            "雖找到 S4G 相關表格，但無法對齊到任何 SPARC 星系或缺少厚度欄。"
            "請嘗試：1) 指定更精確的 `--ids`，2) 檢查名稱規則、3) 用 --verbose 檢視表頭。"
        )

    cand = pd.DataFrame(cand_rows)
    if verbose:
        print(f"[geom] Candidate matches: {len(cand)} (unique galaxies: {cand['nn'].nunique()})")

    # 每星系留最佳一筆（按 score 最大；若 tie 取 h_kpc 較小者）
    cand.sort_values(["nn", "score", "h_kpc"], ascending=[True, False, True], inplace=True)
    best = cand.groupby("nn", as_index=False).first()

    # 僅輸出必要欄位
    out = best[["galaxy", "h_kpc", "h_err_kpc", "RA_deg", "DEC_deg", "band", "source", "profile"]].copy()
    out = out.sort_values("galaxy").reset_index(drop=True)

    # 寫檔
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    if verbose:
        print(f"[geom] Saved h-catalog to: {out_path} (N={len(out)})")

    return out
