# src/ptquat/geometry.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import numpy as np
import pandas as pd

try:
    from astroquery.vizier import Vizier, conf
except Exception as e:
    raise RuntimeError(
        "astroquery 未安裝或不可用；請先 `pip install astroquery` 再重試。"
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
    s = re.sub(r"\b(UGC|NGC|IC|ESO|PGC|DDO|M|MRK|UGCA|KUG)\s*", r"\1 ", s)
    return s

def _find_first_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None

def _guess_thickness_cols(df: pd.DataFrame) -> List[Tuple[str, Optional[str], str]]:
    """
    從 df 中找可能的厚度欄位。
    回傳 [(value_col, err_col_or_None, flavor)],
    其中 flavor ∈ {"thin","thick","unknown"}（從欄名猜測）。
    """
    cols = list(df.columns)
    results = []
    for c in cols:
        lc = c.lower()
        if any(re.match(pat, lc) for pat in _THICKNESS_PATTERNS):
            flavor = "thin" if ("thin" in lc or lc.endswith("1") or lc.endswith("t")) else "unknown"
            err = None
            for pref in _ERR_PREFIX:
                cand_low = f"{pref}{c}".lower()
                for cc in df.columns:
                    if cc.lower() == cand_low:
                        err = cc
                        break
            results.append((c, err, flavor))
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
    簡單啟發式：若數值多在 [0.01, 50] 內，且中位數 < 10，傾向 arcsec。
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

def _load_sparc_distance_map(sparc_tidy_csv: str) -> Tuple[Dict[str,float], Dict[str,str]]:
    df = pd.read_csv(sparc_tidy_csv)
    need = {"galaxy","D_Mpc"}
    miss = sorted(need - set(df.columns))
    if miss:
        raise ValueError(f"{sparc_tidy_csv} 缺少欄位 {miss}")
    g2D = {}
    g2orig = {}
    for name, sub in df.groupby("galaxy"):
        g2D[_norm_name(name)] = float(sub["D_Mpc"].iloc[0])
        g2orig[_norm_name(name)] = str(name)
    return g2D, g2orig

def _fetch_catalog_tables(
    catid: str,
    mirrors: List[str],
    timeout_sec: float,
    retries: int,
    verbose: bool
) -> Dict[str, pd.DataFrame]:
    """逐鏡像+重試抓取一個 catalog 的所有 TableList→DataFrame。"""
    out: Dict[str, pd.DataFrame] = {}
    for mirror in mirrors:
        conf.server = mirror
        conf.timeout = int(round(timeout_sec))  # 必須是 int
        if conf.timeout <= 0:
            conf.timeout = 60
        # 建議順手加上，確保抓「全部欄位」且不限制筆數
        Vizier.ROW_LIMIT = -1
        Vizier.columns = ["**"]
        ok = False
        last_err = None
        for _ in range(max(1, int(retries))):
            try:
                cat = Vizier.get_catalogs(catid)
                ok = True
                break
            except Exception as e:
                last_err = e
                if verbose:
                    print(f"[geom] get_catalogs({catid})@{mirror} failed: {e}")
        if not ok:
            continue
        # 轉 DataFrame（TableList 可 enumerate）
        try:
            for i, tbl in enumerate(cat):
                try:
                    df = tbl.to_pandas()
                    out[f"{catid}[{i}]@{mirror}"] = df
                except Exception as e:
                    if verbose:
                        print(f"[geom] to_pandas fail {catid}[{i}]@{mirror}: {e}")
        except Exception as e:
            if verbose:
                print(f"[geom] TableList parse error for {catid}@{mirror}: {e}")
        if out:
            break  # 此 catalog 已成功至少一張表，換下一個 catalog
    return out

def _gather_s4g_edgeon_tables(
    ids: Optional[List[str]] = None,
    mirror: Optional[str] = None,
    timeout_sec: float = 120.0,
    retries: int = 2,
    verbose: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    以 ids（優先）或關鍵字搜尋彙整可能的 S4G edge-on tables。
    回傳 {key: DataFrame}，key=catid[index]@mirror。
    """
    base_mirrors = [
        "vizier.cfa.harvard.edu",   # US (CfA)
        "vizier.nao.ac.jp",         # Japan (NAOJ)
        "vizier.hia.nrc.ca",        # Canada (HIA)
        "vizier.u-strasbg.fr",      # France (Strasbourg)
    ]
    mirrors: List[str] = []
    if mirror:
        mirrors.append(mirror.strip())
    mirrors += [m for m in base_mirrors if m not in mirrors]

    tables: Dict[str, pd.DataFrame] = {}

    if ids and len(ids) > 0:
        if verbose:
            print(f"[geom] Using explicit VizieR IDs: {ids}")
        for catid in ids:
            got = _fetch_catalog_tables(catid, mirrors, timeout_sec, retries, verbose)
            tables.update(got)
        return tables

    # 否則關鍵字搜尋
    Vizier.ROW_LIMIT = -1
    Vizier.columns = ["**"]
    queries = [
        "S4G edge-on", "Comeron S4G edge-on", "Spitzer S4G edge-on vertical",
        "S4G thick thin disk", "Comeron vertical structure S4G"
    ]
    seen: set[str] = set()
    for q in queries:
        try:
            catmap = Vizier.find_catalogs(q)
        except Exception as e:
            if verbose:
                print(f"[geom] find_catalogs('{q}') failed: {e}")
            continue
        for catid in catmap.keys():
            if catid in seen:
                continue
            seen.add(catid)
            got = _fetch_catalog_tables(catid, mirrors, timeout_sec, retries, verbose)
            tables.update(got)
    return tables

def build_s4g_h_catalog(
    sparc_tidy_csv: str = "dataset/sparc_tidy.csv",
    out_csv: str = "dataset/geometry/h_catalog.csv",
    prefer: str = "thin",                 # "thin" or "thick"
    default_rel_err: float = 0.25,
    ids: Optional[List[str]] = None,      # 例如 ["J/A+A/548/A126","J/A+A/533/A104"]
    mirror: Optional[str] = None,         # 首選鏡像（可為 None）
    timeout_sec: float = 120.0,           # 每請求逾時
    retries: int = 2,                     # 每鏡像重試次數
    verbose: bool = False,
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

    tables = _gather_s4g_edgeon_tables(
        ids=ids, mirror=mirror, timeout_sec=timeout_sec, retries=retries, verbose=verbose
    )
    if not tables:
        raise RuntimeError("在 VizieR 找不到 S4G edge-on 相關可用表格；請稍後再試或檢查網路。")

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

            for (val_col, err_col, flavor) in th_cols:
                try:
                    val = float(row.get(val_col, np.nan))
                except Exception:
                    val = np.nan
                if not np.isfinite(val):
                    continue

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

                # 座標
                RA = float(row.get(ra_col)) if (ra_col and str(row.get(ra_col)) not in ("", "nan", "None")) else np.nan
                DE = float(row.get(de_col)) if (de_col and str(row.get(de_col)) not in ("", "nan", "None")) else np.nan

                # 偏好分數（優先選薄盤；同分選較小 h）
                score = 0.0
                if flavor == "thin":
                    score += 2.0
                if prefer == "thin" and flavor == "thin":
                    score += 2.0
                if prefer == "thick" and ("thick" in str(val_col).lower()):
                    score += 2.0
                score += float(-h_kpc)

                cand_rows.append(dict(
                    nn=nn,
                    galaxy=g2orig[nn],
                    h_kpc=h_kpc,
                    h_err_kpc=float(h_err_kpc),
                    RA_deg=(float(RA) if np.isfinite(RA) else np.nan),
                    DEC_deg=(float(DE) if np.isfinite(DE) else np.nan),
                    profile=("thin" if flavor=="thin" else ("thick" if "thick" in str(val_col).lower() else "unknown")),
                    band="3.6um",
                    source=str(catkey),
                    score=float(score),
                    _val_col=val_col,
                    _err_col=(err_col or "")
                ))

    if not cand_rows:
        raise RuntimeError("雖找到 S4G 相關表格，但無法對齊到任何 SPARC 星系或缺少厚度欄。")

    cand = pd.DataFrame(cand_rows)
    cand.sort_values(["nn", "score", "h_kpc"], ascending=[True, False, True], inplace=True)
    best = cand.groupby("nn", as_index=False).first()

    out = best[["galaxy","h_kpc","h_err_kpc","RA_deg","DEC_deg","band","source","profile"]].copy()
    out = out.sort_values("galaxy").reset_index(drop=True)

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    if verbose:
        print(f"[geom] Wrote {out_path} (N={len(out)})")

    return out
