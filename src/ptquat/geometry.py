# src/ptquat/geometry.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict
import re
import numpy as np
import pandas as pd

ARCSEC_TO_RAD = np.pi / 648000.0  # 1" in rad

# ---- 名稱正規化（盡量把 "ESO 116-G12" / "ESO 116-12" / "ESO116-12" 視為同一）----
def normalize_galname(name: str) -> str:
    s = str(name).strip().upper()
    # 同一化分隔符
    s = re.sub(r"[_.,;:]+", " ", s)
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")
    s = re.sub(r"\s*-\s*", "-", s)
    s = re.sub(r"\s+", " ", s)
    # 常見前綴清洗（保留編號）
    s = s.replace("ESO ", "ESO").replace("UGC ", "UGC").replace("NGC ", "NGC").replace("IC ", "IC")
    s = s.replace("PGC ", "PGC").replace("DDO ", "DDO").replace("M ", "M")
    # "ESO116-G12" -> "ESO116-12"
    s = re.sub(r"ESO(\d+)-G(\d+)", r"ESO\1-\2", s)
    return s

# ---- 從 SPARC tidy 取距離（kpc）以供角尺度轉換 ----
def _sparc_distance_map(sparc_tidy_csv: str) -> Dict[str, float]:
    df = pd.read_csv(sparc_tidy_csv)
    if not {"galaxy","D_Mpc"}.issubset(df.columns):
        raise ValueError(f"{sparc_tidy_csv} 必須包含 columns {{'galaxy','D_Mpc'}}")
    # 一星系一距離
    d1 = df.groupby("galaxy", as_index=False)["D_Mpc"].first()
    d1["norm"] = d1["galaxy"].map(normalize_galname)
    return dict(zip(d1["norm"], (1e3*d1["D_Mpc"]).astype(float)))  # kpc

# ---- 欄位自動偵測（來源 CSV 很雜也能吃）----
_CAND_NAME   = ["galaxy","name","Name","object","Object","ID"]
_CAND_H_EXP  = ["h_kpc","hz_kpc","hz","H_z","H_kpc","scaleheight_kpc"]    # 已是 kpc 或 “exp 等效 h”
_CAND_H_PC   = ["h_pc","hz_pc","z0_pc","scaleheight_pc"]
_CAND_H_ARC  = ["h_arcsec","hz_arcsec","z0_arcsec","scaleheight_arcsec","h_as","hz_as"]
_CAND_ERR    = ["h_err_kpc","hz_err_kpc","err_h_kpc","e_h_kpc","h_err","hz_err","e_h","e_hz"]
_CAND_RA     = ["RA_deg","ra_deg","RAJ2000","RA"]
_CAND_DEC    = ["DEC_deg","dec_deg","DEJ2000","DEC"]
_CAND_BAND   = ["band","Band","filter","Filter","lambda","wavelength"]
_CAND_SOURCE = ["source","Source","ref","Ref","bibcode","BibCode"]
_CAND_COMP   = ["component","Component","disk","Disk"]   # 'thin'/'thick' 可從這看
_CAND_PROF   = ["profile","Profile"]                    # 'exp'/'sech2' 等

# ---- 近紅外優先等分數 ----
_BAND_SCORE = {
    "3.6UM": 100, "IRAC1": 100, "K": 95, "KS": 95, "K_S": 95, "K_S2MASS": 95, "H": 90, "J": 85,
    "I": 70, "R": 60, "R_C": 60, "SDSS I": 70, "SDSS R": 60, "SDSS Z": 65,
    "Z": 65, "Y": 65, "G": 40,
}
def _score_band(band: str | float | int | None) -> int:
    if band is None or (isinstance(band, float) and not np.isfinite(band)):
        return 1
    s = str(band).strip().upper()
    return _BAND_SCORE.get(s, _BAND_SCORE.get(s.replace(" ", ""), 10))

def _score_component(comp: str | None) -> int:
    if comp is None: return 0
    s = str(comp).strip().lower()
    if "thin" in s:  return +20
    if "thick" in s: return -5
    return 0

@dataclass
class HRow:
    galaxy_src: str
    galaxy_norm: str
    h_kpc: float
    h_err_kpc: float | None
    RA_deg: float | None
    DEC_deg: float | None
    band: str | None
    source: str | None
    component: str | None
    profile: str | None
    score: float

def _pick_first(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns: return c
    return None

def _col(df: pd.DataFrame, names: List[str]) -> Optional[pd.Series]:
    cname = _pick_first(df, names)
    return (df[cname] if cname else None)

def _to_float_series(x: pd.Series | None) -> Optional[pd.Series]:
    if x is None: return None
    return pd.to_numeric(x, errors="coerce")

def _to_str_series(x: pd.Series | None) -> Optional[pd.Series]:
    if x is None: return None
    return x.astype(str)

def _infer_profile_from_colname(colname: str) -> Optional[str]:
    cn = colname.lower()
    if "z0" in cn or "z_0" in cn or "sech" in cn: return "sech2"
    if "hz" in cn or re.search(r"(?:^|_)h(?:_|$)", cn): return "exp"
    return None

def _standardize_one_source(path: str | Path,
                            Dkpc_map: Dict[str, float],
                            default_rel_err: float = 0.25,
                            assume_profile_if_unknown: str = "exp") -> List[HRow]:
    p = Path(path)
    df = pd.read_csv(p)
    # 找名稱欄
    name_s = _to_str_series(_col(df, _CAND_NAME))
    if name_s is None:
        raise ValueError(f"{p} 缺少 galaxy/name 欄位（候選: {_CAND_NAME}）")

    # 找厚度欄（多種單位可能共存；一個來源可同時提供多欄，我們逐欄取值）
    cols_h = []
    for group in (_CAND_H_EXP, _CAND_H_PC, _CAND_H_ARC):
        for c in group:
            if c in df.columns:
                cols_h.append(c)
    if not cols_h:
        raise ValueError(f"{p} 無法找到任何厚度欄位，請至少提供一種：kpc/pc/arcsec（候選: {_CAND_H_EXP+_CAND_H_PC+_CAND_H_ARC}）")

    # 其他欄
    err_s   = _to_float_series(_col(df, _CAND_ERR))
    ra_s    = _to_float_series(_col(df, _CAND_RA))
    dec_s   = _to_float_series(_col(df, _CAND_DEC))
    band_s  = _to_str_series(_col(df, _CAND_BAND))
    src_s   = _to_str_series(_col(df, _CAND_SOURCE))
    comp_s  = _to_str_series(_col(df, _CAND_COMP))
    prof_s  = _to_str_series(_col(df, _CAND_PROF))

    out: List[HRow] = []
    for idx, gname in name_s.items():
        gnorm = normalize_galname(gname)
        Dkpc = Dkpc_map.get(gnorm, np.nan)  # 供 arcsec→kpc
        band = band_s.iloc[idx] if band_s is not None and idx in band_s.index else None
        src  = src_s.iloc[idx]  if src_s  is not None and idx in src_s.index  else p.stem
        comp = comp_s.iloc[idx] if comp_s is not None and idx in comp_s.index else None
        prof = prof_s.iloc[idx] if prof_s is not None and idx in prof_s.index else None
        ra   = float(ra_s.iloc[idx])  if (ra_s is not None and np.isfinite(ra_s.iloc[idx]))   else None
        dec  = float(dec_s.iloc[idx]) if (dec_s is not None and np.isfinite(dec_s.iloc[idx])) else None

        # 掃描可用厚度欄，一個 row 可能產生多個候選值（不同定義/單位）
        for c in cols_h:
            val = pd.to_numeric(df.loc[idx, c], errors="coerce")
            if not np.isfinite(val): 
                continue

            # 推斷 profile（若欄名含 z0 則視為 sech2；否則優先 exp）
            prof_here = prof or _infer_profile_from_colname(c) or assume_profile_if_unknown

            # 單位轉 kpc
            ck = c.lower()
            if c in _CAND_H_EXP:            # 已是 exp 定義且為 kpc
                h_kpc = float(val)
            elif c in _CAND_H_PC:           # pc → kpc
                h_kpc = float(val) / 1e3
            elif c in _CAND_H_ARC:          # arcsec → kpc（需要距離）
                if not np.isfinite(Dkpc):
                    # 沒有距離就沒法轉換
                    continue
                h_kpc = float(val) * Dkpc * ARCSEC_TO_RAD
            else:
                # 容錯：依欄名猜測
                if "arcsec" in ck or ck.endswith("_as"):
                    if not np.isfinite(Dkpc): 
                        continue
                    h_kpc = float(val) * Dkpc * ARCSEC_TO_RAD
                elif ck.endswith("_pc"):
                    h_kpc = float(val) / 1e3
                else:
                    h_kpc = float(val)

            # 若 profile=sech2（z0），換算到 exp 等效 h ≈ z0/2
            if prof_here and str(prof_here).lower().startswith("sech"):
                h_kpc = h_kpc / 2.0

            # 誤差（若沒提供，用相對誤差）
            if err_s is not None and np.isfinite(err_s.iloc[idx]):
                herr = float(err_s.iloc[idx])
                # 嘗試判斷單位：如果同欄位代表與厚度同單位（多數情況），則上面已轉成 kpc 之前就該換算；
                # 這裡採保守策略：若 herr/val > 5，視為不同單位不可信，改用相對誤差。
                if val != 0 and (herr/abs(val) < 5.0):
                    # 以“比例”近似帶到 kpc 值
                    rel = abs(herr/val)
                    h_err_kpc = rel * abs(h_kpc)
                else:
                    h_err_kpc = default_rel_err * abs(h_kpc)
            else:
                h_err_kpc = default_rel_err * abs(h_kpc)

            score = _score_band(band) + _score_component(comp)
            out.append(HRow(
                galaxy_src=str(gname), galaxy_norm=gnorm,
                h_kpc=float(h_kpc), h_err_kpc=float(h_err_kpc),
                RA_deg=ra, DEC_deg=dec, band=band, source=src,
                component=comp, profile=prof_here, score=float(score)
            ))
    return out

def assemble_h_catalog(
    sparc_tidy_csv: str,
    source_csvs: Iterable[str | Path],
    out_csv: str | Path = "dataset/geometry/h_catalog.csv",
    prefer_thin: bool = True,
    default_rel_err: float = 0.25,
) -> pd.DataFrame:
    """
    將多個來源 CSV 統一彙整為單一 h catalog（kpc），以 SPARC 距離換算角尺度並對齊星系名稱。
    來源 CSV 欄位可彈性如下（任一即可）： 
      - 厚度：h_kpc / hz_kpc / hz / h_pc / z0_pc / h_arcsec / z0_arcsec / ...
      - 名稱：galaxy|name|Name|object|Object|ID
      - 誤差（選）：h_err_kpc|hz_err_kpc|e_h|...
      - RA/DEC（選）：RA_deg|DEC_deg|RAJ2000|DEJ2000
      - 波段（選）：band|filter|...
      - 組件（選）：component（thin/thick）
      - 簡併形狀（選）：profile（exp/sech2）
    """
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    Dkpc_map = _sparc_distance_map(sparc_tidy_csv)

    rows: List[HRow] = []
    for src in source_csvs:
        rows.extend(_standardize_one_source(src, Dkpc_map, default_rel_err=default_rel_err))

    if not rows:
        raise RuntimeError("沒有從來源 CSV 擷取到任何厚度資料。請確認來源與欄位命名。")

    df = pd.DataFrame([r.__dict__ for r in rows])

    # 僅保留也存在於 SPARC 的星系
    df = df.loc[df["galaxy_norm"].isin(Dkpc_map.keys())].copy()
    if df.empty:
        raise RuntimeError("來源內的星系名稱與 SPARC 對不起來；請檢查命名或補一個 'alias' 欄。")

    # 去重：每星系挑一筆（規則：分數最高；若 tie，prefer thin；再 tie，選 h_kpc 最小值）
    def _pick(group: pd.DataFrame) -> pd.Series:
        g = group.copy()
        g["__rank"] = g["score"].rank(method="max", ascending=False)
        top = g.loc[g["__rank"] == g["__rank"].min()]
        if len(top) > 1 and prefer_thin:
            thin = top.loc[top["component"].astype(str).str.lower().str.contains("thin", na=False)]
            if len(thin) > 0:
                top = thin
        if len(top) > 1:
            top = top.sort_values("h_kpc", ascending=True).iloc[[0]]
        return top.iloc[0]

    picked = df.groupby("galaxy_norm", as_index=False).apply(_pick).reset_index(drop=True)

    # 輸出欄位
    picked["galaxy"] = picked["galaxy_norm"]
    cols = ["galaxy","h_kpc","h_err_kpc","RA_deg","DEC_deg","band","source","component","profile","galaxy_src"]
    picked = picked[cols].copy()

    # 轉回較人類友善的 galaxy 名：用來源名稱（galaxy_src）的大寫修飾
    picked["galaxy"] = picked["galaxy_src"].astype(str)

    picked = picked.rename(columns={"galaxy_src":"matched_from"})
    picked = picked.sort_values("galaxy").reset_index(drop=True)

    picked.to_csv(out_csv, index=False)
    return picked

# ---- 讀取最終 h_catalog（供實驗使用）----
def load_h_catalog(path_csv: str) -> pd.DataFrame:
    df = pd.read_csv(path_csv)
    need = {"galaxy","h_kpc"}
    if not need.issubset(df.columns):
        raise ValueError(f"{path_csv} 必須至少包含欄位 {sorted(need)}")
    # 清理
    df["galaxy_norm"] = df["galaxy"].map(normalize_galname)
    df["h_kpc"] = pd.to_numeric(df["h_kpc"], errors="coerce")
    if "h_err_kpc" in df.columns:
        df["h_err_kpc"] = pd.to_numeric(df["h_err_kpc"], errors="coerce")
    else:
        df["h_err_kpc"] = np.nan
    return df
