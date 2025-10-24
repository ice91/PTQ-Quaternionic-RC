# scripts/suggest_aliases.py
import argparse
import re
from difflib import SequenceMatcher
from collections import defaultdict

import numpy as np
import pandas as pd

try:
    from ptq.data.s4g_h_pipeline import _norm_name
except Exception:
    # 後備：只留 A-Z0-9（與 _norm_name 等價）
    import re as _re
    _keep = _re.compile(r"[A-Z0-9]+")
    def _norm_name(s: str) -> str:
        s = _re.sub(r"\s+", " ", str(s)).upper().strip()
        return "".join(_keep.findall(s))


CAT_RE = {
    "NGC": re.compile(r"^NGC\s*0*(\d+)[A-Z]?$", re.I),
    "IC":  re.compile(r"^IC\s*0*(\d+)[A-Z]?$",  re.I),
    "UGC": re.compile(r"^UGC\s*0*(\d+)$",       re.I),
    "PGC": re.compile(r"^PGC\s*0*(\d+)$",       re.I),
    # ESO 可能寫成 "ESO 138-G005" 或 "ESO138-005"
    "ESO": re.compile(r"^ESO\s*0*(\d+)[-\s]*G?0*(\d+)$", re.I),
}

def to_s4g_style(name: str) -> str | None:
    """把常見的 SPARC 名稱轉成 S4G 風格（零填充/去 G）。命中才回字串，否則 None。"""
    s = str(name).strip().upper()
    for tag, pat in CAT_RE.items():
        m = pat.match(s)
        if not m: 
            continue
        if tag in ("NGC", "IC"):
            n = int(m.group(1))
            return f"{tag}{n:04d}"
        if tag == "UGC":
            n = int(m.group(1))
            # S4G 通常 5 位（UGC06512）
            return f"UGC{n:05d}"
        if tag == "PGC":
            n = int(m.group(1))
            # S4G 常見 6 位（PGC042068）
            return f"PGC{n:06d}"
        if tag == "ESO":
            a, b = int(m.group(1)), int(m.group(2))
            # S4G 風格：ESOddd-sss（無 G，後段 3 位）
            return f"ESO{a:03d}-{b:03d}"
    return None


def best_fuzzy_candidates(target_key: str, s4g_key2name: dict, topk: int = 2):
    """對沒命中的情況，用簡單的 difflib 在 key（A-Z0-9）上做模糊比對，回前 topk 名。"""
    scores = []
    for k in s4g_key2name.keys():
        r = SequenceMatcher(a=target_key, b=k, autojunk=False).ratio()
        scores.append((r, k))
    scores.sort(reverse=True)
    out = []
    for r, k in scores[:topk]:
        out.append((s4g_key2name[k], round(float(r), 3)))
    return out


def main():
    ap = argparse.ArgumentParser(description="Suggest aliases for SPARC↔S4G matching")
    ap.add_argument("--sparc", required=True, help="dataset/sparc_tidy.csv")
    ap.add_argument("--hcat", required=True, help="dataset/geometry/h_catalog.csv")
    ap.add_argument("--unmatched", required=True, help="dataset/geometry/unmatched_galaxies.csv")
    ap.add_argument("--out", required=True, help="輸出：dataset/geometry/aliases_auto.csv")
    ap.add_argument("--suggestions", required=False, default="dataset/geometry/aliases_suggestions.csv",
                    help="輸出：需要人工核對的建議清單")
    args = ap.parse_args()

    # 1) S4G 名錄：key → 原名
    h = pd.read_csv(args.hcat)
    if "Galaxy" not in h.columns:
        raise SystemExit(f"{args.hcat} 缺少 Galaxy 欄位")
    s4g_names = h["Galaxy"].dropna().astype(str)
    s4g_key2name = { _norm_name(g): g for g in s4g_names }

    # 2) SPARC 端：gkey → 原名（取第一個非 NaN）
    sp = pd.read_csv(args.sparc)
    if "galaxy" not in sp.columns:
        raise SystemExit(f"{args.sparc} 缺少 galaxy 欄位")
    sp["gkey"] = sp["galaxy"].map(_norm_name)
    gkey2sname = (sp.groupby("gkey")["galaxy"]
                    .apply(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
                    .to_dict())

    # 3) 未配對清單
    um = pd.read_csv(args.unmatched)
    if "gkey" not in um.columns:
        raise SystemExit(f"{args.unmatched} 需包含 gkey 欄位")
    targets = um["gkey"].dropna().astype(str).unique().tolist()

    auto_rows = []
    sug_rows  = []

    hit_auto = 0
    for gk in targets:
        s_name = gkey2sname.get(gk, gk)  # 沒原名就先用 key 代替
        # 嘗試規則轉換
        guess = to_s4g_style(s_name)
        if guess is not None:
            gk_guess = _norm_name(guess)
            if gk_guess in s4g_key2name:
                auto_rows.append({"galaxy": s_name, "h_galaxy": s4g_key2name[gk_guess]})
                hit_auto += 1
                continue  # 自動命中，不再模糊

        # 模糊建議（僅供人工核對）
        cands = best_fuzzy_candidates(gk, s4g_key2name, topk=2)
        row = {"galaxy": s_name, "h_galaxy": ""}
        if len(cands) > 0:
            row.update({"candidate1": cands[0][0], "score1": cands[0][1]})
        if len(cands) > 1:
            row.update({"candidate2": cands[1][0], "score2": cands[1][1]})
        sug_rows.append(row)

    # 4) 輸出
    if auto_rows:
        pd.DataFrame(auto_rows).to_csv(args.out, index=False)
        print(f"Auto aliases -> {args.out} (rows={len(auto_rows)})")
    else:
        # 仍輸出空表，方便後續流程
        pd.DataFrame(columns=["galaxy","h_galaxy"]).to_csv(args.out, index=False)
        print(f"Auto aliases -> {args.out} (rows=0)")

    if sug_rows:
        pd.DataFrame(sug_rows).to_csv(args.suggestions, index=False)
        print(f"Suggestions -> {args.suggestions} (rows={len(sug_rows)})")

    print(f"Auto-hit from {len(targets)} unmatched: {hit_auto} ({hit_auto/len(targets)*100:.1f}%)")


if __name__ == "__main__":
    main()
