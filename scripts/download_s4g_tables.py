# scripts/download_s4g_tables.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import sys

import pandas as pd
from astroquery.vizier import Vizier, conf

CAT_IDS = [
    "J/A+A/548/A126",  # Comerón+ 2012 (S4G edge-on)
    "J/A+A/533/A104",  # Comerón+ 2011 (S4G edge-on)
]

MIRRORS = [
    "vizier.cfa.harvard.edu",   # US (CfA)
    "vizier.nao.ac.jp",         # Japan (NAOJ)
    "vizier.hia.nrc.ca",        # Canada (HIA)
    "vizier.cds.unistra.fr",    # France (CDS)
]

def fetch_one_catalog(catid: str, outdir: Path, timeout_sec: int = 120, retries: int = 3) -> int:
    Vizier.ROW_LIMIT = -1
    Vizier.columns   = ["**"]
    saved = 0
    for mirror in MIRRORS:
        conf.server  = mirror
        conf.timeout = int(timeout_sec)  # 必須是 int
        last_err = None
        for attempt in range(1, retries+1):
            try:
                tlist = Vizier.get_catalogs(catid)  # TableList
                for i, tbl in enumerate(tlist):
                    try:
                        df = tbl.to_pandas()
                        df = df.copy()
                        # 儲存：<catid>__t<i>__<mirror>.csv
                        safe_cat = catid.replace("/", "_")
                        fn = outdir / f"{safe_cat}__t{i}__{mirror}.csv"
                        df.to_csv(fn, index=False)
                        print(f"[OK] {catid} table#{i} rows={len(df)} -> {fn}")
                        saved += 1
                    except Exception as e:
                        print(f"[WARN] {catid} table#{i} to_pandas fail: {e}")
                if saved > 0:
                    return saved  # 此 catalog 至少有一表成功 → 換下一個 catalog
            except Exception as e:
                last_err = e
                print(f"[RETRY] get_catalogs({catid})@{mirror} attempt {attempt}/{retries} failed: {e}")
        print(f"[SKIP] {catid} on mirror {mirror} all attempts failed: {last_err}")
    return saved

def main():
    outdir = Path("dataset/raw/edgeon")
    outdir.mkdir(parents=True, exist_ok=True)
    total = 0
    for catid in CAT_IDS:
        total += fetch_one_catalog(catid, outdir, timeout_sec=180, retries=3)
    if total == 0:
        print("[ERROR] No tables saved. Check network/firewall or mirrors.")
        sys.exit(2)
    print(f"[DONE] Saved {total} CSV tables under {outdir}")

if __name__ == "__main__":
    main()
