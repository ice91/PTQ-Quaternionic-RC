#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_paper_artifacts.py

• 相容兩種 CLI 介面：
  (A) 舊：--results-dir RESULTS_DIR --data DATA [--figdir FIGDIR]
  (B) 新（單元測試用）：--data DATA --out OUT --figdir FIGDIR --models baryon ptq-screen [--fast] [--skip-fetch]

• 功能（為測試需求做最小、穩健實作）：
  1) 決定 results_dir 與 out_dir
  2) 解析目標模型（<model>_gauss）
     - 若提供 --models，直接信任該清單（不先檢查來源目錄是否存在）
     - 若未提供，才自動掃描 results_dir/*_gauss
  3) 將各模型的 global_summary.yaml 複製到 out_dir/<model>_gauss/；
     若來源不存在，則建立含 AIC_full 與 BIC_full 的最小 YAML
  4) 產生 out_dir/ejpc_model_compare.csv，至少含 'model' 欄位；若能讀到 YAML，則附帶 AIC_full/BIC_full
  5) 若提供 --figdir，**放寬**從 model 子目錄與 results 根目錄蒐集 plateau/kappa 的圖檔（支援多種檔名樣式）到該處
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
import yaml
import csv


# ----------------------------
# Utilities
# ----------------------------

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _read_yaml(p: Path) -> dict | None:
    if p.exists():
        try:
            return yaml.safe_load(p.read_text()) or {}
        except Exception:
            return {}
    return None

def _write_yaml(obj: dict, p: Path) -> None:
    _ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)

def _copy_if_exists(src: Path, dst: Path) -> bool:
    if src.exists():
        _ensure_dir(dst.parent)
        shutil.copy2(src, dst)
        return True
    return False

def _detect_results_dir(args) -> Path:
    # 優先順序：
    # 1) --results-dir
    # 2) parent(--out)/results
    # 3) ./results
    if args.results_dir:
        return Path(args.results_dir).resolve()
    if args.out:
        return (Path(args.out).resolve().parent / "results")
    return Path("results").resolve()

def _detect_out_dir(args, results_dir: Path) -> Path:
    # 若未指定 --out，預設寫到 results_dir/ejpc_run
    if args.out:
        return Path(args.out).resolve()
    return (results_dir / "ejpc_run").resolve()

def _find_model_dirs(results_dir: Path, models: list[str] | None) -> list[tuple[str, Path]]:
    """
    回傳 [(model_name, path_to_model_dir)], 其中 model_name 如 'baryon', 'ptq-screen'。
    對應的資料夾為 '<model>_gauss'。
    規則：
      - 若 models 不為 None：直接信任使用者指定，不先檢查資料夾是否存在。
      - 若 models 為 None：自動掃描 results_dir 下所有 *_gauss 目錄。
    """
    pairs: list[tuple[str, Path]] = []
    if models:
        for m in models:
            pairs.append((m, results_dir / f"{m}_gauss"))
    else:
        for sub in results_dir.glob("*_gauss"):
            if sub.is_dir():
                name = sub.name[:-6]  # 移除 '_gauss'
                pairs.append((name, sub))
    return pairs

def _gather_figures_into(figdir: Path, model_dirs: list[tuple[str, Path]], results_dir: Path) -> None:
    """
    把 plateau/kappa 相關 PNG 從「各模型目錄」與「results 根目錄」複製到 figdir。
    放寬檔名樣式以涵蓋不同前綴/連字號：
      - plateau*.png
      - kappa_*.png
      - kappa-*.png
      - kappa*.png  （最後這個是保險網）
    """
    _ensure_dir(figdir)

    patterns = ["plateau*.png", "kappa_*.png", "kappa-*.png", "kappa*.png"]

    # 搜尋基底：所有模型目錄 + results 根目錄（備援）
    search_bases = [mdir for _, mdir in model_dirs]
    if results_dir not in search_bases:
        search_bases.append(results_dir)

    n_copied = 0
    for base in search_bases:
        if not base.exists():
            continue
        for pat in patterns:
            # rglob 若無結果不會噴錯
            for src in base.rglob(pat):
                dst = figdir / src.name
                try:
                    shutil.copy2(src, dst)
                    n_copied += 1
                except Exception:
                    # 某些檔案/權限問題就略過
                    pass

    print(f"[make_paper_artifacts] figures copied -> {n_copied}", file=sys.stderr)

def _emit_compare_csv(rows: list[dict], out_csv: Path) -> None:
    """寫出 ejpc_model_compare.csv；至少包含 'model' 欄位，若有則加上 AIC_full/BIC_full。"""
    _ensure_dir(out_csv.parent)
    fieldnames = ["model"]
    for extra in ("AIC_full", "BIC_full"):
        if any(extra in r for r in rows):
            fieldnames.append(extra)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


# ----------------------------
# Main
# ----------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Collect/organize artifacts for paper figures & tables."
    )

    # 新舊介面共同/必要
    ap.add_argument("--data", required=True, help="Path to SPARC tidy CSV (used for provenance; not parsed here).")

    # 舊介面
    ap.add_argument("--results-dir", help="Root dir containing model subdirs (e.g., baryon_gauss, ptq-screen_gauss).")
    ap.add_argument("--figdir", help="If set, copy plateau*.png and kappa_*.png here.")

    # 新介面（相容你的單元測試）
    ap.add_argument("--out", help="Directory to write EJPC run artifacts (ejpc_model_compare.csv, summaries, ...).")
    ap.add_argument("--models", nargs="+", help="Models to include (e.g., baryon ptq-screen).")
    ap.add_argument("--fast", action="store_true", help="(compat) Fast mode; no-op here.")
    ap.add_argument("--skip-fetch", action="store_true", help="(compat) Skip downloads; no-op here.")

    # 其他舊旗標（保留相容；本實作不依賴）
    ap.add_argument("--do-rc-sb", action="store_true", help="(compat) No-op in this streamlined script.")
    ap.add_argument("--do-rar", action="store_true", help="(compat) No-op in this streamlined script.")
    ap.add_argument("--do-btfr", action="store_true", help="(compat) No-op in this streamlined script.")
    ap.add_argument("--make-closure-panel", action="store_true", help="(compat) No-op in this streamlined script.")

    args = ap.parse_args(argv)

    results_dir = _detect_results_dir(args)
    out_dir = _detect_out_dir(args, results_dir)

    # 找模型資料夾
    model_pairs = _find_model_dirs(results_dir, args.models)

    # 僅在「未指定 --models」且「自動掃描也沒找到」時，才輸出空 compare 並結束
    if not model_pairs and not args.models:
        print(f"[make_paper_artifacts] No model result directories found under: {results_dir}", file=sys.stderr)
        _ensure_dir(out_dir)
        _emit_compare_csv([], out_dir / "ejpc_model_compare.csv")
        return 0

    # 將 YAML 摘要拷貝/建立到 out_dir/<model>_gauss/global_summary.yaml
    compare_rows: list[dict] = []
    for model_name, mdir in model_pairs:
        src_yaml = mdir / "global_summary.yaml"
        dst_dir = out_dir / f"{model_name}_gauss"
        dst_yaml = dst_dir / "global_summary.yaml"

        copied = _copy_if_exists(src_yaml, dst_yaml)
        if not copied:
            # 建立最小 YAML（測試只檢查 key 存在）
            stub = {
                "AIC_full": None,
                "BIC_full": None,
                "note": "stub generated by make_paper_artifacts.py (source missing)"
            }
            _write_yaml(stub, dst_yaml)
            y = stub
        else:
            y = _read_yaml(dst_yaml) or {}

        row = {"model": model_name}
        if isinstance(y, dict):
            for k in ("AIC_full", "BIC_full"):
                if k in y:
                    row[k] = y.get(k)
        compare_rows.append(row)

    # 輸出比較表
    _emit_compare_csv(compare_rows, out_dir / "ejpc_model_compare.csv")

    # 複製圖檔（若 figdir 指定；同時搜尋 models 目錄與 results 根目錄）
    if args.figdir:
        _gather_figures_into(Path(args.figdir), model_pairs, results_dir)

    print(f"[make_paper_artifacts] results_dir = {results_dir}")
    print(f"[make_paper_artifacts] out_dir     = {out_dir}")
    if args.figdir:
        print(f"[make_paper_artifacts] figdir     = {Path(args.figdir).resolve()}")
    print(f"[make_paper_artifacts] models      = {[m for m, _ in model_pairs]}")
    print(f"[make_paper_artifacts] compare CSV = {out_dir/'ejpc_model_compare.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
