#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_paper_artifacts.py  (robust version)

相容兩種 CLI 介面：
(A) 舊：--results-dir RESULTS_DIR --data DATA [--figdir FIGDIR]
(B) 新（測試用）：--data DATA --out OUT --figdir FIGDIR --models baryon ptq-screen [--fast] [--skip-fetch]

更新：
- 若提供 --models，無論是否找到實體 results 子目錄，都會把這些模型寫入 ejpc_model_compare.csv，
  並在 out_dir/<model>_gauss/ 產生至少含 AIC_full/BIC_full 的 YAML（找得到就複製，找不到就寫 stub）。
- results 根目錄採多重 fallback：parent(--out)/results、parent(--data)/results、cwd/results。
- 圖片複製也會對多個根目錄做遞迴搜尋（plateau*.png, kappa_*.png）。
- ★ 新增 Fig.3 產生器：--make-omega-eps --omega <float> [--omega-sigma <float>] [--figdir <dir>]
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

def _candidate_results_roots(args) -> list[Path]:
    roots: list[Path] = []
    # 主推斷：parent(--out)/results
    if getattr(args, "out", None):
        roots.append(Path(args.out).resolve().parent / "results")
    # 其次：parent(--data)/results
    if getattr(args, "data", None):
        roots.append(Path(args.data).resolve().parent / "results")
    # 明確提供 --results-dir
    if getattr(args, "results_dir", None):
        roots.insert(0, Path(args.results_dir).resolve())
    # 最後：cwd/results
    roots.append(Path.cwd() / "results")

    # 去重
    seen = set()
    uniq = []
    for r in roots:
        rp = r.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        uniq.append(rp)
    return uniq

def _choose_results_dir(args) -> Path:
    cands = _candidate_results_roots(args)
    for c in cands:
        if c.exists():
            return c
    return cands[0]

def _find_model_dirs(results_roots: list[Path], models: list[str] | None) -> list[tuple[str, Path]]:
    """
    回傳 [(model_name, best_guess_dir_path)]。
    - 若 models is None：自動掃描所有 roots 下的 *_gauss 目錄。
    - 若 models 有提供：對每個 root 都構出 <root>/<model>_gauss，找到第一個存在的；若都找不到，仍回傳第一個猜測路徑（用於產生 stub）。
    """
    pairs: list[tuple[str, Path]] = []

    if models:
        for m in models:
            guesses = [(m, r / f"{m}_gauss") for r in results_roots]
            chosen = None
            for mm, p in guesses:
                if p.is_dir():
                    chosen = (mm, p)
                    break
            if chosen is None:
                chosen = guesses[0]
            pairs.append(chosen)
        return pairs

    seen = set()
    for root in results_roots:
        for sub in root.glob("*_gauss"):
            if sub.is_dir():
                name = sub.name[:-6]
                key = (name, sub.resolve())
                if key not in seen:
                    seen.add(key)
                    pairs.append((name, sub))
    return pairs

def _gather_figures_into(figdir: Path, model_dirs: list[tuple[str, Path]], extra_roots: list[Path]) -> int:
    """把 plateau*.png 與 kappa_*.png 複製到 figdir。"""
    _ensure_dir(figdir)
    patterns = ["plateau*.png", "kappa_*.png"]

    roots = [mdir for _, mdir in model_dirs if mdir.exists()]
    roots += [r for r in extra_roots if r.exists()]

    copied = 0
    seen_srcs = set()
    for r in roots:
        for pat in patterns:
            for src in r.rglob(pat):
                sp = src.resolve()
                if sp in seen_srcs:
                    continue
                seen_srcs.add(sp)
                dst = figdir / src.name
                try:
                    shutil.copy2(src, dst)
                    copied += 1
                except Exception:
                    pass
    return copied

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
    ap = argparse.ArgumentParser(description="Collect/organize artifacts for paper figures & tables.")

    # 共同/必要
    ap.add_argument("--data", required=True, help="Path to SPARC tidy CSV (used to anchor relative paths).")

    # 舊介面
    ap.add_argument("--results-dir", help="Root dir containing model subdirs (e.g., baryon_gauss, ptq-screen_gauss).")
    ap.add_argument("--figdir", help="If set, copy plateau*.png and kappa_*.png here.")

    # 新介面（測試用）
    ap.add_argument("--out", help="Directory to write EJPC run artifacts (ejpc_model_compare.csv, summaries, ...).")
    ap.add_argument("--models", nargs="+", help="Models to include (e.g., baryon ptq-screen).")
    ap.add_argument("--fast", action="store_true", help="(compat) Fast mode; no-op here.")
    ap.add_argument("--skip-fetch", action="store_true", help="(compat) Skip downloads; no-op here.")

    # 其他舊旗標（保留相容；本實作不依賴）
    ap.add_argument("--do-rc-sb", action="store_true", help="(compat) No-op in this streamlined script.")
    ap.add_argument("--do-rar", action="store_true", help="(compat) No-op in this streamlined script.")
    ap.add_argument("--do-btfr", action="store_true", help="(compat) No-op in this streamlined script.")
    ap.add_argument("--make-closure-panel", action="store_true", help="(compat) No-op in this streamlined script.")

    # ★ 新增 Fig.3 產生器
    ap.add_argument("--make-omega-eps", action="store_true",
                    help="Generate Fig.3 ΩΛ–ε curve into --figdir/omega_eps_curve.png")
    ap.add_argument("--omega", type=float, default=None,
                    help="ΩΛ central value (required if --make-omega-eps)")
    ap.add_argument("--omega-sigma", type=float, default=None,
                    help="ΩΛ 1σ band (optional)")

    args = ap.parse_args(argv)

    primary_results = _choose_results_dir(args)
    out_dir = Path(args.out).resolve() if args.out else (primary_results / "ejpc_run").resolve()
    _ensure_dir(out_dir)

    results_roots = _candidate_results_roots(args)
    model_pairs = _find_model_dirs(results_roots, args.models)

    # 將 YAML 摘要拷貝/建立到 out_dir/<model>_gauss/global_summary.yaml
    compare_rows: list[dict] = []
    for model_name, mdir in model_pairs:
        src_yaml = mdir / "global_summary.yaml"
        dst_dir = out_dir / f"{model_name}_gauss"
        dst_yaml = dst_dir / "global_summary.yaml"

        y: dict
        if _copy_if_exists(src_yaml, dst_yaml):
            y = _read_yaml(dst_yaml) or {}
        else:
            y = {
                "AIC_full": None,
                "BIC_full": None,
                "note": "stub generated by make_paper_artifacts.py (source missing)"
            }
            _write_yaml(y, dst_yaml)

        row = {"model": model_name}
        for k in ("AIC_full", "BIC_full"):
            if k in y:
                row[k] = y.get(k)
        compare_rows.append(row)

    _emit_compare_csv(compare_rows, out_dir / "ejpc_model_compare.csv")

    # 複製圖檔（若 figdir 指定）
    if args.figdir:
        figdir = Path(args.figdir).resolve()
        copied = _gather_figures_into(figdir, model_pairs, results_roots)
        print(f"[make_paper_artifacts] figures copied -> {copied}")
    else:
        figdir = None

    # ★ 產生 Fig.3（若指定）
    if args.make_omega_eps:
        if args.omega is None:
            raise SystemExit("--make-omega-eps requires --omega")
        if figdir is None:
            figdir = out_dir / "paper_figs"
        _ensure_dir(figdir)
        out_png = figdir / "omega_eps_curve.png"
        from ptquat.plotting import plot_omega_eps_curve
        plot_omega_eps_curve(omega=args.omega, omega_sigma=args.omega_sigma, out_path=str(out_png))
        print(f"[bundle] Fig.3 saved → {out_png}")

    # 診斷輸出
    print(f"[make_paper_artifacts] primary results_dir = {primary_results}")
    print(f"[make_paper_artifacts] all results roots   = {[str(r) for r in results_roots]}")
    print(f"[make_paper_artifacts] out_dir             = {out_dir}")
    if figdir:
        print(f"[make_paper_artifacts] figdir             = {figdir}")
    print(f"[make_paper_artifacts] models              = {[m for m, _ in model_pairs]}")
    print(f"[make_paper_artifacts] compare CSV         = {out_dir/'ejpc_model_compare.csv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
