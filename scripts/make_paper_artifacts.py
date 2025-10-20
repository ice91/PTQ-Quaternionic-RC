#!/usr/bin/env python3
# scripts/make_paper_artifacts.py  (finalized pipeline w/ gates)
from __future__ import annotations
import argparse, json, shutil, subprocess, sys, platform
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import yaml
import pandas as pd

from ptquat.fit_global import run as run_fit_direct
from ptquat import experiments as EXP


# ---------- utils ----------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copyfile(src, dst)

def _git_head() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return None

def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

# ---------- fitting ----------
def _run_fit(model: str, tidy_csv: Path, outdir: Path,
             likelihood: str, fast: bool, prior: str = "galaxies-only") -> Path:
    _ensure_dir(outdir)
    steps = "6000" if not fast else "200"
    nwalk = "4x"   if not fast else "2x"
    argv = [
        f"--data-path={tidy_csv}",
        f"--outdir={outdir}",
        f"--model={model}",
        f"--likelihood={likelihood}",
        f"--steps={steps}",
        f"--nwalkers={nwalk}",
        f"--seed=42",
        f"--prior={prior}",
        f"--sigma-sys=4.0",
    ]
    print("+ ptquat.fit_global.run", " ".join(argv), flush=True)
    run_fit_direct(argv)
    gpath = outdir / "global_summary.yaml"
    if not gpath.exists():
        raise RuntimeError(f"[make] {model} fit finished but {gpath} not found.")
    return gpath

def build_compare_csv(run_root: Path, out_csv: Path, models: list[str], like: str) -> pd.DataFrame:
    rows = []
    for m in models:
        rdir = run_root / f"{m}_{like}"
        gpath = rdir / "global_summary.yaml"
        if not gpath.exists():
            print(f"[make][warn] skip {m}: {gpath} not found", flush=True); continue
        y = yaml.safe_load(gpath.read_text())
        rows.append(dict(
            model=y.get("model", m),
            likelihood=y.get("likelihood", like),
            AIC_full=y.get("AIC_full"),
            BIC_full=y.get("BIC_full"),
            chi2_total=y.get("chi2_total"),
            k_parameters=y.get("k_parameters"),
            N_total=y.get("N_total"),
            outdir=str(rdir)
        ))
    df = pd.DataFrame(rows).sort_values("AIC_full")
    _ensure_dir(out_csv.parent)
    df.to_csv(out_csv, index=False)
    return df

# ---------- diagnostics on a chosen results_dir ----------
def run_diagnostics_for(results_dir: Path,
                        tidy_csv: Path,
                        figdir: Path,
                        omega_lambda: float,
                        eta: float,
                        nbins: int) -> None:
    """
    在 results_dir 上跑 plateau/ppc/closure + κ-profiles（fit & cos）
    + z-space 主圖/單點，並把主圖拷貝到 figdir。
    """
    # 1) plateau
    _, plateau_png = EXP.residual_plateau(str(results_dir), str(tidy_csv), nbins=nbins, out_prefix="plateau")
    # 2) PPC coverage
    ppc_json = EXP.ppc_check(str(results_dir), str(tidy_csv), out_prefix="ppc")
    (results_dir/"ppc_coverage.json").write_text(json.dumps(ppc_json, indent=2))
    # 3) closure
    cl = EXP.closure_test(str(results_dir), omega_lambda=omega_lambda)
    (results_dir/"closure_test.json").write_text(json.dumps(cl, indent=2))

    # 4) κ-profile（fit 與 cos）+ AB 擬合（含 bootstrap）
    for eps_norm, prefix in [("fit", "kappa_profile_A_fit"), ("cos","kappa_profile_A_cos")]:
        _, png = EXP.kappa_radius_resolved(
            results_dir=str(results_dir), data_path=str(tidy_csv),
            eta=eta, omega_lambda=(omega_lambda if eps_norm=="cos" else None),
            nbins=nbins, min_per_bin=20,
            x_kind="r_over_Rd", eps_norm=eps_norm, out_prefix=prefix
        )
        _copy_if_exists(png, figdir / f"{prefix}_{results_dir.name}.png")
        fit_sum = EXP.kappa_profile_fit(
            results_dir=str(results_dir), prefix=prefix, eps_norm=eps_norm,
            omega_lambda=(omega_lambda if eps_norm=="cos" else None), bootstrap=2000, seed=1234
        )
        (results_dir/f"{prefix}_fit_summary.json").write_text(json.dumps(fit_sum, indent=2))

    # 5) per-galaxy κ（obs-debias + cos）
    _ = EXP.kappa_per_galaxy(
        results_dir=str(results_dir), data_path=str(tidy_csv),
        eta=eta, frac_vmax=0.9, nsamp=300, omega_lambda=omega_lambda,
        y_source="obs-debias", eps_norm="cos", rstar_from="obs",
        out_prefix="kappa_gal_obsdebiased_cos"
    )
    _copy_if_exists(results_dir/"kappa_gal_obsdebiased_cos.png", figdir / f"kappa_gal_obsdebiased_cos_{results_dir.name}.png")

    # 6) z-profile（cos, 附理論零參數曲線）
    _, zpng = EXP.z_profile(
        results_dir=str(results_dir), data_path=str(tidy_csv),
        nbins=nbins, min_per_bin=20, eps_norm="cos",
        omega_lambda=omega_lambda, out_prefix="z_profile_cos",
        z_quantile_clip=(0.01, 0.99), do_theory=True
    )
    _copy_if_exists(zpng, figdir / f"z_profile_cos_{results_dir.name}.png")

    # 7) z-gal（每星系 r* 單點，obs-debias + cos）
    _ = EXP.z_per_galaxy(
        results_dir=str(results_dir), data_path=str(tidy_csv),
        frac_vmax=0.9, y_source="obs-debias", rstar_from="obs",
        eps_norm="cos", omega_lambda=omega_lambda, nsamp=300, out_prefix="z_gal_cos"
    )
    _copy_if_exists(results_dir/"z_gal_cos.png", figdir / f"z_gal_cos_{results_dir.name}.png")

    # 8) 收尾拷貝 plateau
    _copy_if_exists(plateau_png, figdir / f"plateau_{results_dir.name}.png")


# ---------- gating / summary ----------
@dataclass
class Gates:
    A_abs_max: float = 0.015      # |A| should be small (A≈0)
    B_min: float = 0.015          # B should be positive and not tiny
    B_max: float = 0.10           # generous upper bound
    ppc68_min: float = 0.20       # PPC 68% coverage reasonable lower bound
    ppc68_max: float = 0.60       # and upper bound (too-wide means overestimated errors)
    expect_closure_pass: bool = False  # we currently expect closure to fail

def _evaluate_and_write_summary(run_root: Path, results_dir: Path,
                                like: str, omega_lambda: float,
                                gates: Gates) -> dict:
    """Collect key numbers, check gates, and write one consolidated JSON + Markdown."""
    # load AB fit (cos)
    fit_json = results_dir / "kappa_profile_A_cos_fit_summary.json"
    if not fit_json.exists():
        raise FileNotFoundError(f"{fit_json} not found; was kappa_profile_fit() run?")
    fit = json.loads(fit_json.read_text())

    # closure
    clo = json.loads((results_dir/"closure_test.json").read_text())

    # ppc
    ppc = json.loads((results_dir/"ppc_coverage.json").read_text())

    # global (for H0)
    gsum = yaml.safe_load((results_dir/"global_summary.yaml").read_text())
    H0_si = float(gsum.get("H0_si"))

    # derive baseline Δa ≈ B*ε_cos*cH0
    eps_cos = float(fit["epsilon_cos"])
    A = float(fit["A"]); B = float(fit["B"])
    c = 299792458.0  # m/s
    delta_a = B * eps_cos * c * H0_si  # m s^-2

    # gates
    checks = {
        "A_small": (abs(A) <= gates.A_abs_max, {"A": A, "thresh": gates.A_abs_max}),
        "B_positive": (B >= gates.B_min, {"B": B, "min": gates.B_min}),
        "B_not_huge": (B <= gates.B_max, {"B": B, "max": gates.B_max}),
        "PPC68_reasonable": (gates.ppc68_min <= ppc["coverage68"] <= gates.ppc68_max,
                             {"coverage68": ppc["coverage68"], "range": [gates.ppc68_min, gates.ppc68_max]}),
        "Closure_expectation": ((clo["pass_within_3sigma"] is True) == gates.expect_closure_pass,
                                {"pass_within_3sigma": clo["pass_within_3sigma"],
                                 "expected": gates.expect_closure_pass})
    }
    status = "PASS" if all(ok for ok, _ in checks.values()) else "WARN"

    summary = dict(
        created_at=_now_iso(),
        platform=dict(python=sys.version.split()[0], system=platform.platform(), git=_git_head()),
        settings=dict(likelihood=like, omega_lambda=omega_lambda),
        paths=dict(results_dir=str(results_dir)),
        metrics=dict(
            A=A, B=B, R2=float(fit["R2"]),
            epsilon_cos=eps_cos, epsilon_fit=float(fit["epsilon_fit"]),
            H0_si=H0_si, delta_a_m_s2=delta_a,
            ppc_coverage68=float(ppc["coverage68"]), ppc_coverage95=float(ppc["coverage95"]),
            closure=clo,
        ),
        gates=dict(
            config=gates.__dict__,
            checks={k: dict(ok=ok, **info) for k,(ok,info) in checks.items()},
            overall=status
        )
    )

    out_json = run_root / "paper_artifacts_summary.json"
    out_md   = run_root / "paper_artifacts_summary.md"
    out_json.write_text(json.dumps(summary, indent=2))

    # a short human-readable MD
    lines = []
    lines.append(f"# Paper artifacts summary ({status})\n")
    lines.append(f"- git: `{summary['platform']['git']}`  | like: **{like}**  | ΩΛ={omega_lambda}\n")
    lines.append(f"- κ-profile (cos) fit: **A={A:.4f}**, **B={B:.4f}**, R²={summary['metrics']['R2']:.2f}")
    lines.append(f"- Δa baseline ≈ **{delta_a:.2e} m s⁻²**  (B·ε_cos·cH₀)\n")
    lines.append(f"- PPC 68/95% coverage: {ppc['coverage68']:.3f} / {ppc['coverage95']:.3f}")
    lines.append(f"- Closure pass_within_3σ: **{clo['pass_within_3sigma']}**  (expected: {gates.expect_closure_pass})\n")
    lines.append("## Gate checks\n")
    for k,(ok,info) in checks.items():
        lines.append(f"- {k}: **{'OK' if ok else 'WARN'}** — {json.dumps(info)}")
    out_md.write_text("\n".join(lines))

    print(f"[make] Wrote summary: {out_json}")
    print(f"[make]          MD : {out_md}")
    print(f"[make] Overall gate: {status}")
    return summary


# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build EJPC artifacts: fits, diagnostics, κ-profiles (fit & cos), z-space, and model-compare; then run gate checks.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--figdir", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--likelihood", choices=["gauss","t"], default="gauss")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--omega-lambda", type=float, default=0.685)
    ap.add_argument("--eta", type=float, default=0.15)
    ap.add_argument("--nbins", type=int, default=24)

    # optional gate overrides
    ap.add_argument("--gate-A-abs-max", type=float, default=Gates.A_abs_max)
    ap.add_argument("--gate-B-min", type=float, default=Gates.B_min)
    ap.add_argument("--gate-B-max", type=float, default=Gates.B_max)
    ap.add_argument("--gate-ppc68-min", type=float, default=Gates.ppc68_min)
    ap.add_argument("--gate-ppc68-max", type=float, default=Gates.ppc68_max)
    ap.add_argument("--gate-closure-should-pass", action="store_true", help="If set, expect closure to pass (defaults to expecting fail).")
    return ap.parse_args()

def main():
    args = parse_args()
    tidy_csv = Path(args.data)
    run_root = Path(args.out)
    figdir   = Path(args.figdir)
    like     = args.likelihood
    _ensure_dir(run_root); _ensure_dir(figdir)

    # 1) 擬合各模型
    results_dirs: dict[str, Path] = {}
    for m in args.models:
        outdir = run_root / f"{m}_{like}"
        _run_fit(model=m, tidy_csv=tidy_csv, outdir=outdir, likelihood=like, fast=args.fast, prior="galaxies-only")
        results_dirs[m] = outdir

    # 2) 診斷（優先 ptq-screen）
    diag_model = "ptq-screen" if "ptq-screen" in results_dirs else list(results_dirs.keys())[-1]
    run_diagnostics_for(results_dirs[diag_model], tidy_csv, figdir,
                        omega_lambda=args.omega_lambda, eta=args.eta, nbins=args.nbins)

    # 3) 模型比較表
    cmp_csv = run_root / "ejpc_model_compare.csv"
    df = build_compare_csv(run_root, cmp_csv, list(results_dirs.keys()), like)
    print(df.to_string(index=False))
    print(f"\n[make] Compare CSV: {cmp_csv}")
    print(f"[make] Figures collected in: {figdir}")

    # 4) Gate checks + consolidated summary
    gates = Gates(
        A_abs_max=args.gate_A_abs_max,
        B_min=args.gate_B_min,
        B_max=args.gate_B_max,
        ppc68_min=args.gate_ppc68_min,
        ppc68_max=args.gate_ppc68_max,
        expect_closure_pass=args.gate_closure_should_pass
    )
    _evaluate_and_write_summary(run_root, results_dirs[diag_model], like, args.omega_lambda, gates)
    # 產生 closure_strict_panel.pdf（與論文一致）
    try:
        from scripts.make_extra_figs import make_closure_panel
        make_closure_panel(results_dirs[diag_model], Path(figdir), out_name="closure_strict_panel.pdf")
        print(f"[make] Closure panel written to: {Path(figdir) / 'closure_strict_panel.pdf'}")
    except Exception as e:
        print("[make][warn] failed to generate closure panel:", repr(e))



if __name__ == "__main__":
    main()
