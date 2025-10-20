#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt

from ptquat.data import load_tidy_sparc, GalaxyData
from ptquat.constants import KPC, KM
from ptquat.models import (
    model_v_baryon, model_v_mond, model_v_nfw1p,
    model_v_ptq, model_v_ptq_split, model_v_ptq_nu, model_v_ptq_screen,
    vbar_squared_kms2
)

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _get_Rd_kpc_safe(g: GalaxyData) -> float:
    for attr in ("Rd_kpc", "R_d_kpc", "R_d", "Rd"):
        if hasattr(g, attr):
            try:
                v = float(getattr(g, attr))
                if np.isfinite(v) and v > 0: return v
            except Exception:
                pass
    i_pk = int(np.argmax(g.v_disk)) if len(g.v_disk)>0 else 0
    rpk  = float(g.r_kpc[i_pk]) if len(g.r_kpc)>0 else 1.0
    return max(rpk/2.2, 0.1)

def _make_vfun(model: str, H0_si: float, per_row: pd.Series, g: GalaxyData,
               eps: Optional[float], q: Optional[float], a0: Optional[float]):
    if model == "ptq":
        U = float(per_row["Upsilon_med"])
        return (lambda rk: model_v_ptq(U, eps, rk, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_si),
                vbar_squared_kms2(U, g.v_disk, g.v_bulge, g.v_gas))
    if model == "ptq-nu":
        U = float(per_row["Upsilon_med"])
        return (lambda rk: model_v_ptq_nu(U, eps, rk, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_si),
                vbar_squared_kms2(U, g.v_disk, g.v_bulge, g.v_gas))
    if model == "ptq-screen":
        U = float(per_row["Upsilon_med"]); qq = float(q) if q is not None else 1.0
        return (lambda rk: model_v_ptq_screen(U, eps, qq, rk, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_si),
                vbar_squared_kms2(U, g.v_disk, g.v_bulge, g.v_gas))
    if model == "ptq-split":
        Ud = float(per_row["Upsilon_med"]); Ub = float(per_row.get("Upsilon_bulge_med", Ud))
        return (lambda rk: model_v_ptq_split(Ud, Ub, eps, rk, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_si),
                Ud*(g.v_disk**2) + Ub*(g.v_bulge**2) + g.v_gas**2)
    if model == "baryon":
        U = float(per_row["Upsilon_med"])
        return (lambda rk: model_v_baryon(U, rk, g.v_disk, g.v_bulge, g.v_gas),
                vbar_squared_kms2(U, g.v_disk, g.v_bulge, g.v_gas))
    if model == "mond":
        U = float(per_row["Upsilon_med"])
        return (lambda rk: model_v_mond(U, a0, rk, g.v_disk, g.v_bulge, g.v_gas),
                vbar_squared_kms2(U, g.v_disk, g.v_bulge, g.v_gas))
    if model == "nfw1p":
        U  = float(per_row["Upsilon_med"]); lM = float(per_row["log10_M200_med"])
        return (lambda rk: model_v_nfw1p(U, lM, rk, g.v_disk, g.v_bulge, g.v_gas, H0_si=H0_si),
                vbar_squared_kms2(U, g.v_disk, g.v_bulge, g.v_gas))
    raise ValueError(model)

def _read_summary(results_dir: Path):
    summ = yaml.safe_load(open(results_dir/"global_summary.yaml"))
    per  = pd.read_csv(results_dir/"per_galaxy_summary.csv").set_index("galaxy")
    return summ, per

# ---------- RC by SB ----------
def _sb_classes_from_csv(csv_path: Path, per: pd.DataFrame, gdict: Dict[str,GalaxyData]) -> Dict[str,str]:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    key = None
    for k in ("sb_class","SBclass","sbClass","SB_CLASS"):
        if k in df.columns: key = k; break
    if key is not None:
        m = df.groupby("galaxy")[key].first().to_dict()
        return {str(k): str(v).upper() for k,v in m.items() if k in per.index}
    proxy = []
    for name, g in gdict.items():
        if name not in per.index: continue
        Rd = _get_Rd_kpc_safe(g)
        idx = int(np.argmin(np.abs(g.r_kpc - 2.2*Rd)))
        Vd2 = float(g.v_disk[idx]**2) if len(g.v_disk)>0 else np.nan
        U   = float(per.loc[name,"Upsilon_med"]) if "Upsilon_med" in per.columns else np.nan
        mu0 = U * Vd2 / max(Rd, 1e-6)
        if np.isfinite(mu0): proxy.append((name, mu0))
    if not proxy: return {}
    vals = np.array([p[1] for p in proxy]); q1, q2 = np.quantile(vals, [1/3, 2/3])
    lab = {}
    for name, mu0 in proxy:
        lab[name] = "LSB" if mu0 <= q1 else ("MSB" if mu0 <= q2 else "HSB")
    return lab

def plot_rc_by_sb(results_dir: Path, data_csv: Path) -> List[Path]:
    out_paths = []
    summ, per = _read_summary(results_dir)
    H0_si  = float(summ.get("H0_si", 2.2e-18))
    eps    = summ.get("epsilon_median"); q = summ.get("q_median"); a0 = summ.get("a0_median")
    model  = str(summ["model"])
    gdict  = load_tidy_sparc(str(data_csv))

    sbmap = _sb_classes_from_csv(data_csv, per, gdict)
    if not sbmap:
        print("[rc-sb][warn] no sb_class in CSV and proxy failed; skip RC-by-SB.")
        return out_paths

    rows = []
    for name, g in gdict.items():
        if name not in per.index or name not in sbmap: continue
        Rd = _get_Rd_kpc_safe(g)
        for r, v in zip(g.r_kpc, g.v_obs):
            rows.append(dict(galaxy=name, cls=sbmap[name], x=float(r/Rd), v=float(v)))
    df = pd.DataFrame(rows)
    if df.empty: return out_paths

    for cls in ("HSB","MSB","LSB"):
        sub = df[df["cls"]==cls].copy()
        if sub.empty: continue
        edges = np.linspace(max(0.1, sub["x"].quantile(0.02)), sub["x"].quantile(0.98), 25)
        mids  = 0.5*(edges[:-1]+edges[1:])
        q16=q50=q84=None
        q16=[]; q50=[]; q84=[]
        for i in range(len(edges)-1):
            m = (sub["x"]>=edges[i]) & (sub["x"]<edges[i+1])
            vv = sub.loc[m,"v"].values
            if vv.size==0:
                q16.append(np.nan); q50.append(np.nan); q84.append(np.nan)
            else:
                q16.append(float(np.nanpercentile(vv,16)))
                q50.append(float(np.nanpercentile(vv,50)))
                q84.append(float(np.nanpercentile(vv,84)))
        fig, ax = plt.subplots(figsize=(5.6,4.0), dpi=150)
        ax.scatter(sub["x"], sub["v"], s=4, alpha=0.08, label=f"{cls} points")
        ax.fill_between(mids, q16, q84, alpha=0.25, label="16–84%")
        ax.plot(mids, q50, linewidth=1.6, label="median")
        ax.set_xlabel("r/Rd"); ax.set_ylabel("Vobs [km/s]")
        ax.grid(True, alpha=0.25); ax.legend()
        out = results_dir/f"plot_{cls}.png"
        fig.tight_layout(); fig.savefig(out); plt.close(fig)
        out_paths.append(out)
    return out_paths

# ---------- RAR ----------
def make_rar(results_dir: Path, data_csv: Path, out_name: str = "rar.png") -> Path:
    summ, per = _read_summary(results_dir)
    model  = str(summ["model"])
    H0_si  = float(summ.get("H0_si", 2.2e-18))
    eps    = summ.get("epsilon_median"); q = summ.get("q_median"); a0 = summ.get("a0_median")

    gdict = load_tidy_sparc(str(data_csv))
    rows = []
    for name, g in gdict.items():
        if name not in per.index: continue
        vfun, vbar2 = _make_vfun(model, H0_si, per.loc[name], g, eps, q, a0)
        r_m  = np.maximum(g.r_kpc*KPC, 1e-30)
        gobs = (g.v_obs**2) * (KM**2) / r_m
        gbar = (vbar2)       * (KM**2) / r_m
        for go, gb in zip(gobs, gbar):
            if np.isfinite(go) and np.isfinite(gb) and gb>0 and go>0:
                rows.append((float(gb), float(go)))
    df = pd.DataFrame(rows, columns=["gbar","gobs"])
    if df.empty: raise RuntimeError("No valid points for RAR.")

    logg = np.log10(df["gbar"].values)
    lo, hi = np.nanpercentile(logg, [2,98])
    edges = np.linspace(lo, hi, 26)
    mids  = 0.5*(edges[:-1]+edges[1:])
    q50=[]; q16=[]; q84=[]
    for i in range(len(edges)-1):
        m = (logg>=edges[i]) & (logg<edges[i+1])
        if m.sum()==0:
            q16.append(np.nan); q50.append(np.nan); q84.append(np.nan)
        else:
            vals = np.log10(df["gobs"].values[m])
            q16.append(float(np.nanpercentile(vals,16)))
            q50.append(float(np.nanpercentile(vals,50)))
            q84.append(float(np.nanpercentile(vals,84)))

    fig, ax = plt.subplots(figsize=(5.2,4.6), dpi=150)
    ax.scatter(df["gbar"], df["gobs"], s=4, alpha=0.08, label="points")
    ax.plot(10**mids, 10**np.asarray(q50), linewidth=1.6, label="median")
    ax.fill_between(10**mids, 10**np.asarray(q16), 10**np.asarray(q84), alpha=0.25, label="16–84%")
    xs = np.logspace(lo, hi, 200)
    ax.plot(xs, xs, linestyle="--", linewidth=1.0, label="1:1")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$g_{\rm bar}\,[{\rm m\,s^{-2}}]$")
    ax.set_ylabel(r"$g_{\rm obs}\,[{\rm m\,s^{-2}}]$")
    ax.grid(True, which="both", alpha=0.25); ax.legend()
    out = results_dir/out_name
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return out

# ---------- BTFR helpers ----------
def _guess_scale_1e9(series: pd.Series) -> float:
    """多數 SPARC 的 L3.6 / MHI 以 10^9 單位提供：若中位數 < 1e6，視為 *1e9。"""
    if series is None or series.empty:
        return 1.0
    vals = pd.to_numeric(series, errors="coerce")
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 1.0
    med = np.median(vals)
    return 1e9 if med < 1e6 else 1.0


def _load_aux_from_table1(table1_path: Path) -> Optional[pd.DataFrame]:
    """從 raw table1 讀 L3.6 -> L36_tot、MHI -> M_HI；如有分量也帶出。"""
    if not table1_path.exists(): return None
    t1 = pd.read_csv(table1_path)
    if "Name" not in t1.columns: return None
    out = pd.DataFrame({"galaxy": t1["Name"]})

    # 精準抓 L3.6 與 MHI（避免把 L16a/L16b 當成 3.6）
    if "L3.6" in t1.columns:
        out["L36_tot"] = pd.to_numeric(t1["L3.6"], errors="coerce")
    if "MHI" in t1.columns:
        out["M_HI"] = pd.to_numeric(t1["MHI"], errors="coerce")

    # 若真的有分量欄位就帶上（少見）
    for c_in, c_out in [("L36_disk","L36_disk"), ("L36_bulge","L36_bulge")]:
        if c_in in t1.columns:
            out[c_out] = pd.to_numeric(t1[c_in], errors="coerce")

    keep = [c for c in ["galaxy","L36_tot","M_HI","L36_disk","L36_bulge"] if c in out.columns]
    return out[keep] if len(keep) >= 2 else None

# ---------- BTFR ----------
def make_btfr(results_dir: Path, data_csv: Path, out_name: str = "btfr.png") -> Optional[Path]:
    summ, per = _read_summary(results_dir)
    gdict = load_tidy_sparc(str(data_csv))
    base = pd.read_csv(data_csv)
    base_grp = base.groupby("galaxy").first()

    # 若 tidy 沒有，就從 raw table1 抽
    if not ({"L36_tot","M_HI"} & set(base_grp.columns)):
        for p in [data_csv.parent/"raw"/"vizier_table1.csv", Path("dataset/raw/vizier_table1.csv")]:
            aux = _load_aux_from_table1(p)
            if aux is not None:
                base_grp = base_grp.merge(aux.set_index("galaxy"), left_index=True, right_index=True, how="left")
                break

    # 估單位縮放（10^9 → 直接乘 1e9）
    s_tot = _guess_scale_1e9(base_grp["L36_tot"]) if "L36_tot" in base_grp.columns else 1.0
    s_d   = _guess_scale_1e9(base_grp["L36_disk"]) if "L36_disk" in base_grp.columns else 1.0
    s_b   = _guess_scale_1e9(base_grp["L36_bulge"]) if "L36_bulge" in base_grp.columns else 1.0
    s_hi  = _guess_scale_1e9(base_grp["M_HI"]) if "M_HI" in base_grp.columns else 1.0

    rows = []
    for name, g in gdict.items():
        if name not in per.index or name not in base_grp.index: 
            continue
        n = len(g.v_obs)
        if n < 5: 
            continue
        take = max(3, n//5)
        Vflat = float(np.nanmedian(g.v_obs[-take:]))

        U  = float(per.loc[name, "Upsilon_med"]) if "Upsilon_med" in per.columns else np.nan
        Ub = float(per.loc[name, "Upsilon_bulge_med"]) if "Upsilon_bulge_med" in per.columns else np.nan

        row = base_grp.loc[name]
        Ld, Lb, Ltot = row.get("L36_disk"), row.get("L36_bulge"), row.get("L36_tot")
        MHI = row.get("M_HI", row.get("MHI"))

        # 單位轉換
        if pd.notna(Ld):   Ld   = float(Ld)   * s_d
        if pd.notna(Lb):   Lb   = float(Lb)   * s_b
        if pd.notna(Ltot): Ltot = float(Ltot) * s_tot
        if pd.notna(MHI):  MHI  = float(MHI)  * s_hi

        # 星質量：優先分量，其次總亮度
        Mstar = np.nan
        if np.isfinite(U) and pd.notna(Ld):
            Mstar = U*float(Ld) + (Ub*float(Lb) if (np.isfinite(Ub) and pd.notna(Lb)) else 0.0)
        elif np.isfinite(U) and pd.notna(Ltot):
            Mstar = U*float(Ltot)

        Mgas = 1.33*float(MHI) if pd.notna(MHI) else 0.0
        Mb = Mstar + Mgas if np.isfinite(Mstar) else np.nan
        if np.isfinite(Mb) and Mb > 0:
            rows.append((Vflat, Mb))

    if not rows:
        print("[btfr][warn] Missing usable L36_tot/L36_* and/or M_HI；skip BTFR.")
        return None

    df = pd.DataFrame(rows, columns=["Vflat","Mb"])
    x = np.log10(df["Vflat"].values); y = np.log10(df["Mb"].values)
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

    fig, ax = plt.subplots(figsize=(5.2,4.6), dpi=150)
    ax.scatter(df["Vflat"], df["Mb"], s=12, alpha=0.6, label=f"galaxies (N={len(df)})")
    xs = np.logspace(np.nanmin(x), np.nanmax(x), 200, base=10.0)
    ax.plot(xs, 10**(slope*np.log10(xs)+intercept),
            linewidth=1.6, label=rf"fit: $\log M_b={slope:.2f}\log V_f+{intercept:.2f}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$V_f\,[{\rm km\,s^{-1}}]$")
    ax.set_ylabel(r"$M_b\,[M_\odot]$")
    ax.grid(True, which="both", alpha=0.25); ax.legend()
    out = results_dir/out_name
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return out

# ---------- closure panel ----------
def make_closure_panel(results_dir: Path, figdir: Path, out_name: str = "closure_strict_panel.pdf") -> Path:
    cjson = results_dir/"closure_test.json"
    clo = json.loads(cjson.read_text()) if cjson.exists() else yaml.safe_load(open(results_dir/"closure_test.yaml"))
    eps_rc = float(clo["epsilon_RC"]); eps_cos= float(clo["epsilon_cos"])
    sig    = float(clo.get("sigma_RC", np.nan))
    passed = bool(clo.get("pass_within_3sigma", False))

    _ensure_dir(figdir)
    fig, ax = plt.subplots(figsize=(4.5,3.6), dpi=150)
    ax.axvline(eps_cos, color="k", linestyle="--", linewidth=1.2, label=r"$\varepsilon_{\rm cos}$")
    ax.errorbar([0],[eps_rc], yerr=[[sig],[sig]], fmt="o", capsize=4, label=r"$\varepsilon_{\rm RC}\pm1\sigma$")
    ax.fill_between([-0.5,0.5], eps_rc-3*sig, eps_rc+3*sig, alpha=0.15, label=r"$\pm3\sigma$")
    ax.set_xlim(-0.6,0.6); ax.set_xticks([]); ax.set_ylabel(r"$\varepsilon$")
    ax.set_title(f"Strict closure: {'PASS' if passed else 'FAIL'}")
    ax.grid(True, alpha=0.25); ax.legend()
    out = figdir/out_name
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    return out

def main():
    ap = argparse.ArgumentParser(description="Extra paper figures: RC by SB, RAR, BTFR, closure panel.")
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--do-rc-sb", action="store_true")
    ap.add_argument("--do-rar", action="store_true")
    ap.add_argument("--do-btfr", action="store_true")
    ap.add_argument("--make-closure-panel", action="store_true")
    ap.add_argument("--figdir", default="paper_figs")
    args = ap.parse_args()

    results = Path(args.results_dir)
    data_csv = Path(args.data)

    if args.do_rc_sb:
        outs = plot_rc_by_sb(results, data_csv)
        print("[rc-sb] wrote:", [str(p) for p in outs])
    if args.do_rar:
        out = make_rar(results, data_csv)
        print("[rar] wrote:", out)
    if args.make_closure_panel:
        out = make_closure_panel(results, Path(args.figdir))
        print("[closure] wrote:", out)
    if args.do_btfr:
        try:
            out = make_btfr(results, data_csv)
            if out is not None:
                print("[btfr] wrote:", out)
        except Exception as e:
            print("[btfr][warn]", repr(e))

if __name__ == "__main__":
    main()
