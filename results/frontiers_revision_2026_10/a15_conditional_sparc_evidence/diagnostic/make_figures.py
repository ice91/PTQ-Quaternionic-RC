"""Labeled SVG rotation curves. Nominal distance and inclination, not the marginal mean."""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import numpy as np

TEST = Path("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/a15_conditional_sparc_evidence")
A10 = Path("/home/rocky/PTQ-Quaternionic-RC/results/frontiers_revision_2026_10/mle_pilot_a10_four_family_science_pilot")
REPORT = Path("/home/rocky/ptq-mond/docs/frontiers_major_revision_2026_10/rev26_p1g_a15_conditional_sparc_evidence/draft/figures")
sys.path.insert(0, str(A10 / "candidate/src"))

from ptquat.data import load_tidy_sparc
from ptquat.models import catalog_velocity_at_distance, vbar_squared_kms2


def fmt(value, digits=3):
    return f"{value:.{digits}g}"


def map_xy(x, y, xlim, ylim, box):
    x0, y0, x1, y1 = box
    px = x0 + (np.asarray(x, float) - xlim[0]) / (xlim[1] - xlim[0]) * (x1 - x0)
    py = y1 - (np.asarray(y, float) - ylim[0]) / (ylim[1] - ylim[0]) * (y1 - y0)
    return px, py


def polyline(x, y, xlim, ylim, box, color, width=1.6, dash=None):
    px, py = map_xy(x, y, xlim, ylim, box)
    pts = " ".join(f"{a:.2f},{b:.2f}" for a, b in zip(px, py))
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<polyline fill="none" stroke="{color}" stroke-width="{width}"{extra} points="{pts}"/>'


def panel(curves, points, xlabel, ylabel, xlim, ylim, box, ticks_x, ticks_y):
    x0, y0, x1, y1 = box
    parts = [f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="white" stroke="#222"/>']
    for tick in ticks_x:
        px, _ = map_xy([tick], [ylim[0]], xlim, ylim, box)
        parts.append(f'<line x1="{px[0]:.2f}" y1="{y1}" x2="{px[0]:.2f}" y2="{y1+5}" stroke="#222"/>')
        parts.append(f'<text x="{px[0]:.2f}" y="{y1+18}" text-anchor="middle" font-size="12" font-family="DejaVu Sans, sans-serif">{html.escape(str(tick))}</text>')
    for tick in ticks_y:
        _, py = map_xy([xlim[0]], [tick], xlim, ylim, box)
        parts.append(f'<line x1="{x0-5}" y1="{py[0]:.2f}" x2="{x0}" y2="{py[0]:.2f}" stroke="#222"/>')
        parts.append(f'<text x="{x0-8}" y="{py[0]+4:.2f}" text-anchor="end" font-size="12" font-family="DejaVu Sans, sans-serif">{html.escape(str(tick))}</text>')
    parts.append(f'<text x="{(x0+x1)/2:.1f}" y="{y1+36}" text-anchor="middle" font-size="13" font-family="DejaVu Sans, sans-serif">{html.escape(xlabel)}</text>')
    parts.append(f'<text transform="translate({x0-48},{(y0+y1)/2:.1f}) rotate(-90)" text-anchor="middle" font-size="13" font-family="DejaVu Sans, sans-serif">{html.escape(ylabel)}</text>')
    for curve in curves:
        parts.append(polyline(curve["x"], curve["y"], xlim, ylim, box, curve["color"], curve.get("width", 1.7), curve.get("dash")))
    if points is not None:
        px, py = map_xy(points["x"], points["y"], xlim, ylim, box)
        ylo, yhi = map_xy(points["x"], points["lo"], xlim, ylim, box)[1], map_xy(points["x"], points["hi"], xlim, ylim, box)[1]
        for a, b, c in zip(px, ylo, yhi):
            parts.append(f'<line x1="{a:.2f}" y1="{b:.2f}" x2="{a:.2f}" y2="{c:.2f}" stroke="#222" stroke-width="1"/>')
        for a, b in zip(px, py):
            parts.append(f'<circle cx="{a:.2f}" cy="{b:.2f}" r="2.4" fill="#111"/>')
    return "\n".join(parts)


def ticks(lo, hi, n=5):
    raw = np.linspace(lo, hi, n)
    return [float(f"{v:.4g}") for v in raw]


def as_catalog(galaxy):
    from types import SimpleNamespace
    return SimpleNamespace(
        name=galaxy.name, r=galaxy.r_kpc, D=float(galaxy.D_Mpc),
        disk=galaxy.v_disk, bulge=galaxy.v_bulge, gas=galaxy.v_gas,
    )


def one_galaxy(galaxy, nu_eps, lin_eps, upsilon, sigma, out):
    pars_nu = {"U": np.full(91, upsilon), "eps": nu_eps, "model": "ptq-nu"}
    pars_lin = {"U": np.full(91, upsilon), "eps": lin_eps, "model": "ptq"}
    g = as_catalog(galaxy)
    v_nu = catalog_velocity_at_distance(g, pars_nu, 0, "ptq-nu", galaxy.D_Mpc)
    v_lin = catalog_velocity_at_distance(g, pars_lin, 0, "ptq", galaxy.D_Mpc)
    vbar2 = vbar_squared_kms2(upsilon, galaxy.v_disk, galaxy.v_bulge, galaxy.v_gas)
    positive = vbar2 > 0
    r = galaxy.r_kpc
    vmax = float(np.nanmax([galaxy.v_obs.max(), v_nu.max(), v_lin.max(), np.sqrt(vbar2[positive]).max() if positive.any() else 0]))
    ylim = (0.0, max(20.0, vmax * 1.15))
    xlim = (0.0, float(r.max()) * 1.05)
    top = (70, 70, 700, 390)
    bot = (70, 460, 700, 760)
    curves = []
    if positive.any():
        curves.append({"x": r[positive], "y": np.sqrt(vbar2[positive]), "color": "#666", "dash": "5 4", "width": 1.4})
    curves.append({"x": r, "y": v_nu, "color": "#1f4e79", "width": 2.0})
    curves.append({"x": r, "y": v_lin, "color": "#b85c38", "width": 2.0})
    res_nu = galaxy.v_obs - v_nu
    res_lin = galaxy.v_obs - v_lin
    rmax = float(np.nanmax(np.abs(np.concatenate([res_nu, res_lin, galaxy.v_err]))))
    rlim = max(10.0, rmax * 1.2)
    body = []
    body.append(panel(
        curves,
        {"x": r, "y": galaxy.v_obs, "lo": galaxy.v_obs - galaxy.v_err, "hi": galaxy.v_obs + galaxy.v_err},
        "radius (kpc)", "velocity (km/s)", xlim, ylim, top, ticks(xlim[0], xlim[1]), ticks(ylim[0], ylim[1]),
    ))
    body.append(panel(
        [
            {"x": r, "y": res_nu, "color": "#1f4e79", "width": 1.7},
            {"x": r, "y": res_lin, "color": "#b85c38", "width": 1.7},
            {"x": [xlim[0], xlim[1]], "y": [0, 0], "color": "#888", "dash": "3 3", "width": 1.0},
        ],
        None,
        "radius (kpc)", "residual v_obs - v_model (km/s)", xlim, (-rlim, rlim), bot,
        ticks(xlim[0], xlim[1]), ticks(-rlim, rlim),
    ))
    legend = f'''
    <rect x="430" y="84" width="250" height="92" fill="white" stroke="#222"/>
    <line x1="442" y1="102" x2="470" y2="102" stroke="#666" stroke-dasharray="5 4"/>
    <text x="478" y="106" font-size="12" font-family="DejaVu Sans, sans-serif">signed baryons</text>
    <line x1="442" y1="124" x2="470" y2="124" stroke="#1f4e79" stroke-width="2"/>
    <text x="478" y="128" font-size="12" font-family="DejaVu Sans, sans-serif">simple-nu conditional</text>
    <line x1="442" y1="146" x2="470" y2="146" stroke="#b85c38" stroke-width="2"/>
    <text x="478" y="150" font-size="12" font-family="DejaVu Sans, sans-serif">linear conditional</text>
    <circle cx="456" cy="166" r="2.4" fill="#111"/>
    <text x="478" y="170" font-size="12" font-family="DejaVu Sans, sans-serif">SPARC with catalog errors</text>
    '''
    title = (
        f"{galaxy.name}: conditional fixed-nuisance curves, not a joint MLE and not a global best fit. "
        f"Upsilon={upsilon} (shared disk and bulge), sigma_sys={sigma:.0f} km/s. "
        f"simple-nu epsilon={nu_eps:.6g}, linear epsilon={lin_eps:.6g}. "
        "Curves use the catalog distance and inclination. The likelihood still marginalizes both."
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="820" viewBox="0 0 760 820">
<rect width="760" height="820" fill="white"/>
<text x="20" y="28" font-size="15" font-family="DejaVu Sans, sans-serif" font-weight="700">{html.escape(galaxy.name)}</text>
<text x="20" y="48" font-size="11" font-family="DejaVu Sans, sans-serif">CONDITIONAL FIXED-NUISANCE FIT. NOT JOINT MLE. NOT GLOBAL BEST FIT.</text>
{''.join(body)}
{legend}
<text x="20" y="808" font-size="10" font-family="DejaVu Sans, sans-serif">{html.escape(title[:220])}</text>
</svg>
'''
    out.write_text(svg)
    return {
        "galaxy": galaxy.name,
        "file": str(out),
        "upsilon": upsilon,
        "sigma_sys_kms": sigma,
        "nu_epsilon": nu_eps,
        "linear_epsilon": lin_eps,
        "n_negative_vbar2": int(np.sum(~positive)),
        "rms_residual_nu": float(np.sqrt(np.mean(res_nu**2))),
        "rms_residual_linear": float(np.sqrt(np.mean(res_lin**2))),
    }


def profile_figure(profiles, study, out):
    # two panels, F relative to each curve's own minimum, versus log10 a0
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="760" height="420" viewBox="0 0 760 420">',
             '<rect width="760" height="420" fill="white"/>',
             '<text x="20" y="24" font-size="14" font-family="DejaVu Sans, sans-serif">Conditional objective F = minus sum of galaxy log likelihoods, relative to each curve minimum.</text>']
    colors = {(0.5, 5.0): "#1f4e79", (0.5, 10.0): "#5b8fb8", (0.7, 5.0): "#b85c38", (0.7, 10.0): "#d08a6a"}
    for i, model in enumerate(["ptq-nu", "ptq"]):
        rows = [p for p in profiles if p["model"] == model]
        a0 = np.asarray(study["grids"][model]["a0_si"], float)
        box = (70 + i * 360, 50, 340 + i * 360, 330)
        xs = np.log10(a0)
        xlim = (float(xs.min()), float(xs.max()))
        ys = []
        for row in rows:
            F = np.asarray(row["F_profile"], float)
            ys.append(F - F.min())
        ymax = max(8.0, min(40.0, float(max(v.max() for v in ys))))
        ylim = (0.0, ymax)
        parts.append(f'<rect x="{box[0]}" y="{box[1]}" width="{box[2]-box[0]}" height="{box[3]-box[1]}" fill="white" stroke="#222"/>')
        parts.append(f'<text x="{box[0]}" y="{box[1]-8}" font-size="13" font-family="DejaVu Sans, sans-serif">{html.escape(model)}</text>')
        for row, rel in zip(rows, ys):
            show = np.minimum(rel, ymax)
            parts.append(polyline(xs, show, xlim, ylim, box, colors[(row["upsilon"], row["sigma_sys_kms"])]))
        for tick in ticks(xlim[0], xlim[1], 4):
            px, _ = map_xy([tick], [0], xlim, ylim, box)
            parts.append(f'<text x="{px[0]:.1f}" y="{box[3]+16}" text-anchor="middle" font-size="11" font-family="DejaVu Sans, sans-serif">{tick:.2f}</text>')
        for tick in (0, 10, 20, 30, 40):
            if tick <= ymax:
                _, py = map_xy([xlim[0]], [tick], xlim, ylim, box)
                parts.append(f'<text x="{box[0]-6}" y="{py[0]+4:.1f}" text-anchor="end" font-size="11" font-family="DejaVu Sans, sans-serif">{tick}</text>')
        parts.append(f'<text x="{(box[0]+box[2])/2:.0f}" y="{box[3]+32}" text-anchor="middle" font-size="12" font-family="DejaVu Sans, sans-serif">log10 a0 (m/s^2)</text>')
        parts.append(f'<text transform="translate({box[0]-36},{(box[1]+box[3])/2:.0f}) rotate(-90)" text-anchor="middle" font-size="12" font-family="DejaVu Sans, sans-serif">Delta F, clipped at 40</text>')
    parts.append('<text x="20" y="400" font-size="11" font-family="DejaVu Sans, sans-serif">Blue family: Upsilon 0.5. Brown family: Upsilon 0.7. Darker line: sigma 5 km/s. Lighter line: sigma 10 km/s. Curves clipped at 40 for display.</text>')
    parts.append("</svg>")
    out.write_text("\n".join(parts))


def main():
    study = json.loads((TEST / "PREDECLARED_STUDY.json").read_text())
    profiles = json.loads((TEST / "CONDITIONAL_PROFILE_RESULTS.json").read_text())
    loaded = load_tidy_sparc(str(A10 / "logs/s_domain13_filtered_tidy.csv"))
    REPORT.mkdir(parents=True, exist_ok=True)
    meta = []
    for upsilon in (0.5, 0.7):
        nu = next(p for p in profiles if p["model"] == "ptq-nu" and p["upsilon"] == upsilon and p["sigma_sys_kms"] == 5.0)
        lin = next(p for p in profiles if p["model"] == "ptq" and p["upsilon"] == upsilon and p["sigma_sys_kms"] == 5.0)
        for name in study["representative_galaxies"]:
            out = REPORT / f"rc_{name}_U{upsilon:.1f}_s5.svg"
            meta.append(one_galaxy(loaded[name], nu["best_epsilon"], lin["best_epsilon"], upsilon, 5.0, out))
    profile_figure(profiles, study, REPORT / "conditional_profiles.svg")
    (TEST / "logs/figure_metrics.json").write_text(json.dumps(meta, indent=2) + "\n")
    # verify legends exist as text
    sample = (REPORT / f"rc_{study['representative_galaxies'][0]}_U0.5_s5.svg").read_text()
    for token in ("simple-nu conditional", "linear conditional", "NOT JOINT MLE", "radius (kpc)"):
        if token not in sample:
            raise SystemExit("missing label " + token)
    print("figures", len(meta))


if __name__ == "__main__":
    main()
