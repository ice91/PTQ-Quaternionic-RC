from pathlib import Path
import ast,json,difflib
B=Path(__file__).resolve().parent;I=B.parents[2];W=B/'isolated_candidate'
if (B/'production_candidate.diff').exists():raise FileExistsError('patch already built')
mp=W/'src/ptquat/models.py';s=mp.read_text();s+='''

# REV26-P1D: fixed-y0 B0 covariance interface; radial laws above are unchanged.
def model_distance_jacobian(model, v_model_kms, r_kpc, D_Mpc, *,
                            log10_M200=None, H0_si=H0_SI,
                            c0=10.0, c_slope=-0.1):
    """Total catalog distance derivative in (km/s)/Mpc at fixed angular samples.

    All baryonic velocity templates scale as sqrt(D/D0), physical radii as
    D/D0, with Upsilon/acceleration/q/halo mass held fixed.  This is a mean
    derivative; the residual derivative has the opposite sign.
    """
    if not (np.isfinite(D_Mpc) and D_Mpc > 0):
        raise ValueError("Distance must be positive and finite")
    v = np.asarray(v_model_kms, float)
    r = np.asarray(r_kpc, float)
    if model in ("baryon", "ptq", "ptq-split", "mond", "ptq-nu",
                 "ptq-screen", "mond-screen"):
        return v / (2.0 * D_Mpc)
    if model != "nfw1p" or log10_M200 is None:
        raise ValueError("Unsupported model or missing restricted NFW mass")
    mass = 10.0**float(log10_M200)
    c = c_powerlaw(mass, c0=c0, beta=c_slope)
    vh = nfw_v_kms_M200_c(r, mass, c, H0_si=H0_si)
    rho = 3.0 * H0_si**2 / (8.0 * np.pi * G_SI)
    R200 = (3.0 * mass * MSUN / (4.0 * np.pi * 200.0 * rho))**(1.0/3.0)
    x = r * KPC * c / R200
    f = np.log1p(x) - x / (1.0 + x)
    eta = x*x / ((1.0+x)**2 * f) - 1.0
    # v^2-vh^2 is the frozen baryonic contribution; no new halo freedom.
    return (v*v - vh*vh + vh*vh*eta) / (2.0 * v * D_Mpc)
''';mp.write_text(s)
p=W/'src/ptquat/likelihood.py';s=p.read_text().replace('sigma_sys_kms: float = 4.0) -> np.ndarray:', 'sigma_sys_kms: float = 4.0, *,\n                     distance_jacobian_kms_per_Mpc: np.ndarray) -> np.ndarray:',1)
a=s.index('    """',s.index('def build_covariance'));b=s.index('    n = len',a)
s=s[:a]+'''    """Corrected B0 covariance in the fixed nominal observed-velocity measure.

    The caller must supply the full catalog-level mean derivative dmu/dD
    in (km/s)/Mpc, including radius AND baryonic-template rescaling.
    J_i = v_model*cot(i0) is per radian; inclination error is converted here.
    The positional v_fun_for_grad remains for caller signature continuity,
    but its historical radius-only derivative is never used by this target.
    This normalized local Gaussian surrogate is not exact truncated nuisance
    marginalization. sigma_sys is additive in the nominal y0 coordinate.
    """
'''+s[b:]
s=s.replace('''        dv_dr = _num_grad_v_wrt_r(r_kpc, v_fun_for_grad)
        J_D = dv_dr * (r_kpc / D_Mpc)''','''        J_D = np.asarray(distance_jacobian_kms_per_Mpc, float)
        if J_D.shape != np.asarray(v_model_kms).shape or not np.all(np.isfinite(J_D)):
            raise ValueError("Full distance Jacobian must be finite and match model shape")''',1)
p.write_text(s)
# Add explicit full-JD keyword to every direct source caller, including diagnostics.
for file in ['fit_global.py','mle.py','experiments.py']:
 p=W/'src/ptquat'/file;s=p.read_text()
 if file=='mle.py':s=s.replace('    from .models import (','    from .models import (\n        model_distance_jacobian,',1)
 else:s=s.replace('from .models import (','from .models import (\n    model_distance_jacobian,',1)
 if file=='experiments.py':
  s=s.replace('    return vfun, vbar2','''    vfun.full_distance_jacobian = lambda D: model_distance_jacobian(
        model, vfun(g.r_kpc), g.r_kpc, D,
        log10_M200=(float(per_row["log10_M200_med"]) if model == "nfw1p" else None),
        H0_si=H0_si)
    return vfun, vbar2''',1)
  # Coverage helper creates its own callable instead of _make_vfun.
  s=s.replace('        v_mod = vfun(g.r_kpc)','''        vfun.full_distance_jacobian = lambda D: model_distance_jacobian(
            model, vfun(g.r_kpc), g.r_kpc, D,
            log10_M200=(lM if model == "nfw1p" else None),
            H0_si=(H0_SI if model == "nfw1p" else H0_si))
        v_mod = vfun(g.r_kpc)''',1)
 tree=ast.parse(s);lines=s.splitlines(keepends=True);offsets=[0]
 for l in lines:offsets.append(offsets[-1]+len(l))
 edits=[]
 for node in ast.walk(tree):
  if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='build_covariance':
   if file=='experiments.py':expr='vfun.full_distance_jacobian(g.D_Mpc)'
   else:
    # The run postprocessing path uses args.model/args.c0; the functions use m/c0.
    runpath=file=='fit_global.py' and node.lineno>500
    mod='args.model' if runpath else 'm';c0='args.c0' if runpath else 'c0';cs='args.c_slope' if runpath else 'c_slope'
    expr=f'model_distance_jacobian({mod}, v_mod, g.r_kpc, g.D_Mpc, log10_M200=(lM if {mod} == "nfw1p" else None), H0_si=H0_si, c0={c0}, c_slope={cs})'
   pos=offsets[node.end_lineno-1]+node.end_col_offset-1
   edits.append((pos,', distance_jacobian_kms_per_Mpc='+expr))
 for pos,addition in sorted(edits,reverse=True):s=s[:pos]+addition+s[pos:]
 ast.parse(s);p.write_text(s)
p=W/'tests/test_covariance.py';s=p.read_text().replace('vfun, sigma_sys_kms=4.0)','vfun, sigma_sys_kms=4.0, distance_jacobian_kms_per_Mpc=v_mod/(2*10.0))');p.write_text(s)
files=['src/ptquat/likelihood.py','src/ptquat/models.py','src/ptquat/fit_global.py','src/ptquat/mle.py','src/ptquat/experiments.py','tests/test_covariance.py'];diff=''
for file in files:diff+=''.join(difflib.unified_diff((I/file).read_text().splitlines(True),(W/file).read_text().splitlines(True),fromfile='a/'+file,tofile='b/'+file))
(B/'production_candidate.diff').write_text(diff)
# Mechanical call-chain coverage and unchanged law-body proof.
source_calls=[]
for p in sorted((W/'src/ptquat').glob('*.py')):
 tree=ast.parse(p.read_text())
 for n in ast.walk(tree):
  if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='build_covariance':
   explicit=any(k.arg=='distance_jacobian_kms_per_Mpc' for k in n.keywords);assert explicit
   source_calls.append(dict(file=str(p.relative_to(W)),line=n.lineno,explicit_full_distance_keyword=True))
old=ast.parse((I/'src/ptquat/models.py').read_text());new=ast.parse(mp.read_text())
for f in old.body:
 if isinstance(f,ast.FunctionDef):assert ast.dump(f,include_attributes=False)==ast.dump(next(x for x in new.body if isinstance(x,ast.FunctionDef) and x.name==f.name),include_attributes=False)
(B/'PATCH_CALL_GRAPH.json').write_text(json.dumps(dict(modified_files=files,source_covariance_calls=source_calls,all_original_model_law_bodies_unchanged=True,canonical_untouched=True,additional_dependency_reason='experiments.py has7 verified direct covariance callers; legacy covariance test must supply new required full-JD keyword',likelihood_version='REV26-P1D-B0-FULLD-GAUSS-CANDIDATE-v1'),indent=2)+'\n')
print('isolated patch',files,'explicit covariance source calls',len(source_calls))
