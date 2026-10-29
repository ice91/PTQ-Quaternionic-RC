from pathlib import Path
import ast
B=Path(__file__).resolve().parent;S=B/'candidate/src/ptquat'
# Allphysical lawbodies remain untouched.
p=S/'mle.py';s=p.read_text();a=s.index('"""');b=s.index('"""',a+3)+3;s=s[:a]+'"""Versioned physical-domain marginal likelihood evaluators; no priors in MLE.\nPure evaluators validate native shape/finite values; optimizer objectives additionally\nenforce the original hard bounds. Legacy unbudgeted fit drivers remain disabled.\n"""'+s[b:];s=s.replace('from .marginal_likelihood import PRIMARY_B0, SENSITIVITY_B1','from .marginal_likelihood import PRIMARY_B0, SENSITIVITY_B1\nfrom .evaluation_contract import validate_native, resolve_galaxy')
a=s.index('    from .marginal_likelihood import require_version',s.index('def objective_neg_loglike'));b=s.index('\n\ndef evaluate_loglike',a)
s=s[:a]+'''    t, _ = validate_native(theta, galaxies, model, likelihood_version=likelihood_version, hard_bounds=True,H0_si=H0_si)
    return -evaluate_loglike(t,galaxies,model,H0_si,c0,c_slope,likelihood_version=likelihood_version)
'''+s[b:]
a=s.index('    from .marginal_likelihood import galaxy_log_marginal',s.index('def galaxy_loglike'));b=s.index('\n\ndef map_theta',a)
s=s[:a]+'''    from .marginal_likelihood import galaxy_log_marginal
    t,L=validate_native(theta,galaxies,model,likelihood_version=likelihood_version)
    gi=resolve_galaxy(galaxies,gi)
    p=_unpack(t,L)
    return float(galaxy_log_marginal(galaxies[gi],p,gi,model,likelihood_version=likelihood_version,
        H0_si=H0_si,c0=c0,c_slope=c_slope)["log_marginal"])
'''+s[b:]
a=s.index('    from .marginal_likelihood import require_version, galaxy_log_marginal',s.index('def evaluate_loglike_components'));b=s.index('    p = _unpack(t, L)',a)
s=s[:a]+'''    from .marginal_likelihood import galaxy_log_marginal
    t,L=validate_native(theta,galaxies,model,likelihood_version=likelihood_version)
'''+s[b:];p.write_text(s)
p=S/'fit_global.py';s=p.read_text();a=s.index('    from .marginal_likelihood import require_version, galaxy_log_marginal',s.index('def log_likelihood_full_per_galaxy'));b=s.index('    pars = _unpack(theta, L)',a)
s=s[:a]+'''    from .marginal_likelihood import galaxy_log_marginal
    from .evaluation_contract import validate_native
    theta,_ = validate_native(theta,galaxies,L["model"],likelihood_version=like_kind,layout=L)
'''+s[b:];p.write_text(s)
p=S/'marginal_likelihood.py';s=p.read_text().replace('Frozen P1C law equations are reused read-only; propagated covariance is never added.','Candidate production model laws are preserved; propagated covariance is never added.')
s=s.replace('    floor = require_version(likelihood_version)\n','''    from .evaluation_contract import validate_decoded,validate_galaxy
    from .warning_telemetry import WarningContext
    floor = require_version(likelihood_version)
    validate_decoded(pars,gi,model)
    validate_galaxy(galaxy)
    if not (np.isfinite(H0_si) and H0_si>0 and np.isfinite(c0) and c0>0 and np.isfinite(c_slope)):
        raise ValueError("Invalid fixed cosmology/halo configuration")
''')
s=s.replace('    record = integrate(g, p, gi, "segmented_quantile", order, floors=(floor,))','''    with WarningContext(model,likelihood_version,pars,gi):
        record = integrate(g, p, gi, "segmented_quantile", order, floors=(floor,))''')
s=s.replace("ll=-.5*(Q+norm[None,:]);assert np.isfinite(ll).all()","ll=-.5*(Q+norm[None,:])\n   if not np.isfinite(ll).all():raise FloatingPointError('Nonfinite conditional Gaussian density')")
p.write_text(s)
p=S/'experiments.py';s=p.read_text();a=s.index('# Primary predictive firewall:');b=s.index('def per_galaxy_log_marginals',a)
# UseAST ONCE atbuildtime torecordexact explicit registry; no runtimeAST.
tree=ast.parse(s);names=set(['_call_fit','ppc_check','kappa_per_galaxy','kappa_radius_resolved','kappa_geom_per_galaxy','kappa_radius_resolved_geom','compare_models_waic','loo_compare_models','_write_pwaic_diagnostics'])
for n in tree.body:
 if isinstance(n,ast.FunctionDef) and any(isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='build_covariance' for x in ast.walk(n)):names.add(n.name)
replacement='''# Explicit disabled registry. No runtime AST, no implicit legacy version fallback.
from .marginal_likelihood import PRIMARY_B0, LEGACY_DIAGONALIZED_SENSITIVITY
from functools import wraps as _wraps
DISABLED_LEGACY_DIAGNOSTICS = '''+repr(tuple(sorted(names)))+'''
def _disabled_legacy(fn):
    @_wraps(fn)
    def disabled(*args, **kwargs):
        raise ValueError("Disabled legacy diagonalized sensitivity: unavailable for physical marginal target")
    return disabled
for _name in DISABLED_LEGACY_DIAGNOSTICS:
    if _name in globals(): globals()[_name]=_disabled_legacy(globals()[_name])

'''
s=s[:a]+replacement+s[b:]
s=s.replace('    p=_unpack(theta,_layout(model,len(galaxies),True,None));rows=[]','    from .evaluation_contract import validate_native\n    theta,L=validate_native(theta,galaxies,model,likelihood_version=PRIMARY_B0)\n    p=_unpack(theta,L);rows=[]');p.write_text(s)
# Disable every inherited optimizer driver; new smoke may only be separately authorized.
p=S/'bounded_driver.py';s=p.read_text();s=s.replace('    out=Path(out);out.mkdir(exist_ok=False)','    raise RuntimeError("P1F-R1: optimizer execution disabled; next smoke requires separate authorization")\n    out=Path(out);out.mkdir(exist_ok=False)');p.write_text(s)
for p in S.glob('*.py'):ast.parse(p.read_text())
print('repaired contracts, explicit diagnostic registry and telemetry boundary; optimizer disabled')
