"""Separate pure-evaluation, hard-bound objective and decoded-local contracts."""
import numpy as np
MODELS=('baryon','mond','ptq','ptq-nu','ptq-screen','mond-screen','nfw1p')
def resolve_galaxy(galaxies,galaxy_id):
    if isinstance(galaxy_id,str):
        ids=[i for i,g in enumerate(galaxies) if g.name==galaxy_id]
        if len(ids)!=1:raise ValueError('Unknown or nonunique galaxy ID')
        return ids[0]
    if isinstance(galaxy_id,(bool,np.bool_)) or not isinstance(galaxy_id,(int,np.integer)) or not 0<=galaxy_id<len(galaxies):
        raise ValueError('Galaxy index outside declared sample')
    return int(galaxy_id)
def validate_galaxy(g):
    arrays=[np.asarray(getattr(g,n),float) for n in ['r_kpc','v_obs','v_err','v_disk','v_bulge','v_gas']]
    if not arrays[0].size or any(a.ndim!=1 or a.shape!=arrays[0].shape or not np.isfinite(a).all() for a in arrays):raise ValueError('Invalid catalog array shape/value')
    geo=np.array([g.D_Mpc,g.D_err_Mpc,g.i_deg,g.i_err_deg],float)
    if not np.isfinite(geo).all() or not (g.D_Mpc>0 and g.D_err_Mpc>=0 and 0<g.i_deg<=90 and g.i_err_deg>=0) or np.any(arrays[2]<0):raise ValueError('Invalid catalog geometry/errors')
def validate_native(theta,galaxies,model,*,likelihood_version,hard_bounds=False,H0_si=None,layout=None):
    from .marginal_likelihood import require_version
    from .fit_global import _layout
    require_version(likelihood_version)
    if model not in MODELS or not len(galaxies):raise ValueError('Unsupported model or empty sample')
    if len({g.name for g in galaxies})!=len(galaxies):raise ValueError('Duplicate galaxy IDs')
    L=_layout(model,len(galaxies),True,None) if layout is None else layout
    expected=_layout(model,len(galaxies),L['sigma_learn'],L['a0_fixed'])
    if L!=expected:raise ValueError('Layout does not match native sample/model contract')
    t=np.asarray(theta,float)
    if t.shape!=(L['k'],) or not np.isfinite(t).all():raise ValueError('Invalid finite native theta shape')
    if hard_bounds:
        from .mle import theta_bounds
        from .constants import H0_SI
        if layout is not None and L!=_layout(model,len(galaxies),True,None):raise ValueError('Optimizer requires original learned-sigma native layout')
        bounds=np.asarray(theta_bounds(model,len(galaxies),H0_SI if H0_si is None else H0_si))
        if np.any(t<bounds[:,0]) or np.any(t>bounds[:,1]):raise ValueError('Native theta outside original hard bounds')
    return t,L

def validate_decoded(p,gi,model):
    if model not in MODELS or p.get('model',model)!=model:raise ValueError('Decoded model mismatch')
    U=np.asarray(p.get('U'),float)
    if U.ndim!=1 or U.size==0 or not np.isfinite(U).all():raise ValueError('Invalid decoded stellar vector')
    if isinstance(gi,(bool,np.bool_)) or not isinstance(gi,(int,np.integer)) or not 0<=gi<len(U):raise ValueError('Invalid decoded local index')
    keys=['sigma']
    if model in ('ptq','ptq-nu','ptq-screen'):keys+=['eps']
    if model in ('mond','mond-screen'):
        keys+=['log10a0'] if p.get('a0_fixed') is None else ['a0_fixed']
    if model in ('ptq-screen','mond-screen'):keys+=['lq']
    for key in keys:
        a=np.asarray(p.get(key),float)
        if a.shape!=() or not np.isfinite(a):raise ValueError('Invalid decoded scalar '+key)
    if p['sigma']<=0:raise ValueError('Nonpositive systematic floor')
    if model=='nfw1p':
        a=np.asarray(p.get('lM'),float)
        if a.shape!=U.shape or not np.isfinite(a).all():raise ValueError('Invalid decoded halo vector')
