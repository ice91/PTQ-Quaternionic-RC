"""Exact galaxy dependency decomposition; numerical finite differences in native units.
No optimizer, no analytic/AD derivative, no persistent parameter-valued cache.
"""
from pathlib import Path
import hashlib,json,time
import numpy as np
from .constants import H0_SI
from .fit_global import _layout,_unpack
from .marginal_likelihood import PRIMARY_B0,galaxy_log_marginal,nuisance_nodes
from .evaluation_contract import validate_native
from .mle import theta_bounds

def source_hashes():
    S=Path(__file__).resolve().parent
    return {n:hashlib.sha256((S/n).read_bytes()).hexdigest() for n in ['models.py','fit_global.py','mle.py','marginal_likelihood.py','evaluation_contract.py','dependency_gradient.py']}
def sample_hash(galaxies):
    h=hashlib.sha256()
    for g in galaxies:
        h.update(g.name.encode())
        for n in ['r_kpc','v_obs','v_err','v_disk','v_bulge','v_gas','D_Mpc','D_err_Mpc','i_deg','i_err_deg']:
            h.update(n.encode());h.update(np.asarray(getattr(g,n),dtype='<f8').tobytes())
    return h.hexdigest()
def ledger(model,galaxies):
    L=_layout(model,len(galaxies),True,None);bounds=theta_bounds(model,len(galaxies));rows=[]
    for j in range(L['k']):
        if L['sl_U'].start<=j<L['sl_U'].stop:
            gi=j-L['sl_U'].start;name='Upsilon_'+galaxies[gi].name;kind='galaxy_local';affected=[gi];transform='identity'
        elif 'sl_lM' in L and L['sl_lM'].start<=j<L['sl_lM'].stop:
            gi=j-L['sl_lM'].start;name='log10_M200_'+galaxies[gi].name;kind='galaxy_local';affected=[gi];transform='M200=10**native; c(M200) fixed prescription'
        else:
            globalkeys=[(key,L[key]) for key in ['i_eps','i_log10a0','i_lq','i_lnsig'] if key in L and L[key]==j]
            if len(globalkeys)!=1:raise ValueError('Unknown dependency slot; fail closed')
            key=globalkeys[0][0];name={'i_eps':'epsilon','i_log10a0':'log10_a0','i_lq':'ln_q','i_lnsig':'ln_sigma_sys'}[key]
            transform={'i_eps':'identity','i_log10a0':'a0=10**native','i_lq':'q=exp(native)','i_lnsig':'sigma=exp(native)'}[key]
            kind='global';affected=list(range(len(galaxies)))
        rows.append(dict(model=model,index=j,name=name,scope=kind,native_transform=transform,bounds=list(bounds[j]),affected_indices=affected,affected_galaxies=[galaxies[i].name for i in affected],cache_invalidation='recompute every affected conditional mean/variance/marginal; geometry-only nodes reusable',source_reason='actual _unpack slice/scalar and catalog_velocity_at_distance U[gi]/lM[gi] indexing; global sigma conditional variance; verified by full recomputation'))
    return rows

def step_stencil(x,bound,relative_step,absolute_step=None,scheme='auto'):
    lo,hi=bound;h=(hi-lo)*relative_step if absolute_step is None else float(absolute_step)
    if not (np.isfinite(h) and h>0 and lo<=x<=hi):raise ValueError('Invalid differentiation step/domain')
    if scheme=='auto' and x-h>=lo and x+h<=hi:return [('minus',x-h),('plus',x+h)],'central'
    if scheme not in ('auto','forward'):raise ValueError('Unsupported difference scheme')
    if x+h<=hi:return [('base',x),('plus',x+h)],'forward_inward'
    if x-h>=lo:return [('minus',x-h),('base',x)],'backward_inward'
    raise ValueError('Step does not fit original bounds; no silent adjustment')

class DependencyGradient:
    def __init__(self,galaxies,model,receipt,*,likelihood_version=PRIMARY_B0,H0_si=H0_SI,c0=10.,c_slope=-.1):
        from .marginal_likelihood import require_version
        require_version(likelihood_version)
        if likelihood_version!=PRIMARY_B0:raise ValueError('Only B0 dependency gradient path is verified; B1 pure evaluators remain supported')
        receipt=json.loads(Path(receipt).read_text()) if not isinstance(receipt,dict) else receipt
        if receipt.get('status')!='PASS' or receipt.get('source_hashes')!=source_hashes() or receipt.get('sample_hash')!=sample_hash(galaxies):raise ValueError('Unverified dependency path/source/sample')
        self.gs=galaxies;self.model=model;self.rows=ledger(model,galaxies)
        if receipt['verified_local_counts'].get(model)!=sum(r['scope']=='galaxy_local' for r in self.rows):raise ValueError('Model dependency coverage missing')
        self.receipt=receipt;self.sample=receipt['sample_hash'];self.version=likelihood_version;self.H0=H0_si;self.c0=c0;self.c_slope=c_slope
    def evaluate(self,theta,*,relative_step=1e-4,absolute_step=None,scheme='auto',order=192,indices=None,full_recompute=False,work_callback=None):
        t,L=validate_native(theta,self.gs,self.model,likelihood_version=self.version,hard_bounds=True,H0_si=self.H0)
        if sample_hash(self.gs)!=self.sample or source_hashes()!=self.receipt['source_hashes']:raise ValueError('Input/source changed after dependency validation')
        requested=list(range(L['k'])) if indices is None else list(indices)
        if len(set(requested))!=len(requested) or any(isinstance(i,bool) or not isinstance(i,(int,np.integer)) or not 0<=i<L['k'] for i in requested):raise ValueError('Invalid requested native indices')
        begin=time.perf_counter();evaluations=[]
        def evaluate_subset(x,ids):
            p=_unpack(x,L);vals=[]
            for gi in ids:
                if work_callback is not None:work_callback(self.model,gi,order)
                r=galaxy_log_marginal(self.gs[gi],p,gi,self.model,likelihood_version=self.version,H0_si=self.H0,c0=self.c0,c_slope=self.c_slope,order=order)
                vals.append(r['log_marginal']);evaluations.append(dict(galaxy=gi,order=order,seconds=r['seconds'],density_nodes=r['density_nodes']))
            return np.array(vals)
        base=evaluate_subset(t,list(range(len(self.gs))));components=[]
        for j in requested:
            row=self.rows[j];ids=list(range(len(self.gs))) if full_recompute else row['affected_indices']
            stencil,kind=step_stencil(t[j],row['bounds'],relative_step,absolute_step,scheme)
            f=[]
            for label,value in stencil:
                if label=='base':values=base[ids]
                else:
                    x=t.copy();x[j]=value;values=evaluate_subset(x,ids)
                f.append(-float(np.sum(values)))
            denom=stencil[1][1]-stencil[0][1];g=(f[1]-f[0])/denom
            if not np.isfinite(g):raise FloatingPointError('Nonfinite native numerical gradient')
            components.append(dict(index=j,name=row['name'],native_gradient=float(g),scaled_gradient=float(g*(row['bounds'][1]-row['bounds'][0])),scheme=kind,native_step=abs(denom)/(2 if kind=='central' else 1),stencil_values=f,denominator=denom,affected_indices=ids))
        return dict(objective=-float(np.sum(base)),baseline_log_marginals=base.tolist(),components=components,
            gradient_native=[v['native_gradient'] for v in components] if requested==list(range(L['k'])) else None,
            requested_indices=requested,model=self.model,likelihood_version=self.version,order=order,scheme=scheme,
            theta_sha256=hashlib.sha256(t.astype('<f8').tobytes()).hexdigest(),source_hashes=self.receipt['source_hashes'],receipt_sample_hash=self.sample,
            actual_galaxy_evaluations=len(evaluations),evaluations=evaluations,wall_seconds=time.perf_counter()-begin,
            cache_policy='baseline only inside this singlecall, exact theta/sample/target/order; no parameter-valued cache survivescall',full_recompute=full_recompute)
    def scipy_value_and_gradient(self,theta,**kwargs):
        if kwargs.get('indices') is not None:raise ValueError('SciPy requires complete native gradient')
        result=self.evaluate(theta,**kwargs)
        return result['objective'],np.asarray(result['gradient_native'],float)
