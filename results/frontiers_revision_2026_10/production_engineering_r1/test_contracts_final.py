"""Fail-closed contract/telemetry controls. Nooptimizer."""
from pathlib import Path
from types import SimpleNamespace
import sys,json,warnings
import numpy as np
B=Path(__file__).resolve().parent;I=B.parents[2];sys.path.insert(0,str(B/'candidate/src'))
from ptquat import mle,fit_global as fg,experiments as exp,marginal_likelihood as ml
from ptquat.data import load_tidy_sparc
from ptquat.evaluation_contract import validate_native
from ptquat.warning_telemetry import WarningCollector,collect_warnings,WarningContext
from ptquat.dependency_gradient import step_stencil,DependencyGradient

def main():
 out=B/'contracts_v2';out.mkdir(exist_ok=False);gd=load_tidy_sparc(str(I/'dataset/sparc_tidy.csv'));gs=[gd[k] for k in sorted(gd)];t=np.load(I/'results/frontiers_revision_2026_10/mle_recovery/original_recovered/01_mle/ptq_screen/theta_mle.npy');L=fg._layout('ptq-screen',91,True,None);records=[]
 def reject(name,fn):
  try:fn()
  except (ValueError,RuntimeError,FloatingPointError):records.append(dict(case=name,status='PASS_REJECTED'))
  else:raise AssertionError(name+' accepted')
 for label,bad in [('missing',t[:-1]),('extra',np.r_[t,1]),('matrix',t[None,:]),('NaN',np.where(np.arange(len(t))==0,np.nan,t)),('infinity',t*np.inf)]:
  for path,fn in [('scalar',lambda:mle.evaluate_loglike(bad,gs,'ptq-screen')),('components',lambda:mle.evaluate_loglike_components(bad,gs,'ptq-screen')),('local',lambda:mle.galaxy_loglike(bad,gs,0,'ptq-screen')),('fg',lambda:fg.log_likelihood_full_per_galaxy(bad,gs,4.,ml.H0_SI,L,10.,-.1,None,ml.PRIMARY_B0,8.)),('descriptive',lambda:exp.descriptive_residuals(bad,gs,'ptq-screen'))]:reject(label+'/'+path,fn)
 for index in [-1,91,1.5,True,'NOT_A_GALAXY']:reject('invalidID/'+str(index),lambda:mle.galaxy_loglike(t,gs,index,'ptq-screen'))
 p=fg._unpack(t,L);reject('decoded_negative_index',lambda:ml.galaxy_log_marginal(gs[0],p,-1,'ptq-screen'));reject('decoded_badU',lambda:ml.galaxy_log_marginal(gs[0],dict(p,U=np.ones((91,1))),0,'ptq-screen'))
 for version in ['gauss','UNKNOWN',None]:reject('badversion/'+str(version),lambda:mle.evaluate_loglike_components(t,gs,'ptq-screen',likelihood_version=version))
 outside=t.copy();outside[L['sl_U'].start]=1.6;validate_native(outside,gs,'ptq-screen',likelihood_version=ml.PRIMARY_B0);reject('hard_bound_no_clip',lambda:mle.objective_neg_loglike(outside,gs,'ptq-screen',ml.H0_SI));assert outside[L['sl_U'].start]==1.6
 for name in exp.DISABLED_LEGACY_DIAGNOSTICS:reject('firewall/'+name,lambda:getattr(exp,name)(likelihood_version=ml.LEGACY_DIAGONALIZED_SENSITIVITY))
 reject('fg_legacy_disabled',lambda:fg.log_likelihood_per_galaxy(t,gs,4.,ml.H0_SI,L,10.,-.1,None,ml.LEGACY_DIAGONALIZED_SENSITIVITY,8.))
 reject('unverified_B1_gradient',lambda:DependencyGradient(gs,'ptq-screen',B/'dependency_v1/CURRENT_VERIFIED_DEPENDENCIES.json',likelihood_version=ml.SENSITIVITY_B1))
 c=WarningCollector()
 with collect_warnings(c):
  with WarningContext('synthetic',ml.PRIMARY_B0,dict(U=np.ones(1),sigma=1.),0):
   for _ in range(17):warnings.warn('repeated warning control',RuntimeWarning)
 assert c.as_dict()['total_events']==17 and c.as_dict()['distinct_keys']==1
 with warnings.catch_warnings(record=True) as ws:
  warnings.simplefilter('always');warnings.warn('outside aggregation remains visible',RuntimeWarning)
 assert len(ws)==1
 for bound,x in [((0.,1.),0.),((0.,1.),1.),((0.,1.),.5)]:
  stencil,kind=step_stencil(x,bound,1e-4);assert all(bound[0]<=v<=bound[1] for _,v in stencil)
 reject('unverified_dependency',lambda:DependencyGradient(gs,'ptq-screen',dict(status='FAIL')))
 # Bad quadrature weights negative control; original function restored, no policy change.
 old=ml.roots_legendre;ml.legendre.cache_clear()
 try:
  ml.roots_legendre=lambda n:(np.zeros(n),-np.ones(n));reject('invalid_quad_weights',lambda:ml.legendre(7))
 finally:ml.roots_legendre=old;ml.legendre.cache_clear()
 (out/'TESTS.json').write_text(json.dumps(dict(status='PASS',cases=records,telemetry=c.as_dict(),pure_evaluation_outside_bounds_not_clipped=True),indent=2)+'\n');print('CONTRACT PASS',len(records),'negative cases and warning counts17/1')
if __name__=='__main__':main()
