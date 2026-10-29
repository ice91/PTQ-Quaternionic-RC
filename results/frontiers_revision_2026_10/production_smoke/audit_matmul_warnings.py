"""Read-only fixed-node warning diagnostics. No target changes, no fitting."""
from pathlib import Path
from types import SimpleNamespace
import sys,json,warnings
import numpy as np
from scipy.special import logsumexp
B=Path(__file__).resolve().parent;I=B.parents[2];sys.path.insert(0,str(B/'candidate/src'))
from ptquat import mle,fit_global as fg,marginal_likelihood as ml
from ptquat.data import load_tidy_sparc

def main():
 out=B/'warning_audit_v1';out.mkdir(exist_ok=False);gd=load_tidy_sparc(str(I/'dataset/sparc_tidy.csv'));gs=[gd[k] for k in sorted(gd)];rows=[];triggered=[]
 t=np.load(I/'results/frontiers_revision_2026_10/mle_recovery/original_recovered/01_mle/baryon/theta_mle.npy');p=fg._unpack(t,fg._layout('baryon',91,True,None))
 for gi,g0 in enumerate(gs):
  with warnings.catch_warnings(record=True) as ws:
   warnings.simplefilter('always');r=ml.galaxy_log_marginal(g0,p,gi,'baryon')
  if ws:triggered.append((gi,g0,[str(w.message) for w in ws]))
 # All triggered real cases are cross-checked, or first predeclared galaxy if no flags inthisprocess.
 selected=triggered if triggered else [(0,gs[0],[])]
 for gi,g0,warnings_in in selected:
  g=SimpleNamespace(r=g0.r_kpc,y=g0.v_obs,e=g0.v_err,disk=g0.v_disk,bulge=g0.v_bulge,gas=g0.v_gas,D=g0.D_Mpc,sD=g0.D_err_Mpc,inc=np.deg2rad(g0.i_deg),si=np.deg2rad(g0.i_err_deg))
  ds,wd,ii,wi,_,_=ml.nuisance_nodes(g,'segmented_quantile',192);pr=dict(p,model='baryon',_H0=ml.H0_SI,_c0=10.,_c_slope=-.1);vd=ml.velocity(g,pr,gi,ds[:,None]);b=np.sin(ii)/np.sin(g.inc)
  for floor,version in [('B0',ml.PRIMARY_B0),('B1',ml.SENSITIVITY_B1)]:
   # Direct residual squared, no sufficient-sum cancellation and no matrix multiplication.
   acc=-np.inf
   for k in range(len(ds)):
    mu=b[:,None]*vd[k][None,:];ll=ml.conditional_logdensity(g,mu,p['sigma'],b[:,None],floor)
    assert np.isfinite(ll).all();acc=np.logaddexp(acc,logsumexp(ll+np.log(wi))+np.log(wd[k]))
   fast=ml.galaxy_log_marginal(g0,p,gi,'baryon',likelihood_version=version)['log_marginal']
   rows.append(dict(galaxy=g0.name,model='baryon',floor=floor,triggered_warnings=warnings_in,fast=fast,direct=float(acc),delta=float(fast-acc),D_nodes=len(ds),i_nodes=len(ii)))
   assert abs(fast-acc)<=1e-8
 summary=dict(status='PASS',triggered_real_cases=len(triggered),checks=rows,max_abs_delta=max(abs(r['delta']) for r in rows),interpretation='warnings retained; finite results independently match direct residual-squared at identical full192 nodes; root cause not proven; no warning suppression or target adjustment')
 (out/'AUDIT.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
