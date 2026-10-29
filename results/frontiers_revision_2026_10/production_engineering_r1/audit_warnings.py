"""Frozen representative full192-node direct residual-squared audit."""
from pathlib import Path
from types import SimpleNamespace
import sys,json
import numpy as np
from scipy.special import logsumexp
B=Path(__file__).resolve().parent;I=B.parents[2];sys.path.insert(0,str(B/'candidate/src'))
from ptquat.data import load_tidy_sparc
from ptquat import mle,fit_global as fg,marginal_likelihood as ml
from ptquat.warning_telemetry import WarningCollector,collect_warnings

def main():
 out=B/'warning_audit_v1';out.mkdir(exist_ok=False);ct=json.loads((B/'PREDECLARED_CONTRACT.json').read_text())['warning_representatives'];gd=load_tidy_sparc(str(I/'dataset/sparc_tidy.csv'));gs=[gd[k] for k in sorted(gd)];names=[g.name for g in gs];rows=[];collector=WarningCollector()
 with collect_warnings(collector):
  for model in ct['models']:
   t=np.load(I/'results/frontiers_revision_2026_10/mle_recovery/original_recovered/01_mle'/model.replace('-','_')/'theta_mle.npy');p=fg._unpack(t,fg._layout(model,91,True,None));p.update(_H0=ml.H0_SI,_c0=10.,_c_slope=-.1)
   for name in ct['galaxies']:
    gi=names.index(name);g0=gs[gi];g=SimpleNamespace(r=g0.r_kpc,y=g0.v_obs,e=g0.v_err,disk=g0.v_disk,bulge=g0.v_bulge,gas=g0.v_gas,D=g0.D_Mpc,sD=g0.D_err_Mpc,inc=np.deg2rad(g0.i_deg),si=np.deg2rad(g0.i_err_deg));ds,wd,ii,wi,ZD,Zi=ml.nuisance_nodes(g,'segmented_quantile',192);vd=ml.velocity(g,p,gi,ds[:,None]);b=np.sin(ii)/np.sin(g.inc)
    for floor,version in [('B0',ml.PRIMARY_B0),('B1',ml.SENSITIVITY_B1)]:
     acc=-np.inf
     for k in range(len(ds)):
      ll=ml.conditional_logdensity(g,b[:,None]*vd[k][None,:],p['sigma'],b[:,None],floor);assert np.isfinite(ll).all();acc=np.logaddexp(acc,logsumexp(ll+np.log(wi))+np.log(wd[k]))
     fast=ml.galaxy_log_marginal(g0,p,gi,model,likelihood_version=version)['log_marginal'];delta=float(fast-acc);assert abs(delta)<=1e-8
     rows.append(dict(model=model,galaxy=name,floor=floor,order=192,fast_logL=fast,direct_logL=float(acc),delta=delta,classification='UNRESOLVED_WARNING_ROOT_CAUSE',numerical_validity='PASS_DIRECT_RESIDUAL_CHECK_AT_THIS_POINT_ONLY'))
 summary=dict(status='PASS',rows=rows,max_abs_delta=max(abs(r['delta']) for r in rows),classification='UNRESOLVED_WARNING_ROOT_CAUSE',corruption_confirmed=False)
 (out/'AUDIT.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'warnings.json').write_text(json.dumps(collector.as_dict(),indent=2)+'\n');print('WARNING DIRECT PASS',len(rows),summary['max_abs_delta'])
if __name__=='__main__':main()
