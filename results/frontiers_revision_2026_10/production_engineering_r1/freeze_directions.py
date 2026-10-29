from pathlib import Path
from types import SimpleNamespace
import sys,json,datetime,hashlib
import numpy as np
B=Path(__file__).resolve().parent;I=B.parents[2];sys.path.insert(0,str(B/'candidate/src'))
from ptquat.data import load_tidy_sparc
from ptquat import mle,fit_global as fg
from ptquat.evaluation_contract import MODELS
from ptquat.marginal_likelihood import nuisance_nodes

def main():
 gd=load_tidy_sparc(str(I/'dataset/sparc_tidy.csv'));gs=[gd[k] for k in sorted(gd)];names=[g.name for g in gs];directions=[];costs={}
 for m in MODELS:
  costs[m]={};total=0
  for order in [192,96]:
   records=[]
   for gi,g0 in enumerate(gs):
    g=SimpleNamespace(D=g0.D_Mpc,sD=g0.D_err_Mpc,inc=np.deg2rad(g0.i_deg),si=np.deg2rad(g0.i_err_deg),y=g0.v_obs,e=g0.v_err)
    ds,wd,ii,wi,_,_=nuisance_nodes(g,'segmented_quantile',order);n=len(g0.v_obs)
    cost=len(ds)*len(ii)*(2*n+10)+(100 if m=='nfw1p' else 30)*len(ds)*n
    records.append(dict(galaxy=g0.name,gi=gi,n=n,nD=len(ds),ni=len(ii),arithmetic_proxy=cost))
   costs[m][str(order)]=records
  norm=sum(r['arithmetic_proxy'] for r in costs[m]['192'])
  for order in ['192','96']:
   for r in costs[m][order]:r['fullsample_equivalent']=r['arithmetic_proxy']/norm
 for m in ['ptq-screen','nfw1p']:
  t=np.load(I/'results/frontiers_revision_2026_10/mle_recovery/original_recovered/01_mle'/m.replace('-','_')/'theta_mle.npy');L=fg._layout(m,91,True,None);bounds=mle.theta_bounds(m,91)
  if m=='ptq-screen':ids=[L['i_eps'],L['i_lq'],L['i_lnsig'],L['sl_U'].start+names.index('F574-1')]
  else:
   contact=[j for j in range(L['sl_U'].start,L['sl_U'].stop) if min(t[j]-bounds[j][0],bounds[j][1]-t[j])<=1e-7*(bounds[j][1]-bounds[j][0])];assert contact,'No frozen boundcontact available'
   ids=[L['i_lnsig'],contact[0],L['sl_lM'].start]
  for j in ids:directions.append(dict(model=m,index=j,theta_value=float(t[j]),bounds=list(bounds[j]),bound_contact=bool(min(t[j]-bounds[j][0],bounds[j][1]-t[j])<=1e-7*(bounds[j][1]-bounds[j][0])),selection='predeclaredrole/fixedF574-1/firstnativeboundcontact/firsthalo; no gradient or residual selection'))
 record=dict(frozen_utc=datetime.datetime.utcnow().isoformat()+'Z',directions=directions,steps=[1e-4,5e-5],orders=[192,96],source_contract_sha256=hashlib.sha256((B/'PREDECLARED_CONTRACT.json').read_bytes()).hexdigest(),budget=dict(max_fullsample_equivalents=96,max_seconds=300),repeat='each model finebaseline twice before directions',archive_comparator=dict(step='exact archived actual percoordinate1e-8 forward orbackward nearbounds',evidence='same-step dependencyFD versus archived fullsum; notprecision proof',roundoff_tolerance='gamma91*2*sum(abs(base_logmarginals))/abs(actualh), native units; nochange frozen directionalcriteria'),benchmark='one full native gradient/model atlegacy1e-8step; reusearchiveddense, no extra95/184fullpasses')
 for name,r in [('DIRECTION_PROTOCOL.json',record),('GALAXY_WORK_WEIGHTS.json',costs)]:
  p=B/name;assert not p.exists();p.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
 print(json.dumps(record,indent=2))
if __name__=='__main__':main()
