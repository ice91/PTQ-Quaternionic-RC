"""Supplemental parity at existing66 predeclared perturbations; no new integrations."""
from pathlib import Path
import sys,json,hashlib,ast
import numpy as np,pandas as pd
B=Path(__file__).resolve().parent;I=B.parents[2];W=B/'isolated_candidate';C=I/'results/frontiers_revision_2026_10/covariance_reference';M=I/'results/frontiers_revision_2026_10/mle_recovery'
sys.path.insert(0,str(C));import reference as ref
sys.path.insert(0,str(W/'src'));from ptquat import mle,fit_global as fg,likelihood as lk
from ptquat.data import load_tidy_sparc

def main():
 out=B/'neighborhood_patch_parity_v1';out.mkdir(exist_ok=False);meta=json.loads((B/'boundary_v1/parameter_points.json').read_text());gsd=load_tidy_sparc(I/'dataset/sparc_tidy.csv');gs=[gsd[k] for k in sorted(gsd)];rs=ref.load_sample(I/'dataset/sparc_tidy.csv');names=sorted(gsd);rows=[]
 tree=ast.parse((W/'src/ptquat/mle.py').read_text());outer=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='_block_coordinate_mle');nested={x.name:x for x in ast.walk(outer) if isinstance(x,ast.FunctionDef)}
 for r in meta:
  m=r['model'];gi=names.index(r['galaxy']);original=np.load(M/'original_recovered/01_mle'/m.replace('-','_')/'theta_mle.npy');t=original.copy()
  for x in r['coordinates']:assert t[x['index']]==x['old'];t[x['index']]=x['new']
  assert hashlib.sha256(t.astype('<f8').tobytes()).hexdigest()==r['theta_float64_sha256']
  captured={};prod=lk.gaussian_loglike
  def record(y,v,Cm):captured['C']=Cm.copy();return prod(y,v,Cm)
  lk.gaussian_loglike=record
  try:val=mle.galaxy_loglike(t,gs,gi,m,ref.H0)
  finally:lk.gaussian_loglike=prod
  p=ref.decode(m,t,91);expected,_,Cm=ref.evaluate(rs[gi],p,gi);delta=val-expected['logL'];cerr=float(np.linalg.norm(captured['C']-Cm)/np.linalg.norm(Cm));assert abs(delta)<=1e-8 and cerr<=1e-9
  L=fg._layout(m,91,True,None);gidx=mle._global_indices(L);name='neg_g' if r['point'] in ['scale1.01','q1.01','sigma1.02'] else 'neg_um' if m=='nfw1p' else 'neg_u';env=dict(np=np,theta=original.copy(),L=L,gidx=gidx,gi=gi,galaxies=gs,model=m,H0_si=ref.H0,galaxy_loglike=mle.galaxy_loglike,neg_full=lambda v,m=m:mle.objective_neg_loglike(v,gs,m,ref.H0))
  exec(compile(ast.Module(body=[nested[name]],type_ignores=[]),'actual_neighborhood_callback','exec'),env)
  x=t[gidx] if name=='neg_g' else np.array([t[L['sl_U'].start+gi],t[L['sl_lM'].start+gi]]) if name=='neg_um' else float(t[L['sl_U'].start+gi]);cb=env[name](x);end=mle.objective_neg_loglike(t,gs,m,ref.H0) if name=='neg_g' else -val;assert abs(cb-end)<=1e-8
  rows.append(dict(model=m,galaxy=r['galaxy'],point=r['point'],candidate_minus_reference=delta,C_relative=cerr,source_callback=name,callback_minus_endpoint=float(cb-end)))
 assert len(rows)==66 and np.isfinite(pd.DataFrame(rows).select_dtypes(include='number').to_numpy()).all()
 pd.DataFrame(rows).to_csv(out/'all66.csv',index=False);result=dict(status='PASS',points=len(rows),max_logL_abs=max(abs(x['candidate_minus_reference']) for x in rows),max_C_relative=max(x['C_relative'] for x in rows),max_callback_delta=max(abs(x['callback_minus_endpoint']) for x in rows),new_integrations=0,optimizer_calls=0)
 (out/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
