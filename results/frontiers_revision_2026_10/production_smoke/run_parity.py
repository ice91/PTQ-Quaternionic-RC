"""Pre-smoke acceptance. No optimizer is called by this script."""
from pathlib import Path
import sys,json,hashlib,time,traceback,platform,subprocess
import numpy as np,pandas as pd
B=Path(__file__).resolve().parent;I=B.parents[2];E=I/'results/frontiers_revision_2026_10/uniform_target_gate'
sys.path.insert(0,str(B/'candidate/src'));sys.path.insert(1,str(B/'candidate/tests'))
from test_uniform_target import run
from ptquat import mle,fit_global as fg,experiments as experiments,marginal_likelihood as ml

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def js(p,r):p.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
def compare_keys(df,cols,expected):
    assert len(df)==expected and not df.duplicated(cols).any()
    assert np.isfinite(df.select_dtypes(include='number').to_numpy()).all()

def main():
 out=B/'parity_v1';out.mkdir(exist_ok=False);start=time.monotonic();attempts=[]
 try:
  gs,vectors=run(out);names=[g.name for g in gs]
  expected=pd.read_csv(E/'acceptance_v1/full_sample_all1274.csv').set_index(['model','galaxy','floor']);rows=[];cache={}
  for model,theta in vectors.items():
   for floor,version in [('B0',ml.PRIMARY_B0),('B1',ml.SENSITIVITY_B1)]:
    for order in [96,192]:
     r=mle.evaluate_loglike_components(theta,gs,model,likelihood_version=version,order=order)
     attempts.append(dict(stage='fullsample',model=model,floor=floor,order=order,status='COMPLETE'))
     for gi,g in enumerate(gs):
      v=r['galaxy_log_marginals'][gi];old=expected.loc[(model,g.name,floor)];ref=float(old.explicit_logL if order==192 else old.coarse_logL)
      rows.append(dict(model=model,galaxy=g.name,floor=floor,order=order,logL=v,reference_logL=ref,parity_delta=v-ref))
      cache[(model,g.name,floor,order)]=v
   print('PARITY full',model,flush=True)
  df=pd.DataFrame(rows);compare_keys(df,['model','galaxy','floor','order'],2548);assert set(df.set_index(['model','galaxy','floor']).index)==set(expected.index)
  df.to_csv(out/'native_parity_all2548.csv',index=False);assert abs(df.parity_delta).max()<=1e-8
  refinements=[]
  for model in vectors:
   for floor in ['B0','B1']:
    d=np.array([cache[(model,g.name,floor,192)]-cache[(model,g.name,floor,96)] for g in gs]);refinements.append(dict(model=model,floor=floor,max_pergal=float(abs(d).max()),sumabs=float(abs(d).sum()),signed=float(d.sum())))
  pd.DataFrame(refinements).to_csv(out/'refinement_by_model.csv',index=False)
  assert max(x['max_pergal'] for x in refinements)<=1e-4 and max(x['sumabs'] for x in refinements)<=.01
  mapping_rows=[]
  maps=[('mond','ptq-nu',mle.map_theta_mond_to_ptqnu),('ptq-nu','mond',mle.map_theta_ptqnu_to_mond),('mond-screen','ptq-screen',mle.map_theta_mondscreen_to_ptqscreen),('ptq-screen','mond-screen',mle.map_theta_ptqscreen_to_mondscreen)]
  for source,target,fun in maps:
   mapped=fun(vectors[source],91)
   for floor,ver in [('B0',ml.PRIMARY_B0),('B1',ml.SENSITIVITY_B1)]:
    r=mle.evaluate_loglike_components(mapped,gs,target,likelihood_version=ver)
    for gi,g in enumerate(gs):mapping_rows.append(dict(source=source,target=target,galaxy=g.name,floor=floor,delta_logL=r['galaxy_log_marginals'][gi]-cache[(source,g.name,floor,192)]))
  md=pd.DataFrame(mapping_rows);compare_keys(md,['source','target','galaxy','floor'],728);assert abs(md.delta_logL).max()<=1e-8;md.to_csv(out/'native_matched_all728.csv',index=False)
  # Frozen P1E neighborhood: reconstruct native theta from original metadata, including declared floor-bound supplement.
  meta=json.loads((I/'results/frontiers_revision_2026_10/patch_boundary_gate/boundary_v1/parameter_points.json').read_text());points=[]
  for p in meta:
   t=vectors[p['model']].copy()
   for c in p['coordinates']:assert t[c['index']]==c['old'];t[c['index']]=c['new']
   assert sha_bytes(t)==p['theta_float64_sha256'];points.append((p['galaxy'],p['model'],p['point'],t))
  prot=json.loads((E/'PREDECLARED_PROTOCOL.json').read_text())
  for p in prot['neighborhood']['near_bound_supplement']:
   val=.001001 if p['galaxy']=='IC4202' else 199.999
   for m in p['models']:
    t=vectors[m].copy();t[-1]=np.log(val);points.append((p['galaxy'],m,'floor_bound_'+str(val),t))
  ne=pd.read_csv(E/'acceptance_v1/neighborhood_all144.csv').set_index(['galaxy','model','point','floor']);nr=[]
  for name,m,point,t in points:
   gi=names.index(name);p=fg._unpack(t,fg._layout(m,91,True,None))
   for floor,ver in [('B0',ml.PRIMARY_B0),('B1',ml.SENSITIVITY_B1)]:
    hi=ml.galaxy_log_marginal(gs[gi],p,gi,m,likelihood_version=ver,order=192)['log_marginal'];lo=ml.galaxy_log_marginal(gs[gi],p,gi,m,likelihood_version=ver,order=96)['log_marginal'];old=ne.loc[(name,m,point,floor)]
    nr.append(dict(galaxy=name,model=m,point=point,floor=floor,logL=hi,parity_delta=hi-float(old.explicit_logL),refinement_delta=hi-lo,theta_sha256=sha_bytes(t)))
  nd=pd.DataFrame(nr);compare_keys(nd,['galaxy','model','point','floor'],144);assert set(nd.set_index(['galaxy','model','point','floor']).index)==set(ne.index);assert abs(nd.parity_delta).max()<=1e-8;assert abs(nd.refinement_delta).max()<=1e-4;nd.to_csv(out/'neighborhood_all144.csv',index=False)
  # Actual public calls, globals, per-galaxy and experiments; no AST substitute for path tests.
  pathrows=[]
  for model,t in vectors.items():
   L=fg._layout(model,91,True,None);args=(t,gs,4.,ml.H0_SI,L,10.,-.1,None,ml.PRIMARY_B0,8.)
   reference=np.array([cache[(model,g.name,'B0',192)] for g in gs]);scalar=float(reference.sum())
   outputs={'mle_objective':-mle.objective_neg_loglike(t,gs,model,ml.H0_SI),'mle_evaluate':mle.evaluate_loglike(t,gs,model),
     'fit_global_scalar':fg.log_likelihood(*args),'fit_global_pergal':np.sum(fg.log_likelihood_full_per_galaxy(*args)),
     'experiments_pergal':np.sum(experiments.per_galaxy_log_marginals(t,gs,model))}
   for name,v in outputs.items():pathrows.append(dict(model=model,path=name,delta=float(v-scalar)));assert abs(v-scalar)<=1e-8
   for gi in [0,90]:assert abs(mle.galaxy_loglike(t,gs,gi,model)-reference[gi])<=1e-8
   # Forced covariance failure proves primary paths do not invoke propagated covariance.
   old=fg.build_covariance
   try:
    fg.build_covariance=lambda *a,**kw:(_ for _ in ()).throw(AssertionError('primary used historical covariance'))
    assert abs(fg.log_likelihood(*args)-scalar)<=1e-8
   finally:fg.build_covariance=old
   descriptive=experiments.descriptive_residuals(t,gs,model);assert len(descriptive)==91 and all(r['interpretation']=='DESCRIPTIVE_RESIDUAL_ONLY_NOT_POINTWISE_MARGINAL' for r in descriptive)
   print('CALL PATH PASS',model,flush=True)
  pd.DataFrame(pathrows).to_csv(out/'actual_call_paths.csv',index=False)
  # Repeat same full91 NFW candidate objective, not two differently optimized points.
  x=mle.evaluate_loglike_components(vectors['nfw1p'],gs,'nfw1p');y=mle.evaluate_loglike_components(vectors['nfw1p'],gs,'nfw1p');repeat=float(np.max(abs(np.array(x['galaxy_log_marginals'])-y['galaxy_log_marginals'])));assert repeat<=1e-12
  before=json.loads((B/'SOURCE_HASHES_BEFORE.json').read_text());assert all(sha(Path(p))==h for p,h in before.items())
  summary=dict(status='PASS',native_B0=637,native_B1=637,native_fine_coarse_rows=len(df),native_matched=len(md),neighborhood=len(nd),actual_public_paths=len(pathrows),max_native_parity=float(abs(df.parity_delta).max()),max_mapping_delta=float(abs(md.delta_logL).max()),max_neighborhood_parity=float(abs(nd.parity_delta).max()),max_refinement_pergal=max(x['max_pergal'] for x in refinements),max_refinement_sumabs=max(x['sumabs'] for x in refinements),repeat_pergal_max=repeat,elapsed_seconds=time.monotonic()-start,source_unchanged=len(before))
  js(out/'PARITY_GATE.json',summary);print(json.dumps(summary,indent=2),flush=True)
  js(B/'CANDIDATE_TESTED_HASHES.json',{str(p):sha(p) for p in (B/'candidate').rglob('*') if p.is_file()})
  js(out/'environment.json',dict(python=sys.version,executable=sys.executable,platform=platform.platform(),singlethread_environment=True));(out/'environment_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
 except Exception as exc:
  js(out/'PARITY_FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),elapsed=time.monotonic()-start));raise
 finally:js(out/'attempts.json',attempts)
def sha_bytes(t):return hashlib.sha256(t.astype('<f8').tobytes()).hexdigest()
if __name__=='__main__':main()
