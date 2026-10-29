"""Production candidate parity gate. Fixed callbacks only; no optimizer/driver invocation."""
from pathlib import Path
import json,sys,hashlib,ast,types,typing,traceback,subprocess,platform
import numpy as np,pandas as pd
B=Path(__file__).resolve().parent;I=B.parents[2];C=I/'results/frontiers_revision_2026_10/covariance_reference';M=I/'results/frontiers_revision_2026_10/mle_recovery';W=B/'isolated_candidate'
sys.path.insert(0,str(C));import reference as ref
sys.path.insert(0,str(W/'src'))
from ptquat import mle,fit_global as fg,likelihood as like,models
from ptquat.data import load_tidy_sparc
from ptquat.constants import H0_SI

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def js(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 out=B/'parity_v1';out.mkdir(exist_ok=False);ct=json.loads((B/'PREDECLARED_PROTOCOL.json').read_text());tol=ct['parity'];gd=load_tidy_sparc(I/'dataset/sparc_tidy.csv');gs=[gd[k] for k in sorted(gd)];rs=ref.load_sample(I/'dataset/sparc_tidy.csv');names=sorted(gd);n=len(gs)
 assert str(W/'src') in str(mle.__file__)
 vectors={m:np.load(M/'original_recovered/01_mle'/m.replace('-','_')/'theta_mle.npy') for m in ref.MODELS};rows=[];maps=[];paths=[];tests=[];fail=[]
 # Explicit hard stop stubs: even accidental optimizer invocation must fail.
 def forbidden(*a,**k):raise RuntimeError('Optimizer prohibited by P1D')
 mle.minimize=forbidden
 import scipy.optimize
 scipy.optimize.minimize=forbidden;scipy.optimize.minimize_scalar=forbidden
 def check(name,fn):
  try:detail=fn();tests.append(dict(test=name,status='PASS',detail=detail));print('PASS',name,flush=True)
  except Exception as e:fail.append(dict(test=name,error=repr(e),traceback=traceback.format_exc()));tests.append(dict(test=name,status='FAIL'));print('FAIL',name,repr(e),flush=True)
 def all637():
  original=like.gaussian_loglike
  for m,t in vectors.items():
   p=ref.decode(m,t,n)
   for gi,g in enumerate(gs):
    rec={}
    def capture(y,v,Cm):
     rec['C']=Cm.copy();rec['v']=v.copy();rec['components']=ref.gaussian(y,v,Cm);return original(y,v,Cm)
    like.gaussian_loglike=capture
    try:value=mle.galaxy_loglike(t,gs,gi,m,H0_SI)
    finally:like.gaussian_loglike=original
    target,v,Cm=ref.evaluate(rs[gi],p,gi);cerr=float(np.linalg.norm(rec['C']-Cm)/np.linalg.norm(Cm));delta=value-target['logL'];comp=rec['components'];assert np.isfinite(value) and all(np.isfinite(x) for x in comp.values());np.linalg.cholesky(rec['C'])
    rows.append(dict(model=m,galaxy=g.name,label='FIXED_HISTORICAL_PARAMETER_POINT',candidate_logL=value,reference_logL=target['logL'],delta_logL=delta,C_relative=cerr,delta_quad=comp['quad']-target['quad'],delta_logdet=comp['logdet']-target['logdet'],normalization=comp['normalization'],decomposition_minus_production=comp['logL']-value))
    assert abs(delta)<=tol['per_galaxy_logL_abs'] and cerr<=tol['C_relative'],(m,g.name,delta,cerr)
  return dict(rows=len(rows),max_abs_logL=max(abs(x['delta_logL']) for x in rows),max_C_relative=max(x['C_relative'] for x in rows))
 check('all637_reference_parity_and_Gaussian_components_SPD',all637)
 def mappings():
  for source,target,fn in [('mond','ptq-nu',mle.map_theta_mond_to_ptqnu),('ptq-nu','mond',mle.map_theta_ptqnu_to_mond),('mond-screen','ptq-screen',mle.map_theta_mondscreen_to_ptqscreen),('ptq-screen','mond-screen',mle.map_theta_ptqscreen_to_mondscreen)]:
   mapped=fn(vectors[source],n,H0_SI)
   for gi,g in enumerate(gs):
    a=mle.galaxy_loglike(vectors[source],gs,gi,source,H0_SI);b=mle.galaxy_loglike(mapped,gs,gi,target,H0_SI);delta=b-a;maps.append(dict(source=source,target=target,galaxy=g.name,delta_logL=delta));assert abs(delta)<=tol['mapped_logL_abs']
  return dict(rows=len(maps),max_abs=max(abs(x['delta_logL']) for x in maps))
 check('all364_native_matched_parameter_mappings',mappings)
 def allpaths():
  for m,t in vectors.items():
   L=fg._layout(m,n,True,None);expected=np.array([r['reference_logL'] for r in rows if r['model']==m]);total=fg.log_likelihood(t,gs,4.,H0_SI,L,10.,-.1,None,'gauss',8.);full=fg.log_likelihood_full_per_galaxy(t,gs,4.,H0_SI,L,10.,-.1,None,'gauss',8.);diag=fg.log_likelihood_per_galaxy(t,gs,4.,H0_SI,L,10.,-.1,None,'gauss',8.);p=ref.decode(m,t,n);diagex=[]
   for gi,g in enumerate(rs):
    comp,v,Cm=ref.evaluate(g,p,gi);d=np.diag(Cm);diagex.extend(-.5*((g.y-v)**2/d+np.log(2*np.pi*d)))
   assert abs(total-expected.sum())<1e-8;assert np.max(abs(full-expected))<1e-8;assert np.max(abs(diag-np.asarray(diagex)))<1e-8
   endpoint=mle.objective_neg_loglike(t,gs,m,H0_SI);assert abs(endpoint+total)<1e-8
   paths.append(dict(model=m,path='global/full_per_galaxy/pointwise_diagonal/objective',full_total_delta=float(total-expected.sum()),max_pergal_delta=float(np.max(abs(full-expected))),max_diagonal_delta=float(np.max(abs(diag-np.asarray(diagex))))))
  # Extract actual nested optimizer callbacks; do not execute optimizer or outer driver.
  tree=ast.parse((W/'src/ptquat/mle.py').read_text());outer=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='_block_coordinate_mle');nested={x.name:x for x in ast.walk(outer) if isinstance(x,ast.FunctionDef)}
  for m,t in vectors.items():
   L=fg._layout(m,n,True,None);gi=0;env=dict(np=np,theta=t.copy(),L=L,gidx=mle._global_indices(L),galaxies=gs,gi=gi,model=m,H0_si=H0_SI,galaxy_loglike=mle.galaxy_loglike,neg_full=lambda z,m=m:mle.objective_neg_loglike(z,gs,m,H0_SI))
   for name in ['neg_g','neg_g2','neg_um' if m=='nfw1p' else 'neg_u']:
    exec(compile(ast.Module(body=[nested[name]],type_ignores=[]),'actual_candidate_callback','exec'),env)
    x=t[env['gidx']] if name.startswith('neg_g') else np.array([t[L['sl_U'].start],t[L['sl_lM'].start]]) if m=='nfw1p' else float(t[L['sl_U'].start]);val=env[name](x)
    expect=mle.objective_neg_loglike(t,gs,m,H0_SI) if name.startswith('neg_g') else -mle.galaxy_loglike(t,gs,0,m,H0_SI)
    assert abs(val-expect)<1e-8;paths.append(dict(model=m,path='source_AST_'+name,full_total_delta=0.,max_pergal_delta=0.,max_diagonal_delta=0.))
  return dict(records=len(paths),optimizer_calls=0)
 check('actual_likelihood_and_optimizer_callback_paths',allpaths)
 def diagnostics_calls():
  # Avoid import of plotting/MCMC/IO-heavy experiments: execute exact factory/call AST only.
  source=ast.parse((W/'src/ptquat/experiments.py').read_text());factory=next(x for x in source.body if isinstance(x,ast.FunctionDef) and x.name=='_make_vfun');env={k:getattr(models,k) for k in dir(models) if not k.startswith('__')};env.update(dict(np=np,pd=pd,GalaxyData=type(gs[0]),Optional=typing.Optional));exec(compile(ast.Module(body=[factory],type_ignores=[]),'candidate_experiment_factory','exec'),env)
  calls=[x for x in ast.walk(source) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='build_covariance'];run_source=ast.parse((W/'src/ptquat/fit_global.py').read_text());run_call=[x for x in ast.walk(run_source) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='build_covariance' and x.lineno>500][0]
  count=0
  for m,t in vectors.items():
   gi=3;g=gs[gi];p=ref.decode(m,t,n);params=fg._unpack(t,fg._layout(m,n,True,None));row=pd.Series(dict(Upsilon_med=p['U'][gi],Upsilon_bulge_med=p['U'][gi],log10_M200_med=p['logM'][gi] if m=='nfw1p' else 11.))
   vf,_=env['_make_vfun'](m,H0_SI,row,g,params.get('eps'),p.get('q'),p.get('a0'));v=vf(g.r_kpc);E=env.copy();E.update(dict(build_covariance=like.build_covariance,vfun=vf,g=g,r=g.r_kpc,v_mod=v,sig_med=p['sigma'],sigma_use=p['sigma'],H0_si=H0_SI,m=m,lM=float(p['logM'][gi]) if m=='nfw1p' else None,c0=10.,c_slope=-.1,args=types.SimpleNamespace(model=m,c0=10.,c_slope=-.1)))
   target=ref.covariance(rs[gi],p,gi)
   for node in calls+[run_call]:
    matrix=eval(compile(ast.Expression(body=node),'actual_covariance_call','eval'),E);assert np.linalg.norm(matrix-target)/np.linalg.norm(target)<1e-9;count+=1
  return dict(experiments_source_calls=len(calls),fit_postprocess_call=1,model_call_combinations=count,scope='exact covariance/factory expressions only; no diagnostic artifact IO, sampling or fit run driver')
 check('all_experiments_and_postprocess_covariance_expressions',diagnostics_calls)
 def derivatives():
  errs=[]
  for name in ['IC4202','UGC07151','F574-1']:
   gi=names.index(name);g=rs[gi]
   for m,t in vectors.items():
    p=ref.decode(m,t,n);v=ref.velocity(g,p,gi);jd=models.model_distance_jacobian(m,v,g.r,g.D,log10_M200=p['logM'][gi] if m=='nfw1p' else None,H0_si=H0_SI)
    ji=v/np.tan(g.inc)
    for rel in tol['FD_steps']:
     h=rel*g.D;hi=rel*max(g.inc,.5);fd=(ref.mean(g,p,gi,g.D+h)-ref.mean(g,p,gi,g.D-h))/(2*h);fi=(ref.mean(g,p,gi,inc=g.inc+hi)-ref.mean(g,p,gi,inc=g.inc-hi))/(2*hi)
     for axis,a,b in [('D',jd,fd),('i',ji,fi)]:
      ae=float(np.max(abs(a-b)));re=float(np.linalg.norm(a-b)/max(np.linalg.norm(a),1e-12));ok=ae<=tol['Jac_near_zero_abs'] or re<=tol['Jac_relative'];errs.append(dict(model=m,galaxy=name,axis=axis,step=rel,abs_error=ae,relative_norm=re,passed=ok))
      if rel==1e-5:assert ok,(m,name,axis,ae,re)
  pd.DataFrame(errs).to_csv(out/'Jacobian_tests.csv',index=False)
  # Minimal baryon: v80 at D10, derivative4, variance16.
  jd=models.model_distance_jacobian('baryon',np.array([80.]),np.array([1.]),10.);assert abs(jd[0]-4.)<1e-12
  C1=like.build_covariance(np.array([80.]),np.array([1.]),np.array([2.]),10.,1.,60.,3.,None,3.,distance_jacobian_kms_per_Mpc=jd);C2=like.build_covariance(np.array([80.]),np.array([1.]),np.array([2.]),10000.,1000.,60.,3.,None,3.,distance_jacobian_kms_per_Mpc=jd/1000);assert np.max(abs(C1-C2))<1e-12
  gi=names.index('IC4202');g=rs[gi];p=ref.decode('mond',vectors['mond'],n);v=ref.velocity(g,p,gi);ji=v/np.tan(g.inc);np.testing.assert_allclose((ji*np.pi/180*np.rad2deg(g.si))**2,(ji*g.si)**2,atol=1e-20)
  # Required-keyword API ensures no unnoticed radius-only fallback.
  try:like.build_covariance(np.array([80.]),np.array([1.]),np.array([2.]),10.,1.,60.,3.,None)
  except TypeError:pass
  else:raise AssertionError('Legacy caller silently accepted')
  return dict(comparisons=len(errs),finest_max_abs=max(x['abs_error'] for x in errs if x['step']==1e-5),finest_failures=sum(not x['passed'] for x in errs if x['step']==1e-5),minimal_variance=16.,unit_checks=True)
 check('Jacobian_NFW_units_minimal_and_fail_closed_API',derivatives)
 def parameterupdate():
  details=[]
  for m,t in vectors.items():
   gi=names.index('F574-1');g=gs[gi];L=fg._layout(m,n,True,None);base=mle.galaxy_loglike(t,gs,gi,m);a=t.copy();a[L['i_lnsig']]+=np.log(1.02);value=mle.galaxy_loglike(a,gs,gi,m);p=ref.decode(m,a,n);target=ref.evaluate(rs[gi],p,gi)[0]['logL'];assert abs(value-target)<1e-8;assert value!=base;details.append(dict(model=m,point='sigma1.02',delta=value-base))
  return details
 check('parameter_dependent_recalculation',parameterupdate)
 def immutable():
  bad=[p for p,h in json.loads((B/'SOURCE_HASHES_BEFORE.json').read_text()).items() if sha(Path(p))!=h];assert not bad,bad;return dict(changed=0,canonical_imports=False)
 check('canonical_and_prior_evidence_immutability',immutable)
 pd.DataFrame(rows).to_csv(out/'all637.csv',index=False);pd.DataFrame(maps).to_csv(out/'all364.csv',index=False);pd.DataFrame(paths).to_csv(out/'call_paths.csv',index=False);js(out/'tests.json',tests);js(out/'failures.json',fail)
 passed=not fail;js(out/'PARITY_GATE.json',dict(status='PASS' if passed else 'FAIL',tests=len(tests),failures=len(fail),rows637=len(rows),maps364=len(maps),max_logL_abs=max(abs(x['delta_logL']) for x in rows),max_C_relative=max(x['C_relative'] for x in rows),max_mapped_logL_abs=max(abs(x['delta_logL']) for x in maps),protocol_sha256=sha(B/'PREDECLARED_PROTOCOL.json'),patch_sha256=sha(B/'production_candidate.diff'),imported_mle=str(mle.__file__),optimizer_invoked=False))
 js(out/'environment.json',dict(python=sys.version,platform=platform.platform(),executable=sys.executable));(out/'environment_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
 if not passed:raise SystemExit(1)
if __name__=='__main__':main()
