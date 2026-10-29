"""Gate-E fixed historical points and bounded Gate-D nuisance sensitivity; no fitting."""
from pathlib import Path
import json,sys,time,hashlib,traceback
import numpy as np,pandas as pd
from reference import *
B=Path(__file__).resolve().parent;I=B.parents[2];M=I/'results/frontiers_revision_2026_10/mle_recovery';P=I/'results/frontiers_revision_2026_10/covariance_gate';START=time.monotonic()
def js(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def guard():
 if time.monotonic()-START>1200:raise RuntimeError('Predeclared20min budget exceeded')
def main():
 gate=json.loads((B/'local_tests_v1/LOCAL_CORRECTNESS_GATE.json').read_text());assert gate['status']=='PASS' and gate['source_sha256']==sha(B/'reference.py')
 out=B/'full_sample_v1';out.mkdir(exist_ok=False)
 for p,h in json.loads((B/'SOURCE_HASHES_BEFORE.json').read_text()).items():assert sha(Path(p))==h,p
 sample=load_sample(I/'dataset/sparc_tidy.csv');assert len(sample)==91 and sum(len(g.r) for g in sample)==1782
 names=[g.name for g in sample];vectors={m:np.load(M/'original_recovered/01_mle'/m.replace('-','_')/'theta_mle.npy') for m in MODELS};pars={m:decode(m,t,len(sample)) for m,t in vectors.items()}
 old=pd.read_csv(M/'derived_from_archive/replay_v1/galaxy_likelihood_components.csv').set_index(['model','galaxy']);oldCs=np.load(M/'derived_from_archive/replay_v1/within_galaxy_covariances.npz',allow_pickle=False)
 rows=[];totals=[];failed=[];matrices={};matched=[];tol=json.loads((B/'PREDECLARED_TEST_CONTRACT.json').read_text())['test_acceptance']
 for m,p in pars.items():
  for gi,g in enumerate(sample):
   guard()
   try:
    comp,v,C=evaluate(g,p,gi);assert all(np.isfinite(x) for x in comp.values());oc=old.loc[(m,g.name)];C0=oldCs[m+'__'+g.name];bU=min(abs(p['U'][gi]-.05),abs(p['U'][gi]-1.5))<=1e-4*(1.5-.05);bM=bool(m=='nfw1p' and min(abs(p['logM'][gi]-9),abs(p['logM'][gi]-13))<=1e-4*4)
    rows.append(dict(label='FIXED_HISTORICAL_PARAMETER_POINT',model=m,galaxy=g.name,n=len(g.r),rmax_kpc=float(max(g.r)),D_Mpc=g.D,relD=g.sD/g.D,i_deg=float(np.rad2deg(g.inc)),i_err_deg=float(np.rad2deg(g.si)),inclination_fraction=float(g.si/np.tan(g.inc)),near_physical_i_boundary=bool(min(g.inc,np.pi/2-g.inc)<=3*g.si),Upsilon=float(p['U'][gi]),U_boundary=bool(bU),halo_boundary=bM,any_boundary=bool(bU or bM),old_logL=float(oc.logL),corrected_logL=comp['logL'],delta_logL=comp['logL']-float(oc.logL),old_quad=float(oc.quad),corrected_quad=comp['quad'],delta_quad=comp['quad']-float(oc.quad),old_logdet=float(oc.logdet),corrected_logdet=comp['logdet'],delta_logdet=comp['logdet']-float(oc.logdet),normalization=comp['normalization'],C_relative_Frobenius=float(np.linalg.norm(C-C0)/np.linalg.norm(C0)),C_difference_Frobenius=float(np.linalg.norm(C-C0)),min_eigenvalue=float(np.linalg.eigvalsh(C).min()),jitter=0.))
    matrices[m+'__'+g.name]=C
   except Exception as exc:failed.append(dict(model=m,galaxy=g.name,error=repr(exc),traceback=traceback.format_exc()))
  sub=[x for x in rows if x['model']==m];totals.append(dict(label='FIXED_HISTORICAL_PARAMETER_POINT',model=m,k=p['k'],galaxies_evaluated=len(sub),galaxies_failed=sum(x['model']==m for x in failed),old_logL=sum(x['old_logL'] for x in sub),corrected_logL=sum(x['corrected_logL'] for x in sub),delta_logL=sum(x['delta_logL'] for x in sub),delta_quad=sum(x['delta_quad'] for x in sub),delta_logdet=sum(x['delta_logdet'] for x in sub),normalization=sum(x['normalization'] for x in sub),max_abs_galaxy_delta=max(abs(x['delta_logL']) for x in sub)))
  print('FIXED_POINT',m,totals[-1]['delta_logL'],flush=True)
 js(out/'failures.json',failed);pd.DataFrame(rows).to_csv(out/'per_galaxy.csv',index=False);pd.DataFrame(totals).to_csv(out/'totals.csv',index=False);np.savez_compressed(out/'corrected_covariances.npz',**matrices)
 if failed:raise RuntimeError('Full sample failures retained; no omissions permitted')
 # Compare new module with authenticated P1B corrected-distance references at overlap.
 overlap=pd.read_csv(P/'run_v1/covariance_effects.csv');overlap=overlap[(overlap.label=='historical')&(overlap.specification=='full_D_fixedframe')]
 diff=[]
 for _,x in overlap.iterrows():
  y=next(z for z in rows if z['model']==x.model and z['galaxy']==x['case']);d=y['corrected_logL']-x.logL;diff.append(dict(model=x.model,galaxy=x['case'],reference_minus_P1B=float(d)));assert abs(d)<tol['logL_abs']
 pd.DataFrame(diff).to_csv(out/'P1B_overlap_validation.csv',index=False)
 # Map raw native theta then decode target, so identity includes floating-point reparameterization.
 for source,target in [('mond','ptq-nu'),('ptq-nu','mond'),('mond-screen','ptq-screen'),('ptq-screen','mond-screen')]:
  t=vectors[source].copy()
  t[0]=10**t[0]/(C_LIGHT*H0) if target.startswith('ptq') else np.log10(t[0]*C_LIGHT*H0)
  targetp=decode(target,t,len(sample))
  for gi,g in enumerate(sample):
   a,_,C=evaluate(g,pars[source],gi);b,_,Ct=evaluate(g,targetp,gi);delta=b['logL']-a['logL'];cd=np.linalg.norm(C-Ct)/np.linalg.norm(C)
   matched.append(dict(label='MATCHED_PHYSICAL_POINT',galaxy=g.name,source=source,target=target,delta_logL=delta,C_relative=float(cd),pass_identity=abs(delta)<=tol['identity_logL_abs'] and cd<=tol['covariance_relative']))
 pd.DataFrame(matched).to_csv(out/'matched_native_vector_identities.csv',index=False)
 if not all(x['pass_identity'] for x in matched):raise RuntimeError('Reference identity failed; STOP')
 # Predeclared population diagnostics, no fit/ranking; do not select favorable cases.
 data=pd.DataFrame(rows);diag=[]
 groups={'relD':pd.cut(data.relD,[-1,.05,.1,.20000001],include_lowest=True),'inclination':pd.cut(data.i_deg,[-1,45,75,90.00001]),'n_points':pd.cut(data.n,[0,10,30,1e9]),'rmax_kpc':pd.cut(data.rmax_kpc,[0,5,15,1e9]),'boundary':data.any_boundary,'inclination_boundary':data.near_physical_i_boundary}
 for axis,labels in groups.items():
  d=data.assign(group=labels.astype(str))
  for (model,group),s in d.groupby(['model','group']):diag.append(dict(model=model,diagnostic=axis,group=group,n_galaxies=len(s),sum_delta=float(s.delta_logL.sum()),min_delta=float(s.delta_logL.min()),max_delta=float(s.delta_logL.max()),median_delta=float(s.delta_logL.median())))
 pd.DataFrame(diag).to_csv(out/'diagnostics.csv',index=False)
 js(out/'GATE_E.json',dict(status='PASS',label='FIXED_HISTORICAL_PARAMETER_POINT',galaxy_model_rows=len(rows),failed=0,matched_checks=len(matched),matched_logL_max_abs=max(abs(x['delta_logL']) for x in matched),matched_C_max=max(x['C_relative'] for x in matched),P1B_overlap_max=max(abs(x['reference_minus_P1B']) for x in diff),optimization=False,no_IC_rankings_computed=True))
 # Gate D: same fixed-coordinate primary data, no propagated nuisance covariance in conditional.
 nq=B/'nuisance_v1';nq.mkdir(exist_ok=False);contract=json.loads((B/'PREDECLARED_TEST_CONTRACT.json').read_text());results=[]
 for name in contract['nuisance_cases']:
  gi=names.index(name);g=sample[gi]
  for m,p in pars.items():
   a=evaluate(g,p,gi)[0]['logL']
   for T in contract['nuisance_tail_T']:
    for order in contract['quadrature_orders']:
     guard()
     for r in quadrature(g,p,gi,T,order):results.append(dict(label='FIXED_HISTORICAL_PARAMETER_POINT',galaxy=name,model=m,A_corr=a,**r))
   print('NUISANCE',name,m,flush=True)
 # Fixed synthetic inclination-boundary sensitivity, not a fitted or residual-selected galaxy.
 g=Galaxy('synthetic_i90',np.arange(1.,5.),np.full(4,100.),np.full(4,2.),np.full(4,100.),np.zeros(4),np.zeros(4),10.,.1,np.pi/2,np.deg2rad(10.));p=dict(model='baryon',U=np.array([1.]),sigma=3.)
 a=evaluate(g,p,0)[0]['logL']
 for T in contract['nuisance_tail_T']:
  for order in contract['quadrature_orders']:
   for r in quadrature(g,p,0,T,order,('B0','linear_B0')):results.append(dict(label='SYNTHETIC_REFERENCE',galaxy=g.name,model='baryon',A_corr=a,**r))
 pd.DataFrame(results).to_csv(nq/'integrals.csv',index=False)
 convergence=[]
 for (name,m,T,construction),s in pd.DataFrame(results).groupby(['galaxy','model','T','construction']):
  s=s.set_index('order');delta=float(s.loc[256].logL-s.loc[128].logL);convergence.append(dict(galaxy=name,model=m,T=T,construction=construction,delta256_128=delta,pass_convergence=abs(delta)<=tol['quadrature_last_two_abs']))
 pd.DataFrame(convergence).to_csv(nq/'convergence.csv',index=False)
 effects=[]
 for (name,m),s in pd.DataFrame(results).groupby(['galaxy','model']):
  s=s[s.order==256];primary=s[s.construction=='B0'].set_index('T');B1=s[s.construction=='B1'].set_index('T');linear=s[s.construction=='linear_B0'].set_index('T')
  for T in contract['nuisance_tail_T']:
   effects.append(dict(galaxy=name,model=m,T=T,A_corr=float(primary.loc[T].A_corr),B0_logL=float(primary.loc[T].logL),B1_logL=float(B1.loc[T].logL) if len(B1) else None,linear_truncated_logL=float(linear.loc[T].logL),nonlinear_minus_matched_linear=float(primary.loc[T].logL-linear.loc[T].logL),linear_truncated_minus_Acorr=float(linear.loc[T].logL-primary.loc[T].A_corr),B0_minus_Acorr=float(primary.loc[T].logL-primary.loc[T].A_corr),floor_B1_minus_B0=float(B1.loc[T].logL-primary.loc[T].logL) if len(B1) else None,tail_B0_T5_minus_T4=float(primary.loc[5.].logL-primary.loc[4.].logL)))
 pd.DataFrame(effects).to_csv(nq/'decomposed_sensitivity.csv',index=False)
 js(nq/'SUMMARY.json',dict(integral_rows=len(results),convergence_cases=len(convergence),nonconverged=[x for x in convergence if not x['pass_convergence']],max_delta256_128=max(abs(x['delta256_128']) for x in convergence),max_prior_mass_error=max(abs(x['prior_weight_sum']-1) for x in results),no_fitting=True,tail_T5_is_numerical_approximation_not_exact_physical_only_integral=True))
 print('COMPLETED seconds',time.monotonic()-START,flush=True)
if __name__=='__main__':main()
