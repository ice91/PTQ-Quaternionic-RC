"""Predeclared independent local correctness tests; no optimization; preserve all attempts."""
from pathlib import Path
import os,sys,json,hashlib,traceback,subprocess,platform
import numpy as np
from scipy.integrate import quad
from reference import *
B=Path(__file__).resolve().parent;I=B.parents[2];M=I/'results/frontiers_revision_2026_10/mle_recovery'
sys.path.insert(0,str(I/'src'))
from ptquat import models as production_models
from ptquat.fit_global import _layout,_unpack

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def js(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 out=B/'local_tests_v1';out.mkdir(exist_ok=False);ct=json.loads((B/'PREDECLARED_TEST_CONTRACT.json').read_text());tol=ct['test_acceptance'];sample=load_sample(I/'dataset/sparc_tidy.csv');names=[g.name for g in sample]
 ps={m:decode(m,np.load(M/'original_recovered/01_mle'/m.replace('-','_')/'theta_mle.npy'),len(sample)) for m in MODELS};records=[];fd=[]
 def check(name,fn,hard=True):
  try:detail=fn();records.append(dict(test=name,status='PASS',hard=hard,detail=detail));print('PASS',name,flush=True)
  except Exception as exc:records.append(dict(test=name,status='FAIL',hard=hard,error=repr(exc),traceback=traceback.format_exc()));print('FAIL',name,repr(exc),flush=True)
 def preservation():
  before=json.loads((B/'SOURCE_HASHES_BEFORE.json').read_text());bad=[p for p,h in before.items() if sha(Path(p))!=h];assert not bad,bad;return dict(files=len(before),changed=0)
 check('source_and_input_immutability_before',preservation)
 def minimal():
  g=Galaxy('synthetic_minimal',np.array([1.]),np.array([80.]),np.array([2.]),np.array([80.]),np.zeros(1),np.zeros(1),10.,1.,np.pi/3,np.deg2rad(3.));p=dict(model='baryon',U=np.ones(1),sigma=3.)
  assert abs(velocity(g,p,0,40.)[0]-160.)<1e-12
  assert abs(derivatives(g,p,0)[0][0]-4.)<1e-12
  return dict(full_mean_derivative=4.,omitted_variance=16.,production_baryon_radius_only_derivative=0.)
 check('minimal_baryon_counterexample',minimal)
 def accounting():
  counts={}
  for m,p in ps.items():
   th=np.load(M/'original_recovered/01_mle'/m.replace('-','_')/'theta_mle.npy');L=_layout(m,len(sample),True,None);prod=_unpack(th,L)
   assert L['k']==p['k']==json.loads((B/'STATISTICAL_SPECIFICATION.json').read_text())['parameter_counts'][m]
   np.testing.assert_array_equal(p['U'],prod['U']);assert abs(p['sigma']-prod['sigma'])<1e-12
   if m=='nfw1p':np.testing.assert_array_equal(p['logM'],prod['lM'])
   counts[m]=p['k']
  return counts
 check('parameter_layout_accounting',accounting)
 def all_fd():
  for name in ct['local_real_cases']:
   gi=names.index(name);g=sample[gi]
   for m,p in ps.items():
    v0=velocity(g,p,gi);jd,ji=derivatives(g,p,gi)
    # Independent production velocity equation for every rescaled template, not derivatives.
    for s in [.7,1.,1.3]:
     args=[p['U'][gi],g.r*s,g.disk*np.sqrt(s),g.bulge*np.sqrt(s),g.gas*np.sqrt(s)]
     if m=='baryon':vp=production_models.model_v_baryon(*args)
     elif m=='ptq':vp=production_models.model_v_ptq(args[0],p['eps'],*args[1:])
     elif m=='mond':vp=production_models.model_v_mond(args[0],p['a0'],*args[1:])
     elif m=='ptq-nu':vp=production_models.model_v_ptq_nu(args[0],p['eps'],*args[1:])
     elif m=='ptq-screen':vp=production_models.model_v_ptq_screen(args[0],p['eps'],p['q'],*args[1:])
     elif m=='mond-screen':vp=production_models.model_v_mond_screen(args[0],p['a0'],p['q'],*args[1:])
     else:vp=production_models.model_v_nfw1p(args[0],p['logM'][gi],*args[1:])
     assert np.max(abs(vp-velocity(g,p,gi,g.D*s)))<tol['mean_comparison_abs']
     if m!='nfw1p':assert np.max(abs(velocity(g,p,gi,g.D*s)-v0*np.sqrt(s)))<1e-9
    for rel in ct['FD_relative_steps']:
     hd=g.D*rel;hi=max(g.inc,.5)*rel
     dfd=(mean(g,p,gi,g.D+hd)-mean(g,p,gi,g.D-hd))/(2*hd)
     ifd=(mean(g,p,gi,inc=g.inc+hi)-mean(g,p,gi,inc=g.inc-hi))/(2*hi)
     for axis,found,expected in [('D',dfd,jd),('i_rad',ifd,ji)]:
      absmax=float(np.max(abs(found-expected)));relerr=float(np.linalg.norm(found-expected)/max(np.linalg.norm(expected),1e-12));passed=absmax<=tol['finest_J_absolute'] or relerr<=tol['finest_J_relative_L2']
      fd.append(dict(galaxy=name,model=m,axis=axis,step=hd if axis=='D' else hi,relative_step=rel,absolute_max=absmax,relative_norm=relerr,pass_numeric=passed))
      if rel==ct['FD_relative_steps'][-1]:assert passed,(name,m,axis,absmax,relerr)
  return dict(rows=len(fd),coarse_threshold_failures=sum(not x['pass_numeric'] for x in fd),finest_threshold_failures=sum(not x['pass_numeric'] for x in fd if x['relative_step']==1e-5))
 check('analytic_distance_inclination_FD_and_independent_laws',all_fd)
 def units():
  g=sample[names.index(ct['local_real_cases'][0])];p=ps['mond'];gi=names.index(g.name);jd,ji=derivatives(g,p,gi)
  # D in kpc rather than Mpc gives derivative /1000 and error *1000, identical C.
  np.testing.assert_allclose(np.outer(jd/1000,jd/1000)*(g.sD*1000)**2,np.outer(jd,jd)*g.sD**2,rtol=1e-14)
  jdegrees=ji*np.pi/180;np.testing.assert_allclose(np.outer(jdegrees,jdegrees)*np.rad2deg(g.si)**2,np.outer(ji,ji)*g.si**2,rtol=1e-14)
  return dict(distance_Mpc_to_kpc=True,angle_rad_to_degree=True)
 check('distance_and_angle_units',units)
 def normal():
  maxerr=0.;min_eigen=np.inf
  for name in ct['local_real_cases']:
   gi=names.index(name);g=sample[gi]
   for m,p in ps.items():
    c,v,C=evaluate(g,p,gi);sg,ld=np.linalg.slogdet(C);assert sg>0
    independent=-.5*((g.y-v)@np.linalg.solve(C,g.y-v)+ld+len(v)*np.log(2*np.pi));maxerr=max(maxerr,abs(independent-c['logL']));assert maxerr<tol['logL_abs'];min_eigen=min(min_eigen,float(np.linalg.eigvalsh(C).min()))
  integral,err=quad(lambda y:np.exp(gaussian(np.array([y]),np.array([2.]),np.array([[9.]]))['logL']),-np.inf,np.inf,epsabs=1e-11)
  assert abs(integral-1)<1e-10
  return dict(max_independent_logL_error=maxerr,min_eigenvalue=min_eigen,one_dimensional_density_integral=integral)
 check('SPD_full_Gaussian_normalization_and_independent_solve',normal)
 def parameter_cov():
  g=sample[names.index('UGC03205')];gi=names.index(g.name);p=ps['mond-screen'];C0=covariance(g,p,gi);changes={}
  for key in ['a0','q','U','sigma']:
   q={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in p.items()}
   if key=='U':q[key][gi]+=.01
   else:q[key]*=1.01
   C=covariance(g,q,gi);changes[key]=float(np.linalg.norm(C-C0)/np.linalg.norm(C0));assert changes[key]>0
  return changes
 check('parameter_dependent_reference_C_recalculation',parameter_cov)
 def identity():
  maximum=0.
  for source,target in [('mond','ptq-nu'),('ptq-nu','mond'),('mond-screen','ptq-screen'),('ptq-screen','mond-screen')]:
   p=ps[source];q=matched_parameters(p,target)
   for name in ct['local_real_cases']:
    gi=names.index(name);g=sample[gi];a,_,C=evaluate(g,p,gi);b,_,Ct=evaluate(g,q,gi);np.testing.assert_allclose(C,Ct,rtol=1e-10,atol=1e-10);maximum=max(maximum,abs(a['logL']-b['logL']));assert maximum<tol['identity_logL_abs']
  return dict(max_delta=maximum,matched_galaxy_checks=12)
 check('matched_PTQ_MOND_corrected_identity',identity)
 def nfw():
  g=Galaxy('synthetic_NFW',np.array([.2,1.,10.,40.]),np.full(4,100.),np.full(4,2.),np.array([20.,40.,60.,30.]),np.zeros(4),np.ones(4)*10,10.,1.,np.pi/3,.05);p=dict(model='nfw1p',U=np.array([.5]),logM=np.array([11.]),sigma=5.);jd=derivatives(g,p,0)[0];h=1e-4
  fd=(velocity(g,p,0,g.D+h)-velocity(g,p,0,g.D-h))/(2*h);error=float(np.linalg.norm(fd-jd)/np.linalg.norm(jd));assert error<1e-6
  sqrt_error=float(np.linalg.norm(jd-velocity(g,p,0)/(2*g.D))/np.linalg.norm(jd));assert sqrt_error>1e-3
  return dict(NFW_analytic_FD_rel_error=error,forcing_sqrt_D_rel_error=sqrt_error)
 check('restricted_NFW_radius_derivative',nfw)
 def coordinate():
  # One fixed y0 point, verify transformed density includes |a| determinant.
  y0=130.;mu_intrinsic=120.;i0=.7;i=.8;a=np.sin(i0)/np.sin(i);b=1/a;e0=4.;sig=6.
  fixed=gaussian(np.array([y0]),np.array([b*mu_intrinsic]),np.array([[e0*e0+sig*sig]]))['logL']
  transformed=gaussian(np.array([a*y0]),np.array([mu_intrinsic]),np.array([[a*a*(e0*e0+sig*sig)]]))['logL']+np.log(a)
  assert abs(fixed-transformed)<1e-10
  return dict(coordinate_log_density_delta=fixed-transformed)
 check('nominal_measure_change_of_variable_Jacobian',coordinate)
 check('source_and_input_immutability_after',preservation)
 js(out/'tests.json',records);pd.DataFrame(fd).to_csv(out/'finite_difference.csv',index=False)
 passed=all(x['status']=='PASS' for x in records if x['hard']);js(out/'LOCAL_CORRECTNESS_GATE.json',dict(status='PASS' if passed else 'FAIL',full_sample_authorized_by_tests=passed,tests=len(records),failed=sum(x['status']=='FAIL' for x in records),spec_sha256=sha(B/'STATISTICAL_SPECIFICATION.json'),contract_sha256=sha(B/'PREDECLARED_TEST_CONTRACT.json'),source_sha256=sha(B/'reference.py')))
 js(out/'environment.json',dict(python=sys.version,platform=platform.platform(),executable=sys.executable));(out/'environment_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
 print('LOCAL_GATE',passed,flush=True)
 if not passed:raise SystemExit(1)
if __name__=='__main__':main()
