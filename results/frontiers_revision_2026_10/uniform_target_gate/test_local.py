"""Gate C: mathematical correctness before full sample; numerical convergence is separate."""
from pathlib import Path
import json,hashlib,sys,traceback,platform,subprocess
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
import explicit_reference as ex
from explicit_reference import laws
B=Path(__file__).resolve().parent;I=B.parents[2];M=I/'results/frontiers_revision_2026_10/mle_recovery'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def js(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 out=B/'local_v1';out.mkdir(exist_ok=False);gs=laws.load_sample(I/'dataset/sparc_tidy.csv');names=[g.name for g in gs];ct=json.loads((B/'PREDECLARED_PROTOCOL.json').read_text());tests=[]
 def check(name,fn):
  try:r=fn();tests.append(dict(test=name,status='PASS',detail=r));print('PASS',name,flush=True)
  except Exception as e:tests.append(dict(test=name,status='FAIL',error=repr(e),traceback=traceback.format_exc()));print('FAIL',name,repr(e),flush=True)
 def normalization():
  maxerror=0.;extreme=0
  for name in ct['pilot_galaxies']:
   g=gs[names.index(name)]
   for method,n in [('quantile',64),('segmented_quantile',24)]:
    ds,wd,is_,wi,ZD,Zi=ex.nuisance_nodes(g,method,n);maxerror=max(maxerror,abs(wd.sum()*wi.sum()-1));assert np.all(ds>0) and np.all(is_>0) and np.all(is_<np.pi/2);assert maxerror<1e-10
   for center,sigma,lo,hi in [(g.D,g.sD,0,np.inf),(g.inc,g.si,0,np.pi/2)]:
    a=(lo-center)/sigma;b=(hi-center)/sigma;u=np.array([1e-14,1e-12,.5,1-1e-12,1-1e-14]);z=ex.quantile_standard(a,b,u);assert np.isfinite(z).all();assert np.all(np.diff(z)>0);extreme+=len(u)
  return dict(max_constant_integral_error=maxerror,extreme_quantile_checks=extreme)
 check('normalized_physical_prior_constant_endpoints',normalization)
 def zero():
  g=laws.Galaxy('zero',np.array([1.,2.]),np.array([80.,82.]),np.array([2.,2.]),np.array([80.,80.]),np.zeros(2),np.zeros(2),10.,0.,np.pi/3,0.);p=dict(model='baryon',U=np.array([1.]),sigma=3.);r=ex.integrate(g,p,0,'quantile',64);mu=laws.velocity(g,p,0);expected=ex.conditional_logdensity(g,mu,3.,1.,'B0');assert abs(r['logL']['B0']-expected)<1e-8 and abs(r['logL']['B1']-expected)<1e-8;return dict(logL_error=r['logL']['B0']-expected)
 check('zero_uncertainty_conditional_limit',zero)
 def full_density():
  g=laws.Galaxy('one',np.array([1.]),np.array([1.]),np.array([2.]),np.array([1.]),np.zeros(1),np.zeros(1),10.,1.,np.pi/3,.1)
  p=dict(model='baryon',U=np.ones(1),sigma=3.)
  integral,err=quad(lambda yy:np.exp(-.5*((yy-1.)**2/13+np.log(2*np.pi*13))),-np.inf,np.inf);assert abs(integral-1)<1e-10
  b=.8;mu=np.array([1.]);B0=ex.conditional_logdensity(g,mu,3.,b,'B0');B1=ex.conditional_logdensity(g,mu,3.,b,'B1');expected=-.5*((g.y-mu)**2/(4+(b*3)**2)+np.log(2*np.pi*(4+(b*3)**2))).sum();assert abs(B1-expected)<1e-12 and B1!=B0
  a=1/b;fixed=ex.conditional_logdensity(g,mu,3.,b,'B0');trans=-.5*(((a*g.y-a*mu)**2/(a*a*13))+np.log(2*np.pi*a*a*13)).sum()+len(g.y)*np.log(a);assert abs(fixed-trans)<1e-10
  # Compare fast sufficient-sum integration with direct3D conditional density on small grid.
  rr=ex.integrate(g,p,0,'quantile',16);ds,wd,incs,wi,_,_=ex.nuisance_nodes(g,'quantile',16);v=laws.velocity(g,p,0,ds[:,None]);bs=np.sin(incs)/np.sin(g.inc)
  for floor in ['B0','B1']:
   vals=np.array([[ex.conditional_logdensity(g,bs[j]*v[k],3.,bs[j],floor) for j in range(len(incs))] for k in range(len(ds))]);val=float(ex.logsumexp(vals+np.log(wd[:,None])+np.log(wi)[None,:]));assert abs(val-rr['logL'][floor])<1e-10
  return dict(conditional_density_integral=integral,fixed_measure=True,fast_vs_direct_grid=True,no_propagated_covariance=True)
 check('normalized_Gaussian_floor_measure_no_double_count',full_density)
 def laws_test():
  gi=names.index('F574-1');g=gs[gi];errs=[]
  for m in ct['models']:
   t=np.load(M/'original_recovered/01_mle'/m.replace('-','_')/'theta_mle.npy');p=laws.decode(m,t,91);v=laws.velocity(g,p,gi);jd,ji=laws.derivatives(g,p,gi);s=1.2
   if m!='nfw1p':np.testing.assert_allclose(laws.velocity(g,p,gi,g.D*s),v*np.sqrt(s),rtol=1e-12)
   hd=g.D*1e-5;fd=(laws.velocity(g,p,gi,g.D+hd)-laws.velocity(g,p,gi,g.D-hd))/(2*hd);err=np.linalg.norm(fd-jd)/np.linalg.norm(jd);assert err<1e-6;errs.append(float(err))
   hi=1e-5;fi=(laws.mean(g,p,gi,inc=g.inc+hi)-laws.mean(g,p,gi,inc=g.inc-hi))/(2*hi);assert np.linalg.norm(fi-ji)/np.linalg.norm(ji)<1e-6
   np.testing.assert_allclose((ji*np.pi/180)*np.rad2deg(g.si),ji*g.si,rtol=1e-14)
  return dict(max_D_error=max(errs),NFW_not_forced_sqrt=True)
 check('full_model_distance_NFW_inclination_units',laws_test)
 def mapped_repeat():
  g=gs[names.index('IC4202')];gi=names.index(g.name);maximum=0.
  for a,b in [('mond','ptq-nu'),('ptq-nu','mond'),('mond-screen','ptq-screen'),('ptq-screen','mond-screen')]:
   t=np.load(M/'original_recovered/01_mle'/a.replace('-','_')/'theta_mle.npy');tt=t.copy();tt[0]=10**t[0]/(laws.C_LIGHT*laws.H0) if b.startswith('ptq') else np.log10(t[0]*laws.C_LIGHT*laws.H0);p=laws.decode(a,t,91);q=laws.decode(b,tt,91);r=ex.integrate(g,p,gi,'quantile',64);s=ex.integrate(g,q,gi,'quantile',64);repeat=ex.integrate(g,p,gi,'quantile',64)
   for f in ['B0','B1']:maximum=max(maximum,abs(r['logL'][f]-s['logL'][f]));assert abs(r['logL'][f]-repeat['logL'][f])<=1e-12
  assert maximum<=1e-8;return dict(mapped_max_error=maximum,repeatability=True)
 check('native_matched_identity_repeatability',mapped_repeat)
 def fail_closed():
  for values in [np.array([0.]),np.array([1.]),np.array([np.nan])]:
   try:ex.quantile_standard(-5.,np.inf,values)
   except (ValueError,FloatingPointError):pass
   else:raise AssertionError('badquantiles accepted')
  g=laws.Galaxy('bad',np.array([1.]),np.array([1.]),np.array([0.]),np.array([1.]),np.zeros(1),np.zeros(1),10.,0.,1.,0.)
  try:ex.conditional_logdensity(g,np.array([1.]),0.,1.,'B0')
  except ValueError:pass
  else:raise AssertionError('nonpositivevariance accepted')
  return dict(invalid_endpoints_rejected=True,nonpositivevariance_rejected=True)
 check('fail_closed_nonfinite_and_variance',fail_closed)
 def immutable():
  before=json.loads((B/'SOURCE_HASHES_BEFORE.json').read_text());bad=[p for p,h in before.items() if sha(Path(p))!=h];assert not bad,bad;return dict(files=len(before),changed=0)
 check('source_data_immutability',immutable)
 passed=all(x['status']=='PASS' for x in tests);js(out/'tests.json',tests);js(out/'LOCAL_GATE.json',dict(status='PASS' if passed else 'FAIL',test_groups=len(tests),failed=sum(x['status']=='FAIL' for x in tests),protocol_sha256=sha(B/'PREDECLARED_PROTOCOL.json'),code_sha256=sha(B/'explicit_reference.py')));js(out/'environment.json',dict(python=sys.version,platform=platform.platform(),executable=sys.executable));(out/'environment_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
 if not passed:raise SystemExit(1)
if __name__=='__main__':main()
