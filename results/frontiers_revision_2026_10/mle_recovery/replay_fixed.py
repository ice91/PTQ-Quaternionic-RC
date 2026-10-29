"""Read-only fixed-vector production replay. No optimizer, posterior or sampler calls.
Run with an explicit NEW --out directory; refuses existing output.
"""
from pathlib import Path
import sys,json,argparse,hashlib,platform,subprocess,csv
import numpy as np
B=Path(__file__).resolve().parent
I=B.parents[2]
sys.path.insert(0,str(I/'src'))
from ptquat import mle,likelihood
from ptquat.data import load_tidy_sparc
from ptquat.fit_global import _layout,_unpack
from ptquat.constants import H0_SI

def write_csv(p,rows):
 with p.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,required=True); args=ap.parse_args()
 provenance=json.loads((B/'SOURCE_MANIFEST.json').read_text())
 for record in provenance['artifacts']:
  source=Path(record['absolute_source_path'])
  recovered=Path(record['recovered_path'])
  # Both preserved bytes and current production dependencies must match authority.
  for path in [source,recovered]:
   if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
    raise RuntimeError('Source drift: '+str(path))
 out=args.out; out.mkdir(parents=True,exist_ok=False)
 # Predeclared BEFORE evaluation; archived JSON has ~15 significant digits.
 # 1e-8 absolute logL allows accumulation/BLAS rounding at |logL|~1e4,
 # is far below displayed 6-decimal logL (0.5e-6) and 3-decimal IC rounding.
 tol={'archive_logL_abs':1e-8,'internal_sum_abs':1e-9,'matched_identity_abs':1e-8,'archive_precision':'full float JSON/CSV + binary float64 theta; manifest logL 6 decimals; printed IC 3 decimals','boundary_relative_span':1e-4}
 (out/'tolerances_predeclared.json').write_text(json.dumps(tol,indent=2))
 gd=load_tidy_sparc(str(I/'dataset/sparc_tidy.csv')); gs=[gd[k] for k in sorted(gd)]
 models=['baryon','mond','ptq','ptq-nu','ptq-screen','mond-screen','nfw1p']
 results={}; vectors={}; params=[]; gals=[]; curves=[]; matrices={}
 original=likelihood.gaussian_loglike
 for m in models:
  src=B/'original_recovered/01_mle'/m.replace('-','_'); th=np.load(src/'theta_mle.npy',allow_pickle=False); vectors[m]=th
  summary=json.loads((src/'mle_summary.json').read_text()); L=_layout(m,len(gs),True,None); p=_unpack(th,L); bounds=mle.theta_bounds(m,len(gs),H0_SI,domain=summary['domain'])
  assert th.shape==(L['k'],)
  direct=mle.evaluate_loglike(th,gs,m,H0_SI)
  for idx,v in enumerate(th):
   name=next((k for k,x in L.items() if k.startswith('i_') and x==idx),None)
   gi=None
   for sl in ['sl_U','sl_lM']:
    if sl in L and L[sl].start<=idx<L[sl].stop: name=sl; gi=idx-L[sl].start
   lo,hi=bounds[idx]
   params.append(dict(model=m,index=idx,coordinate=name,galaxy=gs[gi].name if gi is not None else '',value=float(v),lower=lo,upper=hi,near_boundary=min(abs(v-lo),abs(v-hi))<=1e-4*(hi-lo)))
  for gi,g in enumerate(gs):
   capture={}
   def instrument(y,vm,C):
    # Independently decompose actual production covariance; require SPD without jitter.
    Cr=0.5*(C+C.T); chol=np.linalg.cholesky(Cr); r=y-vm
    quad=float(r@np.linalg.solve(Cr,r)); ld=float(np.linalg.slogdet(Cr)[1]); norm=float(len(r)*np.log(2*np.pi)); ll=float(original(y,vm,C))
    capture.update(quad=quad,logdet=ld,normalization=norm,logL=ll,decomposition_logL=-0.5*(quad+ld+norm),jitter_required=False)
    matrices[m+'__'+g.name]=Cr
    for j in range(len(r)):
     curves.append(dict(model=m,galaxy=g.name,point=j,r_kpc=float(g.r_kpc[j]),v_obs=float(y[j]),v_model=float(vm[j]),residual=float(r[j]),v_disk_template=float(g.v_disk[j]),v_bulge_template=float(g.v_bulge[j]),v_gas_template=float(g.v_gas[j])))
    return ll
   likelihood.gaussian_loglike=instrument
   try: val=mle.galaxy_loglike(th,gs,gi,m,H0_SI)
   finally: likelihood.gaussian_loglike=original
   assert abs(val-capture['decomposition_logL'])<tol['internal_sum_abs']
   gals.append(dict(model=m,galaxy=g.name,**capture))
  group=[x for x in gals if x['model']==m]; total=sum(x['logL'] for x in group)
  boundary=mle.boundary_status(th,bounds)
  results[m]=dict(archived_logL=summary['max_loglike'],production_logL=direct,sum_galaxy_logL=total,delta_logL=direct-summary['max_loglike'],quad=sum(x['quad'] for x in group),logdet=sum(x['logdet'] for x in group),normalization=sum(x['normalization'] for x in group),k=L['k'],archived_k=summary['k'],N_gal=len(gs),N_points=sum(len(g.r_kpc) for g in gs),bounds_satisfied=all(lo<=v<=hi for v,(lo,hi) in zip(th,bounds)),boundary=boundary,boundary_matches_archive=boundary==summary['boundary'],archive_replay_pass=abs(direct-summary['max_loglike'])<=tol['archive_logL_abs'],sum_pass=abs(total-direct)<=tol['internal_sum_abs'],AIC_replayed=-2*direct+2*L['k'],BIC_NPOINTS_replayed=-2*direct+L['k']*np.log(1782),globals={k:float(v) for k,v in p.items() if k not in ['U','lM','model']})
 identities=[]
 for a,b,fn in [('mond','ptq-nu',mle.map_theta_mond_to_ptqnu),('ptq-nu','mond',mle.map_theta_ptqnu_to_mond),('mond-screen','ptq-screen',mle.map_theta_mondscreen_to_ptqscreen),('ptq-screen','mond-screen',mle.map_theta_ptqscreen_to_mondscreen)]:
  mapped=fn(vectors[a],len(gs),H0_SI); ll=mle.evaluate_loglike(mapped,gs,b,H0_SI); delta=ll-results[a]['production_logL']
  identities.append(dict(source=a,target=b,source_logL=results[a]['production_logL'],mapped_logL=ll,delta=delta,pass_tolerance=abs(delta)<=tol['matched_identity_abs']))
 write_csv(out/'galaxy_likelihood_components.csv',gals); write_csv(out/'decoded_parameters.csv',params); write_csv(out/'rotation_curve_points.csv',curves)
 np.savez_compressed(out/'within_galaxy_covariances.npz',**matrices)
 result=dict(status='PASS' if all(x['archive_replay_pass'] and x['sum_pass'] and x['bounds_satisfied'] and x['boundary_matches_archive'] and x['k']==x['archived_k'] for x in results.values()) and all(x['pass_tolerance'] for x in identities) else 'DISCREPANCY',tolerances=tol,models=results,matched_identities=identities,scope='fixed objective consistency only; not validation of residual Jacobian, optimizer globality, statistical validity or nuisance approximation',environment=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,executable=sys.executable),implementation_root=str(I),H0_si=H0_SI,c0=10.0,c_slope=-0.1,priors_used=False,optimization_performed=False)
 (out/'REPLAY_VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n')
 (out/'environment_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
 print(json.dumps({'status':result['status'],'delta_logL':{m:r['delta_logL'] for m,r in results.items()},'identities':identities},indent=2))
if __name__=='__main__': main()
