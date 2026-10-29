"""Post-run finite/normalization/integrity verification on preserved full-sample artifacts."""
from pathlib import Path
import json,hashlib,sys
import numpy as np,pandas as pd
from reference import *
B=Path(__file__).resolve().parent;I=B.parents[2]
def main():
 out=B/'output_validation.json'
 if out.exists():raise FileExistsError(out)
 data=pd.read_csv(B/'full_sample_v1/per_galaxy.csv');assert len(data)==637
 numeric=data.select_dtypes(include='number');assert np.isfinite(numeric.to_numpy()).all()
 sample=load_sample(I/'dataset/sparc_tidy.csv');lookup={g.name:g for g in sample};Cfile=np.load(B/'full_sample_v1/corrected_covariances.npz');maxld=0.;maxq=0.;maxll=0.
 # Use stored C and independently normalized residual squared form, finite before maxima.
 for _,x in data.iterrows():
  C=Cfile[x.model+'__'+x.galaxy];sign,ld=np.linalg.slogdet(C);assert sign==1 and np.isfinite(ld)
  assert np.isfinite(C).all();np.linalg.cholesky(C)
  # From diagonal+rank covariance output one cannot recover sign of residual;
  # reload unchanged native theta and compute independent reference mean once.
  p=Path(I/'results/frontiers_revision_2026_10/mle_recovery/original_recovered/01_mle')/x.model.replace('-','_')/'theta_mle.npy';t=np.load(p);gi=[g.name for g in sample].index(x.galaxy);pars=decode(x.model,t,91);g=lookup[x.galaxy];v=velocity(g,pars,gi);r=g.y-v
  quad=float(r@np.linalg.solve(C,r));ll=-.5*(quad+ld+len(r)*np.log(2*np.pi));assert np.isfinite(quad) and np.isfinite(ll)
  maxld=max(maxld,abs(ld-x.corrected_logdet));maxq=max(maxq,abs(quad-x.corrected_quad));maxll=max(maxll,abs(ll-x.corrected_logL))
 assert maxld<1e-8 and maxq<1e-8 and maxll<1e-8
 nq=pd.read_csv(B/'nuisance_v1/integrals.csv');assert len(nq)==585 and np.isfinite(nq.select_dtypes(include='number').to_numpy()).all()
 assert abs(nq.prior_weight_sum-1).max()<1e-10
 gate=json.loads((B/'full_sample_v1/GATE_E.json').read_text());assert gate['status']=='PASS'
 before=json.loads((B/'SOURCE_HASHES_BEFORE.json').read_text());bad=[s for s,h in before.items() if hashlib.sha256(Path(s).read_bytes()).hexdigest()!=h];assert not bad
 result=dict(status='PASS',finite_full_sample_rows=len(data),finite_nuisance_rows=len(nq),stored_matrices_SPD=len(Cfile.files),independent_logdet_max_abs=maxld,independent_quad_max_abs=maxq,independent_logL_max_abs=maxll,existing_source_files_unchanged=len(before),warning_note='NumPy slogdet warnings retained in execution logs; every returned sign/logdet/quad/logL explicitly finite before maxima')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
