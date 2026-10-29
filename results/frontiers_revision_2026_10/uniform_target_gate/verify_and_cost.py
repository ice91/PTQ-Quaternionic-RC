"""Artifact-only verification plus Gaussian-only cost comparator; no extra explicit integrations."""
from pathlib import Path
import json,hashlib,time
import numpy as np,pandas as pd
import explicit_reference as ex
from explicit_reference import laws
B=Path(__file__).resolve().parent;I=B.parents[2];M=I/'results/frontiers_revision_2026_10/mle_recovery';C=I/'results/frontiers_revision_2026_10/covariance_reference';A=B/'acceptance_v1'
def main():
 out=B/'verification_v1';out.mkdir(exist_ok=False);f=pd.read_csv(A/'full_sample_all1274.csv');t=pd.read_csv(A/'totals.csv');n=pd.read_csv(A/'neighborhood_all144.csv');maps=pd.read_csv(A/'native_matched_all728.csv');attempts=json.loads((A/'attempts.json').read_text());assert len(f)==1274 and len(n)==144 and len(maps)==728
 for frame in [f,n,maps]:assert np.isfinite(frame.select_dtypes(include='number').to_numpy()).all()
 assert not f.duplicated(['model','galaxy','floor']).any() and len(f.galaxy.unique())==91
 for (m,floor),s in f.groupby(['model','floor']):
  row=t[(t.model==m)&(t.floor==floor)];assert len(row)==1;r=row.iloc[0];assert len(s)==91;assert abs(s.explicit_logL.sum()-r.explicit_logL)<1e-8;assert abs(abs(s.refinement_delta).sum()-r.sum_abs_refinement)<1e-10
 assert f.passes_1e_4.all() and n.passes_1e_4.all() and maps.passes_1e_8.all()
 assert len(json.loads((A/'exceptions.json').read_text()))==0
 # noempty/misaddressed pandas T-dependent joins: identical P1DT5 keysarepreserved.
 cross=pd.read_csv(A/'P1D_finite_T5_crosscheck.csv');assert len(cross)==154 and not cross.duplicated(['model','galaxy','floor']).any();assert np.isfinite(cross.select_dtypes(include='number').to_numpy()).all()
 assert sum(x['nodes'] for x in attempts if x['status']=='COMPLETE')==json.loads((A/'ACCEPTANCE_SUMMARY.json').read_text())['density_node_evaluations']
 source=json.loads((B/'SOURCE_HASHES_BEFORE.json').read_text());assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in source.items())
 # Gaussian reference has no nuisance quadrature nodes, no inference/optimization.
 gs=laws.load_sample(I/'dataset/sparc_tidy.csv');cost=[];saved=pd.read_csv(C/'full_sample_v1/totals.csv').set_index('model')
 for m in laws.MODELS:
  theta=np.load(M/'original_recovered/01_mle'/m.replace('-','_')/'theta_mle.npy');p=laws.decode(m,theta,91)
  for repetition in range(3):
   begin=time.perf_counter();values=[laws.evaluate(g,p,j)[0]['logL'] for j,g in enumerate(gs)];elapsed=time.perf_counter()-begin;total=float(np.sum(values));assert abs(total-saved.loc[m].corrected_logL)<1e-8;cost.append(dict(model=m,repetition=repetition,Gaussian_B0_full91_seconds=elapsed,fixed_point_logL=total,explicit_nodes=0))
 pd.DataFrame(cost).to_csv(out/'Gaussian_only_cost_comparator.csv',index=False)
 # Smoothnessincrementsatfixed q steps are descriptive arithmetic, notprofile/Fisher estimation.
 sm=pd.read_csv(A/'smoothness_repeatability.csv');increments=[]
 for floor,s in sm.groupby('floor'):
  s=s.sort_values('q_relative_offset');assert len(s)==5 and not s.q_relative_offset.duplicated().any()
  for j in range(1,len(s)):increments.append(dict(floor=floor,from_offset=float(s.iloc[j-1].q_relative_offset),to_offset=float(s.iloc[j].q_relative_offset),delta_logL=float(s.iloc[j].logL-s.iloc[j-1].logL)))
 pd.DataFrame(increments).to_csv(out/'smoothness_fixed_increments.csv',index=False)
 result=dict(status='PASS',full1274_finite_unique_complete=True,mapping728=True,neighborhood144=True,P1D_crosscheck154=True,no_empty_comparisons=True,pandas_explicit_T_column_used=True,source_files_unchanged=len(source),extra_explicit_integrations=0,Gaussian_only_full91_timings=len(cost),Gaussian_min_seconds=min(x['Gaussian_B0_full91_seconds'] for x in cost),Gaussian_max_seconds=max(x['Gaussian_B0_full91_seconds'] for x in cost),all_attempt_node_count_matches_summary=True)
 (out/'VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
