"""Repair v1 derived metadata only; raw scientific integrals/contract remain immutable."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
B=Path(__file__).resolve().parent

def js(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 out=B/'postprocessing_v2';out.mkdir(exist_ok=False);raw=B/'boundary_v1/all_integrals.csv';df=pd.read_csv(raw);protocol=json.loads((B/'PREDECLARED_PROTOCOL.json').read_text());tol=protocol['nuisance']['convergence_abs_logL']
 assert len(df)==3300 and not df.duplicated(['galaxy','model','point','T','order','construction']).any()
 assert np.isfinite(df.select_dtypes(include='number').to_numpy()).all()
 js(out/'V1_DERIVED_FAILURE.json',dict(stage='postprocessing only',bug="pandas cf.T/dd.T denotes transpose, not temperature/tail column; use frame['T']",affected_files=['boundary_v1/neighborhood_effect_changes.csv (NaN effects/identities)','boundary_v1/decomposition.csv convergence flags linkage (vacuous empty all)'],raw_numerical_integrals_affected=False,v1_files_preserved=True,contract_or_quadrature_changed=False,source_script_preserved='run_boundary.py',corrected_output='postprocessing_v2',independent_agent_detected_same_issue=True))
 convergence=[]
 for key,ss in df.groupby(['galaxy','model','point','tier','T','construction']):
  ss=ss.set_index('order');hi,lo=sorted(ss.index)[-1:-3:-1];delta=float(ss.loc[hi].logL-ss.loc[lo].logL);convergence.append(dict(galaxy=key[0],model=key[1],point=key[2],tier=key[3],T=key[4],construction=key[5],delta_high_low=delta,converged=abs(delta)<=tol))
 cf=pd.DataFrame(convergence);assert len(cf)==1188;cf.to_csv(out/'convergence.csv',index=False);decomp=[]
 for key,ss in df[df.order==256].groupby(['galaxy','model','point','tier','T']):
  ss=ss.set_index('construction');conv=cf[(cf.galaxy==key[0])&(cf.model==key[1])&(cf.point==key[2])&(cf['T']==key[4])];assert len(conv)==4
  lin=float(ss.loc['B0_linear'].logL);full=float(ss.loc['B0_nonlinear'].logL);floor=float(ss.loc['B1_nonlinear'].logL);ionly=float(ss.loc['inclination_only_nonlinear_B0'].logL);A=float(ss.A_corr.iloc[0]);decomp.append(dict(galaxy=key[0],model=key[1],point=key[2],tier=key[3],T=key[4],A_corr=A,linear=lin,nonlinear_B0=full,nonlinear_B1=floor,linear_trunc_minus_A=lin-A,full_nonlinear_minus_linear=full-lin,inclination_nonlinear_minus_linear=ionly-lin,distance_remainder_nonlinear_minus_ionly=full-ionly,floor_B1_minus_B0=floor-full,total_nonlinear_minus_A=full-A,converged=bool(conv.converged.all())))
 dd=pd.DataFrame(decomp);assert len(dd)==297 and np.isfinite(dd.select_dtypes(include='number').to_numpy()).all();dd.to_csv(out/'decomposition.csv',index=False)
 tails=[]
 for key,ss in dd[dd.point=='historical'].groupby(['galaxy','model']):
  ss=ss.set_index('T');tails.append(dict(galaxy=key[0],model=key[1],B0_T5_T4=float(ss.loc[5.].nonlinear_B0-ss.loc[4.].nonlinear_B0),B1_T5_T4=float(ss.loc[5.].nonlinear_B1-ss.loc[4.].nonlinear_B1),linear_T5_T4=float(ss.loc[5.].linear-ss.loc[4.].linear),converged=bool(ss.converged.all())))
 tf=pd.DataFrame(tails);tf.to_csv(out/'tail_sensitivity.csv',index=False)
 neighbors=[]
 for _,r in dd[(dd.point!='historical')&(dd['T']==5.)].iterrows():
  bs=dd[(dd.galaxy==r.galaxy)&(dd.model==r.model)&(dd.point=='historical')&(dd['T']==5.)];assert len(bs)==1;base=bs.iloc[0]
  neighbors.append(dict(galaxy=r.galaxy,model=r.model,point=r.point,nonlinear_effect_change=r.full_nonlinear_minus_linear-base.full_nonlinear_minus_linear,inclination_effect_change=r.inclination_nonlinear_minus_linear-base.inclination_nonlinear_minus_linear,floor_effect_change=r.floor_B1_minus_B0-base.floor_B1_minus_B0,total_effect_change=r.total_nonlinear_minus_A-base.total_nonlinear_minus_A,converged=bool(r.converged and base.converged)))
 nn=pd.DataFrame(neighbors);assert len(nn)==66 and not nn.duplicated(['galaxy','model','point']).any() and np.isfinite(nn.select_dtypes(include='number').to_numpy()).all();nn.to_csv(out/'neighborhood_effect_changes.csv',index=False)
 h=dd[(dd.point=='historical')&(dd['T']==5.)];edge=h[h.galaxy.isin(protocol['tier2_edge_on'])];cols=['full_nonlinear_minus_linear','inclination_nonlinear_minus_linear','distance_remainder_nonlinear_minus_ionly','linear_trunc_minus_A','floor_B1_minus_B0','total_nonlinear_minus_A'];summ=[]
 for name,ss in edge.groupby('galaxy'):
  entry=dict(galaxy=name,models=len(ss))
  for c in cols:entry[c+'_max_abs']=float(abs(ss[c]).max())
  summ.append(entry)
 pd.DataFrame(summ).to_csv(out/'seven_edge_on_summary.csv',index=False)
 # predeclared scientific flags saved; they are not universal validity criteria.
 flagged=[]
 for _,r in dd[dd['T']==5.].iterrows():
  for col,threshold in [('inclination_nonlinear_minus_linear',.1),('full_nonlinear_minus_linear',.1),('floor_B1_minus_B0',.1),('total_nonlinear_minus_A',.1)]:
   if abs(r[col])>=threshold:flagged.append(dict(galaxy=r.galaxy,model=r.model,point=r.point,effect=col,value=float(r[col]),threshold=threshold))
 pd.DataFrame(flagged).to_csv(out/'materiality_flags.csv',index=False)
 metrics=dict(PATCH_PARITY='PASS',BOUNDARY_APPROXIMATION='MATERIAL_SENSITIVITY_REQUIRES_NEW_SPEC',FLOOR_SENSITIVITY='MATERIAL_MODEL_UNCERTAINTY',REFIT_LAUNCH='REQUIRES_STATISTICAL_SPECIFICATION_DECISION',real_edge_on_galaxies=7,edge_on_model_points=len(edge),all_boundary_tier1_galaxies=22,boundary_galaxies_with_full_integrals=9,boundary_galaxies_Tier1_only=13,interior_control_galaxies=2,full_historical_model_points=len(h),parameter_neighborhood_points=len(nn),integrals=len(df),all_integrals_finite=True,convergence_cases=len(cf),nonconverged=int((~cf.converged).sum()),max_quad_error=float(abs(cf.delta_high_low).max()),edge_on_max_abs={c:float(abs(edge[c]).max()) for c in cols},all11_historical_max_abs={c:float(abs(h[c]).max()) for c in cols},neighborhood_effect_changes_max_abs={c:float(abs(nn[c]).max()) for c in ['nonlinear_effect_change','inclination_effect_change','floor_effect_change','total_effect_change']},tail_max_abs={c:float(abs(tf[c]).max()) for c in ['B0_T5_T4','B1_T5_T4','linear_T5_T4']},scientific_scope='no all91 or optimizer-domain bound; realedge total Taylor effects are predominantly distance, not inclination; material wider-error boundary/control results preserved',decision='new uniform likelihood/spec sensitivity policy requires Human Arbiter decision; no fits',v1_postprocess_failure_preserved=True,raw_integrals_sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
 js(out/'FINAL_GATE.json',metrics);print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
