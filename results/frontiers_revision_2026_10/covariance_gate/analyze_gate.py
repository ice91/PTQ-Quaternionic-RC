from pathlib import Path
import json,hashlib,csv,sys,ast
import numpy as np,pandas as pd
B=Path(__file__).resolve().parent;I=B.parents[2];V=B/'run_v1';M=I/'results/frontiers_revision_2026_10/mle_recovery';sys.path.insert(0,str(I/'src'))
from ptquat import mle,fit_global as fg
from ptquat.data import load_tidy_sparc

def js(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
f=pd.read_csv(V/'finite_difference_summary.csv');e=pd.read_csv(V/'covariance_effects.csv');u=pd.read_csv(V/'covariance_updates.csv');q=pd.read_csv(V/'explicit_nuisance.csv');ids=pd.read_csv(V/'matched_identities.csv');design=json.loads((B/'PREDECLARED_DESIGN.json').read_text())
# Quadrature convergence is tested on the predeclared final 64/128 pair; all levels retained.
res=[]
for (case,model,construction),g in q.groupby(['case','model','construction'],sort=True):
 vals=g.set_index('order').logL_integrated;d=float(vals.loc[128]-vals.loc[64]);res.append(dict(case=case,model=model,construction=construction,logL16=float(vals.loc[16]),logL32=float(vals.loc[32]),logL64=float(vals.loc[64]),logL128=float(vals.loc[128]),delta32_16=float(vals.loc[32]-vals.loc[16]),delta64_32=float(vals.loc[64]-vals.loc[32]),delta128_64=d,converged_abs_1e_3=abs(d)<=1e-3))
pd.DataFrame(res).to_csv(V/'quadrature_convergence.csv',index=False)
# Numerical checks use existing outputs, independent from result generation.
assert np.isfinite(f.select_dtypes(include='number')).all().all()
assert np.isfinite(e.select_dtypes(include='number')).all().all()
assert np.isfinite(u.select_dtypes(include='number')).all().all()
assert len(json.loads((V/'failures.json').read_text()))==0
finest=f[f.relative_step==1e-5];assert finest.numerical_pass.all()
assert abs(u.independent_minus_production).max()<1e-8
assert abs(u.callback_delta).max()<1e-8
assert u.C_captured_rel_error.max()<1e-9
assert ids.pass_identity.all()
assert len(f)==405 and len(e)==135 and len(u)==168 and len(ids)==108 and len(q)==168
for key,g in u[u.parameter_point!='original'].groupby('parameter_point'):assert (g.C_change_rel>0).all(),key
npz=np.load(V/'covariances.npz',allow_pickle=False)
for k in npz.files:
 C=npz[k];assert np.isfinite(C).all();np.linalg.cholesky(.5*(C+C.T))
# Exact parameter-point provenance without reevaluation/optimization.
gd=load_tidy_sparc(str(I/'dataset/sparc_tidy.csv'));names=sorted(gd);pp=[]
for row in design['selected']:
 gi=names.index(row['galaxy'])
 for m in design['models']:
  src=M/'original_recovered/01_mle'/m.replace('-','_')/'theta_mle.npy';th=np.load(src);L=fg._layout(m,len(names),True,None);points=[('original',th.copy())]
  for key in ['i_eps','i_log10a0']:
   if key in L:
    z=th.copy();z[L[key]]=z[L[key]]*1.01 if key=='i_eps' else z[L[key]]+np.log10(1.01);points.append(('scale',z))
  if 'i_lq' in L:
   z=th.copy();z[L['i_lq']]+=np.log(1.01);points.append(('q',z))
  z=th.copy();ix=L['sl_U'].start+gi;z[ix]+=.01 if z[ix]+.01<=1.5 else -.01;points.append(('upsilon',z))
  z=th.copy();z[L['i_lnsig']]+=np.log(1.01);points.append(('sigma_sys',z))
  for name,z in points:
   changed=np.flatnonzero(z!=th);bounds=mle.theta_bounds(m,len(names));assert all(lo<=v<=hi for v,(lo,hi) in zip(z,bounds))
   pp.append(dict(case=row['galaxy'],model=m,point=name,source_theta=str(src),source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),theta_float64_bytes_sha256=hashlib.sha256(z.astype('<f8').tobytes()).hexdigest(),changed_coordinates=[dict(index=int(ix),original=float(th[ix]),tested=float(z[ix])) for ix in changed],bounds_satisfied=True))
js(V/'tested_parameter_points.json',pp)
# Nominal geometry provenance, additional input-only cross-check.
t=pd.read_csv(I/'dataset/sparc_tidy.csv');raw=pd.read_csv(I/'dataset/raw/vizier_table1.csv');j=t.merge(raw,left_on='galaxy',right_on='Name',validate='many_to_one');checks={}
for a,b in [('D_Mpc','Dist'),('D_err_Mpc','e_Dist'),('i_deg','i'),('i_err_deg','e_i')]:checks[a]=float(abs(j[a]-j[b]).max());assert checks[a]==0
js(B/'NOMINAL_GEOMETRY_LINK.json',checks)
# Source structure gives globals final-polish identical callback body; no optimization invocation.
tree=ast.parse((I/'src/ptquat/mle.py').read_text());outer=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='_block_coordinate_mle');nested={x.name:x for x in ast.walk(outer) if isinstance(x,ast.FunctionDef)}
js(B/'OPTIMIZER_CALL_CHAIN.json',dict(source=str(I/'src/ptquat/mle.py'),source_sha256=hashlib.sha256((I/'src/ptquat/mle.py').read_bytes()).hexdigest(),paths=[dict(entry='_block_coordinate_mle.neg_g / neg_g2',endpoint='neg_full -> objective_neg_loglike -> fit_global.log_likelihood -> build_covariance -> gaussian_loglike',runtime_exercised='source AST neg_g callback at prescribed global input; 91 fresh covariance calls per invocation'),dict(entry='_block_coordinate_mle.neg_u',endpoint='galaxy_loglike -> likelihood.build_covariance -> gaussian_loglike',runtime_exercised='source AST callback at prescribed U; one fresh covariance call'),dict(entry='_block_coordinate_mle.neg_um',endpoint='galaxy_loglike -> likelihood.build_covariance -> gaussian_loglike',runtime_exercised='source AST NFW callback at prescribed U, fixed logM; one fresh covariance call')],final_polish_callback_body_identical=ast.dump(ast.Module(body=nested['neg_g'].body,type_ignores=[]),include_attributes=False)==ast.dump(ast.Module(body=nested['neg_g2'].body,type_ignores=[]),include_attributes=False),limitations='optimizer iterations/scheduling not executed; source call path and exact callback evaluation demonstrated; no claim of global convergence',no_optimizer_driver_called=True))
# Smallest subcomponent of predeclared synthetic control; no new fitted point.
js(V/'MINIMAL_FAILING_CASE.json',dict(parent_case='synthetic_flat',subset='first distance variance element; same declared synthetic control',Upsilon=1.,D0_Mpc=10.,r0_kpc=1.,v_disk0_kms=80.,sigmaD_Mpc=1.,production_dv_dD=0.,full_dv_dD_kms_per_Mpc=4.,full_residual_dD=-4.,omitted_distance_variance_kms2=16.,derivation='r and all templates scale with D; baryon vm=80 sqrt(D/10); production vfun ignores radius when templates frozen',reproduce_command='PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 /tmp/rev26-p1a-env/bin/python run_gate.py',note='run_v1 must not exist; use a separate copied gate directory under authorized output root for a replay; no production edit'))
real=e[(e.label=='historical')&(e.specification=='full_D_fixedframe')];q128=q[q.order==128];q0=q128[q128.construction=='B0_nominal_floor'];q1=q128[q128.construction=='B1_intrinsic_floor']
summary=dict(overall_verdict='COVARIANCE_CORRECTION_REQUIRED_BEFORE_REFIT',verdicts=dict(HISTORICAL_OBJECTIVE_REPLAY=dict(status='PASS',evidence='authenticated P1A fixed-vector objective replay max delta 4.547473508864641e-12; source hashes unchanged; not rerun/discovery'),FULL_RESIDUAL_JACOBIAN=dict(status='FAIL',evidence='missing distance-template amplitude derivative; transformed-observation Ji differs from model-based Ji; fixed-frame mean Ji matches production sign-oriented'),PARAMETER_DEPENDENT_COVARIANCE=dict(status='PASS',evidence='168 prescribed callback evaluations; updated C/logdet, complete normalization, 0 callback endpoint delta; no optimizer driver'),LINEARIZED_NUISANCE_APPROXIMATION=dict(status='FAIL',evidence='frozen covariance fails source-backed distance propagation; B0-A differences up to about 2.83916 logL for a selected galaxy; conditional corrected-D linearization separate limited finding'),MATCHED_MODEL_IDENTITY=dict(status='PASS',evidence='108 matched point comparisons, complete likelihood/covariance identity within predeclared thresholds')),statistics=dict(finest_fd_analytic_max_abs=float(finest.fd_analytic_abs_max.max()),finest_fd_analytic_max_rel_norm=float(finest.fd_analytic_rel_norm.max()),coarse_failed_rows=int((~f.numerical_pass).sum()),finest_cases_pass=int(finest.numerical_pass.sum()),callback_max_abs_delta=float(abs(u.callback_delta).max()),independent_galaxy_max_abs_delta=float(abs(u.independent_minus_production).max()),captured_C_max_rel=float(u.C_captured_rel_error.max()),matched_C_max_rel=float(ids.covariance_relative.max()),matched_logL_max_abs=float(abs(ids.logL_delta).max()),distance_only_real_min_delta_logL=float(real.delta_logL.min()),distance_only_real_max_delta_logL=float(real.delta_logL.max()),distance_only_real_max_C_relative=float(real.covariance_relative_frobenius.max()),quadrature_max_128_64=float(max(abs(x['delta128_64']) for x in res)),quadrature_converged_cases=sum(x['converged_abs_1e_3'] for x in res),quadrature_total_cases=len(res),B0_minus_A_min=float(q0.delta_B_minus_A.min()),B0_minus_A_max=float(q0.delta_B_minus_A.max()),B0_minus_fullD_max_abs=float(abs(q0.delta_B_minus_fullD).max()),B1_minus_fullD_max_abs=float(abs(q1.delta_B_minus_fullD).max()),prior_weight_max_error=float(abs(q.prior_weight_sum-1).max()),matrices_spd=len(npz.files)),optimization_performed=False,original_MLE_tables='retain immutable historical replay artifacts only; not validated estimates for corrected statistical specification',production_correction='necessary before using full-distance-propagation claims; requires Human Arbiter approval',new_fitting='not authorized or run; authorize only after corrected specification and read-only verification',parameter_and_ranking_impact='UNKNOWN_PENDING_REFIT',observation_scope='catalog-level global D/i convention; image-based inclination/template reconstruction and D/i correlations unavailable',reviewer_request_source='original R1 C3/R2 C2 not found; author supplied numbering only; provisional response cannot quote actual requests')
js(V/'GATE_VERDICTS.json',summary);print(json.dumps(summary,indent=2))
