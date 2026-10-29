"""Bounded predeclared real-galaxy operating-regime audit. No optimizer or fits."""
from pathlib import Path
import sys,json,hashlib,time,traceback,functools
import numpy as np,pandas as pd
from scipy.special import ndtr,logsumexp
from scipy.stats import truncnorm
B=Path(__file__).resolve().parent;I=B.parents[2];C=I/'results/frontiers_revision_2026_10/covariance_reference';M=I/'results/frontiers_revision_2026_10/mle_recovery'
sys.path.insert(0,str(C));import reference as ref
sys.path.insert(0,str(B/'isolated_candidate/src'));from ptquat import fit_global as fg,mle
START=time.monotonic();NODES=0

def js(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
@functools.lru_cache(None)
def legendre(n):return np.polynomial.legendre.leggauss(n)
def quadrature(g,p,gi,T,order):
 global NODES
 protocol=json.loads((B/'PREDECLARED_PROTOCOL.json').read_text())
 if time.monotonic()-START>protocol['budget']['max_runtime_seconds']:raise RuntimeError('predeclared runtime budget exhausted')
 if NODES+4*order**2>protocol['budget']['max_conditional_density_node_evaluations']:raise RuntimeError('predeclared node budget exhausted')
 dl=max(-T,-g.D/g.sD);du=T;il=max(-T,-g.inc/g.si);iu=min(T,(np.pi/2-g.inc)/g.si);z,w=legendre(order)
 zd=dl+(z+1)*(du-dl)/2;zi=il+(z+1)*(iu-il)/2
 wd=w*(du-dl)/2*np.exp(-zd**2/2)/np.sqrt(2*np.pi)/(ndtr(du)-ndtr(dl));wi=w*(iu-il)/2*np.exp(-zi**2/2)/np.sqrt(2*np.pi)/(ndtr(iu)-ndtr(il));weights=np.log(wd)[:,None]+np.log(wi)[None,:]
 v0=ref.velocity(g,p,gi);jd,ji=ref.derivatives(g,p,gi);deltaD=g.sD*zd[:,None,None];deltai=g.si*zi[None,:,None];b=np.sin(g.inc+g.si*zi)/np.sin(g.inc);vd=ref.velocity(g,p,gi,(g.D+g.sD*zd)[:,None]);out=[]
 for kind in ['B0_linear','B0_nonlinear','B1_nonlinear','inclination_only_nonlinear_B0']:
  if kind=='B0_linear':mu=v0[None,None,:]+jd[None,None,:]*deltaD+ji[None,None,:]*deltai
  elif kind=='inclination_only_nonlinear_B0':mu=(v0[None,None,:]+jd[None,None,:]*deltaD)*b[None,:,None]
  else:mu=vd[:,None,:]*b[None,:,None]
  variance=g.e[None,None,:]**2+(p['sigma']**2 if kind!='B1_nonlinear' else (p['sigma']*b[None,:,None])**2)
  logdensity=-.5*np.sum((g.y[None,None,:]-mu)**2/variance+np.log(2*np.pi*variance),axis=-1)
  value=float(logsumexp(logdensity+weights));assert np.isfinite(value)
  NODES+=order**2;out.append(dict(construction=kind,T=T,order=order,logL=value,prior_weight_sum=float(wd.sum()*wi.sum()),zD_low=dl,zD_high=du,zi_low=il,zi_high=iu))
 return out

def points(model,t,L,gi):
 out=[]
 if 'i_eps' in L:
  x=t.copy();x[L['i_eps']]*=1.01;out.append(('scale1.01',x))
 if 'i_log10a0' in L:
  x=t.copy();x[L['i_log10a0']]+=np.log10(1.01);out.append(('scale1.01',x))
 if 'i_lq' in L:
  x=t.copy();x[L['i_lq']]+=np.log(1.01);out.append(('q1.01',x))
 x=t.copy();x[L['i_lnsig']]+=np.log(1.02);out.append(('sigma1.02',x))
 ix=L['sl_U'].start+gi;x=t.copy();x[ix]+=.01 if x[ix]+.01<=1.5 else -.01;out.append(('U_inward_0.01',x))
 if 'sl_lM' in L:
  ix=L['sl_lM'].start+gi;x=t.copy();x[ix]+=.02 if x[ix]+.02<=13 else -.02;out.append(('halo_inward_0.02dex',x))
 return out

def main():
 assert json.loads((B/'parity_v1/PARITY_GATE.json').read_text())['status']=='PASS'
 out=B/'boundary_v1';out.mkdir(exist_ok=False);protocol=json.loads((B/'PREDECLARED_PROTOCOL.json').read_text());gs=ref.load_sample(I/'dataset/sparc_tidy.csv');names=[g.name for g in gs];vectors={m:np.load(M/'original_recovered/01_mle'/m.replace('-','_')/'theta_mle.npy') for m in ref.MODELS};ps={m:ref.decode(m,t,91) for m,t in vectors.items()}
 # Tier1 covers ALL22. Physical Gaussian kernel moments distinguish truncation from nonlinear sine.
 diagnostic=[]
 for row in protocol['tier1']:
  gi=names.index(row['galaxy']);g=gs[gi];a=-g.inc/g.si;b=(np.pi/2-g.inc)/g.si;mean_z,var_z=truncnorm.stats(a,b,moments='mv');z,w=legendre(128);lo=max(-5.,a);hi=min(5.,b);zz=lo+(z+1)*(hi-lo)/2;ww=w*(hi-lo)/2*np.exp(-zz**2/2)/np.sqrt(2*np.pi)/(ndtr(hi)-ndtr(lo));projection=np.sin(g.inc+g.si*zz)/np.sin(g.inc);Eb=float(ww@projection);Vb=float(ww@((projection-Eb)**2))
  diagnostic.append(dict(galaxy=g.name,edge_on=g.name in protocol['tier2_edge_on'],i_deg=float(np.rad2deg(g.inc)),i_err_deg=float(np.rad2deg(g.si)),n=len(g.r),relD=g.sD/g.D,physical_i_prior_mass=float(ndtr(b)-ndtr(a)),physical_D_prior_mass=float(ndtr(g.D/g.sD)),physical_mean_i_shift_rad=float(mean_z*g.si),physical_var_i_rad2=float(var_z*g.si**2),T5_E_projection=Eb,T5_projection_mean_shift=Eb-1.,T5_projection_variance=Vb,untruncated_first_order_projection_variance=float((g.si/np.tan(g.inc))**2),second_order_shift_rough_untruncated=-.5*g.si**2))
 pd.DataFrame(diagnostic).to_csv(out/'tier1_all22.csv',index=False)
 results=[];attempts=[];errors=[];pointmeta=[]
 def runone(g,gi,m,p,point,Ts,orders,tier):
  A,v,Cm=ref.evaluate(g,p,gi)
  for T in Ts:
   for order in orders:
    key=dict(galaxy=g.name,model=m,point=point,T=T,order=order,tier=tier)
    try:
     rr=quadrature(g,p,gi,T,order);attempts.append(dict(**key,status='COMPLETE'))
     for r in rr:results.append(dict(label='FIXED_HISTORICAL_PARAMETER_POINT' if point=='historical' else 'FIXED_PARAMETER_PERTURBATION',galaxy=g.name,model=m,point=point,tier=tier,A_corr=A['logL'],**r))
    except Exception as exc:errors.append(dict(**key,error=repr(exc),traceback=traceback.format_exc()));attempts.append(dict(**key,status='FAILED'))
  return A,Cm
 for name in protocol['integration_galaxies']:
  gi=names.index(name);g=gs[gi];tier='Tier2_edge_on' if name in protocol['tier2_edge_on'] else 'Tier3_geometry_control'
  for m,p in ps.items():runone(g,gi,m,p,'historical',protocol['nuisance']['tails_T'],protocol['nuisance']['orders'],tier)
  print('HISTORICAL',name,'nodes',NODES,flush=True)
 # Limited neighborhoods: no search; one-sided/inward perturbations fixed before outcomes.
 for name in protocol['neighborhood']['galaxies']:
  gi=names.index(name);g=gs[gi]
  for m,t in vectors.items():
   L=fg._layout(m,91,True,None);p0=ps[m];A0,_,C0=ref.evaluate(g,p0,gi)
   for label,tt in points(m,t,L,gi):
    bounds=mle.theta_bounds(m,91,H0_si=ref.H0);valid=all(lo<=v<=hi for v,(lo,hi) in zip(tt,bounds));assert valid,(m,name,label)
    p=ref.decode(m,tt,91);A,Cm=runone(g,gi,m,p,label,protocol['neighborhood']['tails_T'],protocol['neighborhood']['orders'],'GateE_parameter_neighborhood')
    ix=np.flatnonzero(tt!=t);pointmeta.append(dict(galaxy=name,model=m,point=label,theta_float64_sha256=hashlib.sha256(tt.astype('<f8').tobytes()).hexdigest(),coordinates=[dict(index=int(j),old=float(t[j]),new=float(tt[j])) for j in ix],bounds_pass=True,C_change_relative=float(np.linalg.norm(Cm-C0)/np.linalg.norm(C0)),A_corr_delta=A['logL']-A0['logL']))
  print('NEIGHBORHOOD',name,'nodes',NODES,flush=True)
 df=pd.DataFrame(results);df.to_csv(out/'all_integrals.csv',index=False);js(out/'attempt_manifest.json',attempts);js(out/'exceptions.json',errors);js(out/'parameter_points.json',pointmeta)
 # Exact tested/untested combinations must remain explicit, not implied full22 coverage.
 coverage=[]
 for row in protocol['tier1']:
  name=row['galaxy']
  for m in protocol['model_order']:coverage.append(dict(galaxy=name,model=m,tier1_diagnostic=True,historical_full_integration=name in protocol['integration_galaxies'],parameter_neighborhood=name in protocol['neighborhood']['galaxies'],not_tested_reason='' if name in protocol['integration_galaxies'] else 'Tier1 only under predeclared budget; no nonlinear integral'))
 pd.DataFrame(coverage).to_csv(out/'all22_model_coverage.csv',index=False)
 convergence=[];decomp=[]
 for key,sub in df.groupby(['galaxy','model','point','tier','T','construction']):
  ss=sub.set_index('order');low,high=sorted(ss.index)[-2:];delta=float(ss.loc[high].logL-ss.loc[low].logL);convergence.append(dict(galaxy=key[0],model=key[1],point=key[2],tier=key[3],T=key[4],construction=key[5],delta_high_low=delta,converged=abs(delta)<=protocol['nuisance']['convergence_abs_logL']))
 cf=pd.DataFrame(convergence);cf.to_csv(out/'convergence.csv',index=False)
 for key,sub in df[df.order==256].groupby(['galaxy','model','point','tier','T']):
  ss=sub.set_index('construction');conv=cf[(cf.galaxy==key[0])&(cf.model==key[1])&(cf.point==key[2])&(cf.T==key[4])].converged.all();linear=float(ss.loc['B0_linear'].logL);full=float(ss.loc['B0_nonlinear'].logL);floor=float(ss.loc['B1_nonlinear'].logL);ionly=float(ss.loc['inclination_only_nonlinear_B0'].logL);A=float(ss.A_corr.iloc[0]);decomp.append(dict(galaxy=key[0],model=key[1],point=key[2],tier=key[3],T=key[4],A_corr=A,linear=linear,nonlinear_B0=full,nonlinear_B1=floor,linear_trunc_minus_A=linear-A,full_nonlinear_minus_linear=full-linear,inclination_nonlinear_minus_linear=ionly-linear,distance_remainder_nonlinear_minus_ionly=full-ionly,floor_B1_minus_B0=floor-full,total_nonlinear_minus_A=full-A,converged=bool(conv)))
 dd=pd.DataFrame(decomp);dd.to_csv(out/'decomposition.csv',index=False)
 tails=[]
 for key,ss in dd[dd.point=='historical'].groupby(['galaxy','model']):
  ss=ss.set_index('T');tails.append(dict(galaxy=key[0],model=key[1],B0_T5_T4=float(ss.loc[5.].nonlinear_B0-ss.loc[4.].nonlinear_B0),B1_T5_T4=float(ss.loc[5.].nonlinear_B1-ss.loc[4.].nonlinear_B1),linear_T5_T4=float(ss.loc[5.].linear-ss.loc[4.].linear),converged=bool(ss.converged.all())))
 pd.DataFrame(tails).to_csv(out/'tail_sensitivity.csv',index=False)
 # Compare EFFECTS, not raw objective changes, over fixed parameter neighborhoods.
 neighbors=[]
 for _,row in dd[(dd.point!='historical')&(dd.T==5.)].iterrows():
  base=dd[(dd.galaxy==row.galaxy)&(dd.model==row.model)&(dd.point=='historical')&(dd.T==5.)].iloc[0]
  neighbors.append(dict(galaxy=row.galaxy,model=row.model,point=row.point,nonlinear_effect_change=row.full_nonlinear_minus_linear-base.full_nonlinear_minus_linear,inclination_effect_change=row.inclination_nonlinear_minus_linear-base.inclination_nonlinear_minus_linear,floor_effect_change=row.floor_B1_minus_B0-base.floor_B1_minus_B0,total_effect_change=row.total_nonlinear_minus_A-base.total_nonlinear_minus_A,converged=bool(row.converged and base.converged)))
 pd.DataFrame(neighbors).to_csv(out/'neighborhood_effect_changes.csv',index=False)
 js(out/'RUN_SUMMARY.json',dict(tier1_galaxies=len(diagnostic),historical_integrated_galaxies=len(protocol['integration_galaxies']),edge_on_galaxies=len(protocol['tier2_edge_on']),historical_model_points=len(protocol['integration_galaxies'])*7,neighborhood_points=len(pointmeta),integral_rows=len(results),attempted_grids=len(attempts),exceptions=len(errors),convergence_cases=len(convergence),nonconverged=int((~cf.converged).sum()),max_quad_error=float(abs(cf.delta_high_low).max()),node_density_evaluations=NODES,elapsed_seconds=time.monotonic()-START,optimization=False,all_fixed_points=True))
 print('FINISHED',NODES,'nodes',time.monotonic()-START,'seconds',flush=True)
if __name__=='__main__':main()
