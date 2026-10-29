"""Deterministic fixed-point audit. Never invokes an optimizer or writes production files.
All callbacks below are extracted from frozen source and evaluated at prescribed points.
"""
from pathlib import Path
import sys,os,json,hashlib,csv,ast,time,traceback,platform,subprocess
import numpy as np
from scipy.special import logsumexp,ndtr
from scipy.linalg import cho_factor,cho_solve
B=Path(__file__).resolve().parent;I=B.parents[2];M=I/'results/frontiers_revision_2026_10/mle_recovery';sys.path.insert(0,str(I/'src'))
from ptquat import mle,models,likelihood,fit_global as fg
from ptquat.constants import H0_SI,KPC,KM,G_SI,MSUN
from ptquat.data import load_tidy_sparc,GalaxyData
DESIGN=json.loads((B/'PREDECLARED_DESIGN.json').read_text());START=time.monotonic()
def js(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def csvout(p,rows):
 if rows:
  with p.open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def hashfile(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def metrics(a,b):
 d=a-b;den=max(float(np.linalg.norm(b)),1e-12)
 return dict(abs_max=float(np.max(abs(d))),rel_norm=float(np.linalg.norm(d))/den,point_relative_max=float(np.max(abs(d)/np.maximum(abs(b),1e-8))))
def components(y,v,C):
 C=.5*(C+C.T);L=np.linalg.cholesky(C);r=y-v;z=cho_solve((L,True),r);q=float(r@z);ld=float(2*np.log(np.diag(L)).sum());n=float(len(r)*np.log(2*np.pi));return dict(quad=q,logdet=ld,normalization=n,logL=-.5*(q+ld+n),jitter=0.)
def vmodel(m,p,gi,g,scale=1.,radius_only=False):
 s=np.asarray(scale);r=g.r_kpc*s;f=1. if radius_only else np.sqrt(s);d=g.v_disk*f;b=g.v_bulge*f;gas=g.v_gas*f;U=float(p['U'][gi])
 if m=='baryon':return models.model_v_baryon(U,r,d,b,gas)
 if m=='ptq':return models.model_v_ptq(U,p['eps'],r,d,b,gas,H0_si=H0_SI)
 if m=='ptq-nu':return models.model_v_ptq_nu(U,p['eps'],r,d,b,gas,H0_si=H0_SI)
 if m=='ptq-screen':return models.model_v_ptq_screen(U,p['eps'],np.exp(p['lq']),r,d,b,gas,H0_si=H0_SI)
 if m=='mond':return models.model_v_mond(U,10**p['log10a0'],r,d,b,gas)
 if m=='mond-screen':return models.model_v_mond_screen(U,10**p['log10a0'],np.exp(p['lq']),r,d,b,gas)
 if m=='nfw1p':return models.model_v_nfw1p(U,float(p['lM'][gi]),r,d,b,gas,H0_si=H0_SI)
 raise ValueError(m)
def analytic_distance(m,p,gi,g,v):
 if m!='nfw1p':return v/(2*g.D_Mpc)
 mass=10**float(p['lM'][gi]);c=models.c_powerlaw(mass);vh=models.nfw_v_kms_M200_c(g.r_kpc,mass,c,H0_si=H0_SI)
 rhoc=3*H0_SI**2/(8*np.pi*G_SI);r200=(3*mass*MSUN/(4*np.pi*200*rhoc))**(1/3)/KPC;x=c*g.r_kpc/r200;fx=np.log1p(x)-x/(1+x);eta=x*x/((1+x)**2*fx)-1
 bar=models.vbar_squared_kms2(float(p['U'][gi]),g.v_disk,g.v_bulge,g.v_gas)
 return (bar+vh*vh*eta)/(2*v*g.D_Mpc)
def Cprod(m,p,gi,g):
 v=vmodel(m,p,gi,g)
 def vf(r):return vmodel(m,p,gi,g,np.asarray(r)/g.r_kpc,radius_only=True)
 C=likelihood.build_covariance(v,g.r_kpc,g.v_err,g.D_Mpc,g.D_err_Mpc,g.i_deg,g.i_err_deg,vf,sigma_sys_kms=p['sigma'])
 JD=likelihood._num_grad_v_wrt_r(g.r_kpc,vf)*g.r_kpc/g.D_Mpc;Ji=v/np.tan(np.radians(g.i_deg))
 return v,C,JD,Ji
def fresh_cov(g,sigma,JD,Ji):return np.diag(g.v_err**2+sigma**2)+g.D_err_Mpc**2*np.outer(JD,JD)+np.radians(g.i_err_deg)**2*np.outer(Ji,Ji)
def unpack(m,th,n):return fg._unpack(th,fg._layout(m,n,True,None))
def budget():
 if time.monotonic()-START>900:raise RuntimeError('Predeclared 15 minute budget exhausted')
def execute_callback(m,th,gs,gi,kind,thpoint):
 # Compile the actual source callback AST, not an alternate implementation.
 tree=ast.parse((I/'src/ptquat/mle.py').read_text());outer=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='_block_coordinate_mle')
 name={'globals':'neg_g','upsilon':'neg_u','halo':'neg_um'}[kind];node=next(x for x in ast.walk(outer) if isinstance(x,ast.FunctionDef) and x.name==name)
 L=fg._layout(m,len(gs),True,None);gidx=mle._global_indices(L)
 env=dict(np=np,theta=th.copy(),L=L,gidx=gidx,galaxies=gs,gi=gi,model=m,H0_si=H0_SI,galaxy_loglike=mle.galaxy_loglike,neg_full=lambda t:mle.objective_neg_loglike(t,gs,m,H0_SI))
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(I/'src/ptquat/mle.py'),'exec'),env)
 if kind=='globals':x=thpoint[gidx]
 elif kind=='upsilon':x=thpoint[L['sl_U'].start+gi]
 else:x=np.array([thpoint[L['sl_U'].start+gi],thpoint[L['sl_lM'].start+gi]])
 return float(env[name](x))

def main():
 out=B/'run_v1';out.mkdir(exist_ok=False)
 for s,h in json.loads((B/'SOURCE_HASHES_BEFORE.json').read_text()).items():
  assert hashfile(Path(s))==h,s
 js(out/'execution_authority.json',dict(design_sha256=hashfile(B/'PREDECLARED_DESIGN.json'),contract_sha256=hashfile(B/'OBSERVATION_CONTRACT.json'),script_sha256=hashfile(Path(__file__)),python=sys.version,platform=platform.platform(),executable=sys.executable,thread_env={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','OMP_NUM_THREADS']}))
 (out/'environment_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
 gd=load_tidy_sparc(str(I/'dataset/sparc_tidy.csv'));gs=[gd[k] for k in sorted(gd)];names=[g.name for g in gs];families=DESIGN['models'];vectors={m:np.load(M/'original_recovered/01_mle'/m.replace('-','_')/'theta_mle.npy',allow_pickle=False) for m in families}
 # Reuse authenticated historical replay evidence; do not repeat all 91-galaxy evaluations.
 prior=json.loads((M/'REPLAY_VALIDATION.json').read_text());js(out/'historical_replay_authority.json',dict(source=str(M/'REPLAY_VALIDATION.json'),sha256=hashfile(M/'REPLAY_VALIDATION.json'),objective_pass=all(x['archive_replay_pass'] for x in prior['models'].values()),max_abs_delta=max(abs(x['delta_logL']) for x in prior['models'].values()),metadata_discrepancy_preserved=True))
 fd=[];points=[];effects=[];updates=[];identity=[];nuisance=[];failures=[];matrices={};callback_records=[]
 cases=[]
 for row in DESIGN['selected']:
  g=gd[row['galaxy']];gi=names.index(g.name)
  for m in families:cases.append((m,vectors[m],gs,gi,g,'historical'))
 for s in DESIGN['synthetic']:
  r=np.linspace(1.,8.,8);v=80*r**s['beta'];g=GalaxyData(s['name'],r,v+3,np.full(8,2.),v,np.zeros(8),np.zeros(8),10.,1.,s['i_deg'],3.)
  th=np.r_[np.ones(8)[:1],np.log(3.)] # one galaxy U=1; sigma_sys=3
  cases.append(('baryon',th,[g],0,g,'synthetic_reference'))
 for m,th,glist,gi,g,label in cases:
  budget()
  try:
   p=unpack(m,th,len(glist));v,C,JDp,Jip=Cprod(m,p,gi,g);dd=analytic_distance(m,p,gi,g,v);inc=np.radians(g.i_deg);cot=1/np.tan(inc);JDa=-dd;Jia=-g.v_obs*cot;Jina=-v*cot
   refs={};
   for rel in DESIGN['fd_relative_steps']:
    hd=rel*g.D_Mpc;hi=rel*max(inc,.5)
    jd=(g.v_obs-vmodel(m,p,gi,g,1+rel)-(g.v_obs-vmodel(m,p,gi,g,1-rel)))/(2*hd)
    ji=((g.v_obs*np.sin(inc)/np.sin(inc+hi)-v)-(g.v_obs*np.sin(inc)/np.sin(inc-hi)-v))/(2*hi)
    jin=((g.v_obs-v*np.sin(inc+hi)/np.sin(inc))-(g.v_obs-v*np.sin(inc-hi)/np.sin(inc)))/(2*hi)
    refs[rel]=(jd,ji,jin)
    for axis,got,expected,prod in [('D',jd,JDa,-JDp),('i_transformed',ji,Jia,-Jip),('i_fixed_frame',jin,Jina,-Jip)]:
     met=metrics(got,expected);pm=metrics(prod,expected);fd.append(dict(case=g.name,label=label,model=m,axis=axis,step=hd if axis=='D' else hi,relative_step=rel,fd_analytic_abs_max=met['abs_max'],fd_analytic_rel_norm=met['rel_norm'],fd_analytic_point_rel_max=met['point_relative_max'],production_analytic_abs_max=pm['abs_max'],production_analytic_rel_norm=pm['rel_norm'],numerical_pass=met['abs_max']<=1e-5 or met['rel_norm']<=1e-6))
     for j in range(len(v)):points.append(dict(case=g.name,label=label,model=m,axis=axis,relative_step=rel,point=j,finite_difference=float(got[j]),analytic=float(expected[j]),production_residual_oriented=float(prod[j])))
   comps=components(g.v_obs,v,C);Cdist=fresh_cov(g,p['sigma'],JDa,Jina);Ctrans=fresh_cov(g,p['sigma'],JDa,Jia)
   matrices[g.name+'__'+m+'__production']=C;matrices[g.name+'__'+m+'__distance_corrected_fixedframe']=Cdist;matrices[g.name+'__'+m+'__full_transformed']=Ctrans
   for spec,Cx in [('production',C),('full_D_fixedframe',Cdist),('full_D_transformed',Ctrans)]:
    c=components(g.v_obs,v,Cx);effects.append(dict(case=g.name,label=label,model=m,specification=spec,**c,delta_logL=c['logL']-comps['logL'],covariance_relative_frobenius=float(np.linalg.norm(Cx-C)/np.linalg.norm(C)),JD_full_norm=float(np.linalg.norm(JDa)),JD_production_norm=float(np.linalg.norm(JDp))))
   print('FD',g.name,m,flush=True)
  except Exception as exc:failures.append(dict(stage='FD',case=g.name,model=m,error=repr(exc),traceback=traceback.format_exc()))
 # Evaluate production callback endpoints at predeclared points; capture actual C construction.
 for row in DESIGN['selected']:
  g=gd[row['galaxy']];gi=names.index(g.name)
  for m in families:
   budget();th=vectors[m];L=fg._layout(m,len(gs),True,None);p=unpack(m,th,len(gs));v0,C0,_,_=Cprod(m,p,gi,g);pt=[('original',th.copy(),'globals')]
   if 'i_eps' in L:
    z=th.copy();z[L['i_eps']]*=1.01;pt.append(('scale',z,'globals'))
   if 'i_log10a0' in L:
    z=th.copy();z[L['i_log10a0']]+=np.log10(1.01);pt.append(('scale',z,'globals'))
   if 'i_lq' in L:
    z=th.copy();z[L['i_lq']]+=np.log(1.01);pt.append(('q',z,'globals'))
   z=th.copy();ix=L['sl_U'].start+gi;z[ix]+=.01 if z[ix]+.01<=1.5 else -.01;pt.append(('upsilon',z,'halo' if m=='nfw1p' else 'upsilon'))
   z=th.copy();z[L['i_lnsig']]+=np.log(1.01);pt.append(('sigma_sys',z,'globals'))
   for name,z,kind in pt:
    try:
     pp=unpack(m,z,len(gs));v,C,_,_=Cprod(m,pp,gi,g);comp=components(g.v_obs,v,C)
     captured=[];orig=fg.build_covariance;ol=likelihood.build_covariance
     def capture(*a,**kw):
      mat=ol(*a,**kw);captured.append(mat.copy());return mat
     fg.build_covariance=capture;likelihood.build_covariance=capture
     try:cb=execute_callback(m,th,gs,gi,kind,z)
     finally:fg.build_covariance=orig;likelihood.build_covariance=ol
     at=captured[gi] if kind=='globals' else captured[0]
     # Globals callback computes full sample; compare to exact production objective.
     expected=-mle.objective_neg_loglike(z,gs,m,H0_SI) if kind=='globals' else mle.galaxy_loglike(z,gs,gi,m,H0_SI)
     native=mle.galaxy_loglike(z,gs,gi,m,H0_SI)
     updates.append(dict(case=g.name,model=m,parameter_point=name,callback=kind,covariance_calls=len(captured),C_captured_rel_error=float(np.linalg.norm(at-C)/np.linalg.norm(C)),C_change_rel=float(np.linalg.norm(C-C0)/np.linalg.norm(C0)),logdet_change=comp['logdet']-components(g.v_obs,v0,C0)['logdet'],callback_logL=-cb,callback_endpoint_logL=expected,callback_delta=-cb-expected,production_galaxy_logL=native,independent_minus_production=comp['logL']-native,**comp))
     callback_records.append(dict(case=g.name,model=m,parameter_point=name,actual_callback=kind,source_file=str(I/'src/ptquat/mle.py'),optimizer_invoked=False))
     matrices[g.name+'__'+m+'__point_'+name]=C
     # Matched same physical points for both families, in both directions.
     maps={'mond':('ptq-nu',mle.map_theta_mond_to_ptqnu),'ptq-nu':('mond',mle.map_theta_ptqnu_to_mond),'mond-screen':('ptq-screen',mle.map_theta_mondscreen_to_ptqscreen),'ptq-screen':('mond-screen',mle.map_theta_ptqscreen_to_mondscreen)}
     if m in maps:
      target,fn=maps[m];zt=fn(z,len(gs),H0_SI);p2=unpack(target,zt,len(gs));v2,C2,_,_=Cprod(target,p2,gi,g);c2=components(g.v_obs,v2,C2)
      d=np.linalg.norm(C-C2)/np.linalg.norm(C);delta=c2['logL']-comp['logL'];identity.append(dict(case=g.name,source=m,target=target,parameter_point=name,covariance_relative=float(d),logL_delta=delta,pass_identity=abs(delta)<=1e-7 and d<=1e-9))
    except Exception as exc:failures.append(dict(stage='UPDATE',case=g.name,model=m,point=name,error=repr(exc),traceback=traceback.format_exc()))
   print('UPDATE',g.name,m,flush=True)
 # Normalized quadrature on independent truncated Gaussian zD,zi; fixed data density.
 for name in DESIGN['nuisance_galaxies']:
  g=gd[name];gi=names.index(name);inc=np.radians(g.i_deg);si=np.radians(g.i_err_deg)
  dl=max(-3.,-g.D_Mpc/g.D_err_Mpc);du=3.;il=max(-3.,-inc/si);iu=min(3.,(np.pi/2-inc)/si)
  for m in families:
   p=unpack(m,vectors[m],len(gs));v,C,_,_=Cprod(m,p,gi,g);JD=analytic_distance(m,p,gi,g,v);Cref=fresh_cov(g,p['sigma'],-JD,-v/np.tan(inc));A=components(g.v_obs,v,C)['logL'];Alin=components(g.v_obs,v,Cref)['logL']
   for order in DESIGN['quadrature_orders']:
    budget();nodes,w=np.polynomial.legendre.leggauss(order)
    zd=dl+(nodes+1)*(du-dl)/2;zi=il+(nodes+1)*(iu-il)/2
    wd=w*(du-dl)/2*np.exp(-zd**2/2)/np.sqrt(2*np.pi)/(ndtr(du)-ndtr(dl));wi=w*(iu-il)/2*np.exp(-zi**2/2)/np.sqrt(2*np.pi)/(ndtr(iu)-ndtr(il))
    scale=1+zd*g.D_err_Mpc/g.D_Mpc;sinratio=np.sin(inc+zi*si)/np.sin(inc);vs=vmodel(m,p,gi,g,scale[:,None]);mu=vs[:,None,:]*sinratio[None,:,None];r=g.v_obs[None,None,:]-mu
    for floor in ['B0_nominal_floor','B1_intrinsic_floor']:
     var=g.v_err[None,None,:]**2+(p['sigma']**2 if floor=='B0_nominal_floor' else (p['sigma']*sinratio[None,:,None])**2)
     ll=-.5*np.sum(r*r/var+np.log(2*np.pi*var),axis=-1);logint=float(logsumexp(ll+np.log(wd)[:,None]+np.log(wi)[None,:]))
     nuisance.append(dict(case=name,model=m,construction=floor,order=order,nodes=order**2,zD_lo=dl,zD_hi=du,zi_lo=il,zi_hi=iu,prior_weight_sum=float(wd.sum()*wi.sum()),logL_integrated=logint,logL_A_production=A,logL_A_fullD_fixedframe=Alin,delta_B_minus_A=logint-A,delta_B_minus_fullD=logint-Alin,propagated_terms_in_conditional=False))
   print('QUAD',name,m,flush=True)
 csvout(out/'finite_difference_summary.csv',fd);csvout(out/'finite_difference_points.csv',points);csvout(out/'covariance_effects.csv',effects);csvout(out/'covariance_updates.csv',updates);csvout(out/'matched_identities.csv',identity);csvout(out/'explicit_nuisance.csv',nuisance);js(out/'callback_execution.json',callback_records);js(out/'failures.json',failures);np.savez_compressed(out/'covariances.npz',**matrices)
 summary=dict(fd_rows=len(fd),effect_rows=len(effects),update_rows=len(updates),identity_rows=len(identity),quadrature_rows=len(nuisance),failures=len(failures),elapsed_seconds=time.monotonic()-START,optimization_performed=False,source_changed=False)
 js(out/'RUN_SUMMARY.json',summary);print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
