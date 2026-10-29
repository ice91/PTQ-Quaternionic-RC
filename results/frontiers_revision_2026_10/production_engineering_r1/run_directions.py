"""Seven frozen directions, two scaled steps/orders, hard96 work/300s guards."""
from pathlib import Path
import sys,json,time,traceback,resource
import numpy as np,pandas as pd
B=Path(__file__).resolve().parent;I=B.parents[2];sys.path.insert(0,str(B/'candidate/src'))
from ptquat.data import load_tidy_sparc
from ptquat.dependency_gradient import DependencyGradient
from ptquat import mle
from ptquat.warning_telemetry import WarningCollector,collect_warnings

def js(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 out=B/'directions_v1';out.mkdir(exist_ok=False);prot=json.loads((B/'DIRECTION_PROTOCOL.json').read_text());ct=json.loads((B/'PREDECLARED_CONTRACT.json').read_text())['directions'];weights=json.loads((B/'GALAXY_WORK_WEIGHTS.json').read_text());gd=load_tidy_sparc(str(I/'dataset/sparc_tidy.csv'));gs=[gd[k] for k in sorted(gd)]
 begin=time.monotonic();equiv=0.;count=0;worklog=[];rows=[];raw=[];repeat=[];collector=WarningCollector();exception=[]
 def charge(model,gi,order):
  nonlocal equiv,count
  w=weights[model][str(order)][gi]['fullsample_equivalent']
  if equiv+w>ct['max_equivalent_work'] or time.monotonic()-begin>=ct['max_seconds']:raise RuntimeError('FROZEN_DIRECTION_BUDGET_LIMIT')
  equiv+=w;count+=1;worklog.append(dict(model=model,gi=gi,order=order,equivalent=w))
 with collect_warnings(collector):
  try:
   for model in ['ptq-screen','nfw1p']:
    t=np.load(I/'results/frontiers_revision_2026_10/mle_recovery/original_recovered/01_mle'/model.replace('-','_')/'theta_mle.npy');ids=[d['index'] for d in prot['directions'] if d['model']==model];engine=DependencyGradient(gs,model,B/'dependency_v1/CURRENT_VERIFIED_DEPENDENCIES.json')
    # Two identical full192 baseline evaluations before any directional numerator.
    baselines=[]
    for rep in range(2):
     p=mle._unpack(t,mle._layout(model,91,True,None));vs=[]
     from ptquat.marginal_likelihood import galaxy_log_marginal
     for gi,g in enumerate(gs):charge(model,gi,192);vs.append(galaxy_log_marginal(g,p,gi,model)['log_marginal'])
     baselines.append(vs)
    delta=float(np.max(np.abs(np.array(baselines[0])-baselines[1])));repeat.append(dict(model=model,max_abs=delta));assert delta<=ct['repeatability_abs']
    for h in prot['steps']:
     for order in prot['orders']:
      dep=engine.evaluate(t,relative_step=h,order=order,indices=ids,work_callback=charge)
      full=engine.evaluate(t,relative_step=h,order=order,indices=ids,full_recompute=True,work_callback=charge)
      raw.append(dict(model=model,h=h,order=order,dependency=dep,full=full));assert abs(dep['objective']-full['objective'])<=1e-10
      for a,b in zip(dep['components'],full['components']):
       difference=abs(a['scaled_gradient']-b['scaled_gradient']);limit=ct['parity_scaled_abs']+ct['parity_relative']*max(abs(a['scaled_gradient']),abs(b['scaled_gradient']))
       rows.append(dict(model=model,index=a['index'],name=a['name'],h_scaled=h,order=order,scheme=a['scheme'],native_step=a['native_step'],dependency_native=a['native_gradient'],full_native=b['native_gradient'],dependency_scaled=a['scaled_gradient'],full_scaled=b['scaled_gradient'],scaled_parity_difference=difference,scaled_parity_limit=limit,parity_pass=bool(difference<=limit),stencil_values=a['stencil_values'],denominator=a['denominator']))
      print('DIRECTION',model,h,order,'eq',round(equiv,3),'seconds',round(time.monotonic()-begin,2),flush=True)
   assert len(rows)==28
   df=pd.DataFrame(rows);checks=[]
   for (model,index),s in df.groupby(['model','index']):
    for h in prot['steps']:
     fine=s[(s.h_scaled==h)&(s.order==192)].iloc[0];coarse=s[(s.h_scaled==h)&(s.order==96)].iloc[0]
     e=sum(abs(np.asarray(fine.stencil_values)-np.asarray(coarse.stencil_values)));numerator=abs(fine.stencil_values[1]-fine.stencil_values[0]);ratio=e/max(numerator,np.finfo(float).tiny)
     difference=abs(fine.dependency_scaled-coarse.dependency_scaled);limit=ct['precision_scaled_abs']+ct['precision_relative']*max(abs(fine.dependency_scaled),abs(coarse.dependency_scaled))
     checks.append(dict(model=model,index=int(index),comparison='fine_coarse',h=h,scaled_difference=float(difference),limit=float(limit),noise_abs=float(e),numerator_abs=float(numerator),noise_ratio=float(ratio),noise_unresolved=bool(ratio>=ct['noise_unresolved_ratio']),pass_accuracy=bool(difference<=limit)))
    for order in [192,96]:
     a=s[(s.h_scaled==prot['steps'][0])&(s.order==order)].iloc[0];b=s[(s.h_scaled==prot['steps'][1])&(s.order==order)].iloc[0];difference=abs(a.dependency_scaled-b.dependency_scaled);limit=ct['precision_scaled_abs']+ct['precision_relative']*max(abs(a.dependency_scaled),abs(b.dependency_scaled));checks.append(dict(model=model,index=int(index),comparison='two_steps',order=order,scaled_difference=float(difference),limit=float(limit),pass_accuracy=bool(difference<=limit)))
   df.drop(columns=['stencil_values']).to_csv(out/'all28_direction_comparisons.csv',index=False);js(out/'all28_records.json',rows);js(out/'all28_engine_pairs.json',raw);js(out/'precision_checks.json',checks)
   failures=sum(not r['parity_pass'] for r in rows)+sum(not r['pass_accuracy'] for r in checks);unresolved=sum(r.get('noise_unresolved',False) for r in checks)
   js(out/'DIRECTION_GATE.json',dict(status='PASS' if failures==0 and unresolved==0 else ('INCONCLUSIVE' if unresolved and not failures else 'FAIL'),rows=len(rows),directions=7,parity_failures=sum(not r['parity_pass'] for r in rows),precision_failures=sum(not r['pass_accuracy'] for r in checks),noise_unresolved=unresolved,max_scaled_parity_difference=max(r['scaled_parity_difference'] for r in rows),max_scaled_fine_coarse=max(r['scaled_difference'] for r in checks if r['comparison']=='fine_coarse'),max_scaled_two_steps=max(r['scaled_difference'] for r in checks if r['comparison']=='two_steps'),max_noise_ratio=max(r.get('noise_ratio',0) for r in checks),actual_galaxy_evaluations=count,fullsample_equivalent_work=equiv,wall_seconds=time.monotonic()-begin,repeats=repeat,max_process_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,optimization=False))
  except Exception as exc:
   exception.append(dict(error=repr(exc),traceback=traceback.format_exc(),completed=len(rows)));js(out/'DIRECTION_FAILURE.json',exception);raise
  finally:js(out/'work_log.json',worklog);js(out/'warnings.json',collector.as_dict())
if __name__=='__main__':main()
