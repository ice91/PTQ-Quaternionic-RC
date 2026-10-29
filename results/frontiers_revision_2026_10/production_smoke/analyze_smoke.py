"""ENGINEERING_SMOKE_ONLY — NOT CORRECTED MLE. Read-only engineering audit."""
from pathlib import Path
from collections import Counter
import json,statistics
import numpy as np
B=Path(__file__).resolve().parent;S=B/'engineering_smoke_v1';LABEL='ENGINEERING_SMOKE_ONLY — NOT CORRECTED MLE'
def js(p,x):p.write_text(json.dumps(dict(label=LABEL,**x),indent=2,allow_nan=False)+'\n')
def main():
 out=B/'smoke_analysis_v1';out.mkdir(exist_ok=False);results=[];gradients=[];failures=[]
 for p in sorted(S.glob('*_objective_calls.jsonl')):
  model=p.name.removesuffix('_objective_calls.jsonl').replace('_','-');cost=[];warnings=Counter();first=[];total=0;nodes=0;stages=Counter()
  # Stream records: repeated warnings must not require retaining allcalls inRAM.
  for line in p.open():
   r=json.loads(line);total+=1;stages[r['stage']]+=1;warnings.update(w['category']+': '+w['message'] for w in r.get('warnings',[]));nodes+=r.get('density_nodes',0)
   if r['status']!='COMPLETE':failures.append(dict(model=model,call=r['call'],error=r.get('error')))
   if r['stage']=='optimizer':
    cost.append(r['seconds'])
    if len(first)<(94+1 if model=='ptq-screen' else 183+1):first.append({k:r[k] for k in ['theta','negative_logL','optimizer_call']})
  k=len(first[0]['theta']) if first else 0
  if len(first)==k+1:
   x0=np.array(first[0]['theta']);f0=first[0]['negative_logL'];gradient=np.zeros(k);step=np.zeros(k);seen=[]
   for r in first[1:]:
    d=np.array(r['theta'])-x0;indices=np.flatnonzero(d);assert len(indices)==1
    j=int(indices[0]);seen.append(j);step[j]=d[j];gradient[j]=(r['negative_logL']-f0)/d[j]
   assert set(seen)==set(range(k));gradients.append(dict(model=model,kind='forward finite differences of deterministic integrated objective; not analytic derivatives',k=k,complete_initial_gradient=True,steps=step.tolist(),gradient=gradient.tolist(),all_finite=bool(np.isfinite(gradient).all()),scaled_direction_accuracy='NOT_ESTABLISHED_BY_THIS_SMOKE',gradient_refinement='NOT_MEASURED'))
  else:gradients.append(dict(model=model,k=k,complete_initial_gradient=False,calls=len(first)))
  results.append(dict(model=model,total_fullsample_calls=total,optimizer_calls=len(cost),stage_counts=dict(stages),mean_optimizer_seconds=statistics.mean(cost) if cost else None,median_optimizer_seconds=statistics.median(cost) if cost else None,min_optimizer_seconds=min(cost) if cost else None,max_optimizer_seconds=max(cost) if cost else None,
    empirical_initial_gradient_cost_seconds=sum(cost[:k+1]),density_evaluations=nodes,warning_counts=dict(warnings),warning_total=sum(warnings.values()),projected_single_fullgradient_seconds=(k+1)*statistics.mean(cost) if cost else None,
    timing_interpretation='cost projection only; no prediction of iteration count, convergence time, stationary optimum or scientificfit'))
 js(out/'COST_AND_WARNING_AUDIT.json',dict(models=results,failures=failures));js(out/'FINITE_DIFFERENCE_AUDIT.json',dict(models=gradients))
 print(json.dumps({r['model']:{k:v for k,v in r.items() if k not in ['stage_counts','warning_counts']} for r in results},indent=2))
if __name__=='__main__':main()
