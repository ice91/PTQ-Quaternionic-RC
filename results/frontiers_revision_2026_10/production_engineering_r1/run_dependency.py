"""Gate C: all728localcoordinates, genuine full91 recomputation, no optimization."""
from pathlib import Path
import sys,json,time,hashlib,traceback
import numpy as np,pandas as pd
B=Path(__file__).resolve().parent;I=B.parents[2];sys.path.insert(0,str(B/'candidate/src'))
from ptquat.data import load_tidy_sparc
from ptquat import mle,fit_global as fg
from ptquat.evaluation_contract import MODELS
from ptquat.dependency_gradient import ledger,source_hashes,sample_hash
from ptquat.marginal_likelihood import galaxy_log_marginal,PRIMARY_B0
from ptquat.warning_telemetry import WarningCollector,collect_warnings

def js(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
 out=B/'dependency_v1';out.mkdir(exist_ok=False);begin=time.monotonic();ct=json.loads((B/'PREDECLARED_CONTRACT.json').read_text())['dependency'];gd=load_tidy_sparc(str(I/'dataset/sparc_tidy.csv'));gs=[gd[k] for k in sorted(gd)];rows=[];ledgers=[];counts={};allvalues=[];collector=WarningCollector();assert len(gs)==91
 with collect_warnings(collector):
  try:
   for model in MODELS:
    t=np.load(I/'results/frontiers_revision_2026_10/mle_recovery/original_recovered/01_mle'/model.replace('-','_')/'theta_mle.npy');L=fg._layout(model,91,True,None);lr=ledger(model,gs);ledgers.extend(lr)
    base=np.asarray(mle.evaluate_loglike_components(t,gs,model,order=96)['galaxy_log_marginals']);counts[model]=0
    for r in lr:
     if r['scope']!='galaxy_local':continue
     if time.monotonic()-begin>ct['wall_cap_seconds']:raise RuntimeError('Dependency declared900s budget')
     j=r['index'];x=t.copy();lo,hi=r['bounds'];h=(hi-lo)*1e-4;x[j]+=h if x[j]+h<=hi else -h
     full=np.asarray(mle.evaluate_loglike_components(x,gs,model,order=96)['galaxy_log_marginals'])
     gi=r['affected_indices'][0];p=fg._unpack(x,L);local=galaxy_log_marginal(gs[gi],p,gi,model,order=96)['log_marginal'];mask=np.arange(91)!=gi
     unaffected=float(abs(full[mask]-base[mask]).max());diff=float(abs(local-full[gi]));assert unaffected<=ct['unaffected_absolute_tolerance'] and diff<=ct['local_vs_full_absolute_tolerance']
     rebuilt=base.copy();rebuilt[gi]=local;sumdiff=abs(float(rebuilt.sum()-full.sum()));assert sumdiff<=1e-10
     rows.append(dict(model=model,index=j,galaxy=gs[gi].name,native_step=float(x[j]-t[j]),unaffected_max_abs=unaffected,local_full_abs=diff,total_objective_abs=sumdiff,full_galaxy_count=91,local_galaxy_count=1,order=96))
     allvalues.append(dict(model=model,index=j,full_log_marginals=full.tolist()));counts[model]+=1
    print('DEPENDENCY PASS',model,counts[model],'elapsed',round(time.monotonic()-begin,2),flush=True)
   assert len(rows)==728
   pd.DataFrame(rows).to_csv(out/'all728_local_full_recomputations.csv',index=False);js(out/'full_vectors_all728.json',allvalues);js(B/'PARAMETER_DEPENDENCY_LEDGER.json',ledgers)
   matrix={m:[[int(g in r['affected_indices']) for g in range(91)] for r in ledgers if r['model']==m] for m in MODELS};js(B/'PARAMETER_DEPENDENCY_MATRIX.json',dict(galaxies=[g.name for g in gs],matrix=matrix))
   js(out/'VERIFIED_DEPENDENCIES.json',dict(status='PASS',models=list(MODELS),verified_local_counts=counts,local_coordinates=728,source_hashes=source_hashes(),sample_hash=sample_hash(gs),galaxies=[g.name for g in gs],max_unaffected=max(r['unaffected_max_abs'] for r in rows),max_local_full=max(r['local_full_abs'] for r in rows),actual_galaxy_evaluations=7*91+728*92,wall_seconds=time.monotonic()-begin,scope='all728local fullrecomputed ataccepted coarse96; primary192 unchanged; selectedfine192 fullchecks later'))
  except Exception as e:js(out/'FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),completed=len(rows)));raise
  finally:js(out/'warnings.json',collector.as_dict())
if __name__=='__main__':main()
