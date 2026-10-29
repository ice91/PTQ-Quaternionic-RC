"""One fullnative dependencyFD/model; dense comparator reused read-only fromP1F."""
from pathlib import Path
import sys,json,time,resource
import numpy as np
B=Path(__file__).resolve().parent;I=B.parents[2];F=B.parent/'production_smoke';sys.path.insert(0,str(B/'candidate/src'))
from ptquat.data import load_tidy_sparc
from ptquat.dependency_gradient import DependencyGradient
from ptquat.warning_telemetry import WarningCollector,collect_warnings

def main():
 out=B/'benchmark_v2';out.mkdir(exist_ok=False);gd=load_tidy_sparc(str(I/'dataset/sparc_tidy.csv'));gs=[gd[k] for k in sorted(gd)];arch=json.loads((F/'smoke_analysis_v1/FINITE_DIFFERENCE_AUDIT.json').read_text());arch={r['model']:r for r in arch['models']};cost=json.loads((F/'smoke_analysis_v1/COST_AND_WARNING_AUDIT.json').read_text());cost={r['model']:r for r in cost['models']};weights=json.loads((B/'GALAXY_WORK_WEIGHTS.json').read_text());rows=[];collector=WarningCollector()
 try:
  with collect_warnings(collector):
   for model in ['nfw1p']:
    t=np.load(I/'results/frontiers_revision_2026_10/mle_recovery/original_recovered/01_mle'/model.replace('-','_')/'theta_mle.npy');engine=DependencyGradient(gs,model,B/'dependency_v1/CURRENT_VERIFIED_DEPENDENCIES.json');before=collector.as_dict()['total_events'];r=engine.evaluate(t,absolute_step=1e-8,scheme='forward',order=192);old=arch[model];assert old['complete_initial_gradient'];a=np.array(r['gradient_native']);b=np.array(old['gradient']);diff=abs(a-b);denoms=np.array([c['denominator'] for c in r['components']]);assert np.array_equal(abs(denoms),abs(np.array(old['steps'])))
    n=91;gamma=n*np.finfo(float).eps/(1-n*np.finfo(float).eps);bound=gamma*2*np.sum(abs(np.array(r['baseline_log_marginals'])))/abs(denoms);assert np.all(diff<=bound)
    work=sum(weights[model][str(v['order'])][v['galaxy']]['fullsample_equivalent'] for v in r['evaluations']);k=len(t)
    row=dict(model=model,k=k,status='PASS_ARCHIVED_ROUNDOFF_COMPARATOR',dependency_fullsample_base_calls=1,dependency_actual_galaxy_evaluations=r['actual_galaxy_evaluations'],dependency_fullsample_equivalents=work,dependency_wall_seconds=r['wall_seconds'],dependency_warning_events=collector.as_dict()['total_events']-before,dense_source='P1F actual original driver initialcompletegradient, same historicaltheta/steps/policy',dense_fullsample_calls=k+1,dense_actual_galaxy_evaluations=91*(k+1),dense_fullsample_equivalents=k+1,dense_archived_wall_seconds=cost[model]['empirical_initial_gradient_cost_seconds'],work_speedup=(k+1)/work,crossrun_wall_speedup=cost[model]['empirical_initial_gradient_cost_seconds']/r['wall_seconds'],max_native_gradient_delta=float(diff.max()),max_relative_gradient_delta=float(np.max(diff/np.maximum(abs(b),1.))),max_roundoff_allowance=float(bound.max()),gradient_precision_claim='archive tinyFD comparator is correctness evidence subject to fullsum roundoff, notaccuracyproof; 7directionsteps/refinement gate separate',cache='no persistent parameter means/variance cache',max_process_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    rows.append(row);(out/(model.replace('-','_')+'_complete_gradient.json')).write_text(json.dumps(dict(result=r,archived_gradient=old,component_deltas=diff.tolist(),roundoff_allowance=bound.tolist()),indent=2)+'\n');print(json.dumps(row),flush=True)
  (out/'COST_AND_GRADIENT.json').write_text(json.dumps(dict(status='PASS',models=rows,optimization=False),indent=2)+'\n')
 except Exception as exc:
  (out/'FAILURE.json').write_text(json.dumps(dict(error=repr(exc),completed_models=len(rows),warnings_persisted_in_finally=True),indent=2)+'\n')
  raise
 finally:
  (out/'warnings.json').write_text(json.dumps(collector.as_dict(),indent=2)+'\n')
if __name__=='__main__':main()
