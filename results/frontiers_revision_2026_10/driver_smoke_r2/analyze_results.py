"""ENGINEERING_SMOKE_ONLY — NOT CORRECTED MLE. Read-only postprocessing."""
from pathlib import Path
from collections import Counter,defaultdict
import json,math,hashlib,statistics
import numpy as np
B=Path(__file__).resolve().parent;S=B/'execution_v1';LABEL='ENGINEERING_SMOKE_ONLY — NOT CORRECTED MLE'
def ld(p):return json.loads(p.read_text())
def save(p,r):
 if p.exists():raise FileExistsError(p)
 p.write_text(json.dumps(dict(label=LABEL,**r),indent=2,allow_nan=False)+'\n')
def main():
 out=B/'analysis_v1';out.mkdir(exist_ok=False);summary=ld(S/'RUN_SUMMARY.json');budget=summary['budget'];protocol=ld(B/'SMOKE_PROTOCOL.json');weights=ld(B/'GALAXY_WORK_WEIGHTS.json');requests=ld(S/'REQUESTS.json')['records'];groups=defaultdict(list);models=[];node_sum=0;eq=0.;last=0
 with (S/'galaxy_work_ledger.jsonl').open() as f:
  for line in f:
   r=json.loads(line);assert r['id']==last+1;last=r['id'];assert r['order'] in [192,96] and 0<=r['gi']<91
   w=weights[r['model']][str(r['order'])][r['gi']];assert w['galaxy']==r['galaxy'] and w['fullsample_equivalent']==r['equivalent'] and w['nD']*w['ni']==r['density_nodes'];eq+=r['equivalent'];node_sum+=r['density_nodes'];groups[r['request_id']].append(r)
 assert last==budget['actual_galaxy_calls']==budget['completed_galaxy_calls'] and node_sum==budget['charged_density_nodes']==budget['completed_density_nodes'];assert abs(eq-budget['charged_arithmetic_proxy_equivalents'])<=1e-10
 assert [r['id'] for r in requests]==list(range(1,len(requests)+1)) and all(r['status']=='COMPLETE' for r in requests)
 for r in requests:
  assert r['actual_galaxy_calls']==len(groups[r['id']]);assert abs(sum(g['equivalent'] for g in groups[r['id']])-r['equivalent_work'])<=1e-10
  file=S/f'{r["kind"]}_{r["id"]:04d}_{r["model"].replace("-","_")}.json';a=ld(file)
  if r['kind']=='gradient':
   assert a['actual_galaxy_evaluations']==len(groups[r['id']]);assert sum(e['density_nodes'] for e in a['evaluations'])==sum(z['density_nodes'] for z in groups[r['id']]);assert len(a['gradient_native'])==len(a['input_theta']) and np.isfinite(a['gradient_native']).all()
  else:assert len(a['galaxy_log_marginals'])==91 and np.isfinite(a['galaxy_log_marginals']).all()
 for model in ['ptq-screen','nfw1p']:
  result=ld(S/(model.replace('-','_')+'_RESULT.json'));trials=result['trials'];accepted=result['accepted_callbacks'];assert len(accepted)==1 and accepted[0]['advanced'];assert result['scipy']['nit']==1 and result['scipy']['nfev']==result['scipy']['njev']==len(trials)==4
  terminal=result['terminal'];assert terminal['theta_sha256']==accepted[0]['theta_sha256'];chosen=[t for t in trials if t['theta_sha256']==terminal['theta_sha256']];assert len(chosen)==1 and chosen[0]['request_id']==accepted[0]['request_id'];assert len(set(t['theta_sha256'] for t in trials))==len(trials)
  rs=[r for r in requests if r['model']==model];opt=[r for r in rs if r['phase']=='optimizer'];calls=[g for r in opt for g in groups[r['id']]];audit=[]
  for phase in ['initial','terminal']:
   a=ld(S/(model.replace('-','_')+'_'+phase+'_AUDIT.json'));assert a['status']=='PASS' and all(z['pass_direction'] for z in a['directions']);assert all(d['candidate_pass'] and all(c['pass_test'] for c in d['checks']) for d in a['directions']);audit.append(a)
  directions=[r for a in audit for r in a['directions']];max_step=max(c['difference'] for r in directions for c in r['checks'] if c['kind']=='two_steps');max_ratio=max(c['difference']/c['limit'] for r in directions for c in r['checks']);max_noise=max(c['noise_ratio'] for r in directions for c in r['checks'] if c['kind']=='fine_coarse');all_model=[g for r in rs for g in groups[r['id']]]
  row=dict(model=model,status=result['status'],accepted_iterations=1,optimizer_gradient_batches=len(opt),rejected_linesearch_trials=sum(t['acceptance']=='NOT_ACCEPTED_LINESEARCH_TRIAL' for t in trials),optimizer_actual_galaxy_calls=len(calls),optimizer_fullsample_equivalents=sum(c['equivalent'] for c in calls),optimizer_density_nodes=sum(c['density_nodes'] for c in calls),optimizer_wall_seconds=result['optimizer_wall_seconds'],optimizer_mean_gradient_seconds=statistics.mean(r['end_elapsed']-r['start_elapsed'] for r in opt),all_model_actual_galaxy_calls=len(all_model),all_model_equivalents=sum(c['equivalent'] for c in all_model),preflight_wall_seconds=summary['preflight'][model]['wall_seconds'],terminal_precision_wall_seconds=result['terminal_precision_wall_seconds'],objective_improvement=result['objective_improvement'],numerical_error_estimate=result['numerical_error_estimate'],improvement_threshold=result['improvement_threshold'],improvement_over_threshold=result['objective_improvement']/result['improvement_threshold'],claimed_fun_error=result['claimed_fun_error'],max_candidate_independent_scaled_difference=max(d['candidate_difference'] for d in directions),max_direction_step_difference=max_step,max_precision_error_over_limit=max_ratio,max_numerator_noise_ratio=max_noise,initial_precision=audit[0]['objective_precision'],terminal_precision=audit[1]['objective_precision'],projected_gradient_inf_norm=result['scipy']['projected_gradient_inf_norm'],scipy_status=result['scipy']['status'],scipy_message=result['scipy']['message'],new_scientific_MLE=False)
  models.append(row)
 verdict=dict(DRIVER_WIRING='PASS',PTQ_SCREEN_STEP=models[0]['status'],NFW_STEP=models[1]['status'],SCIENTIFIC_CAMPAIGN_READINESS='READY_FOR_BOUNDED_P1G_AUTHORIZATION')
 assert all(r['status']=='ACCEPTED_AND_NUMERICALLY_VALIDATED' for r in models)
 assert budget['actual_galaxy_calls']<=60000 and budget['charged_arithmetic_proxy_equivalents']<=600 and budget['elapsed_wall_seconds']<=1800 and all(v<=21 for v in budget['optimizer_batches'].values())
 save(out/'VALIDATED_RESULTS.json',dict(verdicts=verdict,models=models,total_budget=budget,warning_events=summary['warnings']['total_events'],warning_keys=summary['warnings']['distinct_keys'],warnings_root_cause='UNRESOLVED_WARNING_ROOT_CAUSE',actual_runtime_failures=summary['exception'],requests_complete=len(requests),objective_requests_including_gradient_baselines=len(requests),gradient_requests=sum(r['kind']=='gradient' for r in requests),standalone_full_objective_requests=sum(r['kind']=='objective' for r in requests),ledger_complete=True,synthetic_controls_excluded=True))
 print(json.dumps(dict(verdicts=verdict,models=models,total_galaxy_calls=last,equivalents=eq,density_nodes=node_sum,wall=budget['elapsed_wall_seconds']),indent=2))
if __name__=='__main__':main()
