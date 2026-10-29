"""Version-specific, single-start engineering driver. Never a scientific campaign."""
from pathlib import Path
import hashlib,json,time,warnings
import numpy as np
from scipy.optimize import minimize
from . import mle
from .fit_global import _layout,_unpack
from .marginal_likelihood import PRIMARY_B0
LABEL='ENGINEERING_SMOKE_ONLY — NOT CORRECTED MLE'
class DriverBudgetLimit(RuntimeError): pass

def write_json(path,record):
    path.write_text(json.dumps(dict(label=LABEL,**record),indent=2,allow_nan=False)+'\n')

def run_bounded_smoke(galaxies,vectors,protocol,out):
    raise RuntimeError("P1F-R1: optimizer execution disabled; next smoke requires separate authorization")
    out=Path(out);out.mkdir(exist_ok=False)
    cfg=protocol['smoke'];started=time.monotonic();aggregate=0;results=[]
    def clockguard():
        if time.monotonic()-started>=cfg['max_optimization_wall_seconds_aggregate']:
            raise DriverBudgetLimit('aggregate wall-time cap')
    for model in cfg['models']:
        modelstart=time.monotonic();calls=0;optcalls=0;accepted=[];warnings_seen=[]
        t0=np.array(vectors[model],dtype=float,copy=True);L=_layout(model,len(galaxies),True,None)
        bounds=np.asarray(mle.theta_bounds(model,len(galaxies)),float)
        logs=out/(model.replace('-','_')+'_objective_calls.jsonl')
        exceptions=[];validation=[];trials=[];status='NOT_STARTED';scipy_record=None
        write_json(out/(model.replace('-','_')+'_initial.json'),dict(model=model,theta=t0.tolist(),bounds=bounds.tolist(),likelihood_version=PRIMARY_B0))
        def valid(t):
            x=np.asarray(t,float)
            if x.shape!=t0.shape or not np.isfinite(x).all():raise ValueError('invalid theta')
            if np.any(x<bounds[:,0]) or np.any(x>bounds[:,1]):raise ValueError('theta outside original bounds')
            return x
        def evaluate(t,stage,order=192,optimizer=False):
            nonlocal calls,optcalls,aggregate
            clockguard();x=valid(t)
            if calls>=cfg['max_per_model_fullsample_evaluations'] or aggregate>=cfg['max_fullsample_evaluations_aggregate']:
                raise DriverBudgetLimit('hard full-sample count cap')
            if optimizer and optcalls>=cfg['max_optimizer_calls_per_model']:
                raise DriverBudgetLimit('optimizer count cap; validation reserve retained')
            calls+=1;aggregate+=1
            if optimizer:optcalls+=1
            ts=time.monotonic();record=dict(label=LABEL,model=model,likelihood_version=PRIMARY_B0,stage=stage,
                call=calls,aggregate_call=aggregate,optimizer_call=optcalls,quadrature_order=order,
                theta_sha256=hashlib.sha256(x.astype('<f8').tobytes()).hexdigest(),theta=x.tolist(),
                hard_bound_indices=np.flatnonzero((x==bounds[:,0])|(x==bounds[:,1])).tolist())
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    parts=mle.evaluate_loglike_components(x,galaxies,model,likelihood_version=PRIMARY_B0,order=order,tick=clockguard)
                warning_list=[dict(category=w.category.__name__,message=str(w.message)) for w in caught]
                warnings_seen.extend(warning_list)
                if not np.isfinite(parts['logL']):raise FloatingPointError('nonfinite objective')
                record.update(status='COMPLETE',negative_logL=-parts['logL'],galaxy_log_marginals=parts['galaxy_log_marginals'],
                    density_nodes=sum(z['density_nodes'] for z in parts['records']),warnings=warning_list)
            except Exception as exc:
                record.update(status='FAILED',error=repr(exc));exceptions.append(dict(stage=stage,error=repr(exc)))
                raise
            finally:
                record.update(seconds=time.monotonic()-ts,elapsed_seconds=time.monotonic()-started)
                with logs.open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
            if optimizer and (optcalls==1 or optcalls%20==0):print('OBJECTIVE',model,optcalls,'wall',round(time.monotonic()-started,2),flush=True)
            return parts
        def precision(t,stage):
            fine=evaluate(t,stage+'_fine');coarse=evaluate(t,stage+'_coarse',96);repeat=evaluate(t,stage+'_repeat')
            delta=np.array(fine['galaxy_log_marginals'])-coarse['galaxy_log_marginals']
            rep=np.max(np.abs(np.array(fine['galaxy_log_marginals'])-repeat['galaxy_log_marginals']))
            r=dict(stage=stage,max_abs_refinement=float(abs(delta).max()),sum_abs_refinement=float(abs(delta).sum()),
                signed_refinement=float(delta.sum()),repeat_max_abs=float(rep),
                sum_decomposition_error=abs(float(np.sum(fine['galaxy_log_marginals']))-fine['logL']))
            r['pass']=bool(r['max_abs_refinement']<=protocol['parity']['refinement_pergal'] and r['sum_abs_refinement']<=protocol['parity']['refinement_sample_sumabs'] and rep<=protocol['parity']['repeat_abs'])
            validation.append(r)
            if not r['pass']:raise ArithmeticError('frozen precision gate failed: '+stage)
        terminal=t0.copy()
        try:
            precision(t0,'initial')
            indices=[L['i_eps'],L['i_lq'],L['i_lnsig']] if model=='ptq-screen' else [L['sl_U'].start,L['sl_lM'].start,L['i_lnsig']]
            for idx in indices:
                t=t0.copy();step=1e-6*(bounds[idx,1]-bounds[idx,0]);direction=1 if t[idx]+step<=bounds[idx,1] else -1
                t[idx]+=direction*step;r=evaluate(t,'fixed_perturbation_'+str(idx))
                trials.append(dict(index=idx,step=direction*step,negative_logL=-r['logL']))
            def objective(t):return -evaluate(t,'optimizer',optimizer=True)['logL']
            def callback(t):
                accepted.append(np.asarray(t,float).copy())
                if len(accepted)>cfg['max_accepted_iterations_per_run']:raise RuntimeError('accepted iteration cap violated')
            options=dict(cfg['options'],maxiter=cfg['maxiter'],maxls=cfg['maxls'],maxfun=cfg['max_optimizer_calls_per_model'])
            try:
                result=minimize(objective,t0,method='L-BFGS-B',bounds=bounds,jac=None,callback=callback,options=options)
                terminal=np.array(result.x,copy=True)
                scipy_record=dict(success=bool(result.success),status=int(result.status),message=str(result.message),nit=int(result.nit),nfev=int(result.nfev),njev=int(result.njev))
                status='PASS' if result.nit>=1 else 'NO_ACCEPTED_ADVANCE'
            except DriverBudgetLimit as exc:
                status='DRIVER_SMOKE_BUDGET_LIMIT';exceptions.append(dict(stage='optimizer',error=repr(exc)))
                terminal=accepted[-1] if accepted else t0.copy()
            precision(terminal,'terminal')
        except DriverBudgetLimit as exc:
            status='DRIVER_SMOKE_BUDGET_LIMIT';exceptions.append(dict(stage='outer',error=repr(exc)))
        except Exception as exc:
            status='FAIL';exceptions.append(dict(stage='outer',error=repr(exc)))
        p=_unpack(valid(terminal),L)
        decoded={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in p.items()}
        summary=dict(model=model,likelihood_version=PRIMARY_B0,status=status,fullsample_evaluations=calls,
            optimizer_evaluations=optcalls,accepted_iterations=len(accepted),wall_seconds=time.monotonic()-modelstart,
            parameter_count=L['k'],nominal_gradient_fullsample_cost=L['k']+1,scipy=scipy_record,
            precision=validation,perturbations=trials,exceptions=exceptions,warnings=warnings_seen,
            terminal_semantics='last accepted callback or returned x; original start if no accepted step; never scientific MLE',
            terminal_theta=terminal.tolist(),initial_theta=t0.tolist(),decoded_terminal=decoded,
            accepted_vectors=[x.tolist() for x in accepted])
        write_json(out/(model.replace('-','_')+'_result.json'),summary);results.append(summary)
        print('DRIVER_RESULT',model,status,'nfev',optcalls,'nit',len(accepted),flush=True)
        if status=='FAIL':break
    final=dict(results=results,total_fullsample_evaluations=aggregate,total_wall_seconds=time.monotonic()-started,
        protocol_limits=cfg,optimization_campaign=False)
    write_json(out/'SMOKE_SUMMARY.json',final)
    return final
