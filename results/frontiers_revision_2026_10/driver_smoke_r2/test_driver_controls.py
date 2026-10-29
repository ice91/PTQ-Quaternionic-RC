"""Synthetic controls only. Never invokes SciPy optimizer or actual likelihood."""
from pathlib import Path
from types import SimpleNamespace
import sys,json,hashlib,time,warnings
import numpy as np
B=Path(__file__).resolve().parent;I=B.parents[2];A=B.parent/'production_engineering_r1';sys.path.insert(0,str(B/'candidate/src'))
from ptquat import repaired_smoke_driver as d
from ptquat.dependency_gradient import source_hashes
from ptquat.warning_telemetry import collect_warnings,WarningContext

def main():
 out=B/'controls_v2';out.mkdir(exist_ok=False);protocol=json.loads((B/'SMOKE_PROTOCOL.json').read_text());gs,vectors=d.load_inputs(B);collector=d.Telemetry();tests=[]
 def reject(name,fn,kind=Exception):
  try:fn()
  except kind:tests.append(dict(case=name,status='PASS_REJECTED'))
  else:raise AssertionError(name+' accepted')
 receipt=json.loads((B/'VERIFIED_DEPENDENCIES.json').read_text());assert source_hashes()==receipt['source_hashes']
 # Source identity reusesall prior parity; source filenames same acrossisolated copies.
 r1=json.loads((A/'CANDIDATE_TESTED_HASHES.json').read_text());same=[]
 for path,h in r1.items():
  rel=Path(path).relative_to(A/'candidate');p=B/'candidate'/rel;assert hashlib.sha256(p.read_bytes()).hexdigest()==h,p;same.append(str(rel))
 from ptquat import experiments as exp,fit_global as fg
 for name in exp.DISABLED_LEGACY_DIAGNOSTICS:reject('firewall/'+name,lambda:getattr(exp,name)(),ValueError)
 reject('radius_factorization',lambda:fg.log_likelihood_per_galaxy(),ValueError)
 # Exercise hardcounter guards withsynthetic accounting, noactualmarginal calls.
 lout=out/'synthetic_ledger';lout.mkdir();ledger=d.WorkLedger(B,lout,protocol,collector)
 ledger.begin('ptq-screen',vectors['ptq-screen'],'synthetic_control','objective',192)
 ledger.actual=protocol['budget']['max_actual_galaxy_calls'];reject('actualgal_cap',lambda:ledger.charge('ptq-screen',0,192,'synthetic'),d.BudgetLimit);ledger.actual=0
 ledger.equivalent=protocol['budget']['max_fullsample_equivalents'];reject('equivalent_cap',lambda:ledger.charge('ptq-screen',0,192,'synthetic'),d.BudgetLimit);ledger.equivalent=0
 ledger.start=time.monotonic()-1801;reject('wallcap',ledger.guard,d.BudgetLimit);ledger.start=time.monotonic()
 ledger.optimizer_batches['ptq-screen']=21;reject('21gradient_cap',lambda:ledger.begin('ptq-screen',vectors['ptq-screen'],'optimizer','gradient',192),d.BudgetLimit);ledger.close()
 # Explicitwarning/failuretelemetry retains countandpropagatesexception.
 c=d.Telemetry();c.active_native_hash='SYNTHETIC_NOT_HISTORICAL'
 try:
  with collect_warnings(c):
   with WarningContext('synthetic',d.PRIMARY_B0,dict(U=[1.],sigma=1.),0):
    for _ in range(17):warnings.warn('synthetic repeated control',RuntimeWarning)
    raise FloatingPointError('deliberate test, no real likelihood')
 except FloatingPointError:pass
 assert c.as_dict()['total_events']==17 and c.as_dict()['distinct_keys']==1 and len(c.exceptions)==1
 # Stub real-driver API to test jac=True/acceptedcallback paths WITHOUT anoptimizer.
 mockout=out/'synthetic_wiring';mockout.mkdir();rt=object.__new__(d.Runtime);rt.out=mockout;rt.protocol=protocol;rt.gs=gs;rt.gradient_records={};rt.results=[];rt.ledger=SimpleNamespace(start=time.monotonic(),equivalent=0.,checkpoint=lambda:None,guard=lambda:None)
 t0=vectors['ptq-screen'].copy();count=0
 def fake_gradient(theta,model,phase,order):
  nonlocal count
  count+=1;obj=1000. if np.array_equal(theta,t0) else 990.;r=dict(objective=obj,gradient_native=np.ones(len(theta)).tolist(),request_id=count,phase='optimizer');rt.gradient_records[(model,d.theta_hash(theta))]=r;return r
 rt.gradient=fake_gradient
 def fake_audit(theta,model,phase,record):
  return dict(fine=dict(objective=record['objective'],galaxy_log_marginals=[-record['objective']/91]*91),objective_precision=dict(sumabs_refinement=1e-8))
 rt.audit=fake_audit
 def optimizer_stub(fun,x0,method,jac,bounds,callback,options):
  assert method=='L-BFGS-B' and jac is True and options['maxiter']==1 and options['maxls']==20
  f,g=fun(x0);assert len(g)==len(x0);x=x0.copy();x[0]+=1e-6;f,g=fun(x);callback(x)
  return SimpleNamespace(x=x,fun=f,jac=g,success=False,status=1,message='SYNTHETIC_STUB_NO_OPTIMIZATION',nit=1,nfev=2,njev=2)
 original=d.minimize;d.minimize=optimizer_stub
 try:r=rt.optimize_once(t0,'ptq-screen',fake_audit(t0,'ptq-screen','initial',dict(objective=1000.)))
 finally:d.minimize=original
 assert r['status']=='ACCEPTED_AND_NUMERICALLY_VALIDATED' and len(r['accepted_callbacks'])==1 and r['accepted_callbacks'][0]['advanced'];tests.append(dict(case='synthetic_jacTrue_actual_callback_path_no_optimizer',status='PASS'))
 d.save(out/'CONTROLS.json',dict(status='PASS',tests=tests,warning_control=c.as_dict(),source_identity_files=same,synthetic_only=True,real_optimizer_invocations=0,real_galaxy_likelihood_evaluations=0))
 (B/'DRIVER_UNIT_GATE.json').write_text(json.dumps(dict(status='PASS',cases=len(tests),real_optimizer_invocations=0,source_identity_R1=same),indent=2)+'\n')
 # Bind allactual scientific inputs not justcode.
 paths=[I/'dataset/sparc_tidy.csv',B/'VERIFIED_DEPENDENCIES.json',B/'GALAXY_WORK_WEIGHTS.json',B/'SMOKE_PROTOCOL.json']+[I/'results/frontiers_revision_2026_10/mle_recovery/original_recovered/01_mle'/m.replace('-','_')/'theta_mle.npy' for m in protocol['models']]
 (B/'VERIFIED_INPUT_HASHES.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
 (B/'TESTED_CODE_HASHES.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (B/'candidate').rglob('*.py')},indent=2)+'\n')
 print('DRIVER UNIT PASS',len(tests),'syntheticonly; zerooptimizer zeroactualgal')
if __name__=='__main__':main()
