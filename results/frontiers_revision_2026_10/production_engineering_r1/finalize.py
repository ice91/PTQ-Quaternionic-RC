"""Seal only newR1 evidence. Nooptimizer/Gitmutation."""
from pathlib import Path
import json,hashlib,subprocess,ast
B=Path(__file__).resolve().parent;I=B.parents[2];R=I.parent;D=R/'docs/frontiers_major_revision_2026_10/rev26_p1f_r1_engineering';F=B.parent/'production_smoke'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def ld(p):return json.loads(p.read_text())
def js(p,r):
 if p.exists():raise FileExistsError(p)
 p.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
def git(p,args):return subprocess.check_output(['git','-C',str(p)]+args,text=True)
def main():
 before=ld(B/'SOURCE_HASHES_BEFORE.json');changed=[p for p,h in before.items() if not Path(p).is_file() or sha(Path(p))!=h];assert not changed,changed
 tested=ld(B/'CANDIDATE_TESTED_HASHES.json');assert all(sha(Path(p))==h for p,h in tested.items())
 current=ld(B/'dependency_v1/CURRENT_VERIFIED_DEPENDENCIES.json')
 assert all(sha(B/'candidate/src/ptquat'/p)==h for p,h in current['source_hashes'].items())
 assert (B/'candidate/src/ptquat/models.py').read_bytes()==(F/'candidate/src/ptquat/models.py').read_bytes()
 after={n:{k:git(p,a) for k,a in dict(root=['rev-parse','--show-toplevel'],HEAD=['rev-parse','HEAD'],status=['status','--porcelain=v1'],worktrees=['worktree','list','--porcelain']).items()} for n,p in [('outer',R),('implementation',I)]}
 authority=ld(B/'AUTHORITY.json')
 for n in after:
  assert authority[n]['HEAD']==after[n]['HEAD']
  clean=[s for s in after[n]['status'].splitlines() if s!='?? docs/frontiers_major_revision_2026_10/rev26_p1f_r1_engineering/'];assert clean==authority[n]['status'].splitlines(),(n,clean,authority[n])
 js(B/'GIT_AFTER.json',after);js(B/'PRESERVATION_VALIDATION.json',dict(status='PASS',baseline_files=len(before),changed=changed,candidate_tested_hashes='PASS',current_receipt_hashes='PASS',original_model_source_bytewise='UNCHANGED',canonical_installed=False,optimizer_invocations=0))
 js(B/'FAILURE_MANIFEST.json',dict(actual_scientific_likelihood_failures=[],unexplained_nonfinite_results=[],expected_contract_negative_controls=len(ld(B/'contracts_v2/TESTS.json')['cases']),directional_failures=0,dependency_failures=0,regression_failures=0,
   engineering_incidents=[dict(incident='benchmark_v1 signed-step harness assertion',type='COMPARATOR_HARNESS_FAILURE_NOT_GRADIENT',models=dict(ptq_screen='successful455gal/5eq preserved and reused',nfw1p='complete364gal/4eq beforeassert; v2correctcomparisonpassed'),artifacts=['benchmark_execution.log','run_benchmark_executed_v1.py.txt','benchmark_v1/ptq_screen_complete_gradient.json','ARCHIVED_BACKWARD_STENCIL_PROOF.json'],missing=['failedNFWexactwall','v1percategorywarningtelemetry','failedNFWaggregatewarningcount'],disposition='UNKNOWN retained; no inventedmeasurements; currentharnessfinallypersistsfuturefailures',old_tolerances_changed=False)],
   metadata_only_repairs=['binddependencygradientmodulehash withunchangedledger/step/sampleAST','B0-onlygradient verification','disablepubliclegacyradius withallotherFGfuncASTidentical; olddirectionexecution snapshotpreserved'],
   warning_classification='UNRESOLVED_WARNING_ROOT_CAUSE',numerical_corruption_confirmed=False,raw_P1F_eventlogs='UNCHANGED',future_failure_control='TELEMETRY_EXCEPTION_CONTROL.json',optimizer_executed=False))
 js(B/'SOURCE_MANIFEST.json',dict(repositories=after,inputs='INPUT_MANIFEST.json',baseline='SOURCE_HASHES_BEFORE.json',frozen_contract='PREDECLARED_CONTRACT.json',direction_protocol='DIRECTION_PROTOCOL.json',source_changes='INTEGRATION_FILES.json',candidate_hashes='CANDIDATE_TESTED_HASHES.json',scientific_model_laws='bytewiseequal P1F',likelihood='REV26-P1E-UNIFORM-EXPLICIT-B0-PHYSICAL-v1',source_identity='uncommitted isolatedcopy; baselineHEAD+SHA256 authoritative, HEADalone insufficient',environment=['parity_v1/environment.json','parity_v1/environment_freeze.txt'],receipt_chain=['dependency_v1/VERIFIED_DEPENDENCIES.json','dependency_v1/SEALED_VERIFIED_DEPENDENCIES.json','dependency_v1/CURRENT_VERIFIED_DEPENDENCIES.json'],snapshot_provenance=['DEPENDENCY_PRE_REVIEW_LOGIC_HASHES.json','LEGACY_CLOSURE_AST_PROOF.json','fit_global_before_legacy_closure.py.txt','run_directions_executed_v1.py.txt','run_benchmark_executed_v1.py.txt','run_benchmark_executed_v2.py.txt']))
 js(B/'EXECUTION_INDEX.json',dict(optimizer_calls=0,single_thread_environment=dict(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',OMP_NUM_THREADS='1'),python='/tmp/rev26-p1a-env/bin/python',stages=[dict(stage='bootstrap source/previouschecksums',status='PASS',record='AUTHORITY.json/INPUT_MANIFEST.json/SOURCE_HASHES_BEFORE.json'),dict(stage='contractnegativecontrols',status='PASS',logs=['contracts_execution.log','contracts_execution_v2.log']),dict(stage='full728coarse dependency',status='PASS',log='dependency_execution.log',actual_galaxy_evaluations=67613),dict(stage='7directions',status='PASS',log='directions_execution.log',equivalents=64.26640124933355,wall_seconds=88.276265667),dict(stage='legacyAPIclosure',status='PASS',record='LEGACY_CLOSURE_AST_PROOF.json'),dict(stage='fullgradientbenchmarkv1',status='HARNESS_FAILURE_AFTER_COMPLETENFW',log='benchmark_execution.log',actual_galaxy_evaluations=819,metadata_missing=True),dict(stage='NFWbenchmarkv2',status='PASS',log='benchmark_execution_v2.log',actual_galaxy_evaluations=364),dict(stage='fullregression',status='PASS',log='regression_execution.log'),dict(stage='warningdirect8fixedpoints',status='PASS',log='warning_audit_execution.log'),dict(stage='independentreview',status='COMPLETE_WITH_PROVENANCE_CONDITION',records=['INDEPENDENT_REVIEW_RAW.md','INDEPENDENT_REVIEW_NUMERICAL_ADDENDUM.md'])],all_attempted_benchmark_equivalents=13,no_optimizer_or_dense_refit=True))
 required=['00_EXECUTIVE_VERDICT.md','01_EVALUATOR_AND_BOUNDS_CONTRACT.md','02_PARAMETER_DEPENDENCY_LEDGER.md','03_DEPENDENCY_AWARE_GRADIENT.md','04_FIXED_DIRECTION_PRECISION.md','05_WARNING_TELEMETRY.md','06_FULL_PARITY_RESULTS.md','07_COST_AND_SPEEDUP.md','08_INDEPENDENT_REVIEW.md','09_NEXT_SMOKE_AUTHORIZATION_PACKET.md','10_NEXT_EXECUTION_PROMPT.md','SOURCE_PRESERVATION.md'];assert all((D/p).is_file() for p in required)
 # Directional output precision records must remain complete even afterunrelatedAPIclosure.
 import csv
 for path,n in [('parity_v1/native_parity_all2548.csv',2548),('parity_v1/native_matched_all728.csv',728),('parity_v1/neighborhood_all144.csv',144),('directions_v1/all28_direction_comparisons.csv',28),('dependency_v1/all728_local_full_recomputations.csv',728)]:
  with (B/path).open() as f:assert len(list(csv.DictReader(f)))==n,path
 js(B/'MANIFEST.json',dict(task='REV26-P1F-R1',verdicts=ld(B/'DECISION.json')['verdicts'],numerical_root=str(B),docs_root=str(D),reports=required,isolated_candidate='candidate',patch='integration_from_P1F.diff',source_manifest='SOURCE_MANIFEST.json',failure_manifest='FAILURE_MANIFEST.json',execution='EXECUTION_INDEX.json',checksums='SHA256SUMS',dependency_ledger='PARAMETER_DEPENDENCY_LEDGER.json',parameter_count=742,local_parameter_count=728,native_models=7,full_parity='parity_v1/PARITY_GATE.json',directions='directions_v1/DIRECTION_GATE.json',complete_gradient='BENCHMARK_SUMMARY.json',inactive_next_smoke='NEXT_SMOKE_PROPOSAL.json',optimizer_calls=0,scientific_fitting=False,canonical_installation=False))
 js(D/'ARTIFACT_MANIFEST.json',dict(task='REV26-P1F-R1',numerical_manifest=str(B/'MANIFEST.json'),numerical_checksums=str(B/'SHA256SUMS'),reports=required,verdicts=ld(B/'DECISION.json')['verdicts'],source_preservation='2114preexisting files unchanged',next_prompt='10_NEXT_EXECUTION_PROMPT.md'))
 for root in [B,D]:
  sums=root/'SHA256SUMS';assert not sums.exists();files=sorted(p for p in root.rglob('*') if p.is_file() and p!=sums);sums.write_text(''.join(sha(p)+'  '+str(p.relative_to(root))+'\n' for p in files))
  for line in sums.read_text().splitlines():digest,name=line.split('  ',1);assert sha(root/name)==digest,name
 print(json.dumps(dict(status='SEALED_VERIFIED',baseline_files=len(before),numerical_manifest_sha256=sha(B/'SHA256SUMS'),docs_manifest_sha256=sha(D/'SHA256SUMS'),patch_sha256=sha(B/'integration_from_P1F.diff'),verdicts=ld(B/'DECISION.json')['verdicts']),indent=2))
if __name__=='__main__':main()
