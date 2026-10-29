"""Seal newR2 outputs; only read-only Git/source inspection. No numerical calls."""
from pathlib import Path
import json,hashlib,subprocess,difflib,ast
B=Path(__file__).resolve().parent;I=B.parents[2];R=I.parent;D=R/'docs/frontiers_major_revision_2026_10/rev26_p1f_r2_driver_smoke';A=B.parent/'production_engineering_r1'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def ld(p):return json.loads(p.read_text())
def js(p,r):
 if p.exists():raise FileExistsError(p)
 p.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
def git(p,a):return subprocess.check_output(['git','-C',str(p)]+a,text=True)
def main():
 before=ld(B/'SOURCE_HASHES_BEFORE.json');bad=[p for p,h in before.items() if not Path(p).is_file() or sha(Path(p))!=h];assert not bad,bad
 for manifest in ['TESTED_CODE_HASHES.json','VERIFIED_INPUT_HASHES.json']:
  assert all(Path(p).is_file() and sha(Path(p))==h for p,h in ld(B/manifest).items()),manifest
 receipt=ld(B/'VERIFIED_DEPENDENCIES.json')
 assert all(sha(B/'candidate/src/ptquat'/p)==h for p,h in receipt['source_hashes'].items())
 patch=[];changes=[]
 for p in sorted((B/'candidate').rglob('*.py')):
  rel=p.relative_to(B/'candidate');old=A/'candidate'/rel;s=old.read_text() if old.is_file() else '';new=p.read_text();ast.parse(new)
  if s!=new:
   patch.extend(difflib.unified_diff(s.splitlines(True),new.splitlines(True),fromfile='a/'+str(rel) if old.is_file() else '/dev/null',tofile='b/'+str(rel)));changes.append(dict(relative=str(rel),old_sha256=sha(old) if old.is_file() else None,new_sha256=sha(p)))
 assert {r['relative'] for r in changes}=={'src/ptquat/repaired_smoke_driver.py','scripts/run_repaired_gradient_smoke.py'}
 (B/'integration_from_R1.diff').write_text(''.join(patch));js(B/'INTEGRATION_FILES.json',changes)
 after={n:{k:git(p,v) for k,v in dict(root=['rev-parse','--show-toplevel'],HEAD=['rev-parse','HEAD'],status=['status','--porcelain=v1'],worktrees=['worktree','list','--porcelain']).items()} for n,p in [('outer',R),('implementation',I)]};authority=ld(B/'AUTHORITY.json')
 for n in after:
  assert after[n]['HEAD']==authority[n]['HEAD'];expected=[s for s in after[n]['status'].splitlines() if s!='?? docs/frontiers_major_revision_2026_10/rev26_p1f_r2_driver_smoke/'];assert expected==authority[n]['status'].splitlines(),n
 js(B/'GIT_AFTER.json',after);js(B/'PRESERVATION_VALIDATION.json',dict(status='PASS',baseline_files=len(before),changed=[],tested_code_hashes='PASS',input_hashes='PASS',receipt_hashes='PASS',new_source_only=[r['relative'] for r in changes],canonical_installation=False))
 a=ld(B/'analysis_v1/VALIDATED_RESULTS.json');budget=a['total_budget']
 js(B/'FAILURE_MANIFEST.json',dict(label='ENGINEERING_SMOKE_ONLY — NOT CORRECTED MLE',actual_optimizer_calls=2,actual_optimizer_run_attempts=2,additional_starts=0,retries=0,actual_runtime_exceptions=[],numerical_source_or_budget_failures=[],scipy_iteration_limit_stops=[dict(model=r['model'],status=r['scipy_status'],message=r['scipy_message'],accepted_iterations=1,stationarity_established=False) for r in a['models']],rejected_linesearch_trials={r['model']:r['rejected_linesearch_trials'] for r in a['models']},rejected_trials_are_not_runtime_exceptions=True,all_attempted_marginal_calls_completed=budget['actual_galaxy_calls']==budget['completed_galaxy_calls'],warnings=dict(events=a['warning_events'],keys=a['warning_keys'],classification='UNRESOLVED_WARNING_ROOT_CAUSE',numerical_corruption_confirmed=False,global_suppression=False),synthetic_unit_controls='controls_v1/v2 clearly marked; zeroactualoptimizer/likelihoodcalls',prior_R1_metadata_gap='exact failedNFWbenchmark wall/warningcounts remainUNKNOWN; nofabrication'))
 js(B/'SOURCE_MANIFEST.json',dict(repositories=after,baseline='SOURCE_HASHES_BEFORE.json',baseline_files=len(before),inputs='INPUT_MANIFEST.json',verified_inputs='VERIFIED_INPUT_HASHES.json',tested_code='TESTED_CODE_HASHES.json',source_identity='sameR1 scientific+gradientbytes; uncommittedisolatedcopy; SHAauthoritative',source_changes='INTEGRATION_FILES.json',target='REV26-P1E-UNIFORM-EXPLICIT-B0-PHYSICAL-v1',sample=dict(galaxies=91,points=1782),quadrature=dict(method='P1E segmentedquantile',fine_perpiece=192,coarse_perpiece=96,knots_standard_z=[-6,-4,-2,0,2,4,6],domain='D>0;0<i<pi/2; physicalnormalizedkernels',B0='nominal-coordinate sigmafloor'),dependency_receipt='VERIFIED_DEPENDENCIES.json',workweights='GALAXY_WORK_WEIGHTS.json',environment=['ENVIRONMENT.json','environment_freeze.txt'],preflight='execution_v1/PREFLIGHT_GATE.json',reused_parity='PARITY_REUSE.json'))
 js(B/'EXECUTION_INDEX.json',dict(label='ENGINEERING_SMOKE_ONLY — NOT CORRECTED MLE',actual_entrypoint='candidate/scripts/run_repaired_gradient_smoke.py',actual_command='PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 /tmp/rev26-p1a-env/bin/python -u <newCLI> --bundle <thisroot>',stages=[dict(stage='bootstrap source/data/previousmanifest verification',status='PASS',record='AUTHORITY.json/INPUT_MANIFEST.json'),dict(stage='unitcontrols beforepostwallguard',status='PASS_SYNTHETIC_ONLY',log='unit_execution.log'),dict(stage='unitcontrols finalfrozen source',status='PASS_SYNTHETIC_ONLY',log='unit_execution_v2.log'),dict(stage='actualinitialpreflight+twooptimizers+terminalprecision',status='PASS',log='ENGINEERING_SMOKE_ONLY_execution.log',records='execution_v1',actual_galaxy_calls=budget['actual_galaxy_calls'],equivalents=budget['charged_arithmetic_proxy_equivalents'],wall_seconds=budget['elapsed_wall_seconds']),dict(stage='readonly postprocessing',status='PASS',log='ENGINEERING_SMOKE_ONLY_analysis.log',records='analysis_v1'),dict(stage='independentreview',status='COMPLETE_WITH_LIMITATIONS',records=['INDEPENDENT_REVIEW_RAW.md','INDEPENDENT_REVIEW_NUMERICAL_ADDENDUM.md'])],no_extra_optimizer_or_numerical_calls=True))
 required=['00_EXECUTIVE_VERDICT.md','01_DRIVER_INTEGRATION_AND_CALL_GRAPH.md','02_PREFLIGHT_PARITY.md','03_FROZEN_SMOKE_PROTOCOL.md','04_PTQ_SCREEN_SMOKE.md','05_NFW_SMOKE.md','06_ACCEPTED_STEP_AND_PRECISION_AUDIT.md','07_WORK_AND_WARNING_TELEMETRY.md','08_INDEPENDENT_REVIEW.md','09_SCIENTIFIC_CAMPAIGN_READINESS.md','10_NEXT_EXECUTION_PROMPT.md','SOURCE_PRESERVATION.md'];assert all((D/p).is_file() for p in required)
 js(B/'MANIFEST.json',dict(task='REV26-P1F-R2',label='ENGINEERING_SMOKE_ONLY — NOT CORRECTED MLE',verdicts=ld(B/'DECISION.json')['verdicts'],numerical_root=str(B),reports_root=str(D),reports=required,isolated_source='candidate',patch='integration_from_R1.diff',source_manifest='SOURCE_MANIFEST.json',failure_manifest='FAILURE_MANIFEST.json',checksums='SHA256SUMS',execution='EXECUTION_INDEX.json',actual_optimizer_assets='execution_v1',synthetic_only_assets=['controls_v1','controls_v2'],verified_results='analysis_v1/VALIDATED_RESULTS.json',protocol='SMOKE_PROTOCOL.json',protocol_sha256=sha(B/'SMOKE_PROTOCOL.json'),inactive_next_plan='INACTIVE_P1G_PLAN.json',scientific_fitting_campaign_executed=False,canonical_installed=False))
 js(D/'ARTIFACT_MANIFEST.json',dict(task='REV26-P1F-R2',verdicts=ld(B/'DECISION.json')['verdicts'],numerical_manifest=str(B/'MANIFEST.json'),numerical_checksums=str(B/'SHA256SUMS'),reports=required,source_preserved=len(before),next_prompt='10_NEXT_EXECUTION_PROMPT.md',human_authorization_required_for_P1G=True))
 # Markpostprocessing console log withoutaltering anyexistingcontent.
 with (B/'ENGINEERING_SMOKE_ONLY_analysis.log').open('a') as f:f.write('\nENGINEERING_SMOKE_ONLY — NOT CORRECTED MLE\n')
 for root in [B,D]:
  sums=root/'SHA256SUMS';assert not sums.exists();files=sorted(p for p in root.rglob('*') if p.is_file() and p!=sums);sums.write_text(''.join(sha(p)+'  '+str(p.relative_to(root))+'\n' for p in files))
  for line in sums.read_text().splitlines():digest,name=line.split('  ',1);assert sha(root/name)==digest,name
 print(json.dumps(dict(status='SEALED_VERIFIED',baseline_files=len(before),numerical_SHA256SUMS=sha(B/'SHA256SUMS'),reports_SHA256SUMS=sha(D/'SHA256SUMS'),driver_patch_sha256=sha(B/'integration_from_R1.diff'),protocol_sha256=sha(B/'SMOKE_PROTOCOL.json'),verdicts=ld(B/'DECISION.json')['verdicts']),indent=2))
if __name__=='__main__':main()
