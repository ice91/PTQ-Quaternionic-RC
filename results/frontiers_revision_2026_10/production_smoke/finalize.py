"""Seal new evidence only. Read-only Git/source inspection."""
from pathlib import Path
import hashlib,json,subprocess,sys,platform
B=Path(__file__).resolve().parent;I=B.parents[2];P=I.parent;D=P/'docs/frontiers_major_revision_2026_10/rev26_p1f_production_smoke'
LABEL='ENGINEERING_SMOKE_ONLY — NOT CORRECTED MLE'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def js(p,x):
    if p.exists():raise FileExistsError(p)
    p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args],text=True)
def main():
 before=json.loads((B/'SOURCE_HASHES_BEFORE.json').read_text());assert all(Path(p).is_file() and sha(Path(p))==h for p,h in before.items())
 tested=json.loads((B/'CANDIDATE_TESTED_HASHES.json').read_text());assert all(sha(Path(p))==h for p,h in tested.items())
 after={}
 for name,root in [('outer',P),('implementation',I)]:after[name]={k:git(root,*args) for k,args in [('root',['rev-parse','--show-toplevel']),('HEAD',['rev-parse','HEAD']),('status',['status','--porcelain=v1']),('worktrees',['worktree','list','--porcelain'])]}
 authority=json.loads((B/'AUTHORITY.json').read_text())
 for name in after:
    assert after[name]['HEAD']==authority[name]['HEAD']
    expected=[s for s in after[name]['status'].splitlines() if s!='?? docs/frontiers_major_revision_2026_10/rev26_p1f_production_smoke/']
    assert expected==authority[name]['status'].splitlines(),(name,expected,authority[name]['status'])
 js(B/'GIT_AFTER.json',after)
 failures=dict(engineering_smoke_label=LABEL,numerical_unit_failures=[],parity_failures=[],nonfinite_objective_calls=[],
    expected_budget_exceptions=[dict(model=m,stage='optimizer',error='DriverBudgetLimit: optimizer count cap; validation reserve retained',optimizer_evaluations=190,accepted_iterations=0,retry=False) for m in ['ptq-screen','nfw1p']],
    assembly_incident=dict(stage='before any numeric test or optimizer',artifact='build_candidate_assembly_failure_v0.txt',cause='generated indentation string assembling independent numeric kernel; repaired in new candidate before testing',canonical_affected=False),
    reporting_incident=dict(stage='after all numerical work',artifacts=['report_generator_encoding_failure_v0.txt','report_generation_failure_v0.log'],description='Python3.9 source-reading error on long UTF8 report lines; triple quoted bodies rendered actual newlines; original bytes decodeUTF8; exact parser root cause unknown',scientific_outputs_affected=False),
    warnings=dict(categories=['RuntimeWarning: divide by zero encountered in matmul','RuntimeWarning: overflow encountered in matmul','RuntimeWarning: invalid value encountered in matmul'],count_per_model=1291386,total=2582772,root_cause='UNRESOLVED; warnings retained not suppressed',direct_full192_checks=182,max_direct_logL_delta=8.526512829121202e-13,scope='all91 original baryon fixedpoints bothfloors; not allfuturetheta'))
 js(B/'FAILURE_AND_EXCEPTION_MANIFEST.json',failures)
 js(B/'SOURCE_MANIFEST.json',dict(actual_repositories=after,authority='frozen P1E explicit target and recovered historical theta',input_manifest='INPUT_MANIFEST.json',baseline='SOURCE_HASHES_BEFORE.json',baseline_files=len(before),source_change_count=0,
    code_identity='candidate is uncommitted isolated copy of existing canonical dirty checkout; HEAD alone does not identify allsource; SHA256 authoritative',candidate_hashes='CANDIDATE_TESTED_HASHES.json',patch='integration.diff',independent_reference_imported_by_candidate=False,data_sample=dict(galaxies=91,points=1782,ordering='sorted name; original radius sorting/masks unchanged'),environment='parity_v1/environment.json and environment_freeze.txt',likelihood_B0='REV26-P1E-UNIFORM-EXPLICIT-B0-PHYSICAL-v1',likelihood_B1='REV26-P1E-UNIFORM-EXPLICIT-B1-PHYSICAL-v1'))
 js(B/'EXECUTION_INDEX.json',dict(stages=[dict(stage='source and P1E authority hashes',status='PASS',log='build_execution.log'),dict(stage='initial assembly incident',status='REPAIRED_BEFORE_TESTS',source='build_candidate_assembly_failure_v0.txt'),dict(stage='unit/fullfixedpoint/mapping/neighborhood/actualcallpaths',status='PASS',log='parity_execution_v1.log',records='parity_v1'),dict(stage='warning direct residual audit',status='PASS',log='warning_audit_execution.log'),dict(stage='real optimizer smoke',label=LABEL,status='BUDGET_LIMITED',log='ENGINEERING_SMOKE_ONLY_execution_v1.log',records='engineering_smoke_v1',fullsample_calls=398,wall_seconds=550.493800083),dict(stage='smoke read-only analysis',label=LABEL,status='COMPLETE',log='smoke_analysis_execution.log'),dict(stage='report initial failure',status='ENCODING_READ_FAILURE',log='report_generation_failure_v0.log'),dict(stage='report generation',status='COMPLETE',log='report_generation_v1.log'),dict(stage='independent review',status='COMPLETE_WITH_OBJECTIONS',records=['INDEPENDENT_REVIEW_RAW.md','INDEPENDENT_REVIEW_SMOKE_ADDENDUM.md'])],commands=dict(pre_smoke='PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 /tmp/rev26-p1a-env/bin/python run_parity.py',warningaudit='sameenvironment python audit_matmul_warnings.py',actual_driver='sameenvironment python candidate/scripts/run_uniform_explicit_smoke.py --bundle <this_new_bundle>; exact wrapper uses runpy and prints engineering label',postprocessing='python analyze_smoke.py; python check_preservation.py; python write_reports.py; python finalize.py'),no_optimizer_retries=True))
 required=['00_EXECUTIVE_VERDICT.md','01_INTEGRATION_DIFF_AND_CALL_GRAPH.md','02_FIXED_POINT_PARITY.md','03_LIKELIHOOD_VERSION_AND_DIAGNOSTIC_FIREWALL.md','04_FROZEN_SMOKE_PROTOCOL.md','05_REAL_DRIVER_SMOKE_RESULTS.md','06_PRECISION_AND_GRADIENT_AUDIT.md','07_RUNTIME_AND_FIT_BUDGET.md','08_INDEPENDENT_REVIEW.md','09_INACTIVE_SEVEN_MODEL_CAMPAIGN.md','10_NEXT_EXECUTION_PROMPT.md','SOURCE_PRESERVATION.md']
 assert all((D/n).is_file() for n in required)
 js(B/'MANIFEST.json',dict(task='REV26-P1F',label=LABEL,verdict=json.loads((B/'DECISION.json').read_text())['verdicts'],numerical_root=str(B),docs_root=str(D),source_manifest='SOURCE_MANIFEST.json',source_unchanged=len(before),canonical_installed=False,scientific_fit=False,isolated_changes=json.loads((B/'INTEGRATION_FILES.json').read_text()),reports=[str(D/n) for n in required],checksums=dict(numerical='SHA256SUMS',reports=str(D/'SHA256SUMS')),incidents='FAILURE_AND_EXCEPTION_MANIFEST.json',execution='EXECUTION_INDEX.json',independent_reviews=['INDEPENDENT_REVIEW_RAW.md','INDEPENDENT_REVIEW_SMOKE_ADDENDUM.md']))
 js(D/'ARTIFACT_MANIFEST.json',dict(task='REV26-P1F',numerical_manifest=str(B/'MANIFEST.json'),numerical_checksums=str(B/'SHA256SUMS'),reports=required,report_count=len(required),canonical_changes=0,historical_vector_changes=0,smoke_label=LABEL))
 for root in [B,D]:
    sums=root/'SHA256SUMS'
    if sums.exists():raise FileExistsError(sums)
    files=sorted(p for p in root.rglob('*') if p.is_file() and p!=sums)
    sums.write_text(''.join(sha(p)+'  '+str(p.relative_to(root))+'\n' for p in files))
    for line in sums.read_text().splitlines():
        digest,name=line.split('  ',1);assert sha(root/name)==digest,name
 print(json.dumps(dict(status='SEALED_AND_VERIFIED',baseline_files=len(before),outer_HEAD=after['outer']['HEAD'].strip(),implementation_HEAD=after['implementation']['HEAD'].strip(),numerical_SHA256SUMS=sha(B/'SHA256SUMS'),reports_SHA256SUMS=sha(D/'SHA256SUMS'),candidate_diff_sha256=sha(B/'integration.diff')),indent=2))
if __name__=='__main__':main()
