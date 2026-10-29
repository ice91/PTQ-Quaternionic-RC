from pathlib import Path
import os,json,hashlib,subprocess,ast,sys
import numpy as np,pandas as pd
import explicit_reference as ex
R=Path('/Users/rocky/ptq-mond');I=R/'PTQ-Quaternionic-RC';B=I/'results/frontiers_revision_2026_10/uniform_target_gate';D=R/'docs/frontiers_major_revision_2026_10/rev26_p1e_uniform_target';A=B/'acceptance_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def js(p,x):
 if p.exists():raise FileExistsError(p)
 p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
if (B/'SHA256SUMS').exists():raise FileExistsError('Finalized run exists')
# Direct negative-control guard tests; no density evaluations or new scientific cases.
negative=[]
original=ex.legendre
try:
 ex.legendre=lambda n:(np.array([-.5,.5]),np.array([-1.,3.]))
 try:ex.axis_nodes(11.123,1.01,0.,np.inf,'quantile',2)
 except ValueError:negative.append(dict(test='invalid quadrature weights',fail_closed=True))
 else:raise AssertionError('negative weights accepted')
finally:ex.legendre=original
bad=ex.laws.Galaxy('nonfinite_negative_control',np.array([1.]),np.array([np.nan]),np.array([2.]),np.array([1.]),np.zeros(1),np.zeros(1),10.,1.,1.,.1)
try:ex.integrate(bad,dict(model='baryon',U=np.ones(1),sigma=3.),0,'quantile',2)
except ValueError:negative.append(dict(test='nonfinite observed likelihood input',fail_closed=True))
else:raise AssertionError('nonfinite input accepted')
assert ex.NODE_EVALUATIONS==0
# Ensure dataframe columns use explicit['T'] where required, not accidental transpose.
parsed=ast.parse((B/'run_acceptance.py').read_text());forbidden_names={'prior','qdf','qlo','qhi','fdf','td','df','pilot','s','ss'}
assert not any(isinstance(n,ast.Attribute) and n.attr=='T' and isinstance(n.value,ast.Name) and n.value.id in forbidden_names for n in ast.walk(parsed))
js(B/'FAIL_CLOSED_NEGATIVE_CONTROLS.json',dict(tests=negative,extra_density_nodes=0,pandas_T_static_guard=True))
# All expected actual tests and outputs are present/finite before finalhash.
local=json.loads((B/'local_v1/LOCAL_GATE.json').read_text());summary=json.loads((A/'ACCEPTANCE_SUMMARY.json').read_text());validation=json.loads((B/'verification_v1/VALIDATION.json').read_text());assert local['status']=='PASS' and validation['status']=='PASS' and summary['explicit_numerical_acceptance'];assert local['code_sha256']==sha(B/'explicit_reference.py') and local['protocol_sha256']==sha(B/'PREDECLARED_PROTOCOL.json')
plan=json.loads((A/'EXECUTION_PLAN.json').read_text());assert plan['code_sha256']==sha(B/'explicit_reference.py') and plan['protocol_sha256']==sha(B/'PREDECLARED_PROTOCOL.json')
for name,n,keys in [('full_sample_all1274.csv',1274,['model','galaxy','floor']),('native_matched_all728.csv',728,['source','target','galaxy','floor']),('neighborhood_all144.csv',144,['model','galaxy','point','floor'])]:
 frame=pd.read_csv(A/name);assert len(frame)==n and not frame.duplicated(keys).any();assert np.isfinite(frame.select_dtypes(include='number').to_numpy()).all()
# Rejectoverwritechecks exercise onlyguards,neverrerun numerical comparisons.
guards=[]
for script in ['test_local.py','run_acceptance.py','verify_and_cost.py']:
 proc=subprocess.run(['/tmp/rev26-p1a-env/bin/python',str(B/script)],cwd=B,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','OMP_NUM_THREADS':'1'})
 assert proc.returncode!=0 and 'FileExistsError' in proc.stdout
 guards.append(dict(script=script,existing_outputs_refused=True,exitcode=proc.returncode,console=proc.stdout))
js(B/'OUTPUT_OVERWRITE_GUARDS.json',guards)
# Preserveallsources,includinginvalidP1Dv1incidentoutputs.
before=json.loads((B/'SOURCE_HASHES_BEFORE.json').read_text());changed=[];missing=[]
for s,h in before.items():
 p=Path(s)
 if not p.is_file():missing.append(s)
 elif sha(p)!=h:changed.append(s)
assert not changed and not missing,(changed,missing)
repos={}
for name,cwd in [('outer',R),('implementation',I)]:repos[name]={k:subprocess.check_output(['git']+v,cwd=cwd,text=True) for k,v in dict(root=['rev-parse','--show-toplevel'],HEAD=['rev-parse','HEAD'],status=['status','--porcelain=v1'],worktrees=['worktree','list','--porcelain']).items()}
prior=json.loads((B/'AUTHORITY.json').read_text())
for name in repos:
 for key in ['root','HEAD','worktrees']:assert repos[name][key]==prior[name][key]
js(B/'GIT_AFTER.json',repos)
js(B/'SOURCE_PRESERVATION_VALIDATION.json',dict(existing_files_checked=len(before),changed=changed,missing=missing,canonical_and_historical_theta_data_unchanged=True,previous_P1A_P1D_all_evidence_unchanged=True,failed_P1D_v1_preserved_and_not_used=True,reference_and_protocol_unchanged_since_tests=True,repo_HEAD_worktrees_unchanged=True,no_optimizer=True,new_output_roots=[str(B),str(D)]))
# Decisionsourcepreservationpendingfieldisnotrewrittensilently;explicitfinalvalidationseparate.
qfail=pd.read_csv(B/'UNSEGMENTED_QUANTILE_ACCEPTANCE_FAILURES.csv')
js(B/'FAILURE_AND_INCIDENT_MANIFEST.json',dict(pre_execution_incidents=[dict(source='run_acceptance_syntax_failure_v0.txt',log='acceptance_syntax_failure_v0.log',issue='missingwhitespace else199.999 SyntaxError before any scientificcomputation',resolution='whitespacefixed;statistical/numericalcontractunchanged')],numerical_policy_failures=dict(method='unsegmentedquantile512vs256',failed_rows=len(qfail),source='UNSEGMENTED_QUANTILE_ACCEPTANCE_FAILURES.csv',resolution='predeclaredglobal switch tosame-targetsegmentedquantile'),main_exceptions=json.loads((A/'exceptions.json').read_text()),selected_policy_pergal_acceptance_failures=0,neighborhood_failures=0,all_attempts_retained=True,independent_warning_note='NumPy matmulFPE warningslogged;alloutputsfinite,directdensitycomparisonpasses',prior_P1D_incident='preservedunchanged;onlypostprocessing_v2usedfordecomposition'))
(D/'SOURCE_PRESERVATION.md').write_text('''# Source preservation and execution authority

GateA在任何P1Enumericalcomparison前記錄1,957份既有project檔案SHA256。最後逐份核對0 changed、0 missing；包含canonical source、datasets、原theta、稿件/main2/v5/publicrelease與P1A–P1D evidence（含錯誤P1Dv1incidentfiles）。SOURCE_HASHES_BEFORE與SOURCE_PRESERVATION_VALIDATION保存完整path/hash與結果。排除.git內部、bytecode/cache/venv及本輪兩newroots，未把排除項當sciencepreservation證據。

Outerroot /Users/rocky/ptq-mond，HEAD3eb302666da7eccca19c5843f729e98126d68977；implementationroot /Users/rocky/ptq-mond/PTQ-Quaternionic-RC，HEAD5a8dca57f0fc3484387c643b51e22c7e10d024a8。各自root/HEAD/status/worktrees實測於AUTHORITY/GIT_AFTER；HEAD/worktrees一致，原dirtyfiles與untracked修訂資料全部保留。沒有pull/checkout/reset/stash/commit/push/tag/release。

P1A recovery、P1C reference、P1Dnumerical/docs SHA256SUMS起始逐項通過。P1Dderivedauthority僅postprocessing_v2；v1brokenNaNsummary不進科學結論。原P1Dparameter_points.json是獨立native-coordinate metadata（非invalidderivedsummary），只讀重用以重建66frozenpoints，hash逐點核對。沒有重做archive、P1Bcampaign或Symmetry分析。

newroots uniform_target_gate及rev26_p1e_uniform_target原先不存在，各runoutput拒絕overwrite。canonical source完全未改；只新增explicit_reference與tests/drivers/reports。spec與protocol先凍結，reference codehash在local與main後仍相同。唯一driver修復是pre-execution SyntaxError缺空白，failedsource/log留存；沒有numericoutput覆寫、契約／target變更或降低門檻。

原temporaryPython /tmp/rev26-p1a-env沿用，沒有productioninstall；environment/platform/pipfreeze存local_v1。PYTHONDONTWRITEBYTECODE=1及單BLASthread明示；source/inputhashes、全部attempt/runtime logs、failedpilotcases、finite/error驗證、負控制與outputoverwriteguards保存。七localgroups、1274fullrows、728mappings、144neighborhoodrows完整finiteunique。未進行MLE、MCMC、NFWstarts、line-search、profiles、CI、manuscript表格替換或任何public/journalwrite。

主積分5.978148224Bnode-density、約43秒／122MB；local約0.1M與獨立代理約9.8M小額外查核合計仍低於凍結6Bbudget。後續verification只讀artifacts及Gaussian-onlycostcomparison，沒有新增explicitintegration。獨立review未改檔、未被當proofauthority。

TARGET_DECISION中sourcepreservation欄記錄撰稿時finalvalidationpending，最終本檔及SOURCE_PRESERVATION_VALIDATION正式完成核驗；不靜默改寫已生成決策來源。唯一targetrecommendation維持UNIFORM_EXPLICIT_B0_READY_FOR_FIT_REVIEW。

SHA256SUMS分別涵蓋兩root新檔案除自身；docs/ASSET_MANIFEST.json列其他新資產絕對path/bytes/hash（排除自身與docs checksum避免循環）。finalchecksums逐項通過。真正productionintegration與fitting都未獲本輪執行，待作者另批。
''')
required=['00_EXECUTIVE_DECISION.md','01_FROZEN_TARGET_AND_NUMERICAL_PROTOCOL.md','02_UNIFORM_EXPLICIT_REFERENCE.md','03_NORMALIZATION_AND_QUADRATURE_TESTS.md','04_FULL_SAMPLE_FIXED_POINT_RESULTS.md','05_B0_B1_AND_BOUNDARY_SENSITIVITY.md','06_OBJECTIVE_COST_AND_STABILITY.md','07_INDEPENDENT_REVIEW.md','08_TARGET_DECISION_AND_LIMITATIONS.md','09_INACTIVE_PRODUCTION_AND_REFIT_PLAN.md','10_NEXT_EXECUTION_PROMPT.md','SOURCE_PRESERVATION.md'];assert all((D/n).is_file() for n in required)
for p in B.glob('*.py'):ast.parse(p.read_text())
(B/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(B))+'\n' for p in sorted(B.rglob('*')) if p.is_file() and p!=B/'SHA256SUMS'))
assets=[]
for root in [B,D]:
 for p in sorted(root.rglob('*')):
  if p.is_file() and p not in [D/'ASSET_MANIFEST.json',D/'SHA256SUMS']:assets.append(dict(absolute_path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
js(D/'ASSET_MANIFEST.json',dict(schema_version=1,excludes_self_and_docs_checksum=True,assets=assets))
(D/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(D))+'\n' for p in sorted(D.rglob('*')) if p.is_file() and p!=D/'SHA256SUMS'))
for root in [B,D]:
 for line in (root/'SHA256SUMS').read_text().splitlines():
  h,p=line.split('  ',1);assert sha(root/p)==h
for p in [D/'ASSET_MANIFEST.json',D/'SHA256SUMS',B/'SHA256SUMS']:print(str(p),sha(p))
print('existingfilespreserved',len(before),'newassets',len(assets))
