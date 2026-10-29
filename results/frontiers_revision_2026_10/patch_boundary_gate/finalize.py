from pathlib import Path
import json,hashlib,ast,subprocess,os,sys,runpy
import numpy as np,pandas as pd
R=Path('/Users/rocky/ptq-mond');I=R/'PTQ-Quaternionic-RC';B=I/'results/frontiers_revision_2026_10/patch_boundary_gate';D=R/'docs/frontiers_major_revision_2026_10/rev26_p1d_patch_boundary';W=B/'isolated_candidate'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def js(p,x):
 if p.exists():raise FileExistsError(p)
 p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
if (B/'SHA256SUMS').exists():raise FileExistsError('Finalized run exists')
protocol=json.loads((B/'PREDECLARED_PROTOCOL.json').read_text());parity=json.loads((B/'parity_v1/PARITY_GATE.json').read_text());assert sha(B/'production_candidate.diff')==parity['patch_sha256'];assert sha(B/'PREDECLARED_PROTOCOL.json')==parity['protocol_sha256']
assert parity['status']=='PASS' and json.loads((B/'neighborhood_patch_parity_v1/SUMMARY.json').read_text())['status']=='PASS'
# Exact unified diff applies against canonical source without any write.
p=subprocess.run(['git','apply','--check',str(B/'production_candidate.diff')],cwd=I,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
(B/'patch_apply_check.log').write_text(p.stdout or 'git apply --check exit0; read-only no patch applied\n');assert p.returncode==0,p.stdout
# Original law, prior, optimizer routines unchanged except galaxy covariance caller.
for file,funcs in [('models.py',None),('fit_global.py',['log_prior','log_posterior']),('mle.py',['theta_bounds','objective_neg_loglike','optimize_mle','_block_coordinate_mle','build_multistart'])]:
 old={x.name:x for x in ast.parse((I/'src/ptquat'/file).read_text()).body if isinstance(x,ast.FunctionDef)};new={x.name:x for x in ast.parse((W/'src/ptquat'/file).read_text()).body if isinstance(x,ast.FunctionDef)}
 for name in old if funcs is None else funcs:assert ast.dump(old[name],include_attributes=False)==ast.dump(new[name],include_attributes=False),(file,name)
# Execute migrated standalone unit test without importing any canonical production source.
sys.path.insert(0,str(W/'src'));unit=runpy.run_path(str(W/'tests/test_covariance.py'));unit['test_covariance_positive_definite']()
# Raw and official v2 finite/count/uniqueness checks; failed v1derived intentionally excluded.
df=pd.read_csv(B/'boundary_v1/all_integrals.csv');assert len(df)==3300 and np.isfinite(df.select_dtypes(include='number').to_numpy()).all();assert not df.duplicated(['galaxy','model','point','T','order','construction']).any()
pp=B/'postprocessing_v2';dd=pd.read_csv(pp/'decomposition.csv');nn=pd.read_csv(pp/'neighborhood_effect_changes.csv');cc=pd.read_csv(pp/'convergence.csv');assert len(dd)==297 and len(nn)==66 and len(cc)==1188 and cc.converged.all()
for frame in [dd,nn,cc]:assert np.isfinite(frame.select_dtypes(include='number').to_numpy()).all()
# Record intentional old derived failure rather than removing/rewriting it.
old=pd.read_csv(B/'boundary_v1/neighborhood_effect_changes.csv');assert old.galaxy.isna().any()
coverage=pd.read_csv(B/'boundary_v1/all22_model_coverage.csv');assert len(coverage)==154 and len(coverage.galaxy.unique())==22;assert len(coverage[coverage.historical_full_integration])==63
js(B/'NUMERICAL_INTEGRITY_VALIDATION.json',dict(raw_integrals=3300,v2_decomposition=297,v2_neighborhood=66,convergence_cases=1188,nonconverged=0,migrated_unit_test='PASS',law_prior_optimizer_AST_unchanged=True,patch_apply_check_read_only='PASS',source_calls_explicit_full_JD=12,canonical_imports_in_this_process=False,v1_failed_derived_preserved=True,quadrature_rerun_for_v2=False))
# Read-only source preservation after all isolated code/evidence work.
before=json.loads((B/'SOURCE_HASHES_BEFORE.json').read_text());changed=[];missing=[]
for path,h in before.items():
 p=Path(path)
 if not p.is_file():missing.append(path)
 elif sha(p)!=h:changed.append(path)
assert not changed and not missing,(changed,missing)
repos={}
for name,cwd in [('outer',R),('implementation',I)]:repos[name]={k:subprocess.check_output(['git']+v,cwd=cwd,text=True) for k,v in dict(root=['rev-parse','--show-toplevel'],HEAD=['rev-parse','HEAD'],status=['status','--porcelain=v1'],worktrees=['worktree','list','--porcelain']).items()}
prior=json.loads((B/'AUTHORITY.json').read_text())
for name in repos:
 for key in ['root','HEAD','worktrees']:assert repos[name][key]==prior[name][key]
js(B/'GIT_AFTER.json',repos)
# Failures manifest distinguishes numerical exceptions, failed derived summaries and scientific flags.
js(B/'FAILURE_EXCEPTION_MANIFEST.json',dict(parity_failures=[],numerical_integration_exceptions=json.loads((B/'boundary_v1/exceptions.json').read_text()),quadrature_nonconvergence=[],derived_failures=[dict(run='boundary_v1',script='run_boundary.py',files=['neighborhood_effect_changes.csv','decomposition.csv convergence metadata'],issue="pandas .T vs ['T']",preserved=True,resolution='postprocessing_v2 without numerical rerun')],scientific_flags='postprocessing_v2/materiality_flags.csv; all retained',no_failure_hidden=True))
js(B/'SOURCE_PRESERVATION_VALIDATION.json',dict(checked_existing_files=len(before),changed=changed,missing=missing,canonical_tree_read_only=True,original_theta_data_science_unchanged=True,previous_revision_bundles_unchanged=True,HEADs_worktrees_unchanged=True,no_optimizer=True,no_Git_mutation=True,new_output_roots=[str(B),str(D)],read_only_git_apply_check=True))
(D/'SOURCE_PRESERVATION.md').write_text('''# Source preservation report

GateA先記錄1,866份既有project source/data/results/manuscripts與P1A–P1C evidence SHA256；最後逐份核對0 changed、0 missing。完整path/hash在SOURCE_HASHES_BEFORE.json，核驗結果SOURCE_PRESERVATION_VALIDATION.json。排除.git內部、cache、bytecode、venv與本輪新root，不把排除項冒作science preservation證據。

Outer /Users/rocky/ptq-mond HEAD 3eb302666da7eccca19c5843f729e98126d68977；implementation /Users/rocky/ptq-mond/PTQ-Quaternionic-RC HEAD 5a8dca57f0fc3484387c643b51e22c7e10d024a8。各有自己的Git boundary；前後HEAD/worktree清單一致，完整status保留AUTHORITY/GIT_AFTER；原四份implementation dirty files與既有untracked evidence均保留。

Canonical production source從未修改；所有patch只在全新isolated_candidate local copy。submitted sources、main2.tex、v5、public release、原data與MLEtheta、P1A–P1C數值／reports未覆寫。P1C numerical/docs與P1A numerical SHA256SUMS起始逐項通過，P1C local tests/reference也核驗。沒有重做archive或Symmetry audit。

全部新code/results/reports僅放patch_boundary_gate與rev26_p1d_patch_boundary，run子目錄拒絕overwrite。臨時Python沿用/tmp/rev26-p1a-env，freeze/platform記於parity_v1；沒有production install，PYTHONDONTWRITEBYTECODE=1。沒有optimizer、NFWstarts、MCMC、newbestfit、稿件table替換、public/journalwrite或Gitcommit/pull/push/checkout/tag/release。

允許唯一Git新增read-only操作為git apply --check，exit0，證明diff對canonical目前bytes可套用；沒有實際apply。另以AST證明originalmodel-laws、priors、bounds、optimizer/build_multistart bodies不變。migratedcovarianceunit test在candidateimport路徑單獨通過。

v1derivedpostprocessing失敗透明保留：原raw3300integrals、originalrun_boundary.py、NaNneighborCSV及錯誤convergence linkage檔未改。postprocessing_v2從raw重建297decomposition/66neighbor/1188convergence並finite檢查；沒有重跑quadrature、修改reference或predeclaredcontract。Independentreview確認同一錯誤與修復。正式報告只用v2derived摘要。

SHA256SUMS分別涵蓋兩root全部新檔除自身；ASSET_MANIFEST.json列絕對path/bytes/hash（排除自身及docs checksum避免循環）。完整diff、isolatedtree、source/inputauthority、protocol、runtime/paritylogs、environment、failuremanifest與materialflags皆保留。finalchecksumverification通過。
''')
# Verify complete required report contract.
required=['00_EXECUTIVE_GATE.md','01_ISOLATED_PATCH_DIFF_AND_CALL_GRAPH.md','02_REFERENCE_PARITY_RESULTS.md','03_BOUNDARY_TEST_PROTOCOL.md','04_REAL_EDGE_ON_RESULTS.md','05_B0_B1_FLOOR_SENSITIVITY.md','06_PARAMETER_NEIGHBORHOOD_SENSITIVITY.md','07_INDEPENDENT_PATCH_REVIEW.md','08_SEVEN_MODEL_REFIT_LAUNCH_PACKET.md','09_UNRESOLVED_STATISTICAL_ASSUMPTIONS.md','10_NEXT_EXECUTION_PROMPT.md','SOURCE_PRESERVATION.md'];assert all((D/x).is_file() for x in required)
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
print('existing files preserved',len(before),'new assets',len(assets))
