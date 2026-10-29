from pathlib import Path
import os,json,hashlib,subprocess,ast
import numpy as np
R=Path('/Users/rocky/ptq-mond');I=R/'PTQ-Quaternionic-RC';B=I/'results/frontiers_revision_2026_10/covariance_reference';D=R/'docs/frontiers_major_revision_2026_10/rev26_p1c_covariance_reference';M=I/'results/frontiers_revision_2026_10/mle_recovery'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def js(p,x):
 if p.exists():raise FileExistsError(p)
 p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
if (B/'SHA256SUMS').exists() or (D/'SHA256SUMS').exists():raise FileExistsError('Do not overwrite finalized run')
# Native-coordinate bounds audit at original fixed vectors; no proposal or optimization.
manifest=json.loads((M/'original_recovered/01_mle/parameter_domain_manifest.json').read_text());counts=json.loads((B/'STATISTICAL_SPECIFICATION.json').read_text())['parameter_counts'];bounds_records=[]
for model,k in counts.items():
 t=np.load(M/'original_recovered/01_mle'/model.replace('-','_')/'theta_mle.npy');bounds=[]
 if model in ['ptq','ptq-nu','ptq-screen']:bounds.append(manifest['historical_epsilon_wide_bounds'] if model=='ptq' else manifest['matched_epsilon_bounds'])
 elif model in ['mond','mond-screen']:bounds.append(np.log10(manifest['matched_a0_bounds_si']).tolist())
 if model in ['ptq-screen','mond-screen']:bounds.append(np.log(manifest['q_bounds']).tolist())
 bounds.extend([manifest['upsilon_bounds']]*91)
 if model=='nfw1p':bounds.extend([manifest['log10_M200_bounds']]*91)
 bounds.append(np.log(manifest['sigma_sys_bounds_kms']).tolist());assert len(bounds)==k==len(t)
 violations=[j for j,(v,(lo,hi)) in enumerate(zip(t,bounds)) if not lo<=v<=hi];assert not violations
 bounds_records.append(dict(model=model,k=k,violations=violations,bounds_native=bounds))
js(B/'PARAMETER_DOMAIN_VALIDATION.json',bounds_records)
# All required outputs exist; reference executed bytes still authoritative.
req=['00_EXECUTIVE_DECISION.md','01_STATISTICAL_SPECIFICATION_B0_B1.md','02_OBSERVATION_AND_DERIVATIVE_CONTRACT.md','03_REFERENCE_IMPLEMENTATION_TESTS.md','04_NUISANCE_AND_FLOOR_SENSITIVITY.md','05_FULL_SAMPLE_FIXED_POINT_EFFECTS.md','06_INDEPENDENT_ADVERSARIAL_REVIEW.md','07_PRODUCTION_CHANGE_PROPOSAL.md','08_SEVEN_MODEL_REFIT_PLAN.md','09_REVIEWER_COVARIANCE_RESPONSE_DRAFT.md','10_NEXT_TASK_PROMPT.md']
assert all((D/x).is_file() for x in req)
local=json.loads((B/'local_tests_v1/LOCAL_CORRECTNESS_GATE.json').read_text());assert local['source_sha256']==sha(B/'reference.py') and local['spec_sha256']==sha(B/'STATISTICAL_SPECIFICATION.json') and local['contract_sha256']==sha(B/'PREDECLARED_TEST_CONTRACT.json')
for p in B.glob('*.py'):ast.parse(p.read_text())
# Refusal test ensures reproduction entrypoints do not overwrite preserved scientific outputs.
guards=[]
for file in ['test_reference.py','run_effects.py','verify_outputs.py']:
 p=subprocess.run(['/tmp/rev26-p1a-env/bin/python',str(B/file)],cwd=B,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','OMP_NUM_THREADS':'1'})
 assert p.returncode!=0 and 'FileExistsError' in p.stdout
 guards.append(dict(script=file,exit_code=p.returncode,refused_existing_output=True,console=p.stdout))
js(B/'OUTPUT_OVERWRITE_GUARD_TESTS.json',guards)
before=json.loads((B/'SOURCE_HASHES_BEFORE.json').read_text());changed=[];missing=[]
for s,h in before.items():
 p=Path(s)
 if not p.is_file():missing.append(s)
 elif sha(p)!=h:changed.append(s)
assert not changed and not missing,(changed,missing)
repos={}
for name,cwd in [('outer',R),('implementation',I)]:repos[name]={k:subprocess.check_output(['git']+v,cwd=cwd,text=True) for k,v in dict(root=['rev-parse','--show-toplevel'],HEAD=['rev-parse','HEAD'],status=['status','--porcelain=v1'],worktrees=['worktree','list','--porcelain']).items()}
old=json.loads((B/'AUTHORITY.json').read_text())
for name in repos:
 for field in ['root','HEAD','worktrees']:assert repos[name][field]==old[name][field]
js(B/'GIT_AFTER.json',repos)
js(B/'SOURCE_PRESERVATION_VALIDATION.json',dict(existing_files_checked=len(before),changed=changed,missing=missing,source_PDF_unchanged=True,specification_and_contract_unchanged_since_before_tests=True,reference_source_unchanged_since_local_gate=True,prior_bundles_untouched=True,production_untouched=True,repository_HEADs_worktrees_unchanged=True,optimizer_called=False,Git_mutations=False,output_overwrite_guards_pass=True))
source='''# Source preservation and authority

Gate A開始時封存1,807份既有檔案SHA256，含專案source/data/results/manuscripts、P1A/P1B全部reports/manifests與下載目錄reviewer supporting PDF。最後逐份核對：0 changed、0 missing。SOURCE_HASHES_BEFORE.json保存完整path/hash；SOURCE_PRESERVATION_VALIDATION.json保存結果。排除.git內部資料、cache、bytecode、venv以及本輪兩個全新output roots，未把排除項冒作科學source保全。

Outer root /Users/rocky/ptq-mond，HEAD 3eb302666da7eccca19c5843f729e98126d68977；implementation root /Users/rocky/ptq-mond/PTQ-Quaternionic-RC，HEAD 5a8dca57f0fc3484387c643b51e22c7e10d024a8。各自.git、HEAD、status、worktrees記錄於AUTHORITY/GIT_AFTER；沒有共用Git歷史的假設。兩repo的HEAD/worktree未變；原dirty files與P1A/P1B untracked保留，沒有pull/checkout/reset/stash/commit/push/tag/release。

本輪根目錄covariance_reference與rev26_p1c_covariance_reference原先不存在，建立全新且各run子目錄拒絕overwrite。original theta、datasets、production likelihood.py/models.py/mle.py/fit_global.py、歷史outputs、submitted/main2/v5/paper_release均未更動。spec/test contract先凍結後numerical tests，reference source在local gate後未改。

P1A/P1B兩個numerical bundle及兩個docs SHA256SUMS逐項通過；P1A source manifest source/recovered bytes再次核對。沒有重做archive discovery或P1B完整campaign；重用其已核驗replay／矩陣／callback證據。

Supporting PDF確實在/Users/rocky/Downloads/1964512reviewsupportingfile1349389.PDF，91531 bytes，SHA256 205ef3282749112e5a511b9b62aead058f035d65a0367c1ccdcc8f0f03d54905。只讀取位元核對存在、PDF標頭與hash；本輪文字權威是Human供給的兩段exact passages，未冒作PDF逐字抽取認證。P1B舊來源缺失記錄保留。

臨時環境沿用/tmp/rev26-p1a-env，Python/platform/pip freeze存local_tests_v1，沒有production install。新source、JSON/CSV、npz、console logs與reports只在兩個新root，Python bytecode禁寫。bootstrap實際stdout摘要記錄於EXECUTION_LOG；局部tests、full sample/nuisance、output verification的完整console各自保存。NumPy runtime warnings未刪。

獨立第二代理在local tests後作read-only review；其檢查與異議獨立保存，沒有把代理當作proof authority，也沒有因其意見修改數值契約。沒有任何optimization、fitting、NFW starts、MCMC、稿件修改、表格替換、公開upload或journal contact。

所有原native向量仍在原hard domains且k不變；PARAMETER_DOMAIN_VALIDATION保存逐native-coordinate bounds。完整execution logs、失敗紀錄與nonconvergence都保留：hard local failures0、full sample failures0、nuisance nonconvergence0，科學敏感度旗標不隱藏。

SHA256SUMS分別涵蓋兩root所有檔案除自身。docs/ASSET_MANIFEST.json列出其他新資產絕對path、bytes與hash（排除自身及docs checksum避免循環）；其hash及checksum檔可由最終回覆交叉核驗。全部新output refusal tests通過。
'''
(D/'SOURCE_PRESERVATION.md').write_text(source)
# Human-supplied evidence and all scientific source files have now been verified.
js(B/'FAILURE_REPORT.json',dict(local_hard_failures=0,full_sample_failures=0,quadrature_nonconvergence=0,omitted_galaxies=0,scientific_sensitivity_flags=['B1 vs B0 floor up to1.171402','synthetic_i90 nonlinear vs linear -0.296842','22 near inclination boundary,7 edge-on real cases not explicit-integrated in this phase'],warnings_retained=['NumPy slogdet runtime warnings; all outputs explicitly finite'],stop_conditions_triggered=False))
(B/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(B))+'\n' for p in sorted(B.rglob('*')) if p.is_file() and p!=B/'SHA256SUMS'))
assets=[]
for root in [B,D]:
 for p in sorted(root.rglob('*')):
  if p.is_file() and p not in [D/'ASSET_MANIFEST.json',D/'SHA256SUMS']:assets.append(dict(absolute_path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
js(D/'ASSET_MANIFEST.json',dict(schema_version=1,excluded_self_and_docs_checksum=True,assets=assets))
(D/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(D))+'\n' for p in sorted(D.rglob('*')) if p.is_file() and p!=D/'SHA256SUMS'))
for root in [B,D]:
 for line in (root/'SHA256SUMS').read_text().splitlines():
  h,p=line.split('  ',1);assert sha(root/p)==h
for p in [D/'ASSET_MANIFEST.json',D/'SHA256SUMS',B/'SHA256SUMS']:print(str(p),sha(p))
print('source preserved',len(before),'new assets',len(assets))
