from pathlib import Path
import hashlib,json,subprocess,ast,os
R=Path('/Users/rocky/ptq-mond');I=R/'PTQ-Quaternionic-RC';B=I/'results/frontiers_revision_2026_10/covariance_gate';D=R/'docs/frontiers_major_revision_2026_10/rev26_p1b_covariance_gate';V=B/'run_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def js(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
# Check output-routing-only change against executed numerical source.
s0=(B/'run_gate_executed_v1.py').read_text();s1=(B/'run_gate.py').read_text();old=" out=B/'run_v1';out.mkdir(exist_ok=False)";new=" import argparse\n ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,default=B/'run_v1');args=ap.parse_args()\n out=args.out.resolve()\n if B.resolve() not in out.parents:raise ValueError('Output must remain under covariance_gate')\n out.mkdir(exist_ok=False)"
assert s0.replace(old,new)==s1
assert sha(B/'run_gate_executed_v1.py')==json.loads((V/'execution_authority.json').read_text())['script_sha256']
# Refuse current output before any evaluation; exercises final CLI output guard.
p=subprocess.run(['/tmp/rev26-p1a-env/bin/python',str(B/'run_gate.py')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
assert p.returncode!=0 and 'FileExistsError' in p.stdout
(B/'replay_output_guard_check.log').write_text(p.stdout)
# Actual test assertions represented in result analysis; all attempted cases kept.
assert all((D/n).exists() for n in ['00_GATE_EXECUTIVE_VERDICT.md','01_OBSERVATION_TRANSFORM_AND_JACOBIAN_LEDGER.md','02_CASE_SELECTION_AND_PREDECLARED_TOLERANCES.md','03_FINITE_DIFFERENCE_RESULTS.md','04_COVARIANCE_UPDATE_AND_LOGDET_AUDIT.md','05_EXPLICIT_NUISANCE_SENSITIVITY.md','06_SCIENTIFIC_MATERIALITY_AND_REFIT_GATE.md','07_REVIEWER_RESPONSE_EVIDENCE.md','08_NEXT_TASK_PROMPT.md'])
for pth in B.glob('*.py'):ast.parse(pth.read_text())
before=json.loads((B/'SOURCE_HASHES_BEFORE.json').read_text());changed=[];missing=[]
for pth,h in before.items():
 p=Path(pth)
 if not p.is_file():missing.append(pth)
 elif sha(p)!=h:changed.append(pth)
assert not changed and not missing,(changed,missing)
repos={}
for label,cwd in [('outer',R),('implementation',I)]:
 repos[label]={k:subprocess.check_output(['git']+c,cwd=cwd,text=True) for k,c in dict(root=['rev-parse','--show-toplevel'],HEAD=['rev-parse','HEAD'],status=['status','--porcelain=v1'],worktrees=['worktree','list','--porcelain']).items()}
js(B/'GIT_AFTER.json',repos)
authority=json.loads((B/'GATE0_AUTHORITY.json').read_text())
for name,x in repos.items():
 for key in ['root','HEAD','worktrees']:assert x[key]==authority[name][key]
js(B/'SOURCE_PRESERVATION_VALIDATION.json',dict(existing_files_checked=len(before),changed=changed,missing=missing,repositories_HEAD_and_worktree_unchanged=True,prior_bundles_checksums_verified=True,executed_script_preserved=True,replay_script_only_output_routing_added=True,original_covariance_and_theta_unchanged=True,new_outputs_only_under=[str(B),str(D)],reviewer_original_requests='unavailable; provisional draft explicitly blocked for actual quotation',numerical_computation_exceptions=0,coarse_fd_threshold_failures_retained=45))
(D/'SOURCE_PRESERVATION.md').write_text(f'''# 來源保全與版本紀錄

Gate0在numerical comparisons前記錄{len(before)}份既有專案檔案SHA256，含production source、dataset/raw、frozen results、P0R/P1A報告與其hash manifests、submitted source、main2.tex、public release/v5。最後逐份核對：0 changed、0 missing。清單SOURCE_HASHES_BEFORE.json；結果SOURCE_PRESERVATION_VALIDATION.json。

實際repo roots：{R}（HEAD {repos['outer']['HEAD'].strip()}）及{I}（HEAD {repos['implementation']['HEAD'].strip()}），各一個登錄worktree，前後HEAD與worktree清單相同。Gate0/GIT_AFTER保留完整status，既有四份implementation dirty files均未動，外層原有P1A untracked保留。沒有pull/fetch/checkout/reset/stash/commit/push/tag/release。

原P1A numerical與docs兩份SHA256SUMS在本輪開始均逐項通過；七份original theta與55項source manifest records的source／recovered bytes也核對。shared ν-q boundary discrepancy紀錄保留。沒有重做archive discovery或Symmetry audit。

執行環境沿用 /tmp/rev26-p1a-env/bin/python，沒有安裝或修改production環境；Python/platform/thread設定與pip freeze在run_v1。全部新scripts、numerical outputs、logs只寫本次covariance_gate與docs/rev26_p1b_covariance_gate兩root；bytecode寫入禁用。未做optimizer driver／global refit、MCMC、NFW starts、production patch、manuscript edit或任何公開寫入。

先凍結OBSERVATION_CONTRACT與PREDECLARED_DESIGN後執行。實際run script副本run_gate_executed_v1.py的SHA256與execution_authority.json一致。最終run_gate.py只加可指定全新--out目錄與本root路徑限制，數值body未變；精確source差異程式比對通過，現有輸出目錄拒絕寫入測試通過（replay_output_guard_check.log），沒有重跑numerical comparisons。prepare/freeze scripts事後增加防覆寫guard，舊design bytes保持不變。

SHA256SUMS分別涵蓋兩root所有檔案，排除自身。ASSET_MANIFEST.json列出新資產絕對path/bytes/hash（排除自身及docs checksum以避免循環）；其hash與兩checksum檔hash由最終回覆／delivery chain指向。原稿與歷史結果沒有更新。
''')
# Keep validation JSON machine-readable as the contract's independent verification result.
js(B/'VALIDATION.json',dict(integrity='PASS',source_preservation='PASS',design_frozen_before_comparison=True,matched_points_pass=True,parameter_update_pass=True,FD_finest_pass=135,FD_coarse_threshold_failures=45,quadrature_convergence_pass=42,quadrature_total=42,all_attempted_cases_saved=True,scientific_verdict='COVARIANCE_CORRECTION_REQUIRED_BEFORE_REFIT',original_review_requests_available=False))
(B/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(B))+'\n' for p in sorted(B.rglob('*')) if p.is_file() and p!=B/'SHA256SUMS'))
assets=[]
for root in [B,D]:
 for p in sorted(root.rglob('*')):
  if p.is_file() and p not in [D/'ASSET_MANIFEST.json',D/'SHA256SUMS']:
   assets.append(dict(absolute_path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
js(D/'ASSET_MANIFEST.json',dict(schema_version=1,excluded_self_and_docs_checksum=True,assets=assets))
(D/'SHA256SUMS').write_text(''.join(sha(p)+'  '+str(p.relative_to(D))+'\n' for p in sorted(D.rglob('*')) if p.is_file() and p!=D/'SHA256SUMS'))
for root in [B,D]:
 for line in (root/'SHA256SUMS').read_text().splitlines():
  h,p=line.split('  ',1);assert sha(root/p)==h,p
for p in [D/'ASSET_MANIFEST.json',D/'SHA256SUMS',B/'SHA256SUMS']:print(str(p),sha(p))
print('preserved',len(before),'new assets',len(assets))
