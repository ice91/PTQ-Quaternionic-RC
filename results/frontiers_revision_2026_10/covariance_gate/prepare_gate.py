from pathlib import Path
import os,json,hashlib,subprocess
R=Path('/Users/rocky/ptq-mond');I=R/'PTQ-Quaternionic-RC';B=I/'results/frontiers_revision_2026_10/covariance_gate';D=R/'docs/frontiers_major_revision_2026_10/rev26_p1b_covariance_gate'
if (B/'SOURCE_HASHES_BEFORE.json').exists():raise FileExistsError('Refuse to overwrite frozen gate authority/design')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def js(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
before={}
for root,dirs,files in os.walk(R):
 dirs[:]=[d for d in dirs if d not in ['.git','.cache','__pycache__','.venv'] and Path(root)/d not in [B,D]]
 for f in files:
  p=Path(root)/f
  if p.is_file():before[str(p)]=sha(p)
js(B/'SOURCE_HASHES_BEFORE.json',before)
repos={}
for name,p in [('outer',R),('implementation',I)]:
 repos[name]={k:subprocess.check_output(['git']+args,cwd=p,text=True) for k,args in dict(root=['rev-parse','--show-toplevel'],HEAD=['rev-parse','HEAD'],status=['status','--porcelain=v1'],worktrees=['worktree','list','--porcelain']).items()}
js(B/'GATE0_AUTHORITY.json',repos)
M=I/'results/frontiers_revision_2026_10/mle_recovery';manifest=json.loads((M/'SOURCE_MANIFEST.json').read_text())
for x in manifest['artifacts']:
 for p in [Path(x['absolute_source_path']),Path(x['recovered_path'])]:assert sha(p)==x['sha256'],p
for root in [M,R/'docs/frontiers_major_revision_2026_10/rev26_p1a_mle_recovery']:
 for line in (root/'SHA256SUMS').read_text().splitlines():
  h,p=line.split('  ',1);assert sha(root/p)==h,p
assert (M/'replay_fixed.py').is_file()
js(B/'INPUT_AUTHORITY.json',dict(prior_bundle_checksums_pass=True,theta_vectors=[{'path':str(p),'sha256':sha(p)} for p in sorted((M/'original_recovered/01_mle').glob('*/theta_mle.npy'))],data=[{'path':str(p),'sha256':sha(p)} for p in sorted((I/'dataset').rglob('*.csv'))],source=[{'path':str(p),'sha256':sha(p)} for p in sorted((I/'src/ptquat').glob('*.py'))],prior_replay_sha256=sha(M/'replay_fixed.py')))
print('Gate0 verified',len(before),'existing files')
