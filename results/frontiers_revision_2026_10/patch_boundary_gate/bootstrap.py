from pathlib import Path
import os,json,hashlib,subprocess,datetime,shutil
R=Path('/Users/rocky/ptq-mond');I=R/'PTQ-Quaternionic-RC';B=I/'results/frontiers_revision_2026_10/patch_boundary_gate';D=R/'docs/frontiers_major_revision_2026_10/rev26_p1d_patch_boundary';C=I/'results/frontiers_revision_2026_10/covariance_reference'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def js(p,x):
 if p.exists():raise FileExistsError(p)
 p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
before={}
for root,dirs,files in os.walk(R):
 dirs[:]=[x for x in dirs if x not in ['.git','.cache','__pycache__','.venv'] and Path(root)/x not in [B,D]]
 for name in files:
  p=Path(root)/name
  if p.is_file():before[str(p)]=sha(p)
js(B/'SOURCE_HASHES_BEFORE.json',before)
repos={}
for name,p in [('outer',R),('implementation',I)]:repos[name]={k:subprocess.check_output(['git']+v,cwd=p,text=True) for k,v in dict(root=['rev-parse','--show-toplevel'],HEAD=['rev-parse','HEAD'],status=['status','--porcelain=v1'],worktrees=['worktree','list','--porcelain']).items()}
js(B/'AUTHORITY.json',repos)
for root in [C,I/'results/frontiers_revision_2026_10/mle_recovery',R/'docs/frontiers_major_revision_2026_10/rev26_p1c_covariance_reference']:
 for l in (root/'SHA256SUMS').read_text().splitlines():
  h,p=l.split('  ',1);assert sha(root/p)==h,p
assert json.loads((C/'local_tests_v1/LOCAL_CORRECTNESS_GATE.json').read_text())['status']=='PASS'
js(B/'INPUT_AUTHORITY.json',dict(P1C_checksums_pass=True,canonical_files=[dict(path=str(p),sha256=sha(p)) for p in sorted((I/'src/ptquat').glob('*.py'))],data=dict(path=str(I/'dataset/sparc_tidy.csv'),sha256=sha(I/'dataset/sparc_tidy.csv')),theta=[dict(path=str(p),sha256=sha(p)) for p in sorted((I/'results/frontiers_revision_2026_10/mle_recovery/original_recovered/01_mle').glob('*/theta_mle.npy'))],reference=dict(path=str(C/'reference.py'),sha256=sha(C/'reference.py'))))
W=B/'isolated_candidate';(W/'src').mkdir(parents=True);shutil.copytree(I/'src/ptquat',W/'src/ptquat',ignore=shutil.ignore_patterns('__pycache__'));(W/'tests').mkdir();shutil.copy2(I/'tests/test_covariance.py',W/'tests/test_covariance.py')
print('preservation',len(before),'files; isolated candidate created')
