from pathlib import Path
import os,json,hashlib,subprocess,shutil,datetime
R=Path('/Users/rocky/ptq-mond');I=R/'PTQ-Quaternionic-RC';B=I/'results/frontiers_revision_2026_10/production_smoke';D=R/'docs/frontiers_major_revision_2026_10/rev26_p1f_production_smoke';E=I/'results/frontiers_revision_2026_10/uniform_target_gate'
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
js(B/'SOURCE_HASHES_BEFORE.json',before);repos={}
for name,p in [('outer',R),('implementation',I)]:repos[name]={k:subprocess.check_output(['git']+v,cwd=p,text=True) for k,v in dict(root=['rev-parse','--show-toplevel'],HEAD=['rev-parse','HEAD'],status=['status','--porcelain=v1'],worktrees=['worktree','list','--porcelain']).items()}
js(B/'AUTHORITY.json',repos)
for root in [E,R/'docs/frontiers_major_revision_2026_10/rev26_p1e_uniform_target',I/'results/frontiers_revision_2026_10/mle_recovery',I/'results/frontiers_revision_2026_10/patch_boundary_gate']:
 for l in (root/'SHA256SUMS').read_text().splitlines():
  h,p=l.split('  ',1);assert sha(root/p)==h,str(root/p)
js(B/'INPUT_MANIFEST.json',dict(P1E_hashes_verified=True,P1E_reference=dict(path=str(E/'explicit_reference.py'),sha256=sha(E/'explicit_reference.py')),data=dict(path=str(I/'dataset/sparc_tidy.csv'),sha256=sha(I/'dataset/sparc_tidy.csv')),theta=[dict(path=str(p),sha256=sha(p)) for p in sorted((I/'results/frontiers_revision_2026_10/mle_recovery/original_recovered/01_mle').glob('*/theta_mle.npy'))],canonical_sources=[dict(path=str(p),sha256=sha(p)) for p in sorted((I/'src/ptquat').glob('*.py'))]))
W=B/'candidate';(W/'src').mkdir(parents=True);shutil.copytree(I/'src/ptquat',W/'src/ptquat',ignore=shutil.ignore_patterns('__pycache__'));(W/'scripts').mkdir();(W/'tests').mkdir()
protocol=dict(label='ENGINEERING_SMOKE_ONLY — NOT CORRECTED MLE',frozen_utc=datetime.datetime.utcnow().isoformat()+'Z',likelihood_B0='REV26-P1E-UNIFORM-EXPLICIT-B0-PHYSICAL-v1',likelihood_B1='REV26-P1E-UNIFORM-EXPLICIT-B1-PHYSICAL-v1',policy='segmented_quantile 192 fixedpieceorder;96precisioncomparison;physicalCDFknots -6,-4,-2,0,2,4,6 withfullprobabilitydomain',parity=dict(all637eachfloor=True,native728=True,neighborhood_P1E72points=True,pergal_logL_abs=1e-8,mapping_abs=1e-8,refinement_pergal=1e-4,refinement_sample_sumabs=0.01,repeat_abs=1e-12,nonfinite_allowed=False),smoke=dict(models=['ptq-screen','nfw1p'],starts='one original recovered theta permodel,nootherstarts',optimizer='scipy.optimize.minimize L-BFGS-B,rawnativecoordinates,boundsoriginal,jac=None numericalforwarddifferences',max_accepted_iterations_per_run=1,maxiter=1,maxls=2,max_fullsample_evaluations_aggregate=400,max_per_model_fullsample_evaluations=200,max_optimizer_calls_per_model=190,max_optimization_wall_seconds_aggregate=1200,include_validation_calls_in400=True,initial_validation='fine192,coarse96,repeatfine192,3prescribedperturbations',terminal_validation='fine192,coarse96,repeatfine192 ifbudgetpermits',reserved_validation_calls_per_model=9,options=dict(ftol=1e-12,gtol=1e-6,eps=1e-8,maxcor=10),retry=False,per_galaxy_wallguard=True,gradient_kind='finite differences ofintegratednumericalobjective,notanalyticderivatives',perturbations=dict(ptq_screen=['epsilon +1e-6*nativeboundspan','lnq +1e-6*nativeboundspan','lnsigma +1e-6*nativeboundspan'],nfw1p=['firstU +1e-6span ifinboundelseminus','firstlogM +1e-6span ifinboundelseminus','lnsigma +1e-6span']),terminal_if_budget='lastaccepted callback vector ifavailable,else originalstart;labelmissingvalidation,nevercallitMLE',budget_result='DRIVER_SMOKE_BUDGET_LIMIT;noautomaticretryorcapincrease'),pre_smoke_gates='unit/parity/allactualpubliclikelihoodcallpaths/firewallPASSbeforeanyoptimizer',post_fit_not_authorized=['sevenmodelMLE','NFWmultistart','profiles','MCMC','newICranking','manuscriptupdates','canonicalinstallation'],source_roots=[str(B),str(D)])
js(B/'SMOKE_PROTOCOL.json',protocol)
print('sources',len(before),'protocol SHA256',sha(B/'SMOKE_PROTOCOL.json'))
