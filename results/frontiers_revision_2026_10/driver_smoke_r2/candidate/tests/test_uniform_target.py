"""Independent deterministic integration correctness and real-call firewall tests."""
from pathlib import Path
from types import SimpleNamespace
import sys,json,hashlib,traceback,ast
import numpy as np
from scipy.special import logsumexp
ROOT=Path(__file__).resolve().parents[2];I=ROOT.parents[2]
sys.path.insert(0,str(ROOT/'candidate/src'))
sys.path.insert(1,str(I/'results/frontiers_revision_2026_10/uniform_target_gate'))
import explicit_reference as ex
from ptquat import mle,fit_global as fg,experiments as exp,marginal_likelihood as ml,models
from ptquat.data import load_tidy_sparc

def run(out):
    gs0=load_tidy_sparc(str(I/'dataset/sparc_tidy.csv'));gs=[gs0[k] for k in sorted(gs0)]
    M=I/'results/frontiers_revision_2026_10/mle_recovery/original_recovered/01_mle'
    vectors={m:np.load(M/m.replace('-','_')/'theta_mle.npy') for m in ex.laws.MODELS}
    tests=[]
    def check(name,fn):
        try:r=fn();tests.append(dict(test=name,status='PASS',detail=r));print('UNIT PASS',name,flush=True)
        except Exception as exc:tests.append(dict(test=name,status='FAIL',error=repr(exc),traceback=traceback.format_exc()));print('UNIT FAIL',name,repr(exc),flush=True)
    def rejected(fn):
        try:fn()
        except (ValueError,RuntimeError,FloatingPointError):return
        raise AssertionError('expected fail-closed exception')
    def kernels():
        maxerr=0
        for g0 in gs:
            g=SimpleNamespace(D=g0.D_Mpc,sD=g0.D_err_Mpc,inc=np.deg2rad(g0.i_deg),si=np.deg2rad(g0.i_err_deg),y=g0.v_obs,e=g0.v_err)
            ds,wd,ii,wi,_,_=ml.nuisance_nodes(g,'segmented_quantile',192)
            maxerr=max(maxerr,abs(wd.sum()*wi.sum()-1));assert (ds>0).all() and (ii>0).all() and (ii<np.pi/2).all()
            for center,sigma,lo,hi in [(g.D,g.sD,0,np.inf),(g.inc,g.si,0,np.pi/2)]:
                if sigma:
                    z=ml.quantile_standard((lo-center)/sigma,(hi-center)/sigma,np.array([1e-14,.5,1-1e-14]));assert np.isfinite(z).all()
        assert maxerr<=1e-12
        return dict(galaxies=91,max_constant_integral_error=maxerr)
    check('physical_normalized_kernels_and_i90',kernels)
    def localdensity():
        g=SimpleNamespace(name='synthetic',r=np.array([1.,2.]),y=np.array([80.,82.]),e=np.array([2.,2.]),disk=np.array([80.,80.]),bulge=np.zeros(2),gas=np.zeros(2),D=10.,sD=0.,inc=np.pi/3,si=0.)
        p=dict(model='baryon',U=np.ones(1),sigma=3.,_H0=ml.H0_SI,_c0=10.,_c_slope=-.1)
        v=ml.velocity(g,p,0,g.D);expected=-.5*(((g.y-v)**2/13)+np.log(2*np.pi*13)).sum()
        r=ml.integrate(g,p,0,'segmented_quantile',192)
        assert max(abs(r['logL'][f]-expected) for f in ['B0','B1'])<1e-10
        np.testing.assert_allclose(ml.velocity(g,p,0,1.21*g.D),1.1*v,rtol=1e-13)
        b=.8;expected1=-.5*(((g.y-b*v)**2/(4+9*b*b))+np.log(2*np.pi*(4+9*b*b))).sum()
        assert abs(ml.conditional_logdensity(g,b*v,3.,b,'B1')-expected1)<1e-12
        g.sD=1.;g.si=.1;ds,wd,ii,wi,_,_=ml.nuisance_nodes(g,'quantile',12);vd=ml.velocity(g,p,0,ds[:,None]);b=np.sin(ii)/np.sin(g.inc)
        fast=ml.integrate(g,p,0,'quantile',12)
        for f in ['B0','B1']:
            grid=np.array([[ml.conditional_logdensity(g,b[j]*vd[k],3.,b[j],f) for j in range(len(ii))] for k in range(len(ds))]);direct=float(logsumexp(grid+np.log(wd[:,None])+np.log(wi[None,:])))
            assert abs(direct-fast['logL'][f])<1e-10
        return dict(zero_uncertainty=True,full_normalization=True,B0_B1=True,no_covariance_doublecount=True,independent_direct_grid=True)
    check('conditional_Gaussian_and_baryon_counterexample',localdensity)
    def laws():
        gi=[g.name for g in gs].index('F574-1');g=gs[gi];ref=ex.laws.load_sample(I/'dataset/sparc_tidy.csv')[gi];maxv=0;maxd=0;maxi=0
        view=SimpleNamespace(r=g.r_kpc,disk=g.v_disk,bulge=g.v_bulge,gas=g.v_gas,D=g.D_Mpc)
        for m,t in vectors.items():
            p=fg._unpack(t,fg._layout(m,91,True,None));rp=ex.laws.decode(m,t,91);jd,ji=ex.laws.derivatives(ref,rp,gi)
            fun=lambda d:models.catalog_velocity_at_distance(view,p,gi,m,d)
            v=fun(g.D_Mpc);rv=ex.laws.velocity(ref,rp,gi);maxv=max(maxv,float(abs(v-rv).max()));assert np.max(abs(v-rv))<=1e-9
            for h in [1e-4,1e-5,1e-6]:
                hd=h*g.D_Mpc;fd=(fun(g.D_Mpc+hd)-fun(g.D_Mpc-hd))/(2*hd);err=np.linalg.norm(fd-jd)/max(np.linalg.norm(jd),1e-10);maxd=max(maxd,float(err));assert err<=1e-6
                i=np.deg2rad(g.i_deg);fi=(np.sin(i+h)-np.sin(i-h))/(2*h*np.sin(i))*v;err=np.linalg.norm(fi-ji)/max(np.linalg.norm(ji),1e-10);maxi=max(maxi,float(err));assert err<=1e-6
            if m!='nfw1p':np.testing.assert_allclose(fun(1.2*g.D_Mpc),v*np.sqrt(1.2),rtol=1e-12)
            else:assert not np.allclose(fun(1.2*g.D_Mpc),v*np.sqrt(1.2),rtol=1e-4)
        old=ast.parse((I/'src/ptquat/models.py').read_text());new=ast.parse((ROOT/'candidate/src/ptquat/models.py').read_text());nd={n.name:ast.dump(n) for n in new.body if isinstance(n,ast.FunctionDef)}
        assert all(nd[n.name]==ast.dump(n) for n in old.body if isinstance(n,ast.FunctionDef))
        return dict(all_original_law_bodies_unchanged=True,max_prediction_abs=maxv,max_D_relative=maxd,max_i_relative=maxi,FD_steps=[1e-4,1e-5,1e-6])
    check('all7_native_laws_derivatives_units_NFW',laws)
    def firewalls():
        t=vectors['mond'];g=gs[0];L=fg._layout('mond',91,True,None)
        for f in [exp.compare_models_waic,exp.loo_compare_models,exp.ppc_check,exp.kappa_per_galaxy,exp._call_fit]:rejected(lambda:f())
        rejected(lambda:fg.log_likelihood_per_galaxy(t,gs,4.,ml.H0_SI,L,10.,-.1,None,ml.PRIMARY_B0,8.))
        rejected(lambda:mle.evaluate_loglike(t,gs,'mond',likelihood_version='gauss'))
        rejected(lambda:fg.run([]));rejected(lambda:mle.optimize_mle());rejected(lambda:ml.galaxy_log_marginal(g,fg._unpack(t,L),0,'mond',order=32))
        for u in [0.,1.,np.nan]:rejected(lambda:ml.quantile_standard(-5,np.inf,[u]))
        rejected(lambda:ml.conditional_logdensity(SimpleNamespace(y=np.array([1.]),e=np.array([0.])),np.array([1.]),0.,1.,'B0'))
        bad=t.copy();bad[0]=np.nan;rejected(lambda:mle.evaluate_loglike(bad,gs,'mond'))
        bad=t.copy();bad[-1]=99;rejected(lambda:mle.objective_neg_loglike(bad,gs,'mond',ml.H0_SI))
        # Invalid conditional source must propagate through real production public path.
        import dataclasses
        badg=dataclasses.replace(g,D_Mpc=-1.)
        rejected(lambda:mle.galaxy_loglike(t,[badg]+gs[1:],0,'mond'))
        return dict(primary_radius_diagnostics_blocked=True,unsupported_version_rejected=True,exceptions_propagate=True,MCMC_disabled=True)
    check('real_imports_predictive_firewall_failclosed',firewalls)
    def source():
        h=json.loads((ROOT/'SOURCE_HASHES_BEFORE.json').read_text());bad=[p for p,v in h.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=v];assert not bad,bad
        return dict(verified=len(h),changed=0)
    check('source_immutability',source)
    (out/'unit_tests.json').write_text(json.dumps(tests,indent=2)+'\n')
    assert all(x['status']=='PASS' for x in tests),'local tests fail; no full sample or optimizer authorized'
    return gs,vectors
