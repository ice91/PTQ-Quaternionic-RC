"""Uniform physical-domain B0/B1 target. No optimization; same target for all galaxies.
Node policy is a separate numerical choice. No finite-tail target truncation.
Candidate production model laws are preserved; propagated covariance is never added.
"""
from pathlib import Path
from functools import lru_cache
import sys,time
import numpy as np
from scipy.special import ndtr,ndtri,roots_legendre,logsumexp
from types import SimpleNamespace
from .models import catalog_velocity_at_distance
from .constants import H0_SI
PRIMARY_B0 = "REV26-P1E-UNIFORM-EXPLICIT-B0-PHYSICAL-v1"
SENSITIVITY_B1 = "REV26-P1E-UNIFORM-EXPLICIT-B1-PHYSICAL-v1"
LEGACY_DIAGONALIZED_SENSITIVITY = "LEGACY_DIAGONALIZED_SENSITIVITY"
ORDER_FINE = 192
ORDER_COARSE = 96


def require_version(version):
    if version not in (PRIMARY_B0, SENSITIVITY_B1):
        raise ValueError("Unsupported primary likelihood version: " + str(version))
    return "B0" if version == PRIMARY_B0 else "B1"


def velocity(g, pars, gi, D):
    return catalog_velocity_at_distance(g, pars, gi, pars["model"], D,
        H0_si=pars["_H0"], c0=pars["_c0"], c_slope=pars["_c_slope"])


def validate_prior(g):
 if not (g.D>0 and g.sD>=0 and 0<g.inc<=np.pi/2 and g.si>=0):raise ValueError('Invalid nominal geometry/errors')
 if np.any(~np.isfinite(g.y)) or np.any(~np.isfinite(g.e)) or np.any(g.e<0):raise ValueError('Invalid observations/errors')

def quantile_standard(a,b,u):
 u=np.asarray(u,float)
 if np.any((u<=0)|(u>=1)) or np.any(~np.isfinite(u)):raise ValueError('Quadrature probabilities must be strictly inside(0,1)')
 A=ndtr(a);SB=ndtr(-b);Z=ndtr(b)-A
 if not (np.isfinite(Z) and Z>0):raise ValueError('Empty/unstable physical prior mass')
 # Choose stable CDF or survival branch; never form 1-CDF near upper endpoint.
 p=A+u*Z;q=SB+(1-u)*Z
 z=np.where(p<=.5,ndtri(p),-ndtri(q))
 if np.any(~np.isfinite(z)) or np.any(z<a) or np.any(z>b):raise FloatingPointError('Unrepresentable physical quantile')
 return z

@lru_cache(None)
def legendre(order):
 z,w=roots_legendre(order)
 if np.any(w<=0) or not np.isclose(w.sum(),2.,rtol=0,atol=1e-13):raise ValueError('Invalid quadrature weights')
 return z,w

@lru_cache(None)
def axis_nodes(center,sigma,lo,hi,method,order):
 if sigma==0:return np.array([center]),np.ones(1),1.
 a=(lo-center)/sigma;b=(hi-center)/sigma;Z=float(ndtr(b)-ndtr(a))
 breaks=[0.,1.]
 if method=='segmented_quantile':
  for knot in [-6.,-4.,-2.,0.,2.,4.,6.]:
   if a<knot<b:breaks.append(float((ndtr(knot)-ndtr(a))/Z))
 elif method!='quantile':raise ValueError(method)
 breaks=np.unique(breaks);z,w=legendre(order);u=[];weights=[]
 for left,right in zip(breaks[:-1],breaks[1:]):
  if not right>left:raise ValueError('Invalid probability segment')
  u.append(left+(z+1)*(right-left)/2);weights.append(w*(right-left)/2)
 u=np.concatenate(u);weights=np.concatenate(weights)
 zz=quantile_standard(a,b,u)
 # Lower physical D bound is zero: use difference from lower standardized bound.
 values=sigma*(zz-a)+lo if np.isfinite(lo) else center+sigma*zz
 if np.any(~np.isfinite(values)) or np.any(values<=lo) or np.any(values>=hi):raise FloatingPointError('Physical quadrature node outside open domain')
 if np.any(weights<=0) or abs(weights.sum()-1.)>1e-12:raise ValueError('Weights not positive/normalized')
 return values,weights,Z

def nuisance_nodes(g,method,order):
 validate_prior(g)
 ds,wd,ZD=axis_nodes(float(g.D),float(g.sD),0.,np.inf,method,int(order));incs,wi,Zi=axis_nodes(float(g.inc),float(g.si),0.,float(np.pi/2),method,int(order))
 return ds,wd,incs,wi,ZD,Zi

def conditional_logdensity(g,mu,sigma,b,floor):
 var=g.e*g.e+(sigma*sigma if floor=='B0' else (b*sigma)**2)
 if np.any(var<=0) or np.any(~np.isfinite(var)):raise ValueError('Nonpositive conditional variance')
 return -.5*np.sum((g.y-mu)**2/var+np.log(2*np.pi*var),axis=-1)

def integrate(g,p,gi,method,order,floors=('B0','B1'),linear=False):
 begin=time.perf_counter();ds,wd,incs,wi,ZD,Zi=nuisance_nodes(g,method,order);nn=len(ds)*len(incs)*len(floors)
 if p['sigma']<=0 or not np.isfinite(p['sigma']):raise ValueError('Invalid floor parameter')
 if linear: raise ValueError('Linearized mean is not the primary marginal target')
 vd=velocity(g,p,gi,ds[:,None]);b=np.sin(incs)/np.sin(g.inc)
 accum={floor:-np.inf for floor in floors};nodeshape=(len(ds),len(incs));max_roundoff=0.
 # Efficient exact conditional quadratic uses weighted sufficient sums; blocked grids.
 for floor in floors:
  var=g.e[None,:]**2+(p['sigma']**2 if floor=='B0' else (p['sigma']*b[:,None])**2)
  if var.shape[0]==1:var=np.broadcast_to(var,(len(incs),len(g.y)))
  if np.any(var<=0) or np.any(~np.isfinite(var)):raise ValueError('Invalid conditional variance')
  inv=1/var;yi=g.y[None,:]*inv;Y2=(g.y[None,:]**2*inv).sum(axis=1);norm=np.log(2*np.pi*var).sum(axis=1)
  for start in range(0,len(ds),128):
   end=min(start+128,len(ds));v=vd[start:end]
   cross=v@yi.T;vv=(v*v)@inv.T
   Q=Y2[None,:]-2*b[None,:]*cross+b[None,:]**2*vv
   scale=Y2[None,:]+2*np.abs(b[None,:]*cross)+b[None,:]**2*np.abs(vv)
   tolerance=64*np.finfo(float).eps*np.maximum(scale,1.)
   if np.any(Q < -tolerance):raise FloatingPointError('Quadratic cancellation outside roundoff bound')
   max_roundoff=max(max_roundoff,float(np.max(tolerance)))
   ll=-.5*(Q+norm[None,:])
   if not np.isfinite(ll).all():raise FloatingPointError('Nonfinite conditional Gaussian density')
   value=float(logsumexp(ll+np.log(wd[start:end,None])+np.log(wi)[None,:]));accum[floor]=float(np.logaddexp(accum[floor],value))
 for value in accum.values():
  if not np.isfinite(value):raise FloatingPointError('Non-finite marginal logL')
 return dict(logL=accum,method=method,order=int(order),D_nodes=nodeshape[0],i_nodes=nodeshape[1],density_nodes=nn,ZD=ZD,Zi=Zi,weight_sum=float(wd.sum()*wi.sum()),seconds=time.perf_counter()-begin,max_quadratic_roundoff_bound=max_roundoff,linear=linear)



def galaxy_log_marginal(galaxy, pars, gi, model, *, likelihood_version=PRIMARY_B0,
                        H0_si=H0_SI, c0=10.0, c_slope=-0.1, order=ORDER_FINE):
    from .evaluation_contract import validate_decoded,validate_galaxy
    from .warning_telemetry import WarningContext
    floor = require_version(likelihood_version)
    validate_decoded(pars,gi,model)
    validate_galaxy(galaxy)
    if not (np.isfinite(H0_si) and H0_si>0 and np.isfinite(c0) and c0>0 and np.isfinite(c_slope)):
        raise ValueError("Invalid fixed cosmology/halo configuration")
    if order not in (ORDER_FINE, ORDER_COARSE):
        raise ValueError("Only frozen fine/coarse quadrature orders are supported")
    g = SimpleNamespace(name=galaxy.name, r=galaxy.r_kpc, y=galaxy.v_obs,
        e=galaxy.v_err, disk=galaxy.v_disk, bulge=galaxy.v_bulge,
        gas=galaxy.v_gas, D=float(galaxy.D_Mpc), sD=float(galaxy.D_err_Mpc),
        inc=float(np.deg2rad(galaxy.i_deg)), si=float(np.deg2rad(galaxy.i_err_deg)))
    p = dict(pars, model=model, _H0=H0_si, _c0=c0, _c_slope=c_slope)
    with WarningContext(model,likelihood_version,pars,gi):
        record = integrate(g, p, gi, "segmented_quantile", order, floors=(floor,))
    record["likelihood_version"] = likelihood_version
    record["galaxy"] = galaxy.name
    record["log_marginal"] = record["logL"][floor]
    record["conditional_normalization"] = "full sum(log(2*pi*variance)) at every node"
    return record
