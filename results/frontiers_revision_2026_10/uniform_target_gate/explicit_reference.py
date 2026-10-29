"""Uniform physical-domain B0/B1 target. No optimization; same target for all galaxies.
Node policy is a separate numerical choice. No finite-tail target truncation.
Frozen P1C law equations are reused read-only; propagated covariance is never added.
"""
from pathlib import Path
from functools import lru_cache
import sys,time
import numpy as np
from scipy.special import ndtr,ndtri,roots_legendre,logsumexp
B=Path(__file__).resolve().parent;I=B.parents[2]
sys.path.insert(0,str(I/'results/frontiers_revision_2026_10/covariance_reference'))
import reference as laws
NODE_EVALUATIONS=0
START=time.monotonic()
MAX_NODES=6000000000
MAX_SECONDS=1200
class NumericalBudgetExceeded(RuntimeError):pass

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
 global NODE_EVALUATIONS
 begin=time.perf_counter();ds,wd,incs,wi,ZD,Zi=nuisance_nodes(g,method,order);nn=len(ds)*len(incs)*len(floors)
 if NODE_EVALUATIONS+nn>MAX_NODES or time.monotonic()-START>MAX_SECONDS:raise NumericalBudgetExceeded('Frozen numerical node/time budget reached')
 if p['sigma']<=0 or not np.isfinite(p['sigma']):raise ValueError('Invalid floor parameter')
 vd=laws.velocity(g,p,gi,ds[:,None]);b=np.sin(incs)/np.sin(g.inc);v0=laws.velocity(g,p,gi);jd,ji=laws.derivatives(g,p,gi)
 accum={floor:-np.inf for floor in floors};nodeshape=(len(ds),len(incs));max_roundoff=0.
 # Efficient exact conditional quadratic uses weighted sufficient sums; blocked grids.
 for floor in floors:
  var=g.e[None,:]**2+(p['sigma']**2 if floor=='B0' else (p['sigma']*b[:,None])**2)
  if var.shape[0]==1:var=np.broadcast_to(var,(len(incs),len(g.y)))
  if np.any(var<=0) or np.any(~np.isfinite(var)):raise ValueError('Invalid conditional variance')
  inv=1/var;yi=g.y[None,:]*inv;Y2=(g.y[None,:]**2*inv).sum(axis=1);norm=np.log(2*np.pi*var).sum(axis=1)
  if linear:
   # mu=vdlinear(D)+Ji*delta_i; exact quadratic expansion without 3D grid.
   shift=ji[None,:]*(incs-g.inc)[:,None];obs_shift=g.y[None,:]-shift;yi=obs_shift*inv;Y2=(obs_shift*obs_shift*inv).sum(axis=1)
  for start in range(0,len(ds),128):
   end=min(start+128,len(ds));v=vd[start:end] if not linear else v0[None,:]+jd[None,:]*(ds[start:end,None]-g.D)
   cross=v@yi.T;vv=(v*v)@inv.T
   Q=Y2[None,:]-2*cross+(vv) if linear else Y2[None,:]-2*b[None,:]*cross+b[None,:]**2*vv
   scale=Y2[None,:]+2*np.abs(cross)+(np.abs(vv)) if linear else Y2[None,:]+2*np.abs(b[None,:]*cross)+b[None,:]**2*np.abs(vv)
   tolerance=64*np.finfo(float).eps*np.maximum(scale,1.)
   if np.any(Q < -tolerance):raise FloatingPointError('Quadratic cancellation outside roundoff bound')
   max_roundoff=max(max_roundoff,float(np.max(tolerance)))
   ll=-.5*(Q+norm[None,:]);assert np.isfinite(ll).all()
   value=float(logsumexp(ll+np.log(wd[start:end,None])+np.log(wi)[None,:]));accum[floor]=float(np.logaddexp(accum[floor],value))
 for value in accum.values():
  if not np.isfinite(value):raise FloatingPointError('Non-finite marginal logL')
 NODE_EVALUATIONS+=nn
 return dict(logL=accum,method=method,order=int(order),D_nodes=nodeshape[0],i_nodes=nodeshape[1],density_nodes=nn,ZD=ZD,Zi=Zi,weight_sum=float(wd.sum()*wi.sum()),seconds=time.perf_counter()-begin,max_quadratic_roundoff_bound=max_roundoff,linear=linear)
