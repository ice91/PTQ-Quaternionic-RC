"""Isolated normalized B0 statistical reference; no production imports or optimizer.
Velocity law equations and constants preserve the frozen seven parameterizations.
Physical nuisance marginal and local Gaussian surrogate are deliberately distinct.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.special import ndtr,logsumexp

C_LIGHT=299792458.;KM=1000.;KPC=3.085677581491367e19;G=6.67430e-11;MSUN=1.98847e30
H0=67.4*KM/(1000*KPC)
MODELS=('baryon','mond','ptq','ptq-nu','ptq-screen','mond-screen','nfw1p')
@dataclass
class Galaxy:
 name:str
 r:np.ndarray
 y:np.ndarray
 e:np.ndarray
 disk:np.ndarray
 bulge:np.ndarray
 gas:np.ndarray
 D:float
 sD:float
 inc:float
 si:float

def load_sample(path):
 df=pd.read_csv(path);out=[]
 for name,sub in sorted(df.groupby('galaxy')):
  a=sub.sort_values('r_kpc');first=a.iloc[0]
  out.append(Galaxy(name,*[a[c].to_numpy(float) for c in ['r_kpc','v_obs_kms','v_err_kms','v_disk_kms','v_bulge_kms','v_gas_kms']],float(first.D_Mpc),float(first.D_err_Mpc),np.deg2rad(first.i_deg),np.deg2rad(first.i_err_deg)))
 return out

def decode(model,theta,n):
 t=np.asarray(theta,float);j=0;p={'model':model}
 if model in ['ptq','ptq-nu','ptq-screen']:
  p['eps']=float(t[j]);j+=1
  if model!='ptq':p['a0']=p['eps']*C_LIGHT*H0
 elif model in ['mond','mond-screen']:
  p['a0']=float(10**t[j]);j+=1
 if model in ['ptq-screen','mond-screen']:p['q']=float(np.exp(t[j]));j+=1
 p['U']=t[j:j+n].copy();j+=n
 if model=='nfw1p':p['logM']=t[j:j+n].copy();j+=n
 p['sigma']=float(np.exp(t[j]));j+=1
 if len(t)!=j:raise ValueError('Unexpected parameter count')
 p['k']=j
 return p

def halo_squared(r,logM):
 mass=10.**logM;c=10.*(mass/1e12)**(-.1)
 rho=3*H0**2/(8*np.pi*G);R200=(3*mass*MSUN/(4*np.pi*200*rho))**(1/3)/KPC
 x=np.asarray(r)*c/R200;fx=np.log1p(x)-x/(1+x);fc=np.log1p(c)-c/(1+c)
 w=G*mass*MSUN*fx/(fc*np.asarray(r)*KPC)/KM**2
 eta=x*x/((1+x)**2*fx)-1.
 return w,eta

def velocity(g,p,gi,D=None):
 """Broadcast D as (...,1) to evaluate a fixed angular sample's templates/radii."""
 if D is None:D=g.D
 s=np.asarray(D)/g.D
 if np.any(s<=0):raise ValueError('D must be positive')
 r=g.r*s;disk=g.disk*np.sqrt(s);bulge=g.bulge*np.sqrt(s);gas=g.gas*np.sqrt(s)
 bar=p['U'][gi]*(disk*disk+bulge*bulge)+gas*gas
 m=p['model']
 if m=='baryon':w=bar
 elif m=='ptq':w=bar+p['eps']*C_LIGHT*H0*r*KPC/KM**2
 elif m in ['mond','ptq-nu','ptq-screen','mond-screen']:
  acc=bar*KM**2/np.maximum(r*KPC,1e-12)
  # Same admitted domain/floors as frozen laws; local/full historical points avoid floors.
  a0=max(p['a0'],1e-16 if m=='mond' else 1e-30)
  y=np.maximum(acc/a0,1e-12)
  nu=.5+np.sqrt(.25+np.power(y,-p.get('q',1.)))
  w=bar*nu
 elif m=='nfw1p':w=bar+halo_squared(r,p['logM'][gi])[0]
 else:raise ValueError(m)
 return np.sqrt(w)

def mean(g,p,gi,D=None,inc=None):
 if inc is None:inc=g.inc
 return velocity(g,p,gi,D)*np.sin(inc)/np.sin(g.inc)

def derivatives(g,p,gi):
 v=velocity(g,p,gi)
 if p['model']!='nfw1p':jd=v/(2*g.D)
 else:
  bar=p['U'][gi]*(g.disk*g.disk+g.bulge*g.bulge)+g.gas*g.gas
  h,eta=halo_squared(g.r,p['logM'][gi]);jd=(bar+h*eta)/(2*v*g.D)
 ji=v/np.tan(g.inc)
 return jd,ji

def covariance(g,p,gi):
 jd,ji=derivatives(g,p,gi)
 return np.diag(g.e*g.e+p['sigma']**2)+np.outer(jd,jd)*g.sD**2+np.outer(ji,ji)*g.si**2

def gaussian(y,mu,C):
 """Normalized Gaussian; reject non-SPD input rather than alter target with jitter."""
 C=.5*(C+C.T);L=np.linalg.cholesky(C);r=np.asarray(y)-np.asarray(mu);z=solve_triangular(L,r,lower=True)
 quad=float(z@z);ld=float(2*np.log(np.diag(L)).sum());norm=float(len(r)*np.log(2*np.pi))
 return dict(quad=quad,logdet=ld,normalization=norm,logL=-.5*(quad+ld+norm),jitter=0.)

def evaluate(g,p,gi):
 v=velocity(g,p,gi);C=covariance(g,p,gi)
 return gaussian(g.y,v,C),v,C

def quadrature(g,p,gi,T,order,constructions=('B0','B1','linear_B0')):
 """Fixed y0 density; D/i priors normalized in z coordinates, no double counting."""
 dl=max(-float(T),-g.D/g.sD);du=float(T)
 il=max(-float(T),-g.inc/g.si);iu=min(float(T),(np.pi/2-g.inc)/g.si)
 if dl>=du or il>=iu:raise ValueError('Empty physical nuisance domain')
 z,w=np.polynomial.legendre.leggauss(order)
 zd=dl+(z+1)*(du-dl)/2;zi=il+(z+1)*(iu-il)/2
 Zd=ndtr(du)-ndtr(dl);Zi=ndtr(iu)-ndtr(il)
 wd=w*(du-dl)/2*np.exp(-zd*zd/2)/np.sqrt(2*np.pi)/Zd
 wi=w*(iu-il)/2*np.exp(-zi*zi/2)/np.sqrt(2*np.pi)/Zi
 bd=np.sin(g.inc+g.si*zi)/np.sin(g.inc)
 vs=velocity(g,p,gi,(g.D+g.sD*zd)[:,None])
 jd,ji=derivatives(g,p,gi);v0=velocity(g,p,gi);out=[]
 logweight=np.log(wd)[:,None]+np.log(wi)[None,:]
 for construction in constructions:
  mu=(v0[None,None,:]+jd[None,None,:]*(g.sD*zd[:,None,None])+ji[None,None,:]*(g.si*zi[None,:,None])) if construction=='linear_B0' else vs[:,None,:]*bd[None,:,None]
  var=g.e[None,None,:]**2+(p['sigma']**2 if construction!='B1' else (p['sigma']*bd[None,:,None])**2)
  ll=-.5*np.sum((g.y[None,None,:]-mu)**2/var+np.log(2*np.pi*var),axis=-1)
  out.append(dict(construction=construction,T=T,order=order,logL=float(logsumexp(ll+logweight)),prior_weight_sum=float(wd.sum()*wi.sum()),zD_lower=dl,zD_upper=du,zi_lower=il,zi_upper=iu,ZD=Zd,Zi=Zi,retained_mass_relative_physical=float(Zd*Zi/(ndtr(g.D/g.sD)*(ndtr((np.pi/2-g.inc)/g.si)-ndtr(-g.inc/g.si))))))
 return out

def matched_parameters(p,target):
 """Same physical response parameters, U, q, sigma; no independently optimized values."""
 q={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in p.items()};q['model']=target
 if target.startswith('ptq'):q['eps']=q['a0']/(C_LIGHT*H0)
 # velocity uses a0 for both nu parameterizations; no numerical remapping drift.
 return q
