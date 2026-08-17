#!/usr/bin/env python3
import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.special import comb
from scipy.stats import beta as beta_dist


def cp_bounds(x, n, alpha):
    if x == 0:
        lo = 0.0
    else:
        lo = beta_dist.ppf(alpha/2, x, n-x+1)
    if x == n:
        hi = 1.0
    else:
        hi = beta_dist.ppf(1-alpha/2, x+1, n-x)
    return float(lo), float(hi)


def simultaneous_multinomial_box(counts, alpha=0.05):
    counts=np.asarray(counts,int); n=int(counts.sum()); d=len(counts)
    a=alpha/d
    lows=[]; highs=[]
    for x in counts:
        lo,hi=cp_bounds(int(x),n,a)
        lows.append(lo); highs.append(hi)
    return np.array(lows),np.array(highs)


def margin_rows(grid,beta,gamma,C,tmax=0.5,nr=80):
    d=np.abs(grid-beta)
    rs=np.unique(d[(d>1e-12)&(d<=tmax+1e-12)])
    if len(rs)>nr:
        ids=np.unique(np.round(np.geomspace(1,len(rs),nr)).astype(int)-1)
        rs=rs[ids]
    A=[]; b=[]
    for t in rs:
        rhs=min(1.0,C*(t**gamma))
        A.append((d<=t+1e-12).astype(float)); b.append(rhs)
    return A,b


def tail_ci_from_counts(counts,beta=0.5,alpha=0.05,grid_n=2001,margin=None):
    counts=np.asarray(counts,int); m=len(counts)-1
    x=np.linspace(0,1,grid_n)
    B=np.vstack([comb(m,k)*(x**k)*((1-x)**(m-k)) for k in range(m+1)])
    lo,hi=simultaneous_multinomial_box(counts,alpha)
    Aub=[]; bub=[]
    for k in range(m+1):
        Aub.append(B[k]); bub.append(hi[k])
        Aub.append(-B[k]); bub.append(-lo[k])
    if margin is not None:
        gamma,C=margin
        A2,b2=margin_rows(x,beta,gamma,C)
        Aub.extend(A2); bub.extend(b2)
    Aeq=np.ones((1,grid_n)); beq=np.array([1.0])
    tail=(x>beta).astype(float)
    kw=dict(A_ub=np.asarray(Aub),b_ub=np.asarray(bub),A_eq=Aeq,b_eq=beq,bounds=(0,None),method='highs')
    rlo=linprog(tail,**kw); rhi=linprog(-tail,**kw)
    if not (rlo.success and rhi.success):
        return np.nan,np.nan
    return float(rlo.fun),float(-rhi.fun)


def mix_sampler(rng,n,kind):
    if kind=='beta_2_5': return rng.beta(2,5,n)
    if kind=='beta_5_2': return rng.beta(5,2,n)
    if kind=='bimodal':
        z=rng.random(n)<0.5
        out=np.empty(n); out[z]=rng.beta(2,8,z.sum()); out[~z]=rng.beta(8,2,(~z).sum()); return out
    if kind=='near_threshold': return rng.beta(20,20,n)
    if kind=='asymmetric_mix':
        z=rng.random(n)<0.7
        out=np.empty(n); out[z]=rng.beta(2,10,z.sum()); out[~z]=rng.beta(8,3,(~z).sum()); return out
    raise ValueError(kind)


def true_tail(kind,beta=0.5):
    if kind=='beta_2_5': return 1-beta_dist.cdf(beta,2,5)
    if kind=='beta_5_2': return 1-beta_dist.cdf(beta,5,2)
    if kind=='bimodal': return .5*(1-beta_dist.cdf(beta,2,8))+.5*(1-beta_dist.cdf(beta,8,2))
    if kind=='near_threshold': return 1-beta_dist.cdf(beta,20,20)
    if kind=='asymmetric_mix': return .7*(1-beta_dist.cdf(beta,2,10))+.3*(1-beta_dist.cdf(beta,8,3))


def estimate_margin_C(kind,gamma=1,beta=0.5):
    # numerical population bound C >= sup_t P(|theta-beta|<=t)/t^gamma, t<=0.5
    ts=np.linspace(0.002,0.5,2000)
    def cdf(v):
        v=np.clip(v,0,1)
        if kind=='beta_2_5': return beta_dist.cdf(v,2,5)
        if kind=='beta_5_2': return beta_dist.cdf(v,5,2)
        if kind=='bimodal': return .5*beta_dist.cdf(v,2,8)+.5*beta_dist.cdf(v,8,2)
        if kind=='near_threshold': return beta_dist.cdf(v,20,20)
        if kind=='asymmetric_mix': return .7*beta_dist.cdf(v,2,10)+.3*beta_dist.cdf(v,8,3)
    mass=np.array([cdf(beta+t)-cdf(beta-t) for t in ts])
    return float(np.max(mass/(ts**gamma))*1.02)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--reps',type=int,default=300); ap.add_argument('--grid',type=int,default=1201); ap.add_argument('--seed',type=int,default=20260815); args=ap.parse_args()
    rng=np.random.default_rng(args.seed)
    out=[]
    kinds=['beta_2_5','bimodal','near_threshold','asymmetric_mix']
    for kind in kinds:
        truth=true_tail(kind)
        C=estimate_margin_C(kind,gamma=1)
        for n in [200,800]:
            for m in [3,5,10]:
                cov0=cov1=0; w0=[]; w1=[]
                for r in range(args.reps):
                    th=mix_sampler(rng,n,kind); K=rng.binomial(m,th); counts=np.bincount(K,minlength=m+1)
                    l0,u0=tail_ci_from_counts(counts,grid_n=args.grid,margin=None)
                    l1,u1=tail_ci_from_counts(counts,grid_n=args.grid,margin=(1.0,C))
                    if np.isfinite(l0): cov0 += int(l0-1e-10<=truth<=u0+1e-10); w0.append(u0-l0)
                    if np.isfinite(l1): cov1 += int(l1-1e-10<=truth<=u1+1e-10); w1.append(u1-l1)
                out.append(dict(scenario=kind,n=n,m=m,true_tail=truth,gamma=1,C=C,coverage_unrestricted=cov0/args.reps,mean_width_unrestricted=np.mean(w0),coverage_margin=cov1/args.reps,mean_width_margin=np.mean(w1),width_reduction=1-np.mean(w1)/np.mean(w0)))
                print(out[-1])
    df=pd.DataFrame(out)
    p=Path('experiments/pilot/results/tail_honest_ci'); p.mkdir(parents=True,exist_ok=True)
    df.to_csv(p/'simulation_summary.csv',index=False)
    print('\n',df.to_string(index=False))

if __name__=='__main__': main()
