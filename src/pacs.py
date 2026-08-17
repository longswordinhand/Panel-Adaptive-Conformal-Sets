"""Panel-Adaptive Conformal Sets (PACS).

Practical calibration for ambiguous/soft multi-expert ground truth.
The target for case i is T_i(q): the largest model conformity threshold whose
prediction set still captures at least q of the case's expert plausibility mass.
A regression model predicts log T_i(q) from test-time model-output features;
split conformal calibration of one-sided residuals turns this into an adaptive
lower threshold.  The construction is standard split-conformal on a derived
case target; the contribution is the panel-mass target and adaptive set design.
"""
from __future__ import annotations
import math
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


def normalize_probs(p):
    p=np.asarray(p,float)
    p=np.clip(p,0,None)
    s=p.sum(1,keepdims=True)
    if np.any(s<=0): raise ValueError('zero probability row')
    return p/s


def required_conformity_threshold(conformity, plausibility, q_mass):
    """Largest threshold retaining >=q_mass plausibility for each case."""
    e=np.asarray(conformity,float); lam=normalize_probs(plausibility)
    if e.shape!=lam.shape or not 0<q_mass<=1: raise ValueError('shape/q')
    n,k=e.shape; out=np.empty(n,float)
    for i in range(n):
        order=np.argsort(-e[i],kind='stable')
        cs=np.cumsum(lam[i,order])
        j=min(int(np.searchsorted(cs,q_mass,side='left')),k-1)
        out[i]=e[i,order[j]]
    return np.clip(out,1e-15,1.0)


def output_features(probs,top=30):
    p=normalize_probs(probs); n,k=p.shape
    s=np.sort(p,axis=1)[:,::-1]
    t=min(top,k); st=s[:,:t]
    ent=-(p*np.log(np.clip(p,1e-15,1))).sum(1,keepdims=True)
    entn=ent/max(np.log(k),1e-12)
    margin=(s[:,0]-s[:,1] if k>1 else s[:,0]).reshape(-1,1)
    maxp=s[:,0:1]
    eff=np.exp(ent).reshape(-1,1)
    ks=[1,2,3,5,10,20,50]
    cum=np.column_stack([s[:,:min(j,k)].sum(1) for j in ks])
    logtop=np.log(np.clip(st,1e-12,1))
    return np.column_stack([st,logtop,entn,margin,maxp,eff,cum])


def upper_conformal_quantile(scores,alpha):
    a=np.sort(np.asarray(scores,float)); n=len(a)
    r=int(math.ceil((n+1)*(1-alpha)))
    return math.inf if r>n else float(a[r-1])


def lower_conformal_quantile(values,alpha):
    """One-sided lower tolerance bound: P(V_new >= bound)>=1-alpha."""
    a=np.sort(np.asarray(values,float)); n=len(a)
    r=int(math.floor(alpha*(n+1)))
    return 0.0 if r<1 else float(a[r-1])


def global_panel_quantile_threshold(conformity_cal,plausibility_cal,q_mass,alpha_case):
    t=required_conformity_threshold(conformity_cal,plausibility_cal,q_mass)
    return lower_conformal_quantile(t,alpha_case)


def predict_global(conformity,threshold):
    e=np.asarray(conformity,float)
    return e>=threshold


class PACS:
    def __init__(self,q_mass=.8,alpha_case=.1,random_state=0):
        self.q_mass=q_mass; self.alpha_case=alpha_case; self.random_state=random_state
        self.reg=GradientBoostingRegressor(n_estimators=160,learning_rate=.04,max_depth=2,
            min_samples_leaf=15,loss='huber',random_state=random_state)
        self.correction_=None
    def fit(self,conformity_train,plausibility_train,conformity_cal,plausibility_cal):
        tt=required_conformity_threshold(conformity_train,plausibility_train,self.q_mass)
        self.reg.fit(output_features(conformity_train),np.log(tt))
        tc=required_conformity_threshold(conformity_cal,plausibility_cal,self.q_mass)
        pred=self.reg.predict(output_features(conformity_cal))
        # Need lower predicted threshold <= true T: pred-correction <= log(T).
        resid=pred-np.log(tc)
        self.correction_=upper_conformal_quantile(resid,self.alpha_case)
        return self
    def thresholds(self,conformity):
        if self.correction_ is None: raise RuntimeError('fit first')
        lp=self.reg.predict(output_features(conformity))-self.correction_
        t=np.exp(np.clip(lp,-35,0))
        # Guarantee nonempty by only enlarging sets.
        return np.minimum(t,np.max(conformity,axis=1))
    def predict(self,conformity):
        t=self.thresholds(conformity)
        return np.asarray(conformity,float)>=t[:,None]


def plausibility_mass(mask,plausibility):
    return (np.asarray(mask,bool)*normalize_probs(plausibility)).sum(1)
