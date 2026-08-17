"""PACS v2: efficiency-oriented adaptive panel-mass prediction sets.

Two variants are provided:
1. PACSQuantileThreshold: lower conditional quantile regression for the
   per-case conformity threshold needed to retain q expert plausibility mass.
2. PACSTopK: directly predicts the minimum top-k set size needed to retain q
   expert plausibility mass and conformalizes the one-sided size residual.

The finite-sample guarantee is ordinary marginal split-conformal coverage of
an observed derived case target; novelty is not claimed for the conformal
quantile itself.
"""
from __future__ import annotations
import math
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from .pacs import normalize_probs, output_features, required_conformity_threshold


def upper_conformal_quantile(scores, alpha):
    a=np.sort(np.asarray(scores,float)); n=len(a)
    r=int(math.ceil((n+1)*(1-alpha)))
    return math.inf if r>n else float(a[r-1])


def required_topk(conformity, plausibility, q_mass):
    """Minimum top-k-by-model-score needed to capture q plausibility mass."""
    e=np.asarray(conformity,float); lam=normalize_probs(plausibility)
    if e.shape != lam.shape or not 0 < q_mass <= 1:
        raise ValueError('shape/q')
    order=np.argsort(-e,axis=1,kind='stable')
    mass=np.take_along_axis(lam,order,axis=1)
    cs=np.cumsum(mass,axis=1)
    return np.argmax(cs >= q_mass-1e-15,axis=1)+1


def topk_mask(conformity, k_values):
    e=np.asarray(conformity,float); n,c=e.shape
    kv=np.asarray(k_values,int)
    if kv.ndim==0: kv=np.repeat(kv,n)
    kv=np.clip(kv,1,c)
    order=np.argsort(-e,axis=1,kind='stable')
    out=np.zeros((n,c),bool)
    for i,k in enumerate(kv): out[i,order[i,:k]]=True
    return out


class PACSQuantileThreshold:
    def __init__(self,q_mass=.8,alpha_case=.1,random_state=0,model_quantile=None):
        self.q_mass=q_mass; self.alpha_case=alpha_case; self.random_state=random_state
        aq=alpha_case if model_quantile is None else model_quantile
        self.reg=GradientBoostingRegressor(
            n_estimators=240,learning_rate=.035,max_depth=2,min_samples_leaf=20,
            loss='quantile',alpha=aq,random_state=random_state)
        self.correction_=None
    def fit(self,conformity_train,plausibility_train,conformity_cal,plausibility_cal):
        tt=required_conformity_threshold(conformity_train,plausibility_train,self.q_mass)
        self.reg.fit(output_features(conformity_train),np.log(tt))
        tc=required_conformity_threshold(conformity_cal,plausibility_cal,self.q_mass)
        pred=self.reg.predict(output_features(conformity_cal))
        self.correction_=upper_conformal_quantile(pred-np.log(tc),self.alpha_case)
        return self
    def thresholds(self,conformity):
        lp=self.reg.predict(output_features(conformity))-self.correction_
        t=np.exp(np.clip(lp,-35,0))
        return np.minimum(t,np.max(conformity,axis=1))
    def predict(self,conformity):
        t=self.thresholds(conformity)
        return np.asarray(conformity,float)>=t[:,None]


class PACSTopK:
    def __init__(self,q_mass=.8,alpha_case=.1,random_state=0,model_quantile=None):
        self.q_mass=q_mass; self.alpha_case=alpha_case; self.random_state=random_state
        aq=1-alpha_case if model_quantile is None else model_quantile
        self.reg=GradientBoostingRegressor(
            n_estimators=240,learning_rate=.035,max_depth=2,min_samples_leaf=20,
            loss='quantile',alpha=aq,random_state=random_state)
        self.correction_=None
    def fit(self,conformity_train,plausibility_train,conformity_cal,plausibility_cal):
        kt=required_topk(conformity_train,plausibility_train,self.q_mass).astype(float)
        # log1p stabilizes the long tail in 419-class dermatology.
        self.reg.fit(output_features(conformity_train),np.log1p(kt))
        kc=required_topk(conformity_cal,plausibility_cal,self.q_mass).astype(float)
        pred=np.expm1(self.reg.predict(output_features(conformity_cal)))
        self.correction_=upper_conformal_quantile(kc-pred,self.alpha_case)
        return self
    def k_values(self,conformity):
        pred=np.expm1(self.reg.predict(output_features(conformity)))
        k=np.ceil(pred+self.correction_).astype(int)
        return np.clip(k,1,np.asarray(conformity).shape[1])
    def predict(self,conformity):
        return topk_mask(conformity,self.k_values(conformity))
