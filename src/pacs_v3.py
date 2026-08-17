"""PACS v3: heteroscedastic normalized-conformal top-k sets."""
from __future__ import annotations
import math
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from .pacs import output_features
from .pacs_v2 import required_topk, topk_mask


def upper_conformal_quantile(scores, alpha):
    a=np.sort(np.asarray(scores,float)); n=len(a)
    r=int(math.ceil((n+1)*(1-alpha)))
    return math.inf if r>n else float(a[r-1])


class PACSNormalizedTopK:
    """Predict location and scale of required top-k, then conformalize residual/scale."""
    def __init__(self,q_mass=.8,alpha_case=.1,random_state=0):
        self.q_mass=q_mass; self.alpha_case=alpha_case; self.random_state=random_state
        self.loc=GradientBoostingRegressor(n_estimators=220,learning_rate=.04,max_depth=2,
            min_samples_leaf=18,loss='huber',random_state=random_state)
        self.scale=GradientBoostingRegressor(n_estimators=180,learning_rate=.04,max_depth=2,
            min_samples_leaf=22,loss='huber',random_state=random_state+7919)
        self.q_=None
    def fit(self,conformity_train,plausibility_train,conformity_cal,plausibility_cal):
        xtr=output_features(conformity_train)
        ktr=required_topk(conformity_train,plausibility_train,self.q_mass).astype(float)
        ytr=np.log1p(ktr)
        self.loc.fit(xtr,ytr)
        loc_tr=self.loc.predict(xtr)
        # Predict conditional absolute log-residual scale; floor stabilizes normalization.
        absr=np.abs(ytr-loc_tr)
        self.scale.fit(xtr,np.log(absr+0.05))
        xcal=output_features(conformity_cal)
        kcal=required_topk(conformity_cal,plausibility_cal,self.q_mass).astype(float)
        loc_cal=self.loc.predict(xcal)
        sc=np.exp(self.scale.predict(xcal))+0.05
        # One-sided upper score in log1p(k) space.
        scores=(np.log1p(kcal)-loc_cal)/sc
        self.q_=upper_conformal_quantile(scores,self.alpha_case)
        return self
    def k_values(self,conformity):
        x=output_features(conformity)
        loc=self.loc.predict(x); sc=np.exp(self.scale.predict(x))+0.05
        upper=loc+self.q_*sc
        k=np.ceil(np.expm1(upper)).astype(int)
        return np.clip(k,1,np.asarray(conformity).shape[1])
    def predict(self,conformity):
        return topk_mask(conformity,self.k_values(conformity))
