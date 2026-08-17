"""Efficiency-selected PACS-TopK with outer-calibration validity preserved.

Hyperparameters are selected using only an inner split of the outer training
sample. Each candidate is itself inner-conformalized, and selection minimizes
validation set size. The untouched outer calibration split is then used for the
final split-conformal correction, so hyperparameter selection does not consume
outer calibration validity.
"""
from __future__ import annotations
import numpy as np
from .pacs_v2 import PACSTopK

class PACSTopKEffCV:
    def __init__(self,q_mass=.9,alpha_case=.1,random_state=0,
                 quantile_grid=(.5,.65,.75,.85,.9,.95)):
        self.q_mass=q_mass; self.alpha_case=alpha_case; self.random_state=random_state
        self.quantile_grid=tuple(quantile_grid); self.selected_quantile_=None
        self.inner_results_=None; self.model_=None
    def _select(self,e,lam):
        n=len(e); rng=np.random.default_rng(self.random_state+271828); p=rng.permutation(n)
        nfit=max(1,int(.5*n)); ncal=max(1,int(.25*n)); fit=p[:nfit]; cal=p[nfit:nfit+ncal]; val=p[nfit+ncal:]
        rows=[]
        for mq in self.quantile_grid:
            m=PACSTopK(self.q_mass,self.alpha_case,self.random_state,model_quantile=mq)
            m.fit(e[fit],lam[fit],e[cal],lam[cal]); mask=m.predict(e[val]); sizes=mask.sum(1)
            mass=(mask*np.asarray(lam[val],float)).sum(1)
            rows.append(dict(model_quantile=float(mq),mean_size=float(np.mean(sizes)),
                             p90_size=float(np.quantile(sizes,.9)),success=float(np.mean(mass>=self.q_mass-1e-12))))
        # Efficiency is the tuning objective; final validity comes from the untouched outer calibration split.
        best=min(rows,key=lambda r:(r['mean_size'],r['p90_size'],-r['success']))
        return best['model_quantile'],rows
    def fit(self,conformity_train,plausibility_train,conformity_cal,plausibility_cal):
        mq,rows=self._select(np.asarray(conformity_train,float),np.asarray(plausibility_train,float))
        self.selected_quantile_=float(mq); self.inner_results_=rows
        self.model_=PACSTopK(self.q_mass,self.alpha_case,self.random_state,model_quantile=mq)
        self.model_.fit(conformity_train,plausibility_train,conformity_cal,plausibility_cal)
        return self
    def predict(self,conformity): return self.model_.predict(conformity)
    def k_values(self,conformity): return self.model_.k_values(conformity)
