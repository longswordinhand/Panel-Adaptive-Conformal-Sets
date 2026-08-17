"""Calibration-safe training-only hyperparameter selection for PACS-TopK."""
from __future__ import annotations
import numpy as np
from .pacs import plausibility_mass
from .pacs_v2 import PACSTopK


class PACSTopKCV:
    """Choose the conditional-quantile level using only the outer training split.

    The outer calibration split is untouched until the final conformal fit, so
    split-conformal validity is not spent on hyperparameter selection.
    """
    def __init__(self,q_mass=.9,alpha_case=.1,random_state=0,
                 quantile_grid=(.5,.65,.75,.85,.9,.95),coverage_slack=.01):
        self.q_mass=q_mass; self.alpha_case=alpha_case; self.random_state=random_state
        self.quantile_grid=tuple(quantile_grid); self.coverage_slack=coverage_slack
        self.selected_quantile_=None; self.inner_results_=None; self.model_=None

    def _select(self,e,lam):
        n=len(e); rng=np.random.default_rng(self.random_state+314159)
        p=rng.permutation(n); nfit=max(1,int(.5*n)); ncal=max(1,int(.25*n))
        fit=p[:nfit]; cal=p[nfit:nfit+ncal]; val=p[nfit+ncal:]
        if len(val)<2:
            return self.quantile_grid[len(self.quantile_grid)//2], []
        rows=[]; target=1-self.alpha_case
        for mq in self.quantile_grid:
            m=PACSTopK(self.q_mass,self.alpha_case,self.random_state,model_quantile=mq)
            m.fit(e[fit],lam[fit],e[cal],lam[cal]); mask=m.predict(e[val])
            mass=plausibility_mass(mask,lam[val]); sizes=mask.sum(1)
            rows.append(dict(model_quantile=float(mq),success=float(np.mean(mass>=self.q_mass-1e-12)),
                             mean_size=float(np.mean(sizes)),p90_size=float(np.quantile(sizes,.9))))
        feasible=[r for r in rows if r['success']>=target-self.coverage_slack]
        if feasible:
            best=min(feasible,key=lambda r:(r['mean_size'],r['p90_size'],-r['success']))
        else:
            best=max(rows,key=lambda r:(r['success'],-r['mean_size']))
        return best['model_quantile'],rows

    def fit(self,conformity_train,plausibility_train,conformity_cal,plausibility_cal):
        e=np.asarray(conformity_train,float); lam=np.asarray(plausibility_train,float)
        mq,rows=self._select(e,lam); self.selected_quantile_=float(mq); self.inner_results_=rows
        self.model_=PACSTopK(self.q_mass,self.alpha_case,self.random_state,model_quantile=mq)
        self.model_.fit(conformity_train,plausibility_train,conformity_cal,plausibility_cal)
        return self
    def k_values(self,conformity):
        return self.model_.k_values(conformity)
    def predict(self,conformity):
        return self.model_.predict(conformity)
