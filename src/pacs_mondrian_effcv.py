"""Nested efficiency-selected difficulty-stratified PACS-TopK.

Candidate conditional-quantile regressors are selected using only an inner split
of the outer training sample.  The untouched outer calibration sample is used
once for the final Mondrian one-sided correction.  This prevents test-set or
outer-calibration tuning while targeting tail efficiency (P90 set size) subject
to acceptable expert-mass success on the inner validation split.
"""
from __future__ import annotations
import numpy as np
from .pacs_mondrian import PACSTopKMondrian


class PACSTopKMondrianEffCV:
    def __init__(self, q_mass=.9, alpha_case=.1, random_state=0,
                 quantile_grid=(.55,.65,.75,.85,.9), n_bins=3,
                 min_cal_per_bin=20, success_slack=.03):
        self.q_mass=float(q_mass); self.alpha_case=float(alpha_case)
        self.random_state=int(random_state); self.quantile_grid=tuple(quantile_grid)
        self.n_bins=int(n_bins); self.min_cal_per_bin=int(min_cal_per_bin)
        self.success_slack=float(success_slack)
        self.selected_quantile_=None; self.inner_results_=None; self.model_=None

    @staticmethod
    def _mass(mask, lam):
        lam=np.asarray(lam,float)
        return (np.asarray(mask,bool)*lam).sum(1)

    def _select(self,e,lam):
        n=len(e); rng=np.random.default_rng(self.random_state+314159)
        p=rng.permutation(n)
        nfit=max(1,int(.50*n)); ncal=max(1,int(.25*n))
        fit=p[:nfit]; cal=p[nfit:nfit+ncal]; val=p[nfit+ncal:]
        rows=[]
        target=1-self.alpha_case-self.success_slack
        for mq in self.quantile_grid:
            m=PACSTopKMondrian(
                self.q_mass,self.alpha_case,self.random_state,
                model_quantile=mq,n_bins=self.n_bins,
                min_cal_per_bin=max(8,min(self.min_cal_per_bin,len(cal)//(2*self.n_bins))))
            m.fit(e[fit],lam[fit],e[cal],lam[cal])
            mask=m.predict(e[val]); size=mask.sum(1); mass=self._mass(mask,lam[val])
            rows.append(dict(model_quantile=float(mq),
                             success=float(np.mean(mass>=self.q_mass-1e-12)),
                             mean_size=float(np.mean(size)),
                             p90_size=float(np.quantile(size,.9))))
        feasible=[r for r in rows if r['success']>=target]
        if feasible:
            best=min(feasible,key=lambda r:(r['p90_size'],r['mean_size'],-r['success']))
        else:
            best=min(rows,key=lambda r:(-r['success'],r['p90_size'],r['mean_size']))
        return best['model_quantile'],rows

    def fit(self,conformity_train,plausibility_train,
            conformity_cal,plausibility_cal):
        e=np.asarray(conformity_train,float); lam=np.asarray(plausibility_train,float)
        mq,rows=self._select(e,lam)
        self.selected_quantile_=float(mq); self.inner_results_=rows
        self.model_=PACSTopKMondrian(
            self.q_mass,self.alpha_case,self.random_state,
            model_quantile=mq,n_bins=self.n_bins,
            min_cal_per_bin=self.min_cal_per_bin)
        self.model_.fit(conformity_train,plausibility_train,
                        conformity_cal,plausibility_cal)
        return self

    def predict(self,conformity):
        return self.model_.predict(conformity)

    def k_values(self,conformity):
        return self.model_.k_values(conformity)
