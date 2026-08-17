"""Safe-PACS: training-only selection between adaptive PACS and global fallback."""
from __future__ import annotations
import numpy as np
from .pacs import plausibility_mass, global_panel_quantile_threshold, predict_global
from .pacs_v2 import PACSTopK


class SafePACSTopK:
    """Use adaptive PACS only when inner training validation supports efficiency gain.

    Method/quantile selection uses only the outer training split. The outer
    calibration split remains untouched until the selected method is fitted.
    """
    def __init__(self,q_mass=.9,alpha_case=.1,random_state=0,
                 quantile_grid=(.5,.65,.75,.85,.9,.95),coverage_slack=.01):
        self.q_mass=q_mass; self.alpha_case=alpha_case; self.random_state=random_state
        self.quantile_grid=tuple(quantile_grid); self.coverage_slack=coverage_slack
        self.selected_method_=None; self.selected_quantile_=None
        self.inner_results_=None; self.model_=None; self.global_threshold_=None

    def _eval(self,mask,lam):
        mass=plausibility_mass(mask,lam); sizes=np.asarray(mask).sum(1)
        return float(np.mean(mass>=self.q_mass-1e-12)),float(np.mean(sizes)),float(np.quantile(sizes,.9))

    def fit(self,conformity_train,plausibility_train,conformity_cal,plausibility_cal):
        e=np.asarray(conformity_train,float); lam=np.asarray(plausibility_train,float)
        n=len(e); rng=np.random.default_rng(self.random_state+271828); p=rng.permutation(n)
        nfit=max(1,int(.5*n)); ncal=max(1,int(.25*n)); fit=p[:nfit]; ical=p[nfit:nfit+ncal]; val=p[nfit+ncal:]
        rows=[]; target=1-self.alpha_case
        # Global candidate on inner calibration/validation.
        gt=global_panel_quantile_threshold(e[ical],lam[ical],self.q_mass,self.alpha_case)
        succ,ms,p90=self._eval(predict_global(e[val],gt),lam[val])
        rows.append(dict(method='global',model_quantile=np.nan,success=succ,mean_size=ms,p90_size=p90))
        for mq in self.quantile_grid:
            m=PACSTopK(self.q_mass,self.alpha_case,self.random_state,model_quantile=mq)
            m.fit(e[fit],lam[fit],e[ical],lam[ical]); succ,ms,p90=self._eval(m.predict(e[val]),lam[val])
            rows.append(dict(method='adaptive',model_quantile=float(mq),success=succ,mean_size=ms,p90_size=p90))
        feasible=[r for r in rows if r['success']>=target-self.coverage_slack]
        if feasible:
            best=min(feasible,key=lambda r:(r['mean_size'],r['p90_size'],-r['success']))
        else:
            best=max(rows,key=lambda r:(r['success'],-r['mean_size']))
        self.inner_results_=rows; self.selected_method_=best['method']; self.selected_quantile_=best['model_quantile']
        if self.selected_method_=='global':
            self.global_threshold_=global_panel_quantile_threshold(conformity_cal,plausibility_cal,self.q_mass,self.alpha_case)
        else:
            self.model_=PACSTopK(self.q_mass,self.alpha_case,self.random_state,model_quantile=float(self.selected_quantile_))
            self.model_.fit(conformity_train,plausibility_train,conformity_cal,plausibility_cal)
        return self

    def predict(self,conformity):
        if self.selected_method_=='global': return predict_global(conformity,self.global_threshold_)
        return self.model_.predict(conformity)
