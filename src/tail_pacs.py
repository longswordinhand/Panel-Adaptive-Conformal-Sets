"""Tail-PACS: training-only tail-inefficiency selection under panel-mass coverage."""
from __future__ import annotations
import numpy as np
from .pacs import plausibility_mass, global_panel_quantile_threshold, predict_global
from .pacs_v2 import PACSTopK


def cvar_upper(x, level=.9):
    a=np.sort(np.asarray(x,float)); n=len(a); j=max(0,int(np.floor(level*n)))
    return float(np.mean(a[j:])) if j<n else float(a[-1])


class TailPACSTopK:
    def __init__(self,q_mass=.9,alpha_case=.1,random_state=0,
                 quantile_grid=(.5,.65,.75,.85,.9,.95),coverage_slack=.01,tail_level=.9):
        self.q_mass=q_mass; self.alpha_case=alpha_case; self.random_state=random_state
        self.quantile_grid=tuple(quantile_grid); self.coverage_slack=coverage_slack; self.tail_level=tail_level
        self.selected_method_=None; self.selected_quantile_=None; self.inner_results_=None
        self.model_=None; self.global_threshold_=None
    def _eval(self,mask,lam):
        mass=plausibility_mass(mask,lam); sizes=np.asarray(mask).sum(1)
        return dict(success=float(np.mean(mass>=self.q_mass-1e-12)),mean_size=float(np.mean(sizes)),
                    p90_size=float(np.quantile(sizes,.9)),cvar_size=cvar_upper(sizes,self.tail_level))
    def fit(self,conformity_train,plausibility_train,conformity_cal,plausibility_cal):
        e=np.asarray(conformity_train,float); lam=np.asarray(plausibility_train,float); n=len(e)
        rng=np.random.default_rng(self.random_state+161803); p=rng.permutation(n); nfit=max(1,int(.5*n)); ncal=max(1,int(.25*n)); fit=p[:nfit]; ical=p[nfit:nfit+ncal]; val=p[nfit+ncal:]
        rows=[]; target=1-self.alpha_case
        gt=global_panel_quantile_threshold(e[ical],lam[ical],self.q_mass,self.alpha_case)
        r=self._eval(predict_global(e[val],gt),lam[val]); r.update(method='global',model_quantile=np.nan); rows.append(r)
        for mq in self.quantile_grid:
            m=PACSTopK(self.q_mass,self.alpha_case,self.random_state,model_quantile=mq); m.fit(e[fit],lam[fit],e[ical],lam[ical])
            r=self._eval(m.predict(e[val]),lam[val]); r.update(method='adaptive',model_quantile=float(mq)); rows.append(r)
        feasible=[r for r in rows if r['success']>=target-self.coverage_slack]
        if feasible: best=min(feasible,key=lambda r:(r['cvar_size'],r['p90_size'],r['mean_size'],-r['success']))
        else: best=max(rows,key=lambda r:(r['success'],-r['cvar_size']))
        self.inner_results_=rows; self.selected_method_=best['method']; self.selected_quantile_=best['model_quantile']
        if self.selected_method_=='global':
            self.global_threshold_=global_panel_quantile_threshold(conformity_cal,plausibility_cal,self.q_mass,self.alpha_case)
        else:
            self.model_=PACSTopK(self.q_mass,self.alpha_case,self.random_state,model_quantile=float(self.selected_quantile_)); self.model_.fit(conformity_train,plausibility_train,conformity_cal,plausibility_cal)
        return self
    def predict(self,conformity):
        if self.selected_method_=='global': return predict_global(conformity,self.global_threshold_)
        return self.model_.predict(conformity)
