"""Difficulty-stratified PACS-TopK (Mondrian calibration).

The difficulty partition is learned without outer-calibration labels.  Each
calibration/test case is assigned to a bin using the model-predicted required
Top-K size, and a separate one-sided conformal correction is applied in each
bin.  This preserves ordinary Mondrian split-conformal validity conditional on
the pre-defined bin while reducing correction transfer from hard to easy cases.
"""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from .pacs import output_features
from .pacs_v2 import required_topk, topk_mask, upper_conformal_quantile


class PACSTopKMondrian:
    def __init__(self, q_mass=.9, alpha_case=.1, random_state=0,
                 model_quantile=.75, n_bins=3, min_cal_per_bin=30):
        self.q_mass=float(q_mass); self.alpha_case=float(alpha_case)
        self.random_state=int(random_state); self.model_quantile=float(model_quantile)
        self.n_bins=int(n_bins); self.min_cal_per_bin=int(min_cal_per_bin)
        self.reg=GradientBoostingRegressor(
            n_estimators=260, learning_rate=.035, max_depth=2,
            min_samples_leaf=20, loss='quantile', alpha=self.model_quantile,
            random_state=self.random_state)
        self.edges_=None; self.corrections_=None; self.global_correction_=None

    def _pred_k(self,e):
        return np.expm1(self.reg.predict(output_features(e)))

    def _bin_ids(self,pred):
        # searchsorted on internal edges -> 0..B-1
        return np.searchsorted(self.edges_, np.asarray(pred,float), side='right')

    def fit(self, conformity_train, plausibility_train,
            conformity_cal, plausibility_cal):
        e_tr=np.asarray(conformity_train,float); e_ca=np.asarray(conformity_cal,float)
        kt=required_topk(e_tr,plausibility_train,self.q_mass).astype(float)
        self.reg.fit(output_features(e_tr),np.log1p(kt))
        ptr=self._pred_k(e_tr)
        # Predefine bins from training predictions only, never outer-cal labels.
        probs=np.linspace(0,1,self.n_bins+1)[1:-1]
        edges=np.unique(np.quantile(ptr,probs)) if len(probs) else np.array([])
        self.edges_=np.asarray(edges,float)

        pca=self._pred_k(e_ca)
        kca=required_topk(e_ca,plausibility_cal,self.q_mass).astype(float)
        resid=kca-pca
        self.global_correction_=upper_conformal_quantile(resid,self.alpha_case)
        bids=self._bin_ids(pca)
        corrections=[]
        for b in range(len(self.edges_)+1):
            z=resid[bids==b]
            if len(z)<self.min_cal_per_bin:
                corrections.append(self.global_correction_)
            else:
                corrections.append(upper_conformal_quantile(z,self.alpha_case))
        self.corrections_=np.asarray(corrections,float)
        return self

    def k_values(self,conformity):
        if self.corrections_ is None: raise RuntimeError('fit first')
        e=np.asarray(conformity,float); pred=self._pred_k(e); bids=self._bin_ids(pred)
        corr=self.corrections_[bids]
        k=np.ceil(pred+corr).astype(int)
        return np.clip(k,1,e.shape[1])

    def predict(self,conformity):
        return topk_mask(conformity,self.k_values(conformity))
