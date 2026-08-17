"""Disagreement-Gated Panel-Adaptive Conformal Sets.

The gate predicts whether a singleton is unlikely to capture the requested
expert plausibility mass q. It is learned only from the outer-train sample.
The untouched outer-calibration sample is then calibrated separately inside
low- and high-disagreement strata.

Low-disagreement cases use a stratum-specific global panel threshold. High-
disagreement cases use PACS-TopK with Mondrian residual calibration. Small
strata fall back to the full-calibration global panel threshold.
"""
from __future__ import annotations

import math
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from .pacs import (
    global_panel_quantile_threshold,
    normalize_probs,
    output_features,
    predict_global,
)
from .pacs_mondrian import PACSTopKMondrian


def _lower_order_quantile(values, alpha):
    """Conservative lower empirical quantile for a high-recall gate."""
    a = np.sort(np.asarray(values, float))
    n = len(a)
    if n == 0:
        return 1.0
    r = int(math.floor(alpha * (n + 1)))
    return float(a[0] if r < 1 else a[min(r - 1, n - 1)])


class DisagreementGatedPACS:
    def __init__(
        self,
        q_mass=.8,
        alpha_case=.1,
        alpha_gate=None,
        random_state=0,
        gate_fraction=.5,
        model_quantile=.9,
        n_bins=3,
        min_cal_per_bin=30,
        min_stratum_cal=40,
    ):
        self.q_mass = float(q_mass)
        self.alpha_case = float(alpha_case)
        self.alpha_gate = float(alpha_case if alpha_gate is None else alpha_gate)
        self.random_state = int(random_state)
        self.gate_fraction = float(gate_fraction)
        self.model_quantile = float(model_quantile)
        self.n_bins = int(n_bins)
        self.min_cal_per_bin = int(min_cal_per_bin)
        self.min_stratum_cal = int(min_stratum_cal)

        self.gate_ = GradientBoostingClassifier(
            n_estimators=180,
            learning_rate=.04,
            max_depth=2,
            min_samples_leaf=20,
            random_state=self.random_state,
        )
        self.constant_gate_ = None
        self.gate_threshold_ = None
        self.low_threshold_ = None
        self.high_model_ = None
        self.global_threshold_ = None
        self.use_high_pacs_ = False
        self.train_gate_rate_ = None
        self.cal_gate_rate_ = None

    def _hard_target(self, plausibility):
        lam = normalize_probs(plausibility)
        # If modal expert mass < q, no singleton can satisfy the q-mass target.
        return (np.max(lam, axis=1) < self.q_mass - 1e-12).astype(int)

    def _gate_score(self, conformity):
        n = len(np.asarray(conformity))
        if self.constant_gate_ is not None:
            return np.full(n, float(self.constant_gate_))
        p = self.gate_.predict_proba(output_features(conformity))
        j = int(np.flatnonzero(self.gate_.classes_ == 1)[0])
        return p[:, j]

    def _gate_ids(self, conformity):
        if self.gate_threshold_ is None:
            raise RuntimeError("fit first")
        return self._gate_score(conformity) >= self.gate_threshold_

    def fit(self, conformity_train, plausibility_train,
            conformity_cal, plausibility_cal):
        e_tr = np.asarray(conformity_train, float)
        lam_tr = normalize_probs(plausibility_train)
        e_ca = np.asarray(conformity_cal, float)
        lam_ca = normalize_probs(plausibility_cal)
        if e_tr.shape != lam_tr.shape or e_ca.shape != lam_ca.shape:
            raise ValueError("conformity/plausibility shape mismatch")
        if len(e_tr) < 2 or len(e_ca) < 1:
            raise ValueError("insufficient train/calibration cases")

        rng = np.random.default_rng(self.random_state + 9173)
        perm = rng.permutation(len(e_tr))
        n_fit = int(round(self.gate_fraction * len(e_tr)))
        n_fit = min(max(n_fit, 1), len(e_tr) - 1)
        gf, gc = perm[:n_fit], perm[n_fit:]

        y_fit = self._hard_target(lam_tr[gf])
        unique = np.unique(y_fit)
        if unique.size < 2:
            self.constant_gate_ = int(unique[0])
        else:
            self.constant_gate_ = None
            self.gate_.fit(output_features(e_tr[gf]), y_fit)

        score_gc = self._gate_score(e_tr[gc])
        y_gc = self._hard_target(lam_tr[gc])
        hard_scores = score_gc[y_gc == 1]
        if len(hard_scores) == 0:
            # No train evidence that q requires >1 label: keep adaptive branch off.
            self.gate_threshold_ = 1.0 + 1e-12
        else:
            self.gate_threshold_ = _lower_order_quantile(hard_scores, self.alpha_gate)

        self.global_threshold_ = global_panel_quantile_threshold(
            e_ca, lam_ca, self.q_mass, self.alpha_case
        )
        hi_tr = self._gate_ids(e_tr)
        hi_ca = self._gate_ids(e_ca)
        self.train_gate_rate_ = float(np.mean(hi_tr))
        self.cal_gate_rate_ = float(np.mean(hi_ca))

        lo_ca = ~hi_ca
        if int(np.sum(lo_ca)) >= self.min_stratum_cal:
            self.low_threshold_ = global_panel_quantile_threshold(
                e_ca[lo_ca], lam_ca[lo_ca], self.q_mass, self.alpha_case
            )
        else:
            self.low_threshold_ = self.global_threshold_

        min_train = max(80, 2 * self.min_stratum_cal)
        if int(np.sum(hi_tr)) >= min_train and int(np.sum(hi_ca)) >= self.min_stratum_cal:
            self.high_model_ = PACSTopKMondrian(
                self.q_mass,
                self.alpha_case,
                random_state=self.random_state + 101,
                model_quantile=self.model_quantile,
                n_bins=self.n_bins,
                min_cal_per_bin=self.min_cal_per_bin,
            ).fit(e_tr[hi_tr], lam_tr[hi_tr], e_ca[hi_ca], lam_ca[hi_ca])
            self.use_high_pacs_ = True
        else:
            self.high_model_ = None
            self.use_high_pacs_ = False
        return self

    def predict(self, conformity):
        e = np.asarray(conformity, float)
        hi = self._gate_ids(e)
        out = np.zeros(e.shape, dtype=bool)
        lo = ~hi
        if np.any(lo):
            out[lo] = predict_global(e[lo], self.low_threshold_)
        if np.any(hi):
            if self.use_high_pacs_:
                out[hi] = self.high_model_.predict(e[hi])
            else:
                out[hi] = predict_global(e[hi], self.global_threshold_)
        return out

    def diagnostics(self):
        return {
            "gate_threshold": None if self.gate_threshold_ is None else float(self.gate_threshold_),
            "train_gate_rate": self.train_gate_rate_,
            "cal_gate_rate": self.cal_gate_rate_,
            "use_high_pacs": bool(self.use_high_pacs_),
            "low_threshold": None if self.low_threshold_ is None else float(self.low_threshold_),
            "global_threshold": None if self.global_threshold_ is None else float(self.global_threshold_),
            "constant_gate": self.constant_gate_,
        }
