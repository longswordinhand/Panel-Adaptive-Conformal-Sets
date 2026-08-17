"""Panel-Tolerance Conformal Prediction (PTCP).

Two-layer calibration for finite expert panels.

Layer 1 uses a nonparametric binomial/order-statistic tolerance bound to
upper-bound the latent q-quantile of expert nonconformity scores for each case.
Layer 2 applies split conformal across cases to those tolerance scores.

The generic tolerance/order-statistic and split-conformal ingredients are
classical; this module composes them for finite multi-rater prediction sets.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np
from scipy.stats import binom


@dataclass(frozen=True)
class ToleranceRule:
    m: int
    q_mass: float
    delta: float
    order: int
    confidence: float


def tolerance_order(m: int, q_mass: float, delta: float) -> ToleranceRule:
    """Smallest order r whose upper order statistic covers latent q-quantile.

    For iid panel scores S_1,...,S_m with CDF F and q-quantile Q_q,
    P(S_(r) >= Q_q) >= P(Bin(m,q) <= r-1).

    We choose the smallest r with the RHS >= 1-delta.  If even r=m cannot
    achieve the requested confidence, the requested (m,q,delta) is infeasible.
    """
    if m < 1:
        raise ValueError("m must be >=1")
    if not (0.0 < q_mass < 1.0):
        raise ValueError("q_mass must be in (0,1)")
    if not (0.0 < delta < 1.0):
        raise ValueError("delta must be in (0,1)")
    target = 1.0 - delta
    confs = [float(binom.cdf(r - 1, m, q_mass)) for r in range(1, m + 1)]
    feasible = [r for r, c in enumerate(confs, start=1) if c + 1e-15 >= target]
    if not feasible:
        min_delta = q_mass ** m
        raise ValueError(
            f"infeasible finite-panel tolerance: delta={delta:g} < q_mass^m="
            f"{min_delta:g} for m={m}, q_mass={q_mass:g}"
        )
    r = feasible[0]
    return ToleranceRule(m=m, q_mass=q_mass, delta=delta, order=r,
                         confidence=confs[r - 1])


@dataclass(frozen=True)
class RandomizedToleranceRule:
    m: int
    q_mass: float
    delta: float
    lower_order: int
    upper_order: int
    prob_lower: float
    achieved_confidence: float


def randomized_tolerance_rule(m: int, q_mass: float, delta: float) -> RandomizedToleranceRule:
    """Randomize between adjacent order statistics to hit 1-delta exactly.

    Randomization is independent across cases.  When the target equals an
    attainable deterministic confidence, lower_order==upper_order.
    """
    if m < 1 or not (0 < q_mass < 1) or not (0 < delta < 1):
        raise ValueError("invalid m/q_mass/delta")
    target = 1.0 - delta
    g = np.array([binom.cdf(r - 1, m, q_mass) for r in range(1, m + 1)], float)
    if target > g[-1] + 1e-15:
        raise ValueError(
            f"infeasible finite-panel tolerance: delta={delta:g} < q_mass^m={q_mass**m:g}"
        )
    j = int(np.searchsorted(g, target, side="left"))
    if abs(g[j] - target) <= 1e-14 or j == 0:
        r = j + 1
        return RandomizedToleranceRule(m,q_mass,delta,r,r,1.0,float(g[j]))
    lo, hi = j, j + 1  # order numbers; g index lo-1 and hi-1
    g_lo, g_hi = g[j-1], g[j]
    p_lo = float((g_hi - target) / (g_hi - g_lo))
    achieved = p_lo * g_lo + (1.0-p_lo) * g_hi
    return RandomizedToleranceRule(m,q_mass,delta,lo,hi,p_lo,float(achieved))


def aps_label_scores(probs: np.ndarray) -> np.ndarray:
    """Deterministic APS nonconformity score for every label.

    score_y is cumulative predicted probability of labels ranked at least as
    probable as y.  Ties are deterministically resolved by stable label order.
    """
    p = np.asarray(probs, float)
    if p.ndim != 2:
        raise ValueError("probs must be [n_cases,n_classes]")
    if np.any(p < -1e-12):
        raise ValueError("negative probability")
    rowsum = p.sum(axis=1)
    if not np.allclose(rowsum, 1.0, atol=1e-6):
        raise ValueError("rows must sum to 1")
    n, k = p.shape
    out = np.empty_like(p)
    for i in range(n):
        order = np.argsort(-p[i], kind="stable")
        cs = np.cumsum(p[i, order])
        out[i, order] = cs
    return out


def binary_probability_to_matrix(p_positive: Sequence[float]) -> np.ndarray:
    pp = np.asarray(p_positive, float)
    if pp.ndim != 1 or np.any((pp < 0) | (pp > 1)):
        raise ValueError("p_positive must be one-dimensional in [0,1]")
    return np.column_stack([1.0-pp, pp])


def panel_tolerance_scores(label_scores: np.ndarray,
                           panel_labels: np.ndarray,
                           q_mass: float,
                           delta: float,
                           rng: np.random.Generator | None = None,
                           randomized: bool = False) -> np.ndarray:
    """Compute one finite-panel tolerance score per case.

    label_scores: [n,k], score for candidate labels.
    panel_labels: [n,m], integer expert labels.  Constant m in this function.
    """
    scores = np.asarray(label_scores, float)
    y = np.asarray(panel_labels)
    if scores.ndim != 2 or y.ndim != 2 or len(scores) != len(y):
        raise ValueError("shape mismatch")
    n, k = scores.shape
    if np.any((y < 0) | (y >= k)):
        raise ValueError("panel label outside class range")
    m = y.shape[1]
    obs = np.take_along_axis(scores, y.astype(int), axis=1)
    obs.sort(axis=1)
    if randomized:
        rule = randomized_tolerance_rule(m, q_mass, delta)
        if rule.lower_order == rule.upper_order:
            orders = np.full(n, rule.lower_order, int)
        else:
            if rng is None:
                rng = np.random.default_rng(0)
            use_lo = rng.random(n) < rule.prob_lower
            orders = np.where(use_lo, rule.lower_order, rule.upper_order)
        return obs[np.arange(n), orders - 1]
    rule = tolerance_order(m, q_mass, delta)
    return obs[:, rule.order - 1]


def split_conformal_threshold(case_scores: Sequence[float], alpha_between: float) -> float:
    """Usual split-conformal upper quantile with +infinity convention."""
    a = np.sort(np.asarray(case_scores, float))
    if a.ndim != 1 or len(a) < 1:
        raise ValueError("case_scores must be nonempty vector")
    if not (0 < alpha_between < 1):
        raise ValueError("alpha_between must be in (0,1)")
    rank = int(math.ceil((len(a)+1)*(1.0-alpha_between)))
    if rank > len(a):
        return math.inf
    return float(a[rank-1])


def ptcp_threshold(label_scores_cal: np.ndarray,
                   panel_labels_cal: np.ndarray,
                   q_mass: float,
                   alpha: float,
                   delta: float,
                   randomized: bool=False,
                   rng: np.random.Generator | None=None) -> float:
    if not (0 < delta < alpha < 1):
        raise ValueError("need 0 < delta < alpha < 1")
    u = panel_tolerance_scores(label_scores_cal, panel_labels_cal, q_mass,
                               delta, rng=rng, randomized=randomized)
    return split_conformal_threshold(u, alpha_between=alpha-delta)


def prediction_set_mask(label_scores: np.ndarray, threshold: float) -> np.ndarray:
    return np.asarray(label_scores, float) <= threshold


def expert_mass_captured(mask: np.ndarray, latent_q: np.ndarray) -> np.ndarray:
    m = np.asarray(mask, bool); q = np.asarray(latent_q, float)
    if m.shape != q.shape:
        raise ValueError("mask/q shape mismatch")
    return (m*q).sum(axis=1)
