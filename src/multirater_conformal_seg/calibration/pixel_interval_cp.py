"""Simple nested pixel-interval prediction sets for the QUBIQ pilot.

This module is intentionally architecture-agnostic. It operates on a foreground
probability map and binary target masks.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class PredictionBand:
    lower: np.ndarray
    upper: np.ndarray
    q: float


def validate_probability_map(prob: np.ndarray) -> np.ndarray:
    p = np.asarray(prob, dtype=float)
    if not np.all(np.isfinite(p)):
        raise ValueError("Probability map contains non-finite values")
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("Probability map must lie in [0, 1]")
    return p


def validate_binary_mask(mask: np.ndarray, shape: tuple[int, ...] | None = None) -> np.ndarray:
    y = np.asarray(mask)
    if shape is not None and y.shape != shape:
        raise ValueError(f"Mask shape {y.shape} != probability shape {shape}")
    vals = np.unique(y)
    if not np.all(np.isin(vals, [0, 1])):
        raise ValueError(f"Mask must be binary, observed values {vals.tolist()}")
    return y.astype(bool, copy=False)


def inclusion_score(prob: np.ndarray, mask: np.ndarray) -> float:
    """Minimal q in [0, 0.5] such that mask belongs to C_q(prob).

    C_q is the set of all masks y satisfying L_q subseteq y subseteq U_q, with
    L_q = {p > 0.5+q} and U_q = {p >= 0.5-q}.
    """
    p = validate_probability_map(prob)
    y = validate_binary_mask(mask, p.shape)

    score = 0.0
    if np.any(~y):
        score = max(score, float(np.max(p[~y]) - 0.5))
    if np.any(y):
        score = max(score, float(0.5 - np.min(p[y])))
    return float(np.clip(score, 0.0, 0.5))


def make_prediction_band(prob: np.ndarray, q: float) -> PredictionBand:
    p = validate_probability_map(prob)
    q = float(q)
    if not (0.0 <= q <= 0.5):
        raise ValueError("q must lie in [0, 0.5]")
    lower = p > (0.5 + q)
    upper = p >= (0.5 - q)
    if np.any(lower & ~upper):
        raise AssertionError("Nested-set invariant violated: lower not subset of upper")
    return PredictionBand(lower=lower, upper=upper, q=q)


def mask_is_covered(mask: np.ndarray, band: PredictionBand) -> bool:
    y = validate_binary_mask(mask, band.lower.shape)
    return bool(np.all(~band.lower | y) and np.all(~y | band.upper))


def split_conformal_quantile(scores: Sequence[float] | np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal order statistic for bounded scores.

    The score range is [0, 0.5]. If ceil((n+1)(1-alpha)) > n, q=0.5 is
    returned, which corresponds to the universal prediction set in this family.
    """
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError("alpha must lie in (0, 1)")
    s = np.asarray(list(scores), dtype=float)
    if s.ndim != 1 or len(s) == 0:
        raise ValueError("scores must be a non-empty 1D sequence")
    if not np.all(np.isfinite(s)) or np.any((s < 0.0) | (s > 0.5)):
        raise ValueError("scores must be finite and lie in [0, 0.5]")
    k = int(math.ceil((len(s) + 1) * (1.0 - a)))
    if k > len(s):
        return 0.5
    return float(np.partition(s, k - 1)[k - 1])


def consensus_mask(masks: Sequence[np.ndarray]) -> np.ndarray:
    if len(masks) == 0:
        raise ValueError("At least one mask is required")
    first = validate_binary_mask(masks[0])
    stack = [first.astype(float)]
    for m in masks[1:]:
        stack.append(validate_binary_mask(m, first.shape).astype(float))
    # >= 0.5 means ties (possible with an even number of raters) are included.
    return np.mean(np.stack(stack, axis=0), axis=0) >= 0.5


def calibrate_consensus(prob_maps: Sequence[np.ndarray], masks_by_case: Sequence[Sequence[np.ndarray]], alpha: float) -> float:
    scores = [inclusion_score(p, consensus_mask(ms)) for p, ms in zip(prob_maps, masks_by_case)]
    return split_conformal_quantile(scores, alpha)


def calibrate_naive_annotations(prob_maps: Sequence[np.ndarray], masks_by_case: Sequence[Sequence[np.ndarray]], alpha: float) -> float:
    scores = [inclusion_score(p, m) for p, ms in zip(prob_maps, masks_by_case) for m in ms]
    return split_conformal_quantile(scores, alpha)


def calibrate_random_rater(
    prob_maps: Sequence[np.ndarray],
    masks_by_case: Sequence[Sequence[np.ndarray]],
    alpha: float,
    rng: np.random.Generator,
) -> float:
    scores = []
    for p, ms in zip(prob_maps, masks_by_case):
        if len(ms) == 0:
            raise ValueError("Each calibration case needs at least one expert mask")
        idx = int(rng.integers(0, len(ms)))
        scores.append(inclusion_score(p, ms[idx]))
    return split_conformal_quantile(scores, alpha)


def calibrate_all_raters(prob_maps: Sequence[np.ndarray], masks_by_case: Sequence[Sequence[np.ndarray]], alpha: float) -> float:
    case_scores = []
    for p, ms in zip(prob_maps, masks_by_case):
        if len(ms) == 0:
            raise ValueError("Each calibration case needs at least one expert mask")
        case_scores.append(max(inclusion_score(p, m) for m in ms))
    return split_conformal_quantile(case_scores, alpha)


def ambiguity_metrics(band: PredictionBand, consensus: np.ndarray | None = None) -> dict[str, float]:
    ambiguous = band.upper & ~band.lower
    n = int(ambiguous.size)
    area = int(np.count_nonzero(ambiguous))
    out = {
        "ambiguity_area_px": float(area),
        "ambiguity_fraction_image": float(area / n),
    }
    if consensus is not None:
        c = validate_binary_mask(consensus, band.lower.shape)
        denom = max(int(np.count_nonzero(c)), 1)
        out["ambiguity_to_consensus_ratio"] = float(area / denom)
    return out
