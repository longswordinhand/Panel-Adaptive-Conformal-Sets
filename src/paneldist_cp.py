"""Panel-aware conformal calibration utilities.

This module deliberately separates the post-hoc panel calibration layer from any
image classifier.  A backbone only needs to provide class probabilities.

Important novelty note:
    The finite-sample validity of the calibration step is an application of
    standard split conformal / quantile-risk ideas.  It must not be claimed as a
    new conformal theorem.  The research contribution, if any, must come from
    the multi-rater modeling / set construction / empirical behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


def _check_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 2:
        raise ValueError("probs must have shape (n_cases, n_classes)")
    if np.any(~np.isfinite(probs)) or np.any(probs < 0):
        raise ValueError("probs must be finite and nonnegative")
    row_sums = probs.sum(axis=1)
    if np.any(row_sums <= 0):
        raise ValueError("each probability row must have positive mass")
    return probs / row_sums[:, None]


def _check_panel(panel: Sequence[Sequence[int]], n_cases: int, n_classes: int) -> list[np.ndarray]:
    if len(panel) != n_cases:
        raise ValueError("panel length must equal number of cases")
    out: list[np.ndarray] = []
    for labels in panel:
        arr = np.asarray(labels, dtype=int)
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError("each case must have a non-empty 1D list of rater labels")
        if np.any((arr < 0) | (arr >= n_classes)):
            raise ValueError("panel labels out of class range")
        out.append(arr)
    return out


def aps_label_scores(probs: np.ndarray) -> np.ndarray:
    """Return deterministic APS score for every candidate label.

    score[i, y] is cumulative model probability through label y when classes are
    sorted by descending probability.  Smaller is more conforming.
    """
    probs = _check_probs(probs)
    n, k = probs.shape
    order = np.argsort(-probs, axis=1, kind="stable")
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    cumulative = np.cumsum(sorted_probs, axis=1)
    scores = np.empty_like(cumulative)
    rows = np.arange(n)[:, None]
    scores[rows, order] = cumulative
    return scores


def panel_quantile_scores(
    probs: np.ndarray,
    panel: Sequence[Sequence[int]],
    q: float,
) -> np.ndarray:
    """Case-level score needed to cover at least a q fraction of panel votes.

    For case i with m_i rater labels, let r_i = ceil(q * m_i).  We compute APS
    scores for each observed rater label and return the r_i-th smallest score.
    Repeated votes are intentionally retained: a 3/5 majority label contributes
    three observations because the target is vote mass, not unique-label recall.
    """
    if not (0 < q <= 1):
        raise ValueError("q must be in (0, 1]")
    probs = _check_probs(probs)
    panel_checked = _check_panel(panel, probs.shape[0], probs.shape[1])
    label_scores = aps_label_scores(probs)
    result = np.empty(probs.shape[0], dtype=float)
    for i, labels in enumerate(panel_checked):
        r = int(np.ceil(q * labels.size))
        observed = np.sort(label_scores[i, labels])
        result[i] = observed[r - 1]
    return result


def split_conformal_threshold(case_scores: Iterable[float], alpha: float) -> float:
    """Finite-sample split-conformal upper quantile with +1 correction."""
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0, 1)")
    scores = np.asarray(list(case_scores), dtype=float)
    if scores.ndim != 1 or scores.size == 0 or np.any(~np.isfinite(scores)):
        raise ValueError("case_scores must be a non-empty finite 1D sequence")
    n = scores.size
    rank = int(np.ceil((n + 1) * (1 - alpha)))
    if rank > n:
        return float("inf")
    return float(np.partition(scores, rank - 1)[rank - 1])


def prediction_sets_from_threshold(probs: np.ndarray, threshold: float) -> list[np.ndarray]:
    """Return APS-prefix prediction sets corresponding to a calibrated threshold."""
    probs = _check_probs(probs)
    if np.isnan(threshold):
        raise ValueError("threshold cannot be NaN")
    n, k = probs.shape
    order = np.argsort(-probs, axis=1, kind="stable")
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    cumulative = np.cumsum(sorted_probs, axis=1)
    sets: list[np.ndarray] = []
    for i in range(n):
        if np.isinf(threshold):
            sets.append(order[i].copy())
            continue
        # Include every class whose APS label score is <= threshold.  Guarantee
        # non-empty output by retaining top-1 if threshold is below top-1 score.
        keep = cumulative[i] <= threshold + 1e-15
        count = max(1, int(keep.sum()))
        sets.append(order[i, :count].copy())
    return sets


def panel_recall(prediction_set: Sequence[int], panel_labels: Sequence[int]) -> float:
    """Fraction of panel votes represented in a prediction set."""
    labels = np.asarray(panel_labels, dtype=int)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("panel_labels must be non-empty")
    chosen = set(int(x) for x in prediction_set)
    return float(np.mean([int(y) in chosen for y in labels]))


@dataclass(frozen=True)
class PanelDistCP:
    q: float
    alpha: float
    threshold: float

    @classmethod
    def fit(
        cls,
        calibration_probs: np.ndarray,
        calibration_panel: Sequence[Sequence[int]],
        q: float,
        alpha: float,
    ) -> "PanelDistCP":
        scores = panel_quantile_scores(calibration_probs, calibration_panel, q=q)
        threshold = split_conformal_threshold(scores, alpha=alpha)
        return cls(q=float(q), alpha=float(alpha), threshold=threshold)

    def predict(self, probs: np.ndarray) -> list[np.ndarray]:
        return prediction_sets_from_threshold(probs, self.threshold)
