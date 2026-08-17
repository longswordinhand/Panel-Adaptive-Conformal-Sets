import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multirater_conformal_seg.calibration.pixel_interval_cp import (
    ambiguity_metrics,
    calibrate_all_raters,
    calibrate_consensus,
    calibrate_naive_annotations,
    calibrate_random_rater,
    consensus_mask,
    inclusion_score,
    make_prediction_band,
    mask_is_covered,
    split_conformal_quantile,
)


def test_inclusion_score_matches_membership_threshold():
    p = np.array([[0.9, 0.6], [0.4, 0.1]])
    y = np.array([[1, 1], [0, 0]])
    s = inclusion_score(p, y)
    assert np.isclose(s, 0.0)
    assert mask_is_covered(y, make_prediction_band(p, s))


def test_score_handles_empty_mask():
    p = np.array([[0.9, 0.2], [0.1, 0.05]])
    y = np.zeros((2, 2), dtype=np.uint8)
    s = inclusion_score(p, y)
    assert np.isclose(s, 0.4)
    assert mask_is_covered(y, make_prediction_band(p, s))


def test_universal_band_at_half_covers_empty_and_nonempty():
    p = np.array([[0.9, 0.2], [0.1, 0.05]])
    band = make_prediction_band(p, 0.5)
    assert not band.lower.any()
    assert band.upper.all()
    assert mask_is_covered(np.zeros((2, 2), dtype=np.uint8), band)
    assert mask_is_covered(np.ones((2, 2), dtype=np.uint8), band)


def test_split_conformal_small_n_falls_back_to_universal():
    # n=11, alpha=0.05 => ceil(12*0.95)=12 > 11
    scores = np.linspace(0.0, 0.4, 11)
    assert split_conformal_quantile(scores, 0.05) == 0.5


def test_consensus_ties_are_foreground():
    masks = [
        np.array([[1, 0], [0, 0]]),
        np.array([[0, 0], [0, 0]]),
    ]
    c = consensus_mask(masks)
    assert bool(c[0, 0])


def test_all_rater_q_not_smaller_than_random_rater_for_same_cases():
    probs = [np.array([[0.9, 0.1]]), np.array([[0.8, 0.2]])]
    masks = [
        [np.array([[1, 0]]), np.array([[0, 0]])],
        [np.array([[1, 0]]), np.array([[1, 1]])],
    ]
    rng = np.random.default_rng(1)
    q_random = calibrate_random_rater(probs, masks, 0.5, rng)
    q_all = calibrate_all_raters(probs, masks, 0.5)
    assert q_all >= q_random


def test_calibration_functions_return_valid_q():
    probs = [np.array([[0.9, 0.1]]) for _ in range(4)]
    masks = [[np.array([[1, 0]]), np.array([[1, 0]])] for _ in range(4)]
    rng = np.random.default_rng(0)
    qs = [
        calibrate_consensus(probs, masks, 0.5),
        calibrate_naive_annotations(probs, masks, 0.5),
        calibrate_random_rater(probs, masks, 0.5, rng),
        calibrate_all_raters(probs, masks, 0.5),
    ]
    assert all(0.0 <= q <= 0.5 for q in qs)


def test_ambiguity_metrics():
    p = np.array([[0.9, 0.55], [0.45, 0.1]])
    band = make_prediction_band(p, 0.1)
    m = ambiguity_metrics(band, np.array([[1, 1], [0, 0]]))
    assert m["ambiguity_area_px"] == 2
    assert np.isclose(m["ambiguity_fraction_image"], 0.5)
