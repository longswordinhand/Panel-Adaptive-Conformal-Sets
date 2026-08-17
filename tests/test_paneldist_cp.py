import numpy as np

from src.paneldist_cp import (
    PanelDistCP,
    aps_label_scores,
    panel_quantile_scores,
    panel_recall,
    prediction_sets_from_threshold,
    split_conformal_threshold,
)


def test_aps_scores_follow_probability_order():
    probs = np.array([[0.6, 0.3, 0.1]])
    scores = aps_label_scores(probs)
    np.testing.assert_allclose(scores[0], [0.6, 0.9, 1.0])


def test_panel_quantile_retains_repeated_votes():
    probs = np.array([[0.6, 0.3, 0.1]])
    panel = [[0, 0, 0, 1, 2]]
    # q=.6 -> ceil(3) -> third smallest observed score = score(label 0)=.6
    score = panel_quantile_scores(probs, panel, q=0.6)
    np.testing.assert_allclose(score, [0.6])
    # q=.8 -> fourth smallest -> label 1 score=.9
    score = panel_quantile_scores(probs, panel, q=0.8)
    np.testing.assert_allclose(score, [0.9])


def test_threshold_uses_plus_one_correction():
    scores = np.arange(1, 10, dtype=float) / 10
    # n=9, alpha=.2 => ceil(10*.8)=8 => .8
    assert split_conformal_threshold(scores, alpha=0.2) == 0.8


def test_threshold_is_infinite_when_nominal_rank_exceeds_calibration_size():
    scores = [0.2, 0.4, 0.6]
    assert np.isinf(split_conformal_threshold(scores, alpha=0.1))


def test_prediction_sets_are_probability_prefixes():
    probs = np.array([[0.5, 0.3, 0.2], [0.2, 0.7, 0.1]])
    sets = prediction_sets_from_threshold(probs, threshold=0.81)
    assert sets[0].tolist() == [0, 1]
    assert sets[1].tolist() == [1]


def test_panel_recall_counts_vote_mass_not_unique_labels():
    assert panel_recall([0], [0, 0, 0, 1, 2]) == 0.6
    assert panel_recall([0, 1], [0, 0, 0, 1, 2]) == 0.8


def test_fit_predict_smoke():
    calibration_probs = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.55, 0.35, 0.10],
            [0.2, 0.7, 0.1],
            [0.4, 0.3, 0.3],
            [0.6, 0.2, 0.2],
            [0.1, 0.8, 0.1],
            [0.45, 0.45, 0.10],
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
            [0.5, 0.25, 0.25],
        ]
    )
    panel = [
        [0, 0, 0, 0, 1],
        [0, 0, 1, 1, 1],
        [1, 1, 1, 1, 0],
        [0, 1, 2, 2, 2],
        [0, 0, 0, 2, 2],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 0, 1],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 2, 2],
        [0, 0, 1, 2, 2],
    ]
    model = PanelDistCP.fit(calibration_probs, panel, q=0.8, alpha=0.2)
    sets = model.predict(np.array([[0.55, 0.30, 0.15]]))
    assert len(sets) == 1
    assert sets[0].ndim == 1
    assert sets[0].size >= 1
