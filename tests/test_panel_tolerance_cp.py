import math
import numpy as np
from scipy.stats import binom
import pytest

from src.panel_tolerance_cp import (
    aps_label_scores, binary_probability_to_matrix, tolerance_order,
    randomized_tolerance_rule, panel_tolerance_scores,
    split_conformal_threshold, expert_mass_captured,
)

def test_tolerance_order_guarantee_and_minimality():
    r=tolerance_order(5,0.5,0.1)
    assert r.order==5
    assert r.confidence >= .9
    assert binom.cdf(r.order-2,5,.5) < .9

def test_tolerance_infeasible_floor():
    with pytest.raises(ValueError): tolerance_order(5,.8,.1)

def test_randomized_rule_exact_confidence():
    r=randomized_tolerance_rule(5,.5,.1)
    assert (r.lower_order,r.upper_order)==(4,5)
    assert 0 < r.prob_lower < 1
    assert abs(r.achieved_confidence-.9)<1e-12

def test_aps_scores_binary():
    p=binary_probability_to_matrix([.8,.3])
    s=aps_label_scores(p)
    assert np.allclose(s[0],[1,.8])
    assert np.allclose(s[1],[.7,1])

def test_panel_tolerance_score_is_requested_order_statistic():
    probs=np.array([[.7,.2,.1]])
    s=aps_label_scores(probs)
    y=np.array([[0,0,1,2,0]])
    # q=.5, delta=.1 => r=5, maximum observed score.
    u=panel_tolerance_scores(s,y,.5,.1)
    assert u.shape==(1,)
    assert u[0]==pytest.approx(1.0)

def test_split_conformal_infinity_small_calibration():
    assert math.isinf(split_conformal_threshold([.1,.2,.3],.1))

def test_expert_mass():
    mask=np.array([[1,0,1],[0,1,0]],bool)
    q=np.array([[.2,.3,.5],[.1,.7,.2]])
    assert np.allclose(expert_mass_captured(mask,q),[.7,.7])
