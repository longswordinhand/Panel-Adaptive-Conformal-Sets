import numpy as np
from src.pacs_gated import DisagreementGatedPACS


def _probs(n, k, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.gamma(2, 1, (n, k))
    return x / x.sum(1, keepdims=True)


def test_consensus_constant_gate_closes_adaptive_branch():
    e = _probs(180, 3, 1)
    lam = np.zeros_like(e)
    lam[:, 0] = .95
    lam[:, 1] = .03
    lam[:, 2] = .02
    m = DisagreementGatedPACS(q_mass=.8, random_state=3, min_stratum_cal=10)
    m.fit(e[:120], lam[:120], e[120:], lam[120:])
    assert m.constant_gate_ == 0
    assert m.train_gate_rate_ == 0.0
    assert m.cal_gate_rate_ == 0.0
    assert not m.use_high_pacs_
    out = m.predict(e[120:])
    assert out.shape == (60, 3)
    assert np.all(out.sum(1) >= 1)


def test_mixed_gate_prediction_shape_and_nonempty_sets():
    e = _probs(360, 4, 2)
    lam = np.zeros_like(e)
    lam[:180] = [.92, .03, .03, .02]
    lam[180:] = [.35, .30, .20, .15]
    m = DisagreementGatedPACS(
        q_mass=.8, random_state=4, min_stratum_cal=20, min_cal_per_bin=10
    )
    m.fit(e[:240], lam[:240], e[240:300], lam[240:300])
    out = m.predict(e[300:])
    assert out.shape == (60, 4)
    assert out.dtype == bool
    assert np.all(out.sum(1) >= 1)


def test_hard_target_matches_singleton_feasibility():
    m = DisagreementGatedPACS(q_mass=.8)
    lam_easy = np.tile([.85, .10, .05], (20, 1))
    lam_hard = np.tile([.60, .25, .15], (20, 1))
    assert np.all(m._hard_target(lam_easy) == 0)
    assert np.all(m._hard_target(lam_hard) == 1)
