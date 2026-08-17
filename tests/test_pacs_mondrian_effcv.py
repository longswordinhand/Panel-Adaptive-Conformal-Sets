import numpy as np
from src.pacs_mondrian_effcv import PACSTopKMondrianEffCV


def test_nested_selection_and_prediction():
    rng=np.random.default_rng(12)
    n=320; c=8
    e=rng.dirichlet(np.ones(c),size=n)
    lam=.7*e+.3*rng.dirichlet(np.ones(c),size=n)
    tr=np.arange(0,180); cal=np.arange(180,280); te=np.arange(280,320)
    m=PACSTopKMondrianEffCV(.8,.1,random_state=3,
                            quantile_grid=(.55,.75,.9),n_bins=2,
                            min_cal_per_bin=10)
    m.fit(e[tr],lam[tr],e[cal],lam[cal])
    mask=m.predict(e[te])
    assert mask.shape==(40,c)
    assert m.selected_quantile_ in (.55,.75,.9)
    assert len(m.inner_results_)==3
    assert np.all(mask.sum(1)>=1)
