import numpy as np
from src.pacs_mondrian import PACSTopKMondrian
from src.pacs_v2 import required_topk

def test_required_and_fit_predict_shapes():
    rng=np.random.default_rng(4)
    n=240; c=6
    e=rng.dirichlet(np.ones(c),size=n)
    lam=.65*e+.35*rng.dirichlet(np.ones(c),size=n)
    tr=np.arange(0,120); ca=np.arange(120,200); te=np.arange(200,240)
    m=PACSTopKMondrian(.8,.1,7,model_quantile=.75,n_bins=2,min_cal_per_bin=10)
    m.fit(e[tr],lam[tr],e[ca],lam[ca])
    mask=m.predict(e[te]); kv=m.k_values(e[te])
    assert mask.shape==(40,c)
    assert np.all(mask.sum(1)==kv)
    assert np.all((kv>=1)&(kv<=c))
    assert len(m.corrections_)>=1

def test_more_required_mass_never_needs_fewer_labels():
    rng=np.random.default_rng(5)
    e=rng.dirichlet(np.ones(5),size=30)
    lam=rng.dirichlet(np.ones(5),size=30)
    assert np.all(required_topk(e,lam,.9)>=required_topk(e,lam,.7))
