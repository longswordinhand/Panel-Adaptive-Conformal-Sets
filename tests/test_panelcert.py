import numpy as np
from src.panelcert import lower_latent_mass_fraction, panel_capture_counts, fixed_sequence_panelcert

def test_all_success_certifies_high_fraction():
    # 400 cases, m=5, all five captured: should certify >=90% have p>=.5.
    counts=np.array([0,0,0,0,0,400])
    lo,ok=lower_latent_mass_fraction(counts,.5,.05,1001)
    assert ok and lo>.9

def test_panel_counts():
    scores=np.array([[.2,.8],[.4,.9]])
    y=np.array([[0,1,0],[0,0,1]])
    c=panel_capture_counts(scores,y,.5)
    assert c.tolist()==[0,0,2,0]

def test_fixed_sequence_returns_finite_on_easy_data():
    # all raters use label 0; score of label 0=.2 for every case.
    s=np.tile([[.2,.9]],(400,1)); y=np.zeros((400,5),int)
    r=fixed_sequence_panelcert(s,y,.5,.1,.05,np.linspace(0,1,21),1001)
    assert r.certified and r.threshold<=.25
