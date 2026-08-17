#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from src.panel_tolerance_cp import aps_label_scores, prediction_set_mask, expert_mass_captured
from src.panelcert import fixed_sequence_panelcert
from scripts.run_ptcp_synthetic import sample_latent, noisy_model_probs, draw_panels

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--reps',type=int,default=60); ap.add_argument('--ntest',type=int,default=5000); args=ap.parse_args()
 rows=[]; thresholds=np.linspace(0,1,51)
 for regime in ['mixed','ambiguous']:
  for m in [3,5,10]:
   for rep in range(args.reps):
    seed=2026081700+rep+1000*m+(0 if regime=='mixed' else 100000); rng=np.random.default_rng(seed)
    qcal=sample_latent(rng,400,5,regime); qtest=sample_latent(rng,args.ntest,5,regime)
    pcal=noisy_model_probs(rng,qcal,.8); ptest=noisy_model_probs(rng,qtest,.8)
    scal=aps_label_scores(pcal); stest=aps_label_scores(ptest); panel=draw_panels(rng,qcal,m)
    r=fixed_sequence_panelcert(scal,panel,q_mass=.5,alpha_case=.1,confidence_error=.05,thresholds=thresholds,grid_n=501)
    mask=prediction_set_mask(stest,r.threshold); mass=expert_mass_captured(mask,qtest)
    rows.append(dict(regime=regime,m=m,rep=rep,threshold=r.threshold,certified=r.certified,
      lower_fraction=r.lower_fraction,tested_thresholds=r.tested_thresholds,
      latent_mass_success=float(np.mean(mass>=.5-1e-12)),set_size=float(np.mean(mask.sum(1))),
      full_set=float(np.mean(mask.sum(1)==5)),singleton=float(np.mean(mask.sum(1)==1))))
    if (rep+1)%20==0: print(regime,m,rep+1,flush=True)
 df=pd.DataFrame(rows); Path('experiments/pr_rescue/results').mkdir(parents=True,exist_ok=True)
 df.to_csv('experiments/pr_rescue/results/panelcert_synthetic_raw.csv',index=False)
 s=df.groupby(['regime','m']).agg(success=('latent_mass_success','mean'),success_sd=('latent_mass_success','std'),set_size=('set_size','mean'),full_set=('full_set','mean'),singleton=('singleton','mean'),threshold=('threshold','mean'),cert_rate=('certified','mean'),lower=('lower_fraction','mean')).reset_index()
 s.to_csv('experiments/pr_rescue/results/panelcert_synthetic_summary.csv',index=False)
 print(s.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
if __name__=='__main__': main()
