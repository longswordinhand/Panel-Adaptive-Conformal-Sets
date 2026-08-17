#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from src.pacs import normalize_probs,global_panel_quantile_threshold,predict_global
from src.pacs_mondrian import PACSTopKMondrian
from scripts.run_dermatology_pacs import irn_from_selectors,metrics,mccp,top1_cp
ROOT=Path('third_party/uncertain_ground_truth/data')
ap=argparse.ArgumentParser(); ap.add_argument('--reps',type=int,default=10); ap.add_argument('--q',type=float,default=.9); a=ap.parse_args(); q=a.q
selectors=json.load(open(ROOT/'dermatology_selectors.json')); lam=irn_from_selectors(selectors)
preds=[normalize_probs(np.loadtxt(ROOT/f'dermatology_predictions{i}.txt')) for i in range(4)]
out=Path('experiments/pr_rescue/results'); out.mkdir(parents=True,exist_ok=True); rows=[]; tag=str(q).replace('.','p')
for rep in range(a.reps):
 for model_idx,e in enumerate(preds):
  n=len(e); rng=np.random.default_rng(2026081800+100000*rep+10000*model_idx+int(1000*q)); perm=rng.permutation(n); ntr=int(.4*n); ncal=int(.3*n); tr=perm[:ntr]; cal=perm[ntr:ntr+ncal]; te=perm[ntr+ncal:]
  methods=[]
  th=global_panel_quantile_threshold(e[cal],lam[cal],q,.1); methods.append(('global',predict_global(e[te],th)))
  th=top1_cp(e[cal],lam[cal],.1); methods.append(('top1_cp',e[te]>=th))
  th=mccp(e[cal],lam[cal],.1,rng,10); methods.append(('mccp10',e[te]>=th))
  mq=.90
  pacs=PACSTopKMondrian(q,.1,random_state=rep*101+model_idx,model_quantile=mq,n_bins=3,min_cal_per_bin=40).fit(e[tr],lam[tr],e[cal],lam[cal])
  methods.append(('pacs_mondrian',pacs.predict(e[te])))
  for method,mask in methods:
   r=metrics(mask,lam[te],q); r.update(rep=rep,model=model_idx,method=method); rows.append(r)
 pd.DataFrame(rows).to_csv(out/f'dermatology_pacs_mondrian_fixed_q{tag}_raw.csv',index=False) if 'tag' in globals() else None
 print('done rep',rep,flush=True)
df=pd.DataFrame(rows); df.to_csv(out/f'dermatology_pacs_mondrian_fixed_q{tag}_raw.csv',index=False)
s=df.groupby('method').agg(success=('case_mass_success','mean'),success_sd=('case_mass_success','std'),mean_size=('mean_size','mean'),mean_size_sd=('mean_size','std'),p90_size=('p90_size','mean'),mean_mass=('mean_mass','mean'),minority=('minority_fraction','mean'),runs=('rep','count')).reset_index(); s.to_csv(out/f'dermatology_pacs_mondrian_fixed_q{tag}_summary.csv',index=False); print(s.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
