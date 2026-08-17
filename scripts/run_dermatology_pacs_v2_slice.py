#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from src.pacs import normalize_probs,global_panel_quantile_threshold,predict_global
from src.pacs_v2 import PACSQuantileThreshold,PACSTopK
from scripts.run_dermatology_pacs import irn_from_selectors,metrics
ROOT=Path('third_party/uncertain_ground_truth/data')
ap=argparse.ArgumentParser(); ap.add_argument('--q',type=float,required=True); args=ap.parse_args(); q=args.q
selectors=json.load(open(ROOT/'dermatology_selectors.json')); lam=irn_from_selectors(selectors)
rows=[]
for model_idx in range(4):
 e=normalize_probs(np.loadtxt(ROOT/f'dermatology_predictions{model_idx}.txt')); n=len(e)
 rng=np.random.default_rng(2026081500+10000*model_idx+int(1000*q)); perm=rng.permutation(n); ntr=int(.4*n); ncal=int(.3*n)
 tr=perm[:ntr]; cal=perm[ntr:ntr+ncal]; te=perm[ntr+ncal:]
 th=global_panel_quantile_threshold(e[cal],lam[cal],q,.1)
 methods=[('global',predict_global(e[te],th)),('pacs_qthr',PACSQuantileThreshold(q,.1,0).fit(e[tr],lam[tr],e[cal],lam[cal]).predict(e[te])),('pacs_topk',PACSTopK(q,.1,0).fit(e[tr],lam[tr],e[cal],lam[cal]).predict(e[te]))]
 for method,mask in methods:
  r=metrics(mask,lam[te],q); r.update(q_mass=q,model=model_idx,method=method); rows.append(r)
 print('done',q,model_idx,flush=True)
df=pd.DataFrame(rows); s=df.groupby('method').agg(success=('case_mass_success','mean'),mean_size=('mean_size','mean'),p90_size=('p90_size','mean'),mean_mass=('mean_mass','mean'),minority=('minority_fraction','mean')).reset_index(); print(s.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
out=Path('experiments/pr_rescue/results'); df.to_csv(out/f'dermatology_pacs_v2_q{q:.1f}_raw.csv',index=False); s.to_csv(out/f'dermatology_pacs_v2_q{q:.1f}_summary.csv',index=False)
