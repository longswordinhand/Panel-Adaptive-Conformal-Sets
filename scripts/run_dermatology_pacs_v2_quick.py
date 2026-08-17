#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np, pandas as pd
from src.pacs import normalize_probs, global_panel_quantile_threshold, predict_global, plausibility_mass
from src.pacs_v2 import PACSQuantileThreshold, PACSTopK
from scripts.run_dermatology_pacs import irn_from_selectors, top1_cp, mccp, exact_soft_quantile, metrics

ROOT=Path('third_party/uncertain_ground_truth/data')
selectors=json.load(open(ROOT/'dermatology_selectors.json'))
lam=irn_from_selectors(selectors)
preds=[normalize_probs(np.loadtxt(ROOT/f'dermatology_predictions{i}.txt')) for i in range(4)]
rows=[]
for q_mass in [.5,.7,.8,.9]:
  for model_idx,e in enumerate(preds):
    n=len(e); rep=0
    rng=np.random.default_rng(2026081500+10000*model_idx+int(1000*q_mass))
    perm=rng.permutation(n); ntr=int(.4*n); ncal=int(.3*n)
    tr=perm[:ntr]; cal=perm[ntr:ntr+ncal]; te=perm[ntr+ncal:]
    methods=[]
    th=global_panel_quantile_threshold(e[cal],lam[cal],q_mass,.1)
    methods.append(('global',predict_global(e[te],th)))
    methods.append(('pacs_qthr',PACSQuantileThreshold(q_mass,.1,rep).fit(e[tr],lam[tr],e[cal],lam[cal]).predict(e[te])))
    methods.append(('pacs_topk',PACSTopK(q_mass,.1,rep).fit(e[tr],lam[tr],e[cal],lam[cal]).predict(e[te])))
    for method,mask in methods:
      r=metrics(mask,lam[te],q_mass); r.update(q_mass=q_mass,model=model_idx,method=method); rows.append(r)
    print('done',q_mass,model_idx,flush=True)
df=pd.DataFrame(rows)
s=df.groupby(['q_mass','method']).agg(success=('case_mass_success','mean'),mean_size=('mean_size','mean'),p90_size=('p90_size','mean'),mean_mass=('mean_mass','mean'),minority=('minority_fraction','mean')).reset_index()
print(s.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
out=Path('experiments/pr_rescue/results'); out.mkdir(parents=True,exist_ok=True)
df.to_csv(out/'dermatology_pacs_v2_quick_raw.csv',index=False)
s.to_csv(out/'dermatology_pacs_v2_quick_summary.csv',index=False)
