#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np,pandas as pd
from src.pacs import normalize_probs,global_panel_quantile_threshold,predict_global
from src.pacs_v2 import PACSTopK
from scripts.run_dermatology_pacs import irn_from_selectors,metrics
ROOT=Path('third_party/uncertain_ground_truth/data'); q=.9
selectors=json.load(open(ROOT/'dermatology_selectors.json')); lam=irn_from_selectors(selectors)
rows=[]
for model_idx in range(4):
 e=normalize_probs(np.loadtxt(ROOT/f'dermatology_predictions{model_idx}.txt')); n=len(e)
 rng=np.random.default_rng(2026081500+10000*model_idx+900); perm=rng.permutation(n); ntr=int(.4*n); ncal=int(.3*n)
 tr=perm[:ntr]; cal=perm[ntr:ntr+ncal]; te=perm[ntr+ncal:]
 th=global_panel_quantile_threshold(e[cal],lam[cal],q,.1); r=metrics(predict_global(e[te],th),lam[te],q); r.update(model=model_idx,method='global'); rows.append(r)
 for mq in [.5,.65,.75,.85,.9,.95]:
  mask=PACSTopK(q,.1,0,model_quantile=mq).fit(e[tr],lam[tr],e[cal],lam[cal]).predict(e[te])
  r=metrics(mask,lam[te],q); r.update(model=model_idx,method=f'topk_q{mq:.2f}'); rows.append(r)
 print('done',model_idx,flush=True)
df=pd.DataFrame(rows); s=df.groupby('method').agg(success=('case_mass_success','mean'),mean_size=('mean_size','mean'),p90_size=('p90_size','mean'),mean_mass=('mean_mass','mean'),minority=('minority_fraction','mean')).reset_index().sort_values(['mean_size']); print(s.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
out=Path('experiments/pr_rescue/results'); df.to_csv(out/'topk_quantile_q09_raw.csv',index=False); s.to_csv(out/'topk_quantile_q09_summary.csv',index=False)
