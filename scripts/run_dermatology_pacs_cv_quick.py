#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np,pandas as pd
from src.pacs import normalize_probs,global_panel_quantile_threshold,predict_global
from src.pacs_cv import PACSTopKCV
from scripts.run_dermatology_pacs import irn_from_selectors,metrics
ROOT=Path('third_party/uncertain_ground_truth/data'); q=.9
selectors=json.load(open(ROOT/'dermatology_selectors.json')); lam=irn_from_selectors(selectors)
rows=[]
for model_idx in range(4):
 e=normalize_probs(np.loadtxt(ROOT/f'dermatology_predictions{model_idx}.txt')); n=len(e)
 rng=np.random.default_rng(2026081500+10000*model_idx+900); perm=rng.permutation(n); ntr=int(.4*n); ncal=int(.3*n)
 tr=perm[:ntr]; cal=perm[ntr:ntr+ncal]; te=perm[ntr+ncal:]
 th=global_panel_quantile_threshold(e[cal],lam[cal],q,.1); r=metrics(predict_global(e[te],th),lam[te],q); r.update(model=model_idx,method='global',selected_q=np.nan); rows.append(r)
 m=PACSTopKCV(q,.1,random_state=model_idx).fit(e[tr],lam[tr],e[cal],lam[cal]); mask=m.predict(e[te]); r=metrics(mask,lam[te],q); r.update(model=model_idx,method='pacs_cv',selected_q=m.selected_quantile_); rows.append(r); print('model',model_idx,'selected',m.selected_quantile_,flush=True)
df=pd.DataFrame(rows); print(df[['model','method','selected_q','case_mass_success','mean_size','p90_size','mean_mass','minority_fraction']].to_string(index=False,float_format=lambda x:f'{x:.4f}')); s=df.groupby('method').agg(success=('case_mass_success','mean'),mean_size=('mean_size','mean'),p90_size=('p90_size','mean'),mean_mass=('mean_mass','mean'),minority=('minority_fraction','mean')).reset_index(); print('\n',s.to_string(index=False,float_format=lambda x:f'{x:.4f}')); out=Path('experiments/pr_rescue/results'); df.to_csv(out/'dermatology_pacs_cv_quick_raw.csv',index=False); s.to_csv(out/'dermatology_pacs_cv_quick_summary.csv',index=False)
