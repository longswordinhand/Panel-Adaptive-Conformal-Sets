#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np,pandas as pd
from src.pacs import normalize_probs,global_panel_quantile_threshold,predict_global
from src.tail_pacs import TailPACSTopK,cvar_upper
from scripts.run_dermatology_pacs import irn_from_selectors,metrics
ROOT=Path('third_party/uncertain_ground_truth/data'); q=.9
selectors=json.load(open(ROOT/'dermatology_selectors.json')); lam=irn_from_selectors(selectors); rows=[]
for model_idx in range(4):
 e=normalize_probs(np.loadtxt(ROOT/f'dermatology_predictions{model_idx}.txt')); n=len(e); rng=np.random.default_rng(2026081500+10000*model_idx+900); perm=rng.permutation(n); ntr=int(.4*n); ncal=int(.3*n); tr=perm[:ntr]; cal=perm[ntr:ntr+ncal]; te=perm[ntr+ncal:]
 th=global_panel_quantile_threshold(e[cal],lam[cal],q,.1); gm=predict_global(e[te],th); r=metrics(gm,lam[te],q); r.update(model=model_idx,method='global',selected='global',selected_q=np.nan,cvar90=cvar_upper(gm.sum(1))); rows.append(r)
 m=TailPACSTopK(q,.1,random_state=model_idx).fit(e[tr],lam[tr],e[cal],lam[cal]); mask=m.predict(e[te]); r=metrics(mask,lam[te],q); r.update(model=model_idx,method='tail_pacs',selected=m.selected_method_,selected_q=m.selected_quantile_,cvar90=cvar_upper(mask.sum(1))); rows.append(r); print('model',model_idx,'selected',m.selected_method_,m.selected_quantile_,flush=True)
df=pd.DataFrame(rows); print(df[['model','method','selected','selected_q','case_mass_success','mean_size','p90_size','cvar90']].to_string(index=False,float_format=lambda x:f'{x:.4f}')); print('\n',df.groupby('method').agg(success=('case_mass_success','mean'),mean_size=('mean_size','mean'),p90_size=('p90_size','mean'),cvar90=('cvar90','mean')).reset_index().to_string(index=False,float_format=lambda x:f'{x:.4f}'))
