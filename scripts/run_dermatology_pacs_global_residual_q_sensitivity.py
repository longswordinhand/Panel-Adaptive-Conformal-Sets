#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd
from src.pacs import normalize_probs, global_panel_quantile_threshold, predict_global
from src.pacs_v2 import PACSTopK
from scripts.run_dermatology_pacs import irn_from_selectors, metrics

ROOT=Path('third_party/uncertain_ground_truth/data')
OUT=Path('experiments/pr_rescue/results'); OUT.mkdir(parents=True,exist_ok=True)
selectors=json.load(open(ROOT/'dermatology_selectors.json')); lam=irn_from_selectors(selectors)
preds=[normalize_probs(np.loadtxt(ROOT/f'dermatology_predictions{i}.txt')) for i in range(4)]
for q in (0.7,0.8):
    rows=[]; reps=6
    for rep in range(reps):
        for model_idx,e in enumerate(preds):
            n=len(e); rng=np.random.default_rng(2026081800+100000*rep+10000*model_idx+int(1000*q)); perm=rng.permutation(n)
            ntr=int(.4*n); ncal=int(.3*n); tr=perm[:ntr]; cal=perm[ntr:ntr+ncal]; te=perm[ntr+ncal:]
            th=global_panel_quantile_threshold(e[cal],lam[cal],q,.1)
            methods=[('global_panel',predict_global(e[te],th))]
            pacs=PACSTopK(q,.1,random_state=rep*101+model_idx,model_quantile=.9).fit(e[tr],lam[tr],e[cal],lam[cal])
            methods.append(('pacs_global_residual',pacs.predict(e[te])))
            for method,mask in methods:
                r=metrics(mask,lam[te],q); r.update(rep=rep,model=model_idx,method=method); rows.append(r)
        print('q',q,'done rep',rep,flush=True)
    df=pd.DataFrame(rows); tag=str(q).replace('.','p'); df.to_csv(OUT/f'dermatology_pacs_global_residual_q{tag}_raw.csv',index=False)
    s=df.groupby('method').agg(success=('case_mass_success','mean'),mean_size=('mean_size','mean'),p90_size=('p90_size','mean'),runs=('rep','count')).reset_index(); s.to_csv(OUT/f'dermatology_pacs_global_residual_q{tag}_summary.csv',index=False)
    print(s.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
