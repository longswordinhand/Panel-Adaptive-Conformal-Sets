#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from src.pacs import normalize_probs,global_panel_quantile_threshold,predict_global
from src.pacs_cv import PACSTopKCV
from src.safe_pacs import SafePACSTopK
from scripts.run_dermatology_pacs import irn_from_selectors,metrics,mccp,top1_cp
ROOT=Path('third_party/uncertain_ground_truth/data')
ap=argparse.ArgumentParser(); ap.add_argument('--reps',type=int,default=5); ap.add_argument('--q',type=float,default=.9); a=ap.parse_args(); q=a.q
selectors=json.load(open(ROOT/'dermatology_selectors.json')); lam=irn_from_selectors(selectors)
preds=[normalize_probs(np.loadtxt(ROOT/f'dermatology_predictions{i}.txt')) for i in range(4)]
out=Path('experiments/pr_rescue/results'); rows=[]
for rep in range(a.reps):
 for model_idx,e in enumerate(preds):
  n=len(e); rng=np.random.default_rng(2026081600+100000*rep+10000*model_idx+int(1000*q)); perm=rng.permutation(n); ntr=int(.4*n); ncal=int(.3*n); tr=perm[:ntr]; cal=perm[ntr:ntr+ncal]; te=perm[ntr+ncal:]
  methods=[]
  th=global_panel_quantile_threshold(e[cal],lam[cal],q,.1); methods.append(('global',predict_global(e[te],th),'global',np.nan))
  pcv=PACSTopKCV(q,.1,random_state=rep*101+model_idx).fit(e[tr],lam[tr],e[cal],lam[cal]); methods.append(('pacs_cv',pcv.predict(e[te]),'adaptive',pcv.selected_quantile_))
  safe=SafePACSTopK(q,.1,random_state=rep*101+model_idx).fit(e[tr],lam[tr],e[cal],lam[cal]); methods.append(('safe_pacs',safe.predict(e[te]),safe.selected_method_,safe.selected_quantile_))
  for method,mask,sel,sq in methods:
   r=metrics(mask,lam[te],q); r.update(rep=rep,model=model_idx,method=method,selected_method=sel,selected_q=sq); rows.append(r)
  pd.DataFrame(rows).to_csv(out/f'dermatology_safe_pacs_q{q:.1f}_raw.csv',index=False)
  print('done',rep,model_idx,'safe',safe.selected_method_,safe.selected_quantile_,flush=True)
df=pd.DataFrame(rows); s=df.groupby('method').agg(success=('case_mass_success','mean'),success_sd=('case_mass_success','std'),mean_size=('mean_size','mean'),mean_size_sd=('mean_size','std'),p90_size=('p90_size','mean'),mean_mass=('mean_mass','mean'),minority=('minority_fraction','mean'),runs=('rep','count')).reset_index(); s.to_csv(out/f'dermatology_safe_pacs_q{q:.1f}_summary.csv',index=False); print(s.to_string(index=False,float_format=lambda x:f'{x:.4f}')); print('\nselection counts'); print(df[df.method.eq('safe_pacs')].groupby(['selected_method','selected_q'],dropna=False).size())
