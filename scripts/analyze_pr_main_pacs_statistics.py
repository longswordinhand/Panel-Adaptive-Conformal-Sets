#!/usr/bin/env python3
"""Final paired statistics for the simplified PACS main method (global residual)."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RES=ROOT/'experiments/pr_rescue/results'; OUT=ROOT/'experiments/pr_rescue/statistics'; OUT.mkdir(parents=True,exist_ok=True)
RNG=np.random.default_rng(20260817); B=20000

def boot(d):
    d=np.asarray(d,float); idx=RNG.integers(0,len(d),(B,len(d))); m=d[idx].mean(1); return np.quantile(m,[.025,.975])
def add(rows,ds,q,metric,g,p):
    d=(p-g).sort_index(); lo,hi=boot(d)
    rows.append(dict(dataset=ds,q=q,metric=metric,n_split_units=len(d),global_mean=g.mean(),pacs_mean=p.mean(),delta=d.mean(),ci_lo=lo,ci_hi=hi,relative_delta_pct=100*d.mean()/g.mean(),n_pacs_lower=int((d<0).sum()),n_pacs_higher=int((d>0).sum()),n_ties=int(np.isclose(d,0).sum())))
rows=[]; unit_frames=[]
# Dermatology q=.9
for q in [.7,.8,.9]:
    if q==.9: f=RES/'dermatology_pacs_ablation_q0p9_raw.csv'
    else: f=RES/f'dermatology_pacs_global_residual_q{str(q).replace(".","p")}_raw.csv'
    d=pd.read_csv(f)
    u=d.groupby(['rep','method'])[['case_mass_success','mean_size','p90_size']].mean().reset_index()
    methods=('global_panel','pacs_global_residual')
    wide=[]
    for rep,gp in u.groupby('rep'):
        z=gp.set_index('method'); rec={'dataset':'Dermatology','q':q,'rep':rep}
        for metric,col in [('success','case_mass_success'),('mean_size','mean_size'),('p90_size','p90_size')]:
            rec[f'global_{metric}']=z.loc[methods[0],col]; rec[f'pacs_{metric}']=z.loc[methods[1],col]
            add(rows,'Dermatology',q,metric,pd.Series({rep:z.loc[methods[0],col]}),pd.Series({rep:z.loc[methods[1],col]})) if False else None
        wide.append(rec)
    w=pd.DataFrame(wide).set_index('rep')
    for metric in ['success','mean_size','p90_size']: add(rows,'Dermatology',q,metric,w[f'global_{metric}'],w[f'pacs_{metric}'])
    unit_frames.append(w.reset_index())
# NIH
pv=pd.read_csv(RES/'nih_pacs_global_residual_validation_raw.csv')
for q in [.7,.8,.9]:
    base=pd.read_csv(RES/f'nih_multiclass_pacs_mondrian_ridge_fixed_q{str(q).replace(".","p")}_raw.csv')
    g=base[base.method=='global_panel'].set_index('rep'); p=pv[pv.q==q].set_index('rep')
    rec=pd.DataFrame({'dataset':'NIH','q':q,'rep':g.index,'global_success':g.success,'pacs_success':p.success,'global_mean_size':g.mean_size,'pacs_mean_size':p.mean_size,'global_p90_size':g.p90_size,'pacs_p90_size':p.p90_size}).set_index('rep')
    for metric in ['success','mean_size','p90_size']: add(rows,'NIH',q,metric,rec[f'global_{metric}'],rec[f'pacs_{metric}'])
    unit_frames.append(rec.reset_index())
# CIFAR
base=pd.read_csv(RES/'cifar10h_pacs_fixed_q0p7_raw.csv'); g=base[base.method=='global_panel'].set_index('rep'); p=pd.read_csv(RES/'cifar10h_pacs_global_residual_validation_raw.csv').set_index('rep')
rec=pd.DataFrame({'dataset':'CIFAR-10H','q':.7,'rep':g.index,'global_success':g.success,'pacs_success':p.success,'global_mean_size':g.mean_size,'pacs_mean_size':p.mean_size,'global_p90_size':g.p90_size,'pacs_p90_size':p.p90_size}).set_index('rep')
for metric in ['success','mean_size','p90_size']: add(rows,'CIFAR-10H',.7,metric,rec[f'global_{metric}'],rec[f'pacs_{metric}'])
unit_frames.append(rec.reset_index())
summary=pd.DataFrame(rows); summary.to_csv(OUT/'paired_effects_main_pacs.csv',index=False); pd.concat(unit_frames,ignore_index=True).to_csv(OUT/'paired_split_units_main_pacs.csv',index=False)
print(summary.to_string(index=False,float_format=lambda x:f'{x:.6g}'))
