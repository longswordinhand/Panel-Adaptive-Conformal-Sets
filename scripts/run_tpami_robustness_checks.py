#!/usr/bin/env python3
from pathlib import Path
import itertools
import numpy as np
import pandas as pd
from scipy.stats import linregress

from run_tail_identification_experiments import (
    ROOT, NIH, bernstein_matrix, tail_bounds, nih_vote_matrices
)

OUT = ROOT / 'experiments/pilot/results/tpami_robustness'
OUT.mkdir(parents=True, exist_ok=True)

# 1) Per-subset NIH width variability at beta=.5, grid=2001.
nih = pd.read_csv(NIH)
mats = nih_vote_matrices(nih)
labels = ['Abnormal','Consolidation','Pleural Thickening','Nodule','Pneumothorax','Cardiomegaly']
grid = np.linspace(0,1,2001)
rows=[]
for label in labels:
    vals=mats[label].to_numpy(dtype=int)
    readers=list(mats[label].columns)
    for m in range(1,6):
        for combo in itertools.combinations(range(5),m):
            k=vals[:,combo].sum(axis=1)
            p=np.bincount(k,minlength=m+1).astype(float); p/=p.sum()
            lo,hi,err=tail_bounds(p,m,.5,grid)
            rows.append(dict(label=label,m=m,subset='|'.join(str(readers[j]) for j in combo),lower=lo,upper=hi,width=hi-lo,projection_l1=err))
sub=pd.DataFrame(rows)
sub.to_csv(OUT/'nih_subset_specific_bounds_beta05.csv',index=False)
summary=sub.groupby(['label','m']).agg(width_median=('width','median'),width_min=('width','min'),width_max=('width','max'),width_mean=('width','mean'),projection_l1_max=('projection_l1','max'),n_subsets=('width','size')).reset_index()
summary.to_csv(OUT/'nih_subset_width_summary_beta05.csv',index=False)

# 2) Grid sensitivity for selected endpoints/m values.
grid_rows=[]
for label in ['Pleural Thickening','Cardiomegaly','Nodule']:
    mat=mats[label]
    vals=mat.to_numpy(dtype=int)
    for m in [3,5]:
        counts=np.zeros(m+1)
        combos=list(itertools.combinations(range(5),m))
        for combo in combos:
            k=vals[:,combo].sum(axis=1)
            counts += np.bincount(k,minlength=m+1)
        p=counts/counts.sum()
        for gn in [501,1001,2001,4001,8001]:
            g=np.linspace(0,1,gn)
            lo,hi,err=tail_bounds(p,m,.5,g)
            grid_rows.append(dict(label=label,m=m,grid_n=gn,lower=lo,upper=hi,width=hi-lo,projection_l1=err))
pd.DataFrame(grid_rows).to_csv(OUT/'grid_sensitivity_beta05.csv',index=False)

# 3) Sequential empirical log-log slopes from existing runs.
seq=pd.read_csv(ROOT/'experiments/pilot/results/thinned_sequential_simulation.csv')
agg=seq.groupby(['gamma','epsilon']).agg(mean_budget=('budget','mean'),mean_abs_error=('abs_error','mean'),reps=('budget','size')).reset_index()
slopes=[]
for gamma,g in agg.groupby('gamma'):
    g=g.sort_values('epsilon')
    lr=linregress(np.log(1/g.epsilon.to_numpy()),np.log(g.mean_budget.to_numpy()))
    pred=max(2.0,2.0/gamma)
    slopes.append(dict(gamma=gamma,predicted_exponent=pred,empirical_slope=lr.slope,r2=lr.rvalue**2,n_eps=len(g)))
pd.DataFrame(slopes).to_csv(OUT/'sequential_budget_loglog_slopes.csv',index=False)
agg.to_csv(OUT/'sequential_budget_summary.csv',index=False)

print('subset summary')
print(summary.to_string(index=False))
print('\ngrid sensitivity')
print(pd.DataFrame(grid_rows).to_string(index=False))
print('\nslopes')
print(pd.DataFrame(slopes).to_string(index=False))
