#!/usr/bin/env python3
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import chi2
from run_tail_identification_experiments import ROOT, NIH, nih_vote_matrices
OUT=ROOT/'experiments/pilot/results/tpami_robustness'
OUT.mkdir(parents=True,exist_ok=True)
df=pd.read_csv(NIH); mats=nih_vote_matrices(df)
labels=['Abnormal','Consolidation','Pleural Thickening','Nodule','Pneumothorax','Cardiomegaly']
rows=[]; rates=[]
for label in labels:
    X=mats[label].to_numpy(dtype=float); n,k=X.shape
    col=X.sum(axis=0); row=X.sum(axis=1)
    T=col.sum()
    denom=(row*(k-row)).sum()
    Q=(k*(col**2).sum()-T**2)/denom if denom>0 else np.nan
    p=chi2.sf(Q,k-1) if np.isfinite(Q) else np.nan
    rr=X.mean(axis=0)
    rows.append(dict(label=label,n=n,k=k,cochran_Q=Q,df=k-1,p_value=p,rate_min=rr.min(),rate_max=rr.max(),rate_range=rr.max()-rr.min()))
    for reader,rate in zip(mats[label].columns,rr): rates.append(dict(label=label,reader=reader,positive_rate=rate))
pd.DataFrame(rows).to_csv(OUT/'nih_reader_heterogeneity_tests.csv',index=False)
pd.DataFrame(rates).to_csv(OUT/'nih_reader_positive_rates.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
print('\nRates:')
print(pd.DataFrame(rates).pivot(index='label',columns='reader',values='positive_rate').to_string())
