#!/usr/bin/env python3
"""Validate the simpler PACS global-residual ablation on NIH and CIFAR-10H.

Uses the same split seeds/protocols as the frozen stratified PACS runs, but
replaces difficulty-stratified corrections with one global residual correction.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from src.pacs import normalize_probs, plausibility_mass
from src.pacs_v2 import PACSTopK
from scripts.run_nih_multiclass_pacs_mondrian_ridge_fixed import (
    LAB, FEAT, IDX, expert_mass, fit_ridge, pred, metrics as nih_metrics
)
from scripts.run_cifar10h_pacs_fixed import metrics as cifar_metrics

ROOT = Path('.')
OUT = ROOT / 'experiments/pr_rescue/results'
OUT.mkdir(parents=True, exist_ok=True)


def run_nih():
    X = np.load(FEAT)
    ids = pd.read_csv(IDX)['image_id'].astype(str).tolist()
    raw = pd.read_csv(LAB)
    raw['Image ID'] = raw['Image ID'].astype(str)
    lam = expert_mass(raw, ids)
    case = raw[['Image ID','Patient ID']].drop_duplicates('Image ID').set_index('Image ID').loc[ids].reset_index()
    patients = case['Patient ID'].astype(str).to_numpy()
    up = np.unique(patients)
    rows = []
    for rep in range(30):
        rng = np.random.default_rng(2026081900 + rep)
        pp = rng.permutation(up)
        a = int(.4*len(up)); b = int(.7*len(up))
        trp=set(pp[:a]); cap=set(pp[a:b]); tep=set(pp[b:])
        tr=np.array([i for i,p in enumerate(patients) if p in trp])
        cal=np.array([i for i,p in enumerate(patients) if p in cap])
        te=np.array([i for i,p in enumerate(patients) if p in tep])
        fit=fit_ridge(X[tr],lam[tr]); etr=pred(fit,X[tr]); ecal=pred(fit,X[cal]); ete=pred(fit,X[te])
        for q in (0.7,0.8,0.9):
            pacs=PACSTopK(q,.1,random_state=20260819+rep,model_quantile=.9).fit(etr,lam[tr],ecal,lam[cal])
            r=nih_metrics(pacs.predict(ete),lam[te],q)
            r.update(dataset='NIH',q=q,method='pacs_global_residual',rep=rep,ntrain=len(tr),ncal=len(cal),ntest=len(te))
            rows.append(r)
        print('NIH done',rep,flush=True)
        pd.DataFrame(rows).to_csv(OUT/'nih_pacs_global_residual_validation_raw.csv',index=False)
    df=pd.DataFrame(rows); df.to_csv(OUT/'nih_pacs_global_residual_validation_raw.csv',index=False)
    s=df.groupby(['q','method']).agg(success=('success','mean'),mean_size=('mean_size','mean'),p90_size=('p90_size','mean'),runs=('rep','count')).reset_index()
    s.to_csv(OUT/'nih_pacs_global_residual_validation_summary.csv',index=False)
    print(s.to_string(index=False,float_format=lambda x:f'{x:.4f}'))


def run_cifar():
    hum=ROOT/'data/public/cifar10h/repo/data/cifar10h-probs.npy'
    predp=ROOT/'experiments/pr_rescue/cifar10h/resnet18_test_probs.npy'
    lam=normalize_probs(np.load(hum)); e=normalize_probs(np.load(predp)); q=.7
    rows=[]
    for rep in range(20):
        rng=np.random.default_rng(2026082000+1000*rep+int(q*100)); p=rng.permutation(len(e)); ntr=4000; ncal=3000
        tr=p[:ntr]; cal=p[ntr:ntr+ncal]; te=p[ntr+ncal:]
        pacs=PACSTopK(q,.1,random_state=20260820+rep,model_quantile=.9).fit(e[tr],lam[tr],e[cal],lam[cal])
        r=cifar_metrics(pacs.predict(e[te]),lam[te],q)
        r.update(dataset='CIFAR-10H',q=q,method='pacs_global_residual',rep=rep)
        rows.append(r)
        print('CIFAR done',rep,flush=True)
        pd.DataFrame(rows).to_csv(OUT/'cifar10h_pacs_global_residual_validation_raw.csv',index=False)
    df=pd.DataFrame(rows); df.to_csv(OUT/'cifar10h_pacs_global_residual_validation_raw.csv',index=False)
    s=df.groupby(['q','method']).agg(success=('success','mean'),mean_size=('mean_size','mean'),p90_size=('p90_size','mean'),runs=('rep','count')).reset_index()
    s.to_csv(OUT/'cifar10h_pacs_global_residual_validation_summary.csv',index=False)
    print(s.to_string(index=False,float_format=lambda x:f'{x:.4f}'))


if __name__=='__main__':
    run_nih()
    run_cifar()
