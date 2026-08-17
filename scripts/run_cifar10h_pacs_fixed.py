#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np, pandas as pd
from src.pacs import normalize_probs, global_panel_quantile_threshold, predict_global, plausibility_mass
from src.pacs_mondrian import PACSTopKMondrian

ROOT=Path('.')
HUM=ROOT/'data/public/cifar10h/repo/data/cifar10h-probs.npy'
PRED=ROOT/'experiments/pr_rescue/cifar10h/resnet18_test_probs.npy'
OUT=ROOT/'experiments/pr_rescue/results'; OUT.mkdir(parents=True,exist_ok=True)

def stutz_quantile(scores,alpha):
    a=np.asarray(scores,float); n=len(a); q=np.floor(alpha*(n+1))/n
    return float(np.quantile(a,np.clip(q,0,1),method='midpoint'))
def top1_cp(cal_e,cal_lam,alpha):
    y=np.argmax(cal_lam,1); return stutz_quantile(cal_e[np.arange(len(y)),y],alpha)
def mccp(cal_e,cal_lam,alpha,rng,num_samples=10):
    n,k=cal_e.shape; cs=np.cumsum(cal_lam,axis=1); u=rng.random((num_samples,n)); labels=(u[...,None]>cs[None,:,:]).sum(2)
    scores=np.concatenate([cal_e[np.arange(n),labels[j]] for j in range(num_samples)])
    q=(np.floor(alpha*num_samples*(n+1))-num_samples+1)/(n*num_samples)
    return float(np.quantile(scores,np.clip(q,0,1),method='midpoint'))
def metrics(mask,lam,q):
    mass=plausibility_mass(mask,lam); size=mask.sum(1)
    return dict(success=float(np.mean(mass>=q-1e-12)),mean_size=float(np.mean(size)),
                p90_size=float(np.quantile(size,.9)),p95_size=float(np.quantile(size,.95)),
                mean_mass=float(np.mean(mass)),minority=float(np.mean(mass-np.max(lam,axis=1)*(mask[np.arange(len(mask)),np.argmax(lam,1)]))))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--reps',type=int,default=30); ap.add_argument('--q',type=float,default=.8); args=ap.parse_args()
    lam=normalize_probs(np.load(HUM)); e=normalize_probs(np.load(PRED)); assert e.shape==lam.shape==(10000,10)
    q=float(args.q); alpha=.1; rows=[]; tag=str(q).replace('.','p')
    for rep in range(args.reps):
        rng=np.random.default_rng(2026082000+1000*rep+int(q*100)); p=rng.permutation(len(e)); ntr=4000; ncal=3000
        tr=p[:ntr]; cal=p[ntr:ntr+ncal]; te=p[ntr+ncal:]
        methods=[]
        th=top1_cp(e[cal],lam[cal],alpha); methods.append(('top1_cp',e[te]>=th))
        th=mccp(e[cal],lam[cal],alpha,rng,10); methods.append(('mccp10',e[te]>=th))
        th=global_panel_quantile_threshold(e[cal],lam[cal],q,alpha); methods.append(('global_panel',predict_global(e[te],th)))
        pacs=PACSTopKMondrian(q,alpha,random_state=20260820+rep,model_quantile=.9,n_bins=4,min_cal_per_bin=100).fit(e[tr],lam[tr],e[cal],lam[cal])
        methods.append(('pacs_mondrian',pacs.predict(e[te])))
        for method,mask in methods:
            r=metrics(mask,lam[te],q); r.update(method=method,rep=rep,q=q); rows.append(r)
        pd.DataFrame(rows).to_csv(OUT/f'cifar10h_pacs_fixed_q{tag}_raw.csv',index=False)
        print('done',rep,flush=True)
    df=pd.DataFrame(rows); df.to_csv(OUT/f'cifar10h_pacs_fixed_q{tag}_raw.csv',index=False)
    s=df.groupby('method').agg(success=('success','mean'),success_sd=('success','std'),mean_size=('mean_size','mean'),mean_size_sd=('mean_size','std'),p90_size=('p90_size','mean'),p95_size=('p95_size','mean'),mean_mass=('mean_mass','mean'),minority=('minority','mean'),runs=('rep','count')).reset_index()
    s.to_csv(OUT/f'cifar10h_pacs_fixed_q{tag}_summary.csv',index=False); print(s.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
if __name__=='__main__': main()
