#!/usr/bin/env python3
from pathlib import Path
import argparse, math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from src.pacs import normalize_probs, global_panel_quantile_threshold, predict_global, plausibility_mass
from src.pacs_cv import PACSTopKCV

ROOT=Path('.')
LAB=ROOT/'data/public/nih_expert_labels/raw/all_findings_expert_labels/all_findings_expert_labels_test_individual_readers.csv'
FEAT=ROOT/'experiments/pr_rescue/nih_features/resnet50_imagenet_features.npy'
IDX=ROOT/'experiments/pr_rescue/nih_features/resnet50_imagenet_index.csv'
OUT=ROOT/'experiments/pr_rescue/results'; OUT.mkdir(parents=True,exist_ok=True)
ENDPOINTS=['Atelectasis','Consolidation','Pleural Thickening','Nodule','Effusion','Cardiomegaly']

def as_binary(s):
    z=s.astype(str).str.upper().str.strip()
    return z.isin(['YES','1','TRUE','Y','POSITIVE']).astype(int).to_numpy()

def fit_reader_logistic(X, vote_counts, seed):
    # Exact binomial-logistic likelihood via two weighted duplicate rows/case.
    n=len(X); rows=[]; yy=[]; ww=[]
    for i,k in enumerate(vote_counts.astype(int)):
        if k>0: rows.append(X[i]); yy.append(1); ww.append(float(k))
        if k<5: rows.append(X[i]); yy.append(0); ww.append(float(5-k))
    Xe=np.asarray(rows); y=np.asarray(yy); w=np.asarray(ww)
    sc=StandardScaler()
    Xs=sc.fit_transform(Xe)
    clf=LogisticRegression(C=.05,solver='liblinear',max_iter=1000,random_state=seed)
    clf.fit(Xs,y,sample_weight=w)
    return sc,clf

def predict_probs(sc,clf,X):
    p=clf.predict_proba(sc.transform(X))[:,1]
    p=np.clip(p,1e-6,1-1e-6)
    return np.column_stack([1-p,p])

def stutz_quantile(scores,alpha):
    a=np.asarray(scores,float); n=len(a); q=np.floor(alpha*(n+1))/n
    return float(np.quantile(a,np.clip(q,0,1),method='midpoint'))

def top1_cp(cal_e,cal_lam,alpha):
    y=np.argmax(cal_lam,1); return stutz_quantile(cal_e[np.arange(len(y)),y],alpha)

def mccp(cal_e,cal_lam,alpha,rng,num_samples=10):
    n,k=cal_e.shape; cs=np.cumsum(cal_lam,axis=1)
    u=rng.random((num_samples,n)); labels=(u[...,None]>cs[None,:,:]).sum(2)
    scores=np.concatenate([cal_e[np.arange(n),labels[j]] for j in range(num_samples)])
    q=(np.floor(alpha*num_samples*(n+1))-num_samples+1)/(n*num_samples)
    return float(np.quantile(scores,np.clip(q,0,1),method='midpoint'))

def metrics(mask,lam,q):
    mass=plausibility_mass(mask,lam); size=mask.sum(1)
    oracle_size=np.where(np.max(lam,axis=1)>=q-1e-12,1,2)
    clear=oracle_size==1; amb=oracle_size==2
    return dict(
        success=float(np.mean(mass>=q-1e-12)),
        mean_size=float(np.mean(size)),
        full_set_rate=float(np.mean(size==2)),
        oracle_mean_size=float(np.mean(oracle_size)),
        excess_size=float(np.mean(size-oracle_size)),
        clear_full_rate=float(np.mean(size[clear]==2)) if np.any(clear) else np.nan,
        clear_success=float(np.mean(mass[clear]>=q-1e-12)) if np.any(clear) else np.nan,
        ambiguous_full_rate=float(np.mean(size[amb]==2)) if np.any(amb) else np.nan,
        ambiguous_success=float(np.mean(mass[amb]>=q-1e-12)) if np.any(amb) else np.nan,
        n_clear=int(clear.sum()),n_ambiguous=int(amb.sum()),
        mean_mass=float(np.mean(mass)),
    )

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--reps',type=int,default=5); ap.add_argument('--q',type=float,default=.9); args=ap.parse_args()
    X=np.load(FEAT); ids=pd.read_csv(IDX)['image_id'].astype(str).tolist(); pos={v:i for i,v in enumerate(ids)}
    raw=pd.read_csv(LAB); raw['Image ID']=raw['Image ID'].astype(str)
    case=raw[['Image ID','Patient ID']].drop_duplicates('Image ID').set_index('Image ID').loc[ids].reset_index()
    rows=[]; q=float(args.q); alpha=.1
    for endpoint in ENDPOINTS:
        temp=raw[raw['Image ID'].isin(ids)].copy(); temp['_y']=as_binary(temp[endpoint])
        k=temp.groupby('Image ID')['_y'].sum().reindex(ids).astype(int).to_numpy()
        lam=np.column_stack([1-k/5.0,k/5.0])
        patients=case['Patient ID'].astype(str).to_numpy(); up=np.unique(patients)
        for rep in range(args.reps):
            rng=np.random.default_rng(2026081600+1000*ENDPOINTS.index(endpoint)+rep)
            pperm=rng.permutation(up); npt=len(up); a=int(.4*npt); b=int(.7*npt)
            trp=set(pperm[:a]); cap=set(pperm[a:b]); tep=set(pperm[b:])
            tr=np.array([i for i,p in enumerate(patients) if p in trp],int)
            cal=np.array([i for i,p in enumerate(patients) if p in cap],int)
            te=np.array([i for i,p in enumerate(patients) if p in tep],int)
            sc,clf=fit_reader_logistic(X[tr],k[tr],20260816+rep)
            e_tr=predict_probs(sc,clf,X[tr]); e_cal=predict_probs(sc,clf,X[cal]); e_te=predict_probs(sc,clf,X[te])
            methods=[]
            th=top1_cp(e_cal,lam[cal],alpha); methods.append(('top1_cp',e_te>=th))
            th=mccp(e_cal,lam[cal],alpha,rng,10); methods.append(('mccp10',e_te>=th))
            th=global_panel_quantile_threshold(e_cal,lam[cal],q,alpha); methods.append(('global_panel',predict_global(e_te,th)))
            pacs=PACSTopKCV(q,alpha,random_state=20260816+rep).fit(e_tr,lam[tr],e_cal,lam[cal]); methods.append(('pacs_cv',pacs.predict(e_te)))
            for method,mask in methods:
                r=metrics(mask,lam[te],q); r.update(endpoint=endpoint,rep=rep,method=method,ntrain=len(tr),ncal=len(cal),ntest=len(te),selected_q=(pacs.selected_quantile_ if method=='pacs_cv' else np.nan))
                rows.append(r)
            print('done',endpoint,rep,'n',len(tr),len(cal),len(te),'pacsq',pacs.selected_quantile_,flush=True)
    df=pd.DataFrame(rows); tag=str(q).replace('.','p'); df.to_csv(OUT/f'nih_pacs_external_q{tag}_raw.csv',index=False)
    s=df.groupby('method').agg(success=('success','mean'),success_sd=('success','std'),mean_size=('mean_size','mean'),full_set=('full_set_rate','mean'),oracle_size=('oracle_mean_size','mean'),excess_size=('excess_size','mean'),clear_full=('clear_full_rate','mean'),amb_full=('ambiguous_full_rate','mean'),mean_mass=('mean_mass','mean'),runs=('rep','count')).reset_index()
    s.to_csv(OUT/f'nih_pacs_external_q{tag}_summary.csv',index=False)
    by=df.groupby(['endpoint','method']).agg(success=('success','mean'),mean_size=('mean_size','mean'),full_set=('full_set_rate','mean'),oracle_size=('oracle_mean_size','mean'),excess_size=('excess_size','mean'),clear_full=('clear_full_rate','mean'),amb_full=('ambiguous_full_rate','mean'),runs=('rep','count')).reset_index()
    by.to_csv(OUT/f'nih_pacs_external_q{tag}_by_endpoint.csv',index=False)
    print('\nOVERALL\n',s.to_string(index=False,float_format=lambda z:f'{z:.4f}'))
    print('\nBY ENDPOINT\n',by.to_string(index=False,float_format=lambda z:f'{z:.4f}'))

if __name__=='__main__': main()
