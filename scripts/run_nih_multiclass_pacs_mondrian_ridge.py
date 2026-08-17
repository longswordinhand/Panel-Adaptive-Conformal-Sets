#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from src.pacs import normalize_probs, global_panel_quantile_threshold, predict_global, plausibility_mass
from src.pacs_mondrian import PACSTopKMondrian

ROOT=Path('.')
LAB=ROOT/'data/public/nih_expert_labels/raw/all_findings_expert_labels/all_findings_expert_labels_test_individual_readers.csv'
FEAT=ROOT/'experiments/pr_rescue/nih_features/resnet50_imagenet_features.npy'
IDX=ROOT/'experiments/pr_rescue/nih_features/resnet50_imagenet_index.csv'
OUT=ROOT/'experiments/pr_rescue/results'; OUT.mkdir(parents=True,exist_ok=True)
FINDINGS=['Atelectasis','Cardiomegaly','Effusion','Infiltration','Mass','Nodule','Pneumonia','Pneumothorax','Consolidation','Edema','Emphysema','Fibrosis','Pleural Thickening','Hernia','Other']
CLASSES=['No Finding']+FINDINGS

def yes(v): return str(v).strip().upper() in {'YES','1','TRUE','Y','POSITIVE'}
def expert_mass(raw,ids):
    by={x:i for i,x in enumerate(ids)}; lam=np.zeros((len(ids),len(CLASSES)),float)
    sub=raw[raw['Image ID'].astype(str).isin(by)].copy()
    for _,r in sub.iterrows():
        i=by[str(r['Image ID'])]; picked=[j+1 for j,f in enumerate(FINDINGS) if yes(r[f])]
        if picked:
            for c in picked: lam[i,c]+=1.0/(5.0*len(picked))
        else: lam[i,0]+=1.0/5.0
    if not np.allclose(lam.sum(1),1): raise RuntimeError('mass')
    return lam

def fit_ridge(X,lam):
    sc=StandardScaler(); Z=sc.fit_transform(X)
    reg=Ridge(alpha=25.0).fit(Z,lam)
    return sc,reg
def pred(fit,X):
    sc,reg=fit; p=reg.predict(sc.transform(X)); p=np.clip(p,1e-6,None); return normalize_probs(p)
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
    return dict(success=float(np.mean(mass>=q-1e-12)),mean_size=float(np.mean(size)),p90_size=float(np.quantile(size,.9)),mean_mass=float(np.mean(mass)))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--reps',type=int,default=10); ap.add_argument('--q',type=float,default=.9); args=ap.parse_args()
    X=np.load(FEAT); ids=pd.read_csv(IDX)['image_id'].astype(str).tolist(); raw=pd.read_csv(LAB); raw['Image ID']=raw['Image ID'].astype(str); lam=expert_mass(raw,ids)
    case=raw[['Image ID','Patient ID']].drop_duplicates('Image ID').set_index('Image ID').loc[ids].reset_index(); patients=case['Patient ID'].astype(str).to_numpy(); up=np.unique(patients)
    rows=[]; q=float(args.q); alpha=.1
    for rep in range(args.reps):
        rng=np.random.default_rng(2026081900+rep); pp=rng.permutation(up); a=int(.4*len(up)); b=int(.7*len(up)); trp=set(pp[:a]); cap=set(pp[a:b]); tep=set(pp[b:])
        tr=np.array([i for i,p in enumerate(patients) if p in trp]); cal=np.array([i for i,p in enumerate(patients) if p in cap]); te=np.array([i for i,p in enumerate(patients) if p in tep])
        fit=fit_ridge(X[tr],lam[tr]); etr=pred(fit,X[tr]); ecal=pred(fit,X[cal]); ete=pred(fit,X[te])
        methods=[]
        th=top1_cp(ecal,lam[cal],alpha); methods.append(('top1_cp',ete>=th))
        th=mccp(ecal,lam[cal],alpha,rng,10); methods.append(('mccp10',ete>=th))
        th=global_panel_quantile_threshold(ecal,lam[cal],q,alpha); methods.append(('global_panel',predict_global(ete,th)))
        for mq in (.65,.75,.85):
            pacs=PACSTopKMondrian(q,alpha,random_state=20260819+rep,model_quantile=mq,n_bins=3,min_cal_per_bin=20).fit(etr,lam[tr],ecal,lam[cal])
            methods.append((f'pacs_mondrian_q{mq:.2f}',pacs.predict(ete)))
        for method,mask in methods:
            r=metrics(mask,lam[te],q); r.update(method=method,rep=rep,ntrain=len(tr),ncal=len(cal),ntest=len(te)); rows.append(r)
        print('done',rep,flush=True)
    df=pd.DataFrame(rows); tag=str(q).replace('.','p'); df.to_csv(OUT/f'nih_multiclass_pacs_mondrian_ridge_q{tag}_raw.csv',index=False)
    s=df.groupby('method').agg(success=('success','mean'),success_sd=('success','std'),mean_size=('mean_size','mean'),mean_size_sd=('mean_size','std'),p90_size=('p90_size','mean'),mean_mass=('mean_mass','mean'),runs=('rep','count')).reset_index(); s.to_csv(OUT/f'nih_multiclass_pacs_mondrian_ridge_q{tag}_summary.csv',index=False); print(s.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
if __name__=='__main__': main()
