#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.pacs import normalize_probs, global_panel_quantile_threshold, predict_global, plausibility_mass
from src.pacs_effcv import PACSTopKEffCV

ROOT=Path('.')
LAB=ROOT/'data/public/nih_expert_labels/raw/all_findings_expert_labels/all_findings_expert_labels_test_individual_readers.csv'
FEAT=ROOT/'experiments/pr_rescue/nih_features/resnet50_imagenet_features.npy'
IDX=ROOT/'experiments/pr_rescue/nih_features/resnet50_imagenet_index.csv'
OUT=ROOT/'experiments/pr_rescue/results'; OUT.mkdir(parents=True,exist_ok=True)
FINDINGS=['Atelectasis','Cardiomegaly','Effusion','Infiltration','Mass','Nodule','Pneumonia','Pneumothorax','Consolidation','Edema','Emphysema','Fibrosis','Pleural Thickening','Hernia','Other']
CLASSES=['No Finding']+FINDINGS

def yes(v): return str(v).strip().upper() in {'YES','1','TRUE','Y','POSITIVE'}

def expert_mass(raw, ids):
    by={x:i for i,x in enumerate(ids)}; lam=np.zeros((len(ids),len(CLASSES)),float)
    sub=raw[raw['Image ID'].astype(str).isin(by)].copy()
    for _,r in sub.iterrows():
        i=by[str(r['Image ID'])]
        picked=[j+1 for j,f in enumerate(FINDINGS) if yes(r[f])]
        if picked:
            w=1.0/len(picked)
            for c in picked: lam[i,c]+=w/5.0
        else:
            lam[i,0]+=1.0/5.0
    if not np.allclose(lam.sum(1),1): raise RuntimeError((lam.sum(1).min(),lam.sum(1).max()))
    return lam

def fit_soft_multinomial(X,lam,seed):
    torch.manual_seed(seed); np.random.seed(seed)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mu=X.mean(0,keepdims=True).astype('float32'); sd=(X.std(0,keepdims=True)+1e-5).astype('float32')
    Z=((X-mu)/sd).astype('float32'); Y=lam.astype('float32')
    class Head(nn.Module):
        def __init__(self,d,k):
            super().__init__(); self.net=nn.Sequential(nn.Linear(d,256),nn.ReLU(),nn.Dropout(.10),nn.Linear(256,k))
        def forward(self,x): return self.net(x)
    model=Head(Z.shape[1],Y.shape[1]).to(dev); opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    xt=torch.from_numpy(Z).to(dev); yt=torch.from_numpy(Y).to(dev); n=len(Z); gen=torch.Generator().manual_seed(seed)
    best=None; bestloss=1e9; stale=0
    for ep in range(220):
        model.train(); perm=torch.randperm(n,generator=gen); losses=[]
        for a in range(0,n,64):
            ix=perm[a:a+64].to(dev); logits=model(xt[ix]); loss=-(yt[ix]*F.log_softmax(logits,1)).sum(1).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
        cur=float(np.mean(losses))
        if cur<bestloss-1e-5:
            bestloss=cur; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale=0
        else: stale+=1
        if stale>=25 and ep>=80: break
    if best is not None: model.load_state_dict(best)
    model.eval(); return mu,sd,model,dev,bestloss

def pred(fit,X):
    mu,sd,model,dev,_=fit; Z=((X-mu)/sd).astype('float32')
    with torch.inference_mode(): p=torch.softmax(model(torch.from_numpy(Z).to(dev)),1).cpu().numpy()
    return normalize_probs(p)

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
    # Oracle top-k size under model ranking is not available without model scores; report distributional support burden.
    support=(lam>1e-12).sum(1)
    return dict(success=float(np.mean(mass>=q-1e-12)),mean_size=float(np.mean(size)),median_size=float(np.median(size)),p90_size=float(np.quantile(size,.9)),mean_mass=float(np.mean(mass)),minority=float(np.mean((mask*lam).sum(1)-mask[np.arange(len(mask)),np.argmax(lam,1)]*lam[np.arange(len(lam)),np.argmax(lam,1)])),support=float(np.mean(support)))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--reps',type=int,default=3); ap.add_argument('--q',type=float,default=.8); args=ap.parse_args()
    X=np.load(FEAT); ids=pd.read_csv(IDX)['image_id'].astype(str).tolist(); raw=pd.read_csv(LAB); raw['Image ID']=raw['Image ID'].astype(str)
    lam=expert_mass(raw,ids)
    case=raw[['Image ID','Patient ID']].drop_duplicates('Image ID').set_index('Image ID').loc[ids].reset_index(); patients=case['Patient ID'].astype(str).to_numpy(); up=np.unique(patients)
    rows=[]; alpha=.1; q=float(args.q)
    for rep in range(args.reps):
        rng=np.random.default_rng(2026081700+rep); pp=rng.permutation(up); a=int(.4*len(up)); b=int(.7*len(up)); trp=set(pp[:a]); cap=set(pp[a:b]); tep=set(pp[b:])
        tr=np.array([i for i,p in enumerate(patients) if p in trp]); cal=np.array([i for i,p in enumerate(patients) if p in cap]); te=np.array([i for i,p in enumerate(patients) if p in tep])
        fit=fit_soft_multinomial(X[tr],lam[tr],20260817+rep); etr=pred(fit,X[tr]); ecal=pred(fit,X[cal]); ete=pred(fit,X[te])
        methods=[]
        th=top1_cp(ecal,lam[cal],alpha); methods.append(('top1_cp',ete>=th))
        th=mccp(ecal,lam[cal],alpha,rng,10); methods.append(('mccp10',ete>=th))
        th=global_panel_quantile_threshold(ecal,lam[cal],q,alpha); methods.append(('global_panel',predict_global(ete,th)))
        pacs=PACSTopKEffCV(q,alpha,random_state=20260817+rep).fit(etr,lam[tr],ecal,lam[cal]); methods.append(('pacs_effcv',pacs.predict(ete)))
        for method,mask in methods:
            r=metrics(mask,lam[te],q); r.update(method=method,rep=rep,q=q,ntrain=len(tr),ncal=len(cal),ntest=len(te),selected_q=(pacs.selected_quantile_ if method=='pacs_effcv' else np.nan)); rows.append(r)
        print('done',rep,'n',len(tr),len(cal),len(te),'pacsq',pacs.selected_quantile_,flush=True)
    df=pd.DataFrame(rows); tag=str(q).replace('.','p'); df.to_csv(OUT/f'nih_multiclass_pacs_effcv_q{tag}_raw.csv',index=False)
    s=df.groupby('method').agg(success=('success','mean'),success_sd=('success','std'),mean_size=('mean_size','mean'),mean_size_sd=('mean_size','std'),p90_size=('p90_size','mean'),mean_mass=('mean_mass','mean'),support=('support','mean'),runs=('rep','count')).reset_index(); s.to_csv(OUT/f'nih_multiclass_pacs_effcv_q{tag}_summary.csv',index=False)
    print('\n',s.to_string(index=False,float_format=lambda z:f'{z:.4f}'))
    print('mass sparsity support mean',float((lam>1e-12).sum(1).mean()),'classes',len(CLASSES))

if __name__=='__main__': main()
