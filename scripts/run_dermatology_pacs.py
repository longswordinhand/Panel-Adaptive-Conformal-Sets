#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path
import numpy as np, pandas as pd
from src.pacs import PACS, normalize_probs, required_conformity_threshold, global_panel_quantile_threshold, predict_global, plausibility_mass

ROOT=Path('third_party/uncertain_ground_truth/data')

def irn_from_selectors(selectors,k=419):
    out=np.zeros((len(selectors),k),float)
    for i,readers in enumerate(selectors):
        for reader in readers:
            for b,group in enumerate(reader):
                for y in group: out[i,int(y)] += 1.0/(1+b)
    z=out.sum(1,keepdims=True)
    if np.any(z==0): raise ValueError('empty IRN row')
    return out/z

def stutz_conformal_quantile(scores,alpha):
    a=np.asarray(scores,float); n=len(a)
    q=np.floor(alpha*(n+1))/n
    return float(np.quantile(a,q,method='midpoint'))

def top1_cp(cal_e,cal_lam,alpha):
    y=np.argmax(cal_lam,1); return stutz_conformal_quantile(cal_e[np.arange(len(y)),y],alpha)

def mccp(cal_e,cal_lam,alpha,rng,num_samples=10):
    n,k=cal_e.shape
    cs=np.cumsum(cal_lam,axis=1)
    u=rng.random((num_samples,n)); labels=(u[...,None]>cs[None,:,:]).sum(2)
    sc=np.empty((num_samples,n),float)
    for j in range(num_samples): sc[j]=cal_e[np.arange(n),labels[j]]
    scores=sc.reshape(-1)
    q=(np.floor(alpha*num_samples*(n+1))-num_samples+1)/(n*num_samples)
    q=float(np.clip(q,0,1))
    return float(np.quantile(scores,q,method='midpoint'))

def exact_soft_quantile(cal_e,cal_lam,alpha):
    # Empirical exact replacement for MC sampling; descriptive baseline, not an exact CP theorem.
    vals=cal_e.ravel(); w=cal_lam.ravel()/cal_e.shape[0]
    o=np.argsort(vals); vals=vals[o]; w=w[o]; cs=np.cumsum(w)
    j=np.searchsorted(cs,alpha,side='left'); return float(vals[min(j,len(vals)-1)])

def metrics(mask,lam,q):
    mass=plausibility_mass(mask,lam); sizes=mask.sum(1); top=np.argmax(lam,1)
    topcov=mask[np.arange(len(mask)),top]
    # disagreement proxy: 1-max plausibility, and low-plausibility mass captured.
    minor=lam.copy(); minor[np.arange(len(lam)),top]=0
    minorden=minor.sum(1); minorcap=(mask*minor).sum(1)
    minorfrac=np.divide(minorcap,minorden,out=np.ones_like(minorden),where=minorden>1e-12)
    return dict(case_mass_success=float(np.mean(mass>=q-1e-12)),mean_mass=float(mass.mean()),
      p10_mass=float(np.quantile(mass,.1)),mean_size=float(sizes.mean()),median_size=float(np.median(sizes)),
      p90_size=float(np.quantile(sizes,.9)),top1_coverage=float(topcov.mean()),minority_fraction=float(minorfrac.mean()),
      empty=float(np.mean(sizes==0)))

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--reps',type=int,default=50); args=ap.parse_args()
 selectors=json.load(open(ROOT/'dermatology_selectors.json')); lam=irn_from_selectors(selectors)
 preds=[normalize_probs(np.loadtxt(ROOT/f'dermatology_predictions{i}.txt')) for i in range(4)]
 rows=[]
 for q_mass in [.5,.7,.8,.9]:
  for alpha_case in [.1]:
   for model_idx,e in enumerate(preds):
    n=len(e)
    for rep in range(args.reps):
      rng=np.random.default_rng(2026081500+10000*model_idx+100*q_mass.__hash__()%100+rep)
      perm=rng.permutation(n); ntr=int(.4*n); ncal=int(.3*n)
      tr=perm[:ntr]; cal=perm[ntr:ntr+ncal]; te=perm[ntr+ncal:]
      # baseline top1 CP and MCCP target marginal 1-alpha_case
      b=[]
      th=top1_cp(e[cal],lam[cal],alpha_case); b.append(('top1_cp',e[te]>=th))
      th=mccp(e[cal],lam[cal],alpha_case,rng,10); b.append(('mccp10',e[te]>=th))
      th=exact_soft_quantile(e[cal],lam[cal],alpha_case); b.append(('soft_exact',e[te]>=th))
      th=global_panel_quantile_threshold(e[cal],lam[cal],q_mass,alpha_case); b.append(('panel_quantile_global',predict_global(e[te],th)))
      pacs=PACS(q_mass,alpha_case,random_state=rep).fit(e[tr],lam[tr],e[cal],lam[cal]); b.append(('pacs',pacs.predict(e[te])))
      for method,mask in b:
       r=metrics(mask,lam[te],q_mass); r.update(method=method,q_mass=q_mass,alpha_case=alpha_case,model=model_idx,rep=rep,ntrain=len(tr),ncal=len(cal),ntest=len(te)); rows.append(r)
    print('done',q_mass,model_idx,flush=True)
 df=pd.DataFrame(rows); out=Path('experiments/pr_rescue/results'); out.mkdir(parents=True,exist_ok=True)
 df.to_csv(out/'dermatology_pacs_raw.csv',index=False)
 s=df.groupby(['q_mass','method']).agg(case_success=('case_mass_success','mean'),case_success_sd=('case_mass_success','std'),mean_mass=('mean_mass','mean'),p10_mass=('p10_mass','mean'),mean_size=('mean_size','mean'),p90_size=('p90_size','mean'),top1_cov=('top1_coverage','mean'),minority=('minority_fraction','mean'),runs=('rep','count')).reset_index()
 s.to_csv(out/'dermatology_pacs_summary.csv',index=False)
 print('\n',s.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
 # also dataset facts
 print('readers per case min/median/max',min(map(len,selectors)),np.median(list(map(len,selectors))),max(map(len,selectors)))
 print('cases',len(selectors),'classes',lam.shape[1])
if __name__=='__main__': main()
