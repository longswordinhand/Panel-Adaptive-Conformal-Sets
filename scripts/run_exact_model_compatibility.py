#!/usr/bin/env python3
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import linprog
from scipy.special import comb
from scipy.stats import beta as beta_dist

DATA=Path('data/public/nih_expert_labels/raw/all_findings_expert_labels/all_findings_expert_labels_test_individual_readers.csv')
OUT=Path('experiments/pilot/results/model_compatibility'); OUT.mkdir(parents=True,exist_ok=True)

def cp(x,n,a):
    lo=0 if x==0 else beta_dist.ppf(a/2,x,n-x+1)
    hi=1 if x==n else beta_dist.ppf(1-a/2,x+1,n-x)
    return float(lo),float(hi)

def feasible(counts,alpha=.05,grid_n=4001):
    counts=np.asarray(counts,int); n=counts.sum(); m=len(counts)-1; d=m+1
    x=np.linspace(0,1,grid_n)
    B=np.vstack([comb(m,k)*x**k*(1-x)**(m-k) for k in range(d)])
    A=[]; b=[]
    for k,N in enumerate(counts):
        lo,hi=cp(int(N),int(n),alpha/d)
        A.append(B[k]); b.append(hi); A.append(-B[k]); b.append(-lo)
    r=linprog(np.zeros(grid_n),A_ub=np.asarray(A),b_ub=np.asarray(b),A_eq=np.ones((1,grid_n)),b_eq=[1],bounds=(0,None),method='highs')
    return r.success

df=pd.read_csv(DATA)
labels=['Abnormal','Atelectasis','Cardiomegaly','Effusion','Consolidation','Nodule','Pleural Thickening','Pneumothorax']
rows=[]
for label in labels:
    vals=df[label].map(lambda z: int(z) if label=='Abnormal' else int(str(z).upper()=='YES'))
    tmp=pd.DataFrame({'Image ID':df['Image ID'],'v':vals})
    K=tmp.groupby('Image ID')['v'].sum().astype(int)
    counts=np.bincount(K,minlength=6)
    rows.append({'label':label,'n':int(counts.sum()),'counts':';'.join(map(str,counts)),'compatible_95':feasible(counts,.05,4001),'compatible_99':feasible(counts,.01,4001)})
out=pd.DataFrame(rows); out.to_csv(OUT/'nih_m5_exact_compatibility.csv',index=False)
print(out.to_string(index=False))
