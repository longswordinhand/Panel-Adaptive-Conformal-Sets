#!/usr/bin/env python3
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import linprog
from scipy.special import comb
from scipy.stats import beta as beta_dist

NIH=Path('data/public/nih_expert_labels/raw/all_findings_expert_labels/all_findings_expert_labels_test_individual_readers.csv')
VINDR=Path('data/public/vindr_cxr/raw/train.csv')
OUT=Path('experiments/pilot/results/tail_honest_ci'); OUT.mkdir(parents=True,exist_ok=True)

def cp(x,n,a):
    return (0 if x==0 else beta_dist.ppf(a/2,x,n-x+1), 1 if x==n else beta_dist.ppf(1-a/2,x+1,n-x))

def ci(counts,beta=.5,alpha=.05,grid_n=2001,gamma=None,C=None):
    counts=np.asarray(counts,int); n=counts.sum(); m=len(counts)-1; d=m+1
    x=np.linspace(0,1,grid_n); B=np.vstack([comb(m,k)*x**k*(1-x)**(m-k) for k in range(d)])
    A=[]; b=[]
    for k,N in enumerate(counts):
        lo,hi=cp(int(N),int(n),alpha/d); A += [B[k],-B[k]]; b += [hi,-lo]
    if gamma is not None:
        dist=np.abs(x-beta); radii=np.unique(dist[(dist>0)&(dist<=.5)])
        if len(radii)>100:
            ids=np.unique(np.round(np.geomspace(1,len(radii),100)).astype(int)-1); radii=radii[ids]
        for t in radii:
            A.append((dist<=t+1e-12).astype(float)); b.append(min(1.,C*t**gamma))
    kw=dict(A_ub=np.asarray(A),b_ub=np.asarray(b),A_eq=np.ones((1,grid_n)),b_eq=[1],bounds=(0,None),method='highs')
    tail=(x>beta).astype(float); lo=linprog(tail,**kw); hi=linprog(-tail,**kw)
    return (np.nan,np.nan,False) if not(lo.success and hi.success) else (float(lo.fun),float(-hi.fun),True)

def nih_counts(label):
    df=pd.read_csv(NIH); v=df[label].map(lambda z:int(z) if label=='Abnormal' else int(str(z).upper()=='YES'))
    K=pd.DataFrame({'id':df['Image ID'],'v':v}).groupby('id').v.sum().astype(int)
    return np.bincount(K,minlength=6)

def vindr_counts(label):
    df=pd.read_csv(VINDR); z=df[['image_id','rad_id','class_name']].drop_duplicates(); imgs=df.image_id.drop_duplicates()
    pos=z[z.class_name.eq(label)][['image_id','rad_id']].drop_duplicates().assign(v=1)
    # all image-reader pairs
    pairs=df[['image_id','rad_id']].drop_duplicates().merge(pos,on=['image_id','rad_id'],how='left').fillna({'v':0})
    K=pairs.groupby('image_id').v.sum().reindex(imgs,fill_value=0).astype(int)
    return np.bincount(K,minlength=4)

rows=[]
for dataset, labels, getter in [
 ('NIH',['Abnormal','Cardiomegaly','Consolidation','Nodule','Pleural Thickening','Pneumothorax'],nih_counts),
 ('VinDr',['Pleural thickening','Lung Opacity','Nodule/Mass','Cardiomegaly','Aortic enlargement'],vindr_counts)]:
    for label in labels:
        counts=getter(label)
        for beta in [.5]:
            for gamma,C,tag in [(None,None,'unrestricted'),(1.,1.,'margin_g1_C1'),(1.,2.,'margin_g1_C2'),(1.,4.,'margin_g1_C4'),(1.,8.,'margin_g1_C8')]:
                l,u,ok=ci(counts,beta=beta,gamma=gamma,C=C,grid_n=2001)
                rows.append(dict(dataset=dataset,label=label,n=int(counts.sum()),m=len(counts)-1,beta=beta,assumption=tag,lower=l,upper=u,width=u-l if ok else np.nan,feasible=ok))
out=pd.DataFrame(rows); out.to_csv(OUT/'real_data_honest_ci_sensitivity.csv',index=False)
print(out.to_string(index=False))
