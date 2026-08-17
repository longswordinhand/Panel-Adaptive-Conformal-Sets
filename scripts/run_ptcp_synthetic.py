#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from src.panel_tolerance_cp import (
    aps_label_scores, panel_tolerance_scores, split_conformal_threshold,
    ptcp_threshold, prediction_set_mask, expert_mass_captured,
)


def sample_latent(rng,n,k,regime):
    if regime=='mixed':
        comp=rng.random(n)<0.55
        q=np.empty((n,k))
        # confident and ambiguous cases mixed
        q[comp]=rng.dirichlet(np.full(k,.18),comp.sum())
        q[~comp]=rng.dirichlet(np.full(k,1.5),(~comp).sum())
        return q
    if regime=='ambiguous': return rng.dirichlet(np.full(k,1.5),n)
    if regime=='confident': return rng.dirichlet(np.full(k,.18),n)
    raise ValueError(regime)

def noisy_model_probs(rng,q,sigma=.8):
    # Case-dependent misspecification preserving some ranking signal.
    z=np.log(np.clip(q,1e-8,1))+rng.normal(0,sigma,q.shape)
    z-=z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)

def draw_panels(rng,q,m):
    n,k=q.shape
    u=rng.random((n,m)); cs=np.cumsum(q,axis=1)
    return (u[...,None] > cs[:,None,:]).sum(axis=2).astype(int)

def majority_label(panel,k):
    # deterministic stable tie break; multiclass majority/plurality
    out=[]
    for row in panel:
        out.append(int(np.argmax(np.bincount(row,minlength=k))))
    return np.array(out)

def calibrate_majority(scores,panel,alpha):
    y=majority_label(panel,scores.shape[1]); a=scores[np.arange(len(y)),y]
    return split_conformal_threshold(a,alpha)

def calibrate_random_rater(scores,panel,alpha,rng):
    j=rng.integers(0,panel.shape[1],size=len(panel)); y=panel[np.arange(len(panel)),j]
    a=scores[np.arange(len(y)),y]; return split_conformal_threshold(a,alpha)

def calibrate_naive_panel(scores,panel,q_mass,alpha):
    m=panel.shape[1]; r=max(1,min(m,int(math.ceil(q_mass*m))))
    obs=np.take_along_axis(scores,panel,axis=1); obs.sort(axis=1)
    return split_conformal_threshold(obs[:,r-1],alpha)

def evaluate(name,t,scores_test,q_test,q_mass):
    mask=prediction_set_mask(scores_test,t)
    mass=expert_mass_captured(mask,q_test)
    return dict(method=name,latent_mass_success=np.mean(mass+1e-12>=q_mass),
                mean_mass=np.mean(mass),mean_set_size=np.mean(mask.sum(1)),
                singleton=np.mean(mask.sum(1)==1),full_set=np.mean(mask.sum(1)==mask.shape[1]))

def one_rep(seed,m,ncal,ntest,k,q_mass,alpha,delta,regime,sigma):
    rng=np.random.default_rng(seed)
    qcal=sample_latent(rng,ncal,k,regime); qtest=sample_latent(rng,ntest,k,regime)
    pcal=noisy_model_probs(rng,qcal,sigma); ptest=noisy_model_probs(rng,qtest,sigma)
    scal=aps_label_scores(pcal); stest=aps_label_scores(ptest)
    panel=draw_panels(rng,qcal,m)
    rows=[]
    t=calibrate_majority(scal,panel,alpha); rows.append(evaluate('majority_cp',t,stest,qtest,q_mass))
    t=calibrate_random_rater(scal,panel,alpha,rng); rows.append(evaluate('random_rater_cp',t,stest,qtest,q_mass))
    t=calibrate_naive_panel(scal,panel,q_mass,alpha); rows.append(evaluate('naive_panel_cp',t,stest,qtest,q_mass))
    try:
        t=ptcp_threshold(scal,panel,q_mass,alpha,delta,False,rng); rows.append(evaluate('ptcp',t,stest,qtest,q_mass))
        # independent RNG for randomized tolerance
        rr=np.random.default_rng(seed+10_000_003)
        t=ptcp_threshold(scal,panel,q_mass,alpha,delta,True,rr); rows.append(evaluate('ptcp_randomized',t,stest,qtest,q_mass))
    except ValueError:
        pass
    for r in rows: r.update(seed=seed,m=m,ncal=ncal,ntest=ntest,k=k,q_mass=q_mass,alpha=alpha,delta=delta,regime=regime,sigma=sigma)
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--reps',type=int,default=300); ap.add_argument('--ntest',type=int,default=5000)
    args=ap.parse_args()
    allr=[]
    for regime in ['mixed','ambiguous']:
      for m in [3,5,10]:
       for rep in range(args.reps):
        allr += one_rep(2026081500+rep+1000*m+(0 if regime=='mixed' else 100000),m,300,args.ntest,5,.5,.1,.05,regime,.8)
    df=pd.DataFrame(allr); out=Path('experiments/pr_rescue/results/ptcp_synthetic_raw.csv'); df.to_csv(out,index=False)
    summ=(df.groupby(['regime','m','method']).agg(success_mean=('latent_mass_success','mean'),success_sd=('latent_mass_success','std'),set_size=('mean_set_size','mean'),full_set=('full_set','mean'),singleton=('singleton','mean'),reps=('seed','count')).reset_index())
    summ.to_csv('experiments/pr_rescue/results/ptcp_synthetic_summary.csv',index=False)
    print(summ.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    # Across-replicate probability that empirical test success itself reaches target 1-alpha.
    guar=(df.assign(hit=lambda x:x.latent_mass_success>=1-x.alpha).groupby(['regime','m','method']).hit.mean().reset_index(name='rep_fraction_success_ge_target'))
    print('\nREPLICATE TARGET HIT FRACTION\n',guar.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    json.dump({'reps':args.reps,'ntest':args.ntest,'seed_base':2026081500},open('experiments/pr_rescue/results/ptcp_synthetic_meta.json','w'),indent=2)
if __name__=='__main__': main()
