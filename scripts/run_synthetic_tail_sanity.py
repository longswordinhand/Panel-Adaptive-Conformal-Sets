#!/usr/bin/env python3
from pathlib import Path
import math
import numpy as np
import pandas as pd
from scipy.special import betaln, comb
from scipy.stats import beta as beta_dist

from run_tail_identification_experiments import tail_bounds

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/pilot/results/tail_identification/synthetic_population_bounds.csv'

# Each scenario is a finite mixture of Beta(a,b).
SCENARIOS = {
    'beta_2_5': [(1.0, 2.0, 5.0)],
    'beta_5_2': [(1.0, 5.0, 2.0)],
    'bimodal_2_8__8_2': [(0.5, 2.0, 8.0), (0.5, 8.0, 2.0)],
    'near_threshold_20_20': [(1.0, 20.0, 20.0)],
    'asymmetric_mix': [(0.7, 2.0, 10.0), (0.3, 10.0, 3.0)],
}


def beta_binom_pmf(m, k, a, b):
    return comb(m, k) * math.exp(betaln(k+a, m-k+b) - betaln(a,b))


def observable_p(m, mix):
    p=np.zeros(m+1)
    for w,a,b in mix:
        p += w*np.array([beta_binom_pmf(m,k,a,b) for k in range(m+1)])
    return p/p.sum()


def true_tail(beta, mix):
    return sum(w*(1-beta_dist.cdf(beta,a,b)) for w,a,b in mix)


def main():
    grid=np.linspace(0,1,4001)
    rows=[]
    for name,mix in SCENARIOS.items():
        truth=true_tail(0.5,mix)
        for m in [1,2,3,5,8,12,20]:
            p=observable_p(m,mix)
            lo,hi,err=tail_bounds(p,m,0.5,grid)
            rows.append(dict(scenario=name,m=m,beta=0.5,true_tail=truth,lower=lo,upper=hi,width=hi-lo,contains_truth=(lo-1e-6<=truth<=hi+1e-6),projection_l1=err))
    d=pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    d.to_csv(OUT,index=False)
    print(d.to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    print('\nAll contain truth:', bool(d.contains_truth.all()))
    print('\nWidth at m=1 vs m=20:')
    print(d.pivot(index='scenario',columns='m',values='width')[[1,3,5,12,20]].to_string(float_format=lambda x:f'{x:.6f}'))

if __name__=='__main__':
    main()
