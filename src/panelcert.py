"""PanelCert: finite-panel latent expert-mass certification.

For a fixed nested prediction-set threshold t, observed panel capture counts
K_i(t) are a binomial mixture over case-specific latent capture probabilities
p_i(t).  A simultaneous confidence box for the observable count law is
propagated through a measure LP to lower-bound P(p_i(t) >= q_mass).

A deterministic threshold grid is tested from largest to smallest with a
fixed-sequence stopping rule.  Because validity is monotone in t, the first
true null controls the family-wise probability of false certification without
multiplicity correction across thresholds.

The measure problem is numerically discretized on [0,1]; grid convergence must
be audited in experiments.  This implementation uses conservative Bonferroni
Clopper-Pearson intervals within each count histogram.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linprog
from scipy.special import comb
from scipy.stats import beta as beta_dist


def _cp_interval(x:int,n:int,err:float):
    lo=0.0 if x==0 else float(beta_dist.ppf(err/2,x,n-x+1))
    hi=1.0 if x==n else float(beta_dist.ppf(1-err/2,x+1,n-x))
    return lo,hi


def bernstein_matrix(m:int, grid:np.ndarray)->np.ndarray:
    x=np.asarray(grid,float)
    return np.vstack([comb(m,k)*x**k*(1-x)**(m-k) for k in range(m+1)])


def lower_latent_mass_fraction(counts, q_mass:float, confidence_error:float=.05,
                               grid_n:int=2001):
    """Conservative lower CI for P(p >= q_mass) from mixture-binomial counts."""
    counts=np.asarray(counts,int)
    if counts.ndim!=1 or np.any(counts<0) or counts.sum()<1:
        raise ValueError('invalid counts')
    if not 0<q_mass<1 or not 0<confidence_error<1:
        raise ValueError('q_mass/confidence_error')
    n=int(counts.sum()); m=len(counts)-1; d=m+1
    x=np.linspace(0,1,grid_n); B=bernstein_matrix(m,x)
    A=[]; b=[]
    for k,N in enumerate(counts):
        lo,hi=_cp_interval(int(N),n,confidence_error/d)
        A.extend([B[k],-B[k]]); b.extend([hi,-lo])
    # >= q.  For a grid, include equality point.
    tail=(x>=q_mass-1e-15).astype(float)
    r=linprog(tail,A_ub=np.asarray(A),b_ub=np.asarray(b),
              A_eq=np.ones((1,grid_n)),b_eq=[1.0],bounds=(0,None),method='highs')
    if not r.success:
        return np.nan,False
    return float(r.fun),True


def panel_capture_counts(label_scores:np.ndarray,panel_labels:np.ndarray,threshold:float):
    s=np.asarray(label_scores,float); y=np.asarray(panel_labels,int)
    obs=np.take_along_axis(s,y,axis=1)
    k=(obs<=threshold).sum(axis=1)
    return np.bincount(k,minlength=y.shape[1]+1)

@dataclass(frozen=True)
class PanelCertResult:
    threshold: float
    lower_fraction: float
    certified: bool
    tested_thresholds: int


def certify_threshold(label_scores,panel_labels,threshold,q_mass,target_fraction,
                      confidence_error=.05,grid_n=2001):
    counts=panel_capture_counts(label_scores,panel_labels,threshold)
    lo,ok=lower_latent_mass_fraction(counts,q_mass,confidence_error,grid_n)
    return lo, bool(ok and lo>=target_fraction)


def fixed_sequence_panelcert(label_scores,panel_labels,q_mass:float,
                             alpha_case:float=.1, confidence_error:float=.05,
                             thresholds=None, grid_n:int=2001)->PanelCertResult:
    """Select smallest certified threshold using monotone fixed-sequence testing.

    thresholds must be deterministic / prespecified independently of calibration
    labels.  They are sorted and tested from largest to smallest.  Testing stops
    at the first non-certification and returns the previous (larger) threshold.
    """
    if thresholds is None:
        thresholds=np.linspace(0,1,101)
    ts=np.unique(np.asarray(thresholds,float))
    if np.any((ts<0)|(ts>1)): raise ValueError('threshold grid must be in [0,1]')
    target=1.0-alpha_case
    last_t=np.nan; last_lo=np.nan; tested=0
    for t in ts[::-1]:
        lo,cert=certify_threshold(label_scores,panel_labels,float(t),q_mass,target,
                                  confidence_error,grid_n)
        tested+=1
        if cert:
            last_t=float(t); last_lo=float(lo)
        else:
            break
    if np.isnan(last_t):
        return PanelCertResult(float('inf'),float('nan'),False,tested)
    return PanelCertResult(last_t,last_lo,True,tested)
