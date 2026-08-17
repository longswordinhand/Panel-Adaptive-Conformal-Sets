#!/usr/bin/env python3
import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd


def sample_theta(rng, n, gamma, beta=0.5, d0=0.4):
    # P(D <= t) = (t/d0)^gamma on [0,d0], saturating a gamma-margin law.
    u = rng.random(n)
    d = d0 * u ** (1.0 / gamma)
    sign = np.where(rng.random(n) < 0.5, -1.0, 1.0)
    return beta + sign * d


def run_once(rng, eps, gamma, n_mult=10.0, beta=0.5, d0=0.4, delta=0.05):
    n = int(math.ceil(n_mult / eps**2))
    h = d0 * eps ** (1.0 / gamma)
    J = max(0, int(math.ceil(math.log(d0 / h, 2))))
    x = 2.0 ** (-np.arange(J + 1, dtype=float))
    d = d0 * x
    A = 1.0 + float(np.sum(x[1:] ** (gamma - 1.0))) if J >= 1 else 1.0
    S = np.minimum(1.0, A * x)
    S[0] = 1.0
    L = math.log(8.0 * n * (J + 1) / delta)
    # r_s <= d_j/2, enough to resolve gaps larger than about d_j on valid paths.
    s = np.ceil(2.0 * L / np.maximum(d, 1e-12) ** 2).astype(np.int64)

    theta = sample_theta(rng, n, gamma, beta=beta, d0=d0)
    active = np.ones(n, dtype=bool)
    heads = np.zeros(n, dtype=np.int64)
    prev_s = 0
    z = np.zeros(n, dtype=float)
    budget = 0

    for j in range(J + 1):
        if j > 0:
            idx = np.flatnonzero(active)
            keep_prob = min(1.0, S[j] / S[j-1])
            keep = rng.random(len(idx)) < keep_prob
            active[idx[~keep]] = False
        idx = np.flatnonzero(active)
        if len(idx) == 0:
            prev_s = int(s[j])
            continue
        inc = int(s[j] - prev_s)
        if inc > 0:
            heads[idx] += rng.binomial(inc, theta[idx])
            budget += inc * len(idx)
        means = heads[idx] / float(s[j])
        rad = math.sqrt(L / (2.0 * float(s[j])))
        pos = means - rad > beta
        neg = means + rad < beta
        resolved = pos | neg
        if np.any(pos):
            z[idx[pos]] = 1.0 / S[j]
        active[idx[resolved]] = False
        prev_s = int(s[j])

    est = float(z.mean())
    return {
        'epsilon': eps, 'gamma': gamma, 'n': n, 'J': J, 'h': h,
        'A': A, 'S_J': float(S[-1]), 'estimate': est,
        'abs_error': abs(est - 0.5), 'budget': int(budget),
        'budget_scaled': budget * eps ** max(2.0, 2.0/gamma),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=50)
    ap.add_argument('--seed', type=int, default=20260815)
    ap.add_argument('--out', default='experiments/pilot/results/thinned_sequential_simulation.csv')
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    rows=[]
    for gamma in [0.5, 0.75, 1.0, 1.5]:
        for eps in [0.2, 0.15, 0.1]:
            for _ in range(args.reps):
                rows.append(run_once(rng, eps, gamma))
    df=pd.DataFrame(rows)
    out=Path(args.out); out.parent.mkdir(parents=True, exist_ok=True); df.to_csv(out,index=False)
    summary=(df.groupby(['gamma','epsilon'])
             .agg(reps=('estimate','size'), mean_abs_error=('abs_error','mean'),
                  q90_abs_error=('abs_error',lambda x: np.quantile(x,.9)),
                  mean_budget=('budget','mean'), mean_scaled_budget=('budget_scaled','mean'),
                  mean_SJ=('S_J','mean'))
             .reset_index())
    print(summary.to_string(index=False))

if __name__=='__main__':
    main()
