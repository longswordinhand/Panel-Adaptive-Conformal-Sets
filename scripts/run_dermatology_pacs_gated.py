#!/usr/bin/env python3
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd

from src.pacs import normalize_probs, global_panel_quantile_threshold, predict_global
from src.pacs_gated import DisagreementGatedPACS
from scripts.run_dermatology_pacs import irn_from_selectors, metrics, mccp, top1_cp

ROOT = Path('third_party/uncertain_ground_truth/data')
OUT = Path('experiments/pr_rescue/results')
OUT.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=12)
    ap.add_argument('--q', type=float, default=.9)
    args = ap.parse_args()
    q = float(args.q)
    alpha = .1

    selectors = json.load(open(ROOT / 'dermatology_selectors.json'))
    lam = irn_from_selectors(selectors)
    preds = [normalize_probs(np.loadtxt(ROOT / f'dermatology_predictions{i}.txt')) for i in range(4)]
    rows = []
    tag = str(q).replace('.', 'p')

    for rep in range(args.reps):
        for model_idx, e in enumerate(preds):
            n = len(e)
            rng = np.random.default_rng(2026082700 + 100000 * rep + 10000 * model_idx + int(q * 1000))
            perm = rng.permutation(n)
            ntr = int(.4 * n)
            ncal = int(.3 * n)
            tr, cal, te = perm[:ntr], perm[ntr:ntr+ncal], perm[ntr+ncal:]

            methods = []
            th = global_panel_quantile_threshold(e[cal], lam[cal], q, alpha)
            methods.append(('global', predict_global(e[te], th), {}))
            th = top1_cp(e[cal], lam[cal], alpha)
            methods.append(('top1_cp', e[te] >= th, {}))
            th = mccp(e[cal], lam[cal], alpha, rng, 10)
            methods.append(('mccp10', e[te] >= th, {}))

            gated = DisagreementGatedPACS(
                q_mass=q,
                alpha_case=alpha,
                alpha_gate=alpha,
                random_state=20260827 + 101 * rep + model_idx,
                gate_fraction=.5,
                model_quantile=.9,
                n_bins=3,
                min_cal_per_bin=30,
                min_stratum_cal=40,
            ).fit(e[tr], lam[tr], e[cal], lam[cal])
            methods.append(('pacs_gated', gated.predict(e[te]), gated.diagnostics()))

            for method, mask, diag in methods:
                r = metrics(mask, lam[te], q)
                r.update(rep=rep, model=model_idx, method=method, q=q)
                for key, val in diag.items():
                    r[f'gate_{key}'] = val
                rows.append(r)
            pd.DataFrame(rows).to_csv(OUT / f'dermatology_pacs_gated_q{tag}_raw.csv', index=False)
            print('done', rep, model_idx, gated.diagnostics(), flush=True)

    df = pd.DataFrame(rows)
    summary = df.groupby('method').agg(
        success=('case_mass_success', 'mean'),
        success_sd=('case_mass_success', 'std'),
        mean_size=('mean_size', 'mean'),
        mean_size_sd=('mean_size', 'std'),
        p90_size=('p90_size', 'mean'),
        mean_mass=('mean_mass', 'mean'),
        minority=('minority_fraction', 'mean'),
        runs=('rep', 'count'),
    ).reset_index()
    summary.to_csv(OUT / f'dermatology_pacs_gated_q{tag}_summary.csv', index=False)
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.4f}'))


if __name__ == '__main__':
    main()
