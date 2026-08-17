#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.pacs import normalize_probs, global_panel_quantile_threshold, predict_global, plausibility_mass
from src.pacs_gated import DisagreementGatedPACS

ROOT = Path('.')
LAB = ROOT / 'data/public/nih_expert_labels/raw/all_findings_expert_labels/all_findings_expert_labels_test_individual_readers.csv'
FEAT = ROOT / 'experiments/pr_rescue/nih_features/resnet50_imagenet_features.npy'
IDX = ROOT / 'experiments/pr_rescue/nih_features/resnet50_imagenet_index.csv'
OUT = ROOT / 'experiments/pr_rescue/results'
OUT.mkdir(parents=True, exist_ok=True)

FINDINGS = ['Atelectasis','Cardiomegaly','Effusion','Infiltration','Mass','Nodule','Pneumonia','Pneumothorax','Consolidation','Edema','Emphysema','Fibrosis','Pleural Thickening','Hernia','Other']
CLASSES = ['No Finding'] + FINDINGS


def yes(v):
    return str(v).strip().upper() in {'YES','1','TRUE','Y','POSITIVE'}


def expert_mass(raw, ids):
    by = {x:i for i,x in enumerate(ids)}
    lam = np.zeros((len(ids), len(CLASSES)), float)
    sub = raw[raw['Image ID'].astype(str).isin(by)].copy()
    for _, r in sub.iterrows():
        i = by[str(r['Image ID'])]
        picked = [j+1 for j,f in enumerate(FINDINGS) if yes(r[f])]
        if picked:
            w = 1.0 / len(picked)
            for c in picked:
                lam[i,c] += w / 5.0
        else:
            lam[i,0] += 1.0 / 5.0
    if not np.allclose(lam.sum(1), 1):
        raise RuntimeError((lam.sum(1).min(), lam.sum(1).max()))
    return lam


def fit_ridge_soft(X, lam):
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    model.fit(X, lam)
    return model


def pred(model, X):
    z = np.asarray(model.predict(X), float)
    z = z - np.minimum(z.min(axis=1, keepdims=True), 0.0)
    z = np.clip(z, 1e-8, None)
    return normalize_probs(z)


def stutz_quantile(scores, alpha):
    a = np.asarray(scores, float)
    n = len(a)
    q = np.floor(alpha * (n + 1)) / n
    return float(np.quantile(a, np.clip(q,0,1), method='midpoint'))


def top1_cp(cal_e, cal_lam, alpha):
    y = np.argmax(cal_lam,1)
    return stutz_quantile(cal_e[np.arange(len(y)),y], alpha)


def mccp(cal_e, cal_lam, alpha, rng, num_samples=10):
    n,k = cal_e.shape
    cs = np.cumsum(cal_lam,axis=1)
    u = rng.random((num_samples,n))
    labels = (u[...,None] > cs[None,:,:]).sum(2)
    scores = np.concatenate([cal_e[np.arange(n),labels[j]] for j in range(num_samples)])
    q = (np.floor(alpha*num_samples*(n+1))-num_samples+1)/(n*num_samples)
    return float(np.quantile(scores,np.clip(q,0,1),method='midpoint'))


def metrics(mask, lam, q):
    mass = plausibility_mass(mask, lam)
    size = mask.sum(1)
    support = (lam > 1e-12).sum(1)
    return dict(
        success=float(np.mean(mass >= q - 1e-12)),
        mean_size=float(np.mean(size)),
        p90_size=float(np.quantile(size,.9)),
        mean_mass=float(np.mean(mass)),
        support=float(np.mean(support)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=30)
    ap.add_argument('--q', type=float, default=.8)
    args = ap.parse_args()
    q = float(args.q)
    alpha = .1

    X = np.load(FEAT)
    ids = pd.read_csv(IDX)['image_id'].astype(str).tolist()
    if X.shape[0] != len(ids) or len(ids) != 810:
        raise RuntimeError(f'expected complete 810-case feature set, got {X.shape[0]} / {len(ids)}')
    raw = pd.read_csv(LAB)
    raw['Image ID'] = raw['Image ID'].astype(str)
    lam = expert_mass(raw, ids)
    case = raw[['Image ID','Patient ID']].drop_duplicates('Image ID').set_index('Image ID').loc[ids].reset_index()
    patients = case['Patient ID'].astype(str).to_numpy()
    up = np.unique(patients)

    rows = []
    tag = str(q).replace('.','p')
    for rep in range(args.reps):
        rng = np.random.default_rng(2026082800 + rep)
        pp = rng.permutation(up)
        a = int(.4 * len(up)); b = int(.7 * len(up))
        trp, cap, tep = set(pp[:a]), set(pp[a:b]), set(pp[b:])
        tr = np.array([i for i,p in enumerate(patients) if p in trp])
        cal = np.array([i for i,p in enumerate(patients) if p in cap])
        te = np.array([i for i,p in enumerate(patients) if p in tep])

        model = fit_ridge_soft(X[tr], lam[tr])
        etr, ecal, ete = pred(model, X[tr]), pred(model, X[cal]), pred(model, X[te])
        methods = []
        th = top1_cp(ecal, lam[cal], alpha)
        methods.append(('top1_cp', ete >= th, {}))
        th = mccp(ecal, lam[cal], alpha, rng, 10)
        methods.append(('mccp10', ete >= th, {}))
        th = global_panel_quantile_threshold(ecal, lam[cal], q, alpha)
        methods.append(('global_panel', predict_global(ete, th), {}))

        gated = DisagreementGatedPACS(
            q_mass=q,
            alpha_case=alpha,
            alpha_gate=alpha,
            random_state=20260828 + rep,
            gate_fraction=.5,
            model_quantile=.9,
            n_bins=3,
            min_cal_per_bin=20,
            min_stratum_cal=25,
        ).fit(etr, lam[tr], ecal, lam[cal])
        methods.append(('pacs_gated', gated.predict(ete), gated.diagnostics()))

        for method, mask, diag in methods:
            r = metrics(mask, lam[te], q)
            r.update(method=method, rep=rep, q=q, ntrain=len(tr), ncal=len(cal), ntest=len(te))
            for key,val in diag.items():
                r[f'gate_{key}'] = val
            rows.append(r)
        pd.DataFrame(rows).to_csv(OUT / f'nih_multiclass_pacs_gated_q{tag}_raw.csv', index=False)
        print('done', rep, gated.diagnostics(), flush=True)

    df = pd.DataFrame(rows)
    summary = df.groupby('method').agg(
        success=('success','mean'), success_sd=('success','std'),
        mean_size=('mean_size','mean'), mean_size_sd=('mean_size','std'),
        p90_size=('p90_size','mean'), mean_mass=('mean_mass','mean'),
        support=('support','mean'), runs=('rep','count'),
    ).reset_index()
    summary.to_csv(OUT / f'nih_multiclass_pacs_gated_q{tag}_summary.csv', index=False)
    print(summary.to_string(index=False, float_format=lambda z:f'{z:.4f}'))


if __name__ == '__main__':
    main()
