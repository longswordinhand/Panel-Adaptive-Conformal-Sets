#!/usr/bin/env python3
"""Controlled PACS ablation on the dermatology benchmark.

All methods use identical model-by-split partitions. The ablation isolates:
1) Global panel: no case-specific demand model.
2) PACS-GlobalResidual: predicts case-specific top-k demand, but applies one
   global conformal residual correction to all cases.
3) PACS-Stratified: same demand model and quantile, plus train-defined
   difficulty strata with stratum-specific residual correction (main PACS).
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.pacs import normalize_probs, global_panel_quantile_threshold, predict_global
from src.pacs_v2 import PACSTopK
from src.pacs_mondrian import PACSTopKMondrian
from scripts.run_dermatology_pacs import irn_from_selectors, metrics

ROOT = Path("third_party/uncertain_ground_truth/data")
OUT = Path("experiments/pr_rescue/results")
OUT.mkdir(parents=True, exist_ok=True)
Q = 0.9
ALPHA = 0.1
REPS = 12
MODEL_QUANTILE = 0.9

selectors = json.load(open(ROOT / "dermatology_selectors.json"))
lam = irn_from_selectors(selectors)
preds = [normalize_probs(np.loadtxt(ROOT / f"dermatology_predictions{i}.txt")) for i in range(4)]
rows = []

for rep in range(REPS):
    for model_idx, e in enumerate(preds):
        n = len(e)
        rng = np.random.default_rng(2026081800 + 100000 * rep + 10000 * model_idx + int(1000 * Q))
        perm = rng.permutation(n)
        ntr = int(0.4 * n)
        ncal = int(0.3 * n)
        tr = perm[:ntr]
        cal = perm[ntr:ntr+ncal]
        te = perm[ntr+ncal:]

        th = global_panel_quantile_threshold(e[cal], lam[cal], Q, ALPHA)
        global_mask = predict_global(e[te], th)

        pacs_global = PACSTopK(
            Q, ALPHA, random_state=rep*101+model_idx, model_quantile=MODEL_QUANTILE
        ).fit(e[tr], lam[tr], e[cal], lam[cal])
        global_resid_mask = pacs_global.predict(e[te])

        pacs_strat = PACSTopKMondrian(
            Q, ALPHA, random_state=rep*101+model_idx, model_quantile=MODEL_QUANTILE,
            n_bins=3, min_cal_per_bin=40
        ).fit(e[tr], lam[tr], e[cal], lam[cal])
        strat_mask = pacs_strat.predict(e[te])

        for method, mask in [
            ("global_panel", global_mask),
            ("pacs_global_residual", global_resid_mask),
            ("pacs_stratified", strat_mask),
        ]:
            r = metrics(mask, lam[te], Q)
            r.update(rep=rep, model=model_idx, method=method)
            rows.append(r)
    print("done rep", rep, flush=True)

raw = pd.DataFrame(rows)
raw_path = OUT / "dermatology_pacs_ablation_q0p9_raw.csv"
raw.to_csv(raw_path, index=False)
summary = raw.groupby("method").agg(
    success=("case_mass_success", "mean"),
    success_sd=("case_mass_success", "std"),
    mean_size=("mean_size", "mean"),
    mean_size_sd=("mean_size", "std"),
    p90_size=("p90_size", "mean"),
    mean_mass=("mean_mass", "mean"),
    runs=("rep", "count"),
).reset_index()
summary_path = OUT / "dermatology_pacs_ablation_q0p9_summary.csv"
summary.to_csv(summary_path, index=False)
print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
