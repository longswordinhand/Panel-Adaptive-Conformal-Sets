#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from multirater_conformal_seg.calibration.pixel_interval_cp import (
    consensus_mask, inclusion_score, split_conformal_quantile
)

CACHE = ROOT / "data/processed/prostate_512"
SPLITS = ROOT / "experiments/pilot/splits/prostate_5fold_patient_splits.csv"
PRED_ROOT = ROOT / "experiments/pilot/predictions/main"
AUDIT = ROOT / "data/processed/audit_prostate/case_task_audit.csv"
OUT = ROOT / "experiments/pilot/results/main"
ALPHAS = (0.10, 0.20, 0.30)
RANDOM_REPS = 200
BASE_SEED = 20260815
EXTREME_TASK02 = {"case07", "case50"}


def load_case(fold: int, task: str, case_id: str) -> dict:
    prob = np.load(PRED_ROOT / f"fold{fold}" / f"task{task}" / f"{case_id}.npy").astype(np.float32)
    with np.load(CACHE / f"{case_id}.npz") as d:
        masks = d[f"task{task}_masks"].astype(np.uint8)
    if prob.shape != (512, 512) or masks.ndim != 3 or masks.shape[1:] != prob.shape:
        raise RuntimeError(f"Shape mismatch fold={fold} task={task} case={case_id}")
    scores = np.array([inclusion_score(prob, m) for m in masks], dtype=float)
    cons = consensus_mask([m for m in masks])
    cons_score = float(inclusion_score(prob, cons))
    # Ambiguity band is exactly |p-0.5| <= q. Sorting makes every later q lookup O(log V).
    abs_margin_sorted = np.sort(np.abs(prob.ravel().astype(np.float32) - 0.5))
    cons_area = max(int(np.count_nonzero(cons)), 1)
    return {
        "scores": scores,
        "consensus_score": cons_score,
        "margins": abs_margin_sorted,
        "n_pixels": int(prob.size),
        "consensus_area": cons_area,
    }


def metrics_from_cache(c: dict, q: float) -> dict[str, float]:
    covered = c["scores"] <= (q + 1e-12)
    area = int(np.searchsorted(c["margins"], q + 1e-12, side="right"))
    return {
        "random_rater_coverage": float(covered.mean()),
        "mean_expert_miss_rate": float(1.0 - covered.mean()),
        "all_rater_coverage": float(covered.all()),
        "consensus_coverage": float(c["consensus_score"] <= q + 1e-12),
        "ambiguity_area_px": float(area),
        "ambiguity_fraction_image": float(area / c["n_pixels"]),
        "ambiguity_to_consensus_ratio": float(area / c["consensus_area"]),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    split_df = pd.read_csv(SPLITS)
    audit = pd.read_csv(AUDIT)
    audit["task"] = audit["task"].astype(int).map(lambda x: f"{x:02d}")
    case_rows, threshold_rows, required_rows = [], [], []

    for task in ("01", "02"):
        for fold in range(5):
            fdf = split_df[split_df.fold == fold]
            cal_cases = sorted(fdf.loc[fdf.role == "calibration", "case_id"].tolist())
            test_cases = sorted(fdf.loc[fdf.role == "test", "case_id"].tolist())
            cal = {c: load_case(fold, task, c) for c in cal_cases}
            test = {c: load_case(fold, task, c) for c in test_cases}

            for case_id, cc in test.items():
                q_all = float(cc["scores"].max())
                req = metrics_from_cache(cc, q_all)
                ar = audit[(audit.task == task) & (audit.case_id == case_id)].iloc[0]
                required_rows.append({
                    "task": task, "fold": fold, "case_id": case_id,
                    "q_required_all": q_all,
                    "q_required_random_mean": float(cc["scores"].mean()),
                    "required_all_ambiguity_fraction": req["ambiguity_fraction_image"],
                    "required_all_ambiguity_ratio": req["ambiguity_to_consensus_ratio"],
                    "audit_disagreement_1_minus_min_dice": float(1-ar.pairwise_dice_min),
                    "audit_mean_dice": float(ar.pairwise_dice_mean),
                    "audit_volume_cv": float(ar.volume_cv),
                    "extreme_task02": task == "02" and case_id in EXTREME_TASK02,
                })

            cons_scores = [cal[c]["consensus_score"] for c in cal_cases]
            naive_scores = [s for c in cal_cases for s in cal[c]["scores"]]
            all_scores = [float(cal[c]["scores"].max()) for c in cal_cases]

            for alpha in ALPHAS:
                qs = {
                    "consensus": split_conformal_quantile(cons_scores, alpha),
                    "naive_annotation": split_conformal_quantile(naive_scores, alpha),
                    "all_rater": split_conformal_quantile(all_scores, alpha),
                }
                for method, q in qs.items():
                    threshold_rows.append({"task":task,"fold":fold,"alpha":alpha,"method":method,"replicate":-1,"q":q})
                    for case_id, cc in test.items():
                        case_rows.append({"task":task,"fold":fold,"alpha":alpha,"method":method,"replicate":-1,"case_id":case_id,
                                          "q":q,"extreme_task02":task=="02" and case_id in EXTREME_TASK02, **metrics_from_cache(cc,q)})

                for rep in range(RANDOM_REPS):
                    seed = BASE_SEED + int(task)*100000 + fold*1000 + int(round(alpha*100))*10 + rep
                    rng = np.random.default_rng(seed)
                    sampled = [cal[c]["scores"][int(rng.integers(0, len(cal[c]["scores"])))] for c in cal_cases]
                    q = split_conformal_quantile(sampled, alpha)
                    threshold_rows.append({"task":task,"fold":fold,"alpha":alpha,"method":"random_rater","replicate":rep,"q":q})
                    for case_id, cc in test.items():
                        case_rows.append({"task":task,"fold":fold,"alpha":alpha,"method":"random_rater","replicate":rep,"case_id":case_id,
                                          "q":q,"extreme_task02":task=="02" and case_id in EXTREME_TASK02, **metrics_from_cache(cc,q)})

    cases = pd.DataFrame(case_rows)
    thresholds = pd.DataFrame(threshold_rows)
    required = pd.DataFrame(required_rows)
    cases.to_csv(OUT/"case_level_metrics.csv", index=False)
    thresholds.to_csv(OUT/"calibration_thresholds.csv", index=False)
    required.to_csv(OUT/"required_q_by_case.csv", index=False)

    det = cases[cases.method != "random_rater"].copy()
    rr = cases[cases.method == "random_rater"].groupby(["task","fold","alpha","method","case_id","extreme_task02"], as_index=False).agg(
        q=("q","mean"), random_rater_coverage=("random_rater_coverage","mean"), mean_expert_miss_rate=("mean_expert_miss_rate","mean"),
        all_rater_coverage=("all_rater_coverage","mean"), consensus_coverage=("consensus_coverage","mean"), ambiguity_area_px=("ambiguity_area_px","mean"),
        ambiguity_fraction_image=("ambiguity_fraction_image","mean"), ambiguity_to_consensus_ratio=("ambiguity_to_consensus_ratio","mean"))
    combined = pd.concat([det, rr], ignore_index=True)
    combined.to_csv(OUT/"case_level_metrics_replicate_averaged.csv", index=False)

    summary = []
    metric_cols = ["random_rater_coverage","mean_expert_miss_rate","all_rater_coverage","consensus_coverage","ambiguity_area_px","ambiguity_fraction_image","ambiguity_to_consensus_ratio"]
    for (task, alpha, method), g in combined.groupby(["task","alpha","method"]):
        rec = {"task":task,"alpha":float(alpha),"method":method,"n_cases":len(g)}
        rec.update({m:float(g[m].mean()) for m in metric_cols})
        summary.append(rec)
        if task == "02":
            gs = g[~g.extreme_task02]
            rec = {"task":task,"alpha":float(alpha),"method":method+"__sensitivity_no_case07_case50","n_cases":len(gs)}
            rec.update({m:float(gs[m].mean()) for m in metric_cols})
            summary.append(rec)
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUT/"summary_metrics.csv", index=False)

    assoc = []
    for task, g in required.groupby("task"):
        for outcome in ["q_required_all","required_all_ambiguity_fraction","required_all_ambiguity_ratio"]:
            x = g.audit_disagreement_1_minus_min_dice.to_numpy(float); y = g[outcome].to_numpy(float)
            pr=pearsonr(x,y); sr=spearmanr(x,y)
            assoc.append({"task":task,"outcome":outcome,"n":len(g),"pearson_r":float(pr.statistic),"pearson_p":float(pr.pvalue),"spearman_rho":float(sr.statistic),"spearman_p":float(sr.pvalue)})
            if task == "02":
                gs=g[~g.extreme_task02]; x=gs.audit_disagreement_1_minus_min_dice.to_numpy(float); y=gs[outcome].to_numpy(float)
                pr=pearsonr(x,y); sr=spearmanr(x,y)
                assoc.append({"task":task,"outcome":outcome+"__sensitivity_no_case07_case50","n":len(gs),"pearson_r":float(pr.statistic),"pearson_p":float(pr.pvalue),"spearman_rho":float(sr.statistic),"spearman_p":float(sr.pvalue)})
    assoc_df=pd.DataFrame(assoc); assoc_df.to_csv(OUT/"disagreement_associations.csv",index=False)

    integrity={"tasks":["01","02"],"folds":5,"alphas":list(ALPHAS),"random_rater_replicates":RANDOM_REPS,
               "expected_oof_test_cases_per_task":55,"case_metric_rows":len(cases),"threshold_rows":len(thresholds),"required_rows":len(required)}
    (OUT/"integrity.json").write_text(json.dumps(integrity,indent=2))
    print("SUMMARY\n"+summary_df.to_string(index=False))
    print("\nASSOCIATIONS\n"+assoc_df.to_string(index=False))
    print(f"\nOUT={OUT}")

if __name__ == "__main__": main()
