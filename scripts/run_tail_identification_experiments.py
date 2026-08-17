#!/usr/bin/env python3
"""Descriptive finite-rater identification experiments for NIH and VinDr.

This script does not assume the latent mixing distribution G is observed. For each
binary finding and reader count m, it estimates the empirical distribution of K,
projects that distribution onto the grid-discretized binomial-mixture model, and
computes sharp grid-LP bounds on tau_beta(G)=P_G(theta>beta) conditional on the
projected observable distribution.

The projection step is explicit because a finite-sample empirical K distribution
need not lie exactly in the convex hull of Binomial(m, theta) distributions.
These are descriptive identified-set experiments, not finite-sample confidence
intervals.
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.special import comb

ROOT = Path(__file__).resolve().parents[1]
NIH = ROOT / "data/public/nih_expert_labels/raw/all_findings_expert_labels/all_findings_expert_labels_test_individual_readers.csv"
VINDR = ROOT / "data/public/vindr_cxr/raw/train.csv"
OUTDIR = ROOT / "experiments/pilot/results/tail_identification"


def bernstein_matrix(m: int, grid: np.ndarray) -> np.ndarray:
    rows = []
    for k in range(m + 1):
        rows.append(comb(m, k) * np.power(grid, k) * np.power(1.0 - grid, m - k))
    return np.asarray(rows, dtype=float)


def l1_project_to_mixture(p: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, float]:
    """L1-project p onto {B w: w simplex}."""
    r = B.shape[0]
    q = B.shape[1]
    # x=[w(q), u(r), v(r)], B w - u + v = p, sum w=1
    c = np.r_[np.zeros(q), np.ones(r), np.ones(r)]
    Aeq = np.zeros((r + 1, q + 2 * r))
    Aeq[:r, :q] = B
    Aeq[:r, q:q+r] = -np.eye(r)
    Aeq[:r, q+r:] = np.eye(r)
    Aeq[r, :q] = 1.0
    beq = np.r_[p, 1.0]
    bounds = [(0.0, None)] * (q + 2 * r)
    res = linprog(c, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"Projection LP failed: {res.message}")
    w = res.x[:q]
    pproj = B @ w
    return pproj, float(np.abs(pproj - p).sum())


def tail_bounds(p: np.ndarray, m: int, beta: float, grid: np.ndarray) -> tuple[float, float, float]:
    B = bernstein_matrix(m, grid)
    pproj, l1err = l1_project_to_mixture(p, B)
    Aeq = np.vstack([B, np.ones((1, len(grid)))])
    beq = np.r_[pproj, 1.0]
    tail = (grid > beta).astype(float)
    bounds = [(0.0, None)] * len(grid)
    lo = linprog(tail, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
    hi = linprog(-tail, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
    if not lo.success or not hi.success:
        raise RuntimeError(f"Bounds LP failed: lo={lo.message}; hi={hi.message}")
    return float(lo.fun), float(-hi.fun), l1err


def nih_vote_matrices(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    id_cols = {"Image ID", "Patient ID", "Reader ID"}
    labels = [c for c in df.columns if c not in id_cols]
    out = {}
    for label in labels:
        d = df[["Image ID", "Reader ID", label]].copy()
        if label == "Abnormal":
            d["vote"] = d[label].astype(int)
        else:
            d["vote"] = d[label].map({"NO": 0, "YES": 1})
        mat = d.pivot(index="Image ID", columns="Reader ID", values="vote")
        if mat.isna().any().any() or mat.shape[1] != 5:
            raise ValueError(f"NIH {label}: expected complete 5-reader panel, got {mat.shape}")
        out[label] = mat.sort_index(axis=1)
    return out


def exact_subset_k_distribution(mat: pd.DataFrame, m: int) -> np.ndarray:
    vals = mat.to_numpy(dtype=int)
    n, M = vals.shape
    combos = list(itertools.combinations(range(M), m))
    counts = np.zeros(m + 1, dtype=float)
    for combo in combos:
        k = vals[:, combo].sum(axis=1)
        counts += np.bincount(k, minlength=m + 1)
    counts /= len(combos)
    return counts / counts.sum()


def vindr_vote_distribution(df: pd.DataFrame, label: str) -> np.ndarray:
    # Deduplicate image-reader-class because multiple boxes can occur for one class.
    d = df[["image_id", "rad_id", "class_name"]].copy()
    pairs = d[["image_id", "rad_id"]].drop_duplicates()
    pos = d.loc[d["class_name"] == label, ["image_id", "rad_id"]].drop_duplicates()
    pos = pos.assign(vote=1)
    merged = pairs.merge(pos, on=["image_id", "rad_id"], how="left")
    merged["vote"] = merged["vote"].fillna(0).astype(int)
    mat = merged.pivot(index="image_id", columns="rad_id", values="vote")
    # Different images have different reader IDs; sum votes rowwise, requiring exactly 3 readers.
    reader_n = mat.notna().sum(axis=1)
    if not (reader_n == 3).all():
        raise ValueError("VinDr expected exactly 3 readers per image")
    k = mat.fillna(0).sum(axis=1).astype(int).to_numpy()
    cnt = np.bincount(k, minlength=4).astype(float)
    return cnt / cnt.sum()


def run(grid_n: int = 2001) -> pd.DataFrame:
    grid = np.linspace(0.0, 1.0, grid_n)
    rows = []

    nih = pd.read_csv(NIH)
    mats = nih_vote_matrices(nih)
    nih_labels = [
        "Abnormal", "Atelectasis", "Consolidation", "Pleural Thickening",
        "Effusion", "Nodule", "Pneumothorax", "Cardiomegaly"
    ]
    for label in nih_labels:
        mat = mats[label]
        for m in range(1, 6):
            p = exact_subset_k_distribution(mat, m)
            for beta in (0.2, 0.5):
                lo, hi, err = tail_bounds(p, m, beta, grid)
                rows.append({
                    "dataset": "NIH-all-findings", "label": label, "n_images": len(mat),
                    "m": m, "beta": beta, "lower": lo, "upper": hi,
                    "width": hi - lo, "projection_l1": err,
                    "p_k": ";".join(f"{x:.10f}" for x in p),
                })

    vindr = pd.read_csv(VINDR)
    vindr_labels = ["Pleural thickening", "Lung Opacity", "Nodule/Mass", "Cardiomegaly", "Aortic enlargement"]
    for label in vindr_labels:
        p = vindr_vote_distribution(vindr, label)
        for beta in (0.2, 0.5):
            lo, hi, err = tail_bounds(p, 3, beta, grid)
            rows.append({
                "dataset": "VinDr-CXR", "label": label, "n_images": 15000,
                "m": 3, "beta": beta, "lower": lo, "upper": hi,
                "width": hi - lo, "projection_l1": err,
                "p_k": ";".join(f"{x:.10f}" for x in p),
            })

    out = pd.DataFrame(rows)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTDIR / "identified_tail_bounds.csv", index=False)

    # NIH contraction summary: m=1 vs m=5 and monotonicity on projected descriptive bounds.
    nih05 = out[(out.dataset == "NIH-all-findings") & (out.beta == 0.5)].copy()
    summary = []
    for label, g in nih05.groupby("label"):
        g = g.sort_values("m")
        widths = g["width"].to_numpy()
        summary.append({
            "label": label,
            "width_m1": widths[0], "width_m3": widths[2], "width_m5": widths[4],
            "m1_to_m5_reduction": widths[0] - widths[4],
            "m1_to_m5_ratio": widths[4] / widths[0] if widths[0] > 0 else np.nan,
            "monotone_nonincreasing": bool(np.all(np.diff(widths) <= 1e-8)),
            "max_projection_l1": g["projection_l1"].max(),
        })
    pd.DataFrame(summary).to_csv(OUTDIR / "nih_m_contraction_summary_beta05.csv", index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=2001)
    args = ap.parse_args()
    out = run(args.grid)
    print(out.to_string(index=False, max_rows=40))
    print(f"\nWrote {OUTDIR / 'identified_tail_bounds.csv'}")
    print(f"Wrote {OUTDIR / 'nih_m_contraction_summary_beta05.csv'}")


if __name__ == "__main__":
    main()
