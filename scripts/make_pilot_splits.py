#!/usr/bin/env python3
"""Create deterministic patient-level pilot splits for QUBIQ prostate.

Design:
- 55 labeled cases (official train + validation combined).
- 5 outer folds; each case is test exactly once (11 test cases/fold).
- For each fold, 11 calibration cases are selected from the remaining 44,
  leaving 33 training cases.
- Assignment is balanced over Task02 inter-rater disagreement rank.
- Task01 and Task02 share exactly the same case assignments.

No annotation-level splitting is permitted.
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "data/processed/audit_prostate/case_task_audit.csv"
OUTDIR = ROOT / "experiments/pilot/splits"
SEED = 20260815
N_FOLDS = 5


def balanced_round_robin(cases: list[str], score: dict[str, float], rng: np.random.Generator) -> dict[str, int]:
    """Assign ranked cases to folds in randomized serpentine blocks."""
    ordered = sorted(cases, key=lambda c: (score[c], c))
    fold_of: dict[str, int] = {}
    for b, start in enumerate(range(0, len(ordered), N_FOLDS)):
        block = ordered[start:start + N_FOLDS]
        perm = list(rng.permutation(N_FOLDS))
        if b % 2:
            perm = perm[::-1]
        for case, fold in zip(block, perm):
            fold_of[case] = int(fold)
    return fold_of


def select_calibration(remaining: list[str], score: dict[str, float], n_cal: int, rng: np.random.Generator) -> set[str]:
    """Select calibration cases spread across the full disagreement range."""
    ordered = sorted(remaining, key=lambda c: (score[c], c))
    # Split ordered cases into n_cal nearly equal rank bins; sample one per bin.
    bins = np.array_split(np.array(ordered, dtype=object), n_cal)
    chosen = set()
    for b in bins:
        idx = int(rng.integers(0, len(b)))
        chosen.add(str(b[idx]))
    assert len(chosen) == n_cal
    return chosen


def main() -> None:
    df = pd.read_csv(AUDIT)
    d2 = df[df["task"].astype(int) == 2].copy()
    if len(d2) != 55:
        raise RuntimeError(f"Expected 55 Task02 cases, found {len(d2)}")

    # Higher score = more disagreement. Use 1 - minimum pairwise Dice;
    # this remains finite even for empty/non-empty expert conflicts.
    score = {r.case_id: float(1.0 - r.pairwise_dice_min) for r in d2.itertuples()}
    official_split = {r.case_id: r.split for r in d2.itertuples()}
    cases = sorted(score)

    rng_outer = np.random.default_rng(SEED)
    outer = balanced_round_robin(cases, score, rng_outer)

    rows = []
    fold_summary = {}
    for fold in range(N_FOLDS):
        test = {c for c in cases if outer[c] == fold}
        if len(test) != 11:
            raise RuntimeError(f"Fold {fold}: expected 11 test, got {len(test)}")
        remaining = [c for c in cases if c not in test]
        rng_cal = np.random.default_rng(SEED + 1000 + fold)
        cal = select_calibration(remaining, score, n_cal=11, rng=rng_cal)
        train = set(remaining) - cal
        assert len(train) == 33 and len(cal) == 11 and len(test) == 11
        assert not (train & cal or train & test or cal & test)
        assert train | cal | test == set(cases)

        for c in cases:
            role = "test" if c in test else "calibration" if c in cal else "train"
            rows.append({
                "fold": fold,
                "case_id": c,
                "role": role,
                "official_split": official_split[c],
                "task02_disagreement": score[c],
            })

        def stat(group: set[str]) -> dict:
            vals = np.array([score[c] for c in group], dtype=float)
            return {
                "n": int(len(vals)),
                "mean_disagreement": float(vals.mean()),
                "median_disagreement": float(np.median(vals)),
                "max_disagreement": float(vals.max()),
                "contains_case07": "case07" in group,
                "contains_case50": "case50" in group,
            }
        fold_summary[str(fold)] = {
            "train": stat(train),
            "calibration": stat(cal),
            "test": stat(test),
        }

    out = pd.DataFrame(rows).sort_values(["fold", "role", "case_id"])
    # Safety: each case must be test exactly once across five outer folds.
    test_counts = out[out.role == "test"].groupby("case_id").size()
    assert len(test_counts) == 55 and (test_counts == 1).all()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTDIR / "prostate_5fold_patient_splits.csv", index=False)
    with open(OUTDIR / "prostate_5fold_patient_splits.json", "w") as f:
        json.dump({
            "seed": SEED,
            "n_folds": N_FOLDS,
            "unit": "patient/case",
            "n_cases": 55,
            "roles_per_fold": {"train": 33, "calibration": 11, "test": 11},
            "stratification_variable": "Task02 disagreement = 1 - minimum pairwise expert Dice",
            "folds": fold_summary,
        }, f, indent=2)

    print(out.groupby(["fold", "role"]).size().unstack(fill_value=0))
    print("\nTest fold disagreement summary:")
    print(out[out.role == "test"].groupby("fold")["task02_disagreement"].agg(["mean", "median", "min", "max"]).round(4))
    print(f"\nWrote: {OUTDIR / 'prostate_5fold_patient_splits.csv'}")


if __name__ == "__main__":
    main()
