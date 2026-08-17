#!/usr/bin/env python3
"""Paired split-level statistics for the Pattern Recognition PACS manuscript.

The inferential/resampling unit is the repeated split (rep). For dermatology,
metrics are first averaged over the four released base-model prediction matrices
within each split to avoid treating model-by-split rows sharing the same cases as
independent replicates. For NIH and CIFAR-10H, the split itself is the unit.

Outputs are descriptive paired effect estimates with nonparametric bootstrap
intervals across split units. Classical paired t and Wilcoxon p-values are also
reported as sensitivity summaries, but should not be interpreted as population
inference from independent newly sampled datasets because repeated splits reuse
observations.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "pr_rescue" / "results"
OUT = ROOT / "experiments" / "pr_rescue" / "statistics"
OUT.mkdir(parents=True, exist_ok=True)
RNG_SEED = 20260817
N_BOOT = 20000


@dataclass(frozen=True)
class Spec:
    dataset: str
    q: float
    path: str
    global_method: str
    pacs_method: str
    aggregate_models_within_rep: bool = False


SPECS = [
    Spec("Dermatology", 0.9, "dermatology_pacs_mondrian_fixed_q0p9_raw.csv", "global", "pacs_mondrian", True),
    Spec("Dermatology", 0.8, "dermatology_pacs_mondrian_fixed_q0p8_raw.csv", "global", "pacs_mondrian", True),
    Spec("Dermatology", 0.7, "dermatology_pacs_mondrian_fixed_q0p7_raw.csv", "global", "pacs_mondrian", True),
    Spec("NIH", 0.7, "nih_multiclass_pacs_mondrian_ridge_fixed_q0p7_raw.csv", "global_panel", "pacs_mondrian_q0.90"),
    Spec("NIH", 0.8, "nih_multiclass_pacs_mondrian_ridge_fixed_q0p8_raw.csv", "global_panel", "pacs_mondrian_q0.90"),
    Spec("NIH", 0.9, "nih_multiclass_pacs_mondrian_ridge_fixed_q0p9_raw.csv", "global_panel", "pacs_mondrian_q0.90"),
    Spec("CIFAR-10H", 0.7, "cifar10h_pacs_fixed_q0p7_raw.csv", "global_panel", "pacs_mondrian"),
]

METRIC_MAP = {
    "Dermatology": {"success": "case_mass_success", "mean_size": "mean_size", "p90_size": "p90_size"},
    "NIH": {"success": "success", "mean_size": "mean_size", "p90_size": "p90_size"},
    "CIFAR-10H": {"success": "success", "mean_size": "mean_size", "p90_size": "p90_size"},
}


def bootstrap_mean_ci(d: np.ndarray, seed: int) -> tuple[float, float]:
    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(seed)
    n = d.size
    idx = rng.integers(0, n, size=(N_BOOT, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def safe_wilcoxon(d: np.ndarray) -> float:
    d = np.asarray(d, dtype=float)
    if np.allclose(d, 0):
        return 1.0
    try:
        return float(stats.wilcoxon(d, alternative="two-sided", method="auto").pvalue)
    except ValueError:
        return float("nan")


def paired_units(spec: Spec) -> pd.DataFrame:
    df = pd.read_csv(RESULTS / spec.path)
    mmap = METRIC_MAP[spec.dataset]
    keep = ["rep", "method"] + list(mmap.values())
    if spec.aggregate_models_within_rep:
        keep.append("model")
    df = df[keep].copy()
    df = df[df["method"].isin([spec.global_method, spec.pacs_method])]

    if spec.aggregate_models_within_rep:
        # Average each method over the four released model matrices inside the
        # same split. The resulting rep-level pair is the resampling unit.
        unit = (
            df.groupby(["rep", "method"], as_index=False)[list(mmap.values())]
            .mean()
        )
    else:
        unit = df

    rows = []
    for rep, g in unit.groupby("rep"):
        if set(g.method) != {spec.global_method, spec.pacs_method}:
            raise RuntimeError(f"Unmatched methods for {spec.dataset} q={spec.q}, rep={rep}")
        gr = g.set_index("method")
        row = {"rep": int(rep)}
        for nice, col in mmap.items():
            row[f"global_{nice}"] = float(gr.loc[spec.global_method, col])
            row[f"pacs_{nice}"] = float(gr.loc[spec.pacs_method, col])
            row[f"delta_{nice}"] = row[f"pacs_{nice}"] - row[f"global_{nice}"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("rep").reset_index(drop=True)


def summarize(spec: Spec, units: pd.DataFrame) -> list[dict]:
    out = []
    seed_base = abs(hash((spec.dataset, spec.q))) % 1_000_000
    for j, metric in enumerate(["success", "mean_size", "p90_size"]):
        g = units[f"global_{metric}"].to_numpy(float)
        p = units[f"pacs_{metric}"].to_numpy(float)
        d = p - g
        lo, hi = bootstrap_mean_ci(d, RNG_SEED + seed_base + j)
        t = stats.ttest_rel(p, g)
        denom = np.std(d, ddof=1)
        dz = float(np.mean(d) / denom) if denom > 0 else float("nan")
        rel = float(np.mean(d) / np.mean(g) * 100.0) if np.mean(g) != 0 else float("nan")
        out.append({
            "dataset": spec.dataset,
            "q": spec.q,
            "metric": metric,
            "n_split_units": int(len(units)),
            "global_mean": float(np.mean(g)),
            "pacs_mean": float(np.mean(p)),
            "paired_delta_mean": float(np.mean(d)),
            "paired_delta_median": float(np.median(d)),
            "paired_delta_sd": float(np.std(d, ddof=1)),
            "bootstrap95_lo": lo,
            "bootstrap95_hi": hi,
            "relative_delta_pct": rel,
            "paired_dz": dz,
            "paired_t_p": float(t.pvalue),
            "wilcoxon_p": safe_wilcoxon(d),
            "n_pacs_gt_global": int(np.sum(d > 0)),
            "n_pacs_lt_global": int(np.sum(d < 0)),
            "n_ties": int(np.sum(np.isclose(d, 0))),
        })
    return out


def fmt_ci(row: pd.Series, scale: float = 1.0, digits: int = 4) -> str:
    m = row.paired_delta_mean * scale
    lo = row.bootstrap95_lo * scale
    hi = row.bootstrap95_hi * scale
    return f"{m:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def main() -> None:
    all_rows = []
    unit_frames = []
    for spec in SPECS:
        units = paired_units(spec)
        units.insert(0, "dataset", spec.dataset)
        units.insert(1, "q", spec.q)
        unit_frames.append(units)
        all_rows.extend(summarize(spec, units))

    summary = pd.DataFrame(all_rows)
    units_all = pd.concat(unit_frames, ignore_index=True)
    summary.to_csv(OUT / "paired_effects.csv", index=False)
    units_all.to_csv(OUT / "paired_split_units.csv", index=False)

    # Machine-readable headline facts.
    facts = []
    for _, r in summary.iterrows():
        facts.append({
            "dataset": r.dataset,
            "q": float(r.q),
            "metric": r.metric,
            "n_split_units": int(r.n_split_units),
            "global_mean": float(r.global_mean),
            "pacs_mean": float(r.pacs_mean),
            "delta_pacs_minus_global": float(r.paired_delta_mean),
            "bootstrap95": [float(r.bootstrap95_lo), float(r.bootstrap95_hi)],
            "relative_delta_pct": float(r.relative_delta_pct),
            "wilcoxon_p_sensitivity": float(r.wilcoxon_p),
        })
    (OUT / "paired_effects.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")

    lines = [
        "# Paired split-level statistical analysis for PACS vs global panel calibration",
        "",
        "## Analysis unit",
        "",
        "- Differences are always PACS minus Global.",
        "- Dermatology: the four released base-model matrices are averaged within each repeated split before inference; n=12 split units at q=0.9 and n=6 at q=0.7/0.8.",
        "- NIH: n=30 patient-level repeated split units at each q.",
        "- CIFAR-10H: n=20 repeated split units at q=0.7.",
        "- 95% intervals are percentile bootstrap intervals for the mean paired split-level difference (20,000 resamples, fixed seed).",
        "- Paired t and Wilcoxon p-values are saved as sensitivity summaries only. Because repeated splits reuse observations, the manuscript should emphasize paired effect sizes and resampling intervals rather than treat these p-values as independent-sample population inference.",
        "",
        "## Headline paired effects",
        "",
    ]

    for dataset, q in [("Dermatology",0.9),("NIH",0.7),("NIH",0.8),("NIH",0.9),("CIFAR-10H",0.7),("Dermatology",0.8),("Dermatology",0.7)]:
        sub = summary[(summary.dataset==dataset)&(summary.q==q)].set_index("metric")
        lines.append(f"### {dataset}, q={q:.1f} (n={int(sub.iloc[0].n_split_units)} split units)")
        s=sub.loc["success"]
        ms=sub.loc["mean_size"]
        p90=sub.loc["p90_size"]
        lines.append(f"- Success delta: {fmt_ci(s, scale=100, digits=2)} percentage points; global={s.global_mean:.4f}, PACS={s.pacs_mean:.4f}.")
        lines.append(f"- Mean-size delta: {fmt_ci(ms, digits=3)} classes ({ms.relative_delta_pct:+.2f}% relative).")
        lines.append(f"- P90-size delta: {fmt_ci(p90, digits=3)} classes ({p90.relative_delta_pct:+.2f}% relative).")
        lines.append("")

    (OUT / "PAIRED_STATISTICS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print(f"\nWrote {OUT/'paired_effects.csv'}")
    print(f"Wrote {OUT/'paired_split_units.csv'}")
    print(f"Wrote {OUT/'PAIRED_STATISTICS.md'}")


if __name__ == "__main__":
    main()
