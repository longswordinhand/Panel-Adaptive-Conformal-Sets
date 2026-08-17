#!/usr/bin/env python3
"""Comprehensive, read-only audit for QUBIQ 2021 prostate multi-rater data."""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/processed/qubiq2021_manifest.csv"
OUTDIR = ROOT / "data/processed/audit_prostate"
REPORT = ROOT / "docs/QUBIQ_PROSTATE_AUDIT.md"


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    na, nb = int(a.sum()), int(b.sum())
    if na == 0 and nb == 0:
        return 1.0
    if na == 0 or nb == 0:
        return 0.0
    return 2.0 * float(np.logical_and(a, b).sum()) / float(na + nb)


def surface(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return mask
    eroded = ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(mask.ndim, 1), border_value=0)
    return np.logical_and(mask, np.logical_not(eroded))


def hd95_mm(a: np.ndarray, b: np.ndarray, spacing: tuple[float, ...]) -> float:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    if not a.any() and not b.any():
        return 0.0
    if not a.any() or not b.any():
        return math.inf
    sa, sb = surface(a), surface(b)
    # Distance transform is zero on target foreground surface and physical distance elsewhere.
    d_to_b = ndimage.distance_transform_edt(~sb, sampling=spacing)
    d_to_a = ndimage.distance_transform_edt(~sa, sampling=spacing)
    d_ab = d_to_b[sa]
    d_ba = d_to_a[sb]
    vals = np.concatenate([d_ab, d_ba])
    return float(np.percentile(vals, 95))


def robust_stats(vals: list[float]) -> dict[str, float | int | None]:
    x = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    n_inf = int(sum(not np.isfinite(v) for v in vals))
    if x.size == 0:
        return {"n": len(vals), "n_finite": 0, "n_inf": n_inf, "mean": None, "sd": None, "median": None, "q1": None, "q3": None, "min": None, "max": None}
    return {
        "n": len(vals), "n_finite": int(x.size), "n_inf": n_inf,
        "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if x.size > 1 else 0.0,
        "median": float(np.median(x)), "q1": float(np.percentile(x, 25)), "q3": float(np.percentile(x, 75)),
        "min": float(x.min()), "max": float(x.max()),
    }


def parse_paths(s: str) -> list[Path]:
    return [ROOT / p for p in str(s).split(";") if p]


def parse_ids(s: str) -> list[str]:
    return [p for p in str(s).split(";") if p]


def fmt(x, digits=3):
    if x is None:
        return "NA"
    if isinstance(x, float) and not np.isfinite(x):
        return "inf"
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(MANIFEST, dtype={"segmentation_task": str, "case_id": str})
    df = df[(df.dataset_task == "prostate") & (df.split.isin(["train", "valid"]))].copy()
    df["segmentation_task"] = df["segmentation_task"].astype(str).str.zfill(2)
    df = df.sort_values(["split", "case_id", "segmentation_task"]).reset_index(drop=True)

    image_cache: dict[str, tuple[np.ndarray, nib.Nifti1Image]] = {}
    case_rows = []
    mask_rows = []
    pair_rows = []
    issues = []

    for _, row in df.iterrows():
        split = row["split"]
        case = row["case_id"]
        task = row["segmentation_task"]
        img_path = ROOT / row["image_path"]
        key = str(img_path)
        if key not in image_cache:
            ni = nib.load(str(img_path))
            arr = np.asanyarray(ni.dataobj)
            image_cache[key] = (arr, ni)
        img_arr, img_nii = image_cache[key]
        shape = tuple(int(v) for v in img_nii.shape)
        zooms = tuple(float(v) for v in img_nii.header.get_zooms()[:len(shape)])
        affine = np.asarray(img_nii.affine)
        orient = "".join(nib.aff2axcodes(affine))
        voxel_mm3 = float(np.prod(zooms[:3])) if len(zooms) >= 3 else float(np.prod(zooms))

        finite = np.isfinite(img_arr)
        image_nonfinite = int((~finite).sum())
        if image_nonfinite:
            issues.append({"severity":"error","split":split,"case_id":case,"task":task,"item":"image_nonfinite","detail":str(image_nonfinite)})
        finite_vals = img_arr[finite]

        ann_paths = parse_paths(row["annotation_paths"])
        rater_ids = parse_ids(row["rater_ids"])
        if len(ann_paths) != len(rater_ids):
            issues.append({"severity":"error","split":split,"case_id":case,"task":task,"item":"manifest_count_mismatch","detail":f"paths={len(ann_paths)} ids={len(rater_ids)}"})

        masks = []
        volumes_cc = []
        binary_flags = []
        exact_hashes = []
        for rid, mp in zip(rater_ids, ann_paths):
            mn = nib.load(str(mp))
            ma_raw = np.asanyarray(mn.dataobj)
            mshape = tuple(int(v) for v in mn.shape)
            mzooms = tuple(float(v) for v in mn.header.get_zooms()[:len(mshape)])
            maff = np.asarray(mn.affine)
            morient = "".join(nib.aff2axcodes(maff))
            unique = np.unique(ma_raw)
            is_binary = bool(np.all(np.isin(unique, [0, 1])))
            binary_flags.append(is_binary)
            if not is_binary:
                issues.append({"severity":"error","split":split,"case_id":case,"task":task,"item":"nonbinary_mask","detail":f"rater={rid} values={unique[:20].tolist()}"})
            if mshape != shape:
                issues.append({"severity":"error","split":split,"case_id":case,"task":task,"item":"shape_mismatch","detail":f"rater={rid} image={shape} mask={mshape}"})
            if not np.allclose(maff, affine, rtol=0, atol=1e-4):
                issues.append({"severity":"warning","split":split,"case_id":case,"task":task,"item":"affine_mismatch","detail":f"rater={rid}"})
            if not np.allclose(mzooms, zooms, rtol=0, atol=1e-5):
                issues.append({"severity":"warning","split":split,"case_id":case,"task":task,"item":"spacing_mismatch","detail":f"rater={rid} image={zooms} mask={mzooms}"})
            mask = np.asarray(ma_raw > 0.5, dtype=bool)
            masks.append(mask)
            nvox = int(mask.sum())
            vol_cc = nvox * voxel_mm3 / 1000.0
            volumes_cc.append(vol_cc)
            exact_hashes.append(hash(mask.tobytes()))
            mask_rows.append({
                "split":split,"case_id":case,"task":task,"rater_id":rid,
                "shape":"x".join(map(str,mshape)),"spacing_mm":"x".join(f"{z:.6g}" for z in mzooms),"orientation":morient,
                "n_voxels":nvox,"volume_cc":vol_cc,"empty":int(nvox==0),"is_binary":int(is_binary),
                "path":str(mp.relative_to(ROOT)),
            })

        ds, hs = [], []
        for ia, ib in combinations(range(len(masks)), 2):
            d = dice(masks[ia], masks[ib])
            h = hd95_mm(masks[ia], masks[ib], zooms)
            ds.append(d); hs.append(h)
            pair_rows.append({
                "split":split,"case_id":case,"task":task,
                "rater_a":rater_ids[ia],"rater_b":rater_ids[ib],"dice":d,"hd95_mm":h,
                "volume_a_cc":volumes_cc[ia],"volume_b_cc":volumes_cc[ib],
                "abs_volume_diff_cc":abs(volumes_cc[ia]-volumes_cc[ib]),
            })

        vols = np.asarray(volumes_cc, dtype=float)
        vol_mean = float(vols.mean()) if len(vols) else math.nan
        vol_cv = float(vols.std(ddof=1) / vol_mean) if len(vols)>1 and vol_mean>0 else 0.0
        consensus = np.sum(np.stack(masks, axis=0), axis=0) >= math.ceil(len(masks)/2) if masks else np.zeros(shape,bool)
        duplicate_pairs = 0
        for a,b in combinations(range(len(exact_hashes)),2):
            if exact_hashes[a] == exact_hashes[b] and np.array_equal(masks[a], masks[b]):
                duplicate_pairs += 1

        case_rows.append({
            "split":split,"case_id":case,"task":task,
            "image_shape":"x".join(map(str,shape)),"spacing_mm":"x".join(f"{z:.6g}" for z in zooms),"orientation":orient,
            "image_min":float(finite_vals.min()) if finite_vals.size else math.nan,
            "image_max":float(finite_vals.max()) if finite_vals.size else math.nan,
            "image_mean":float(finite_vals.mean()) if finite_vals.size else math.nan,
            "image_nonfinite":image_nonfinite,"n_raters":len(masks),"n_empty_masks":int(sum(not m.any() for m in masks)),
            "all_binary":int(all(binary_flags)),"volume_mean_cc":vol_mean,"volume_sd_cc":float(vols.std(ddof=1)) if len(vols)>1 else 0.0,
            "volume_cv":vol_cv,"volume_min_cc":float(vols.min()) if len(vols) else math.nan,"volume_max_cc":float(vols.max()) if len(vols) else math.nan,
            "pairwise_dice_mean":float(np.mean(ds)) if ds else math.nan,"pairwise_dice_min":float(np.min(ds)) if ds else math.nan,
            "pairwise_hd95_mean_mm":float(np.mean(hs)) if hs else math.nan,"pairwise_hd95_max_mm":float(np.max(hs)) if hs else math.nan,
            "consensus_volume_cc":float(consensus.sum())*voxel_mm3/1000.0,
            "exact_duplicate_rater_pairs":duplicate_pairs,
        })

    case_df = pd.DataFrame(case_rows)
    mask_df = pd.DataFrame(mask_rows)
    pair_df = pd.DataFrame(pair_rows)
    issues_df = pd.DataFrame(issues, columns=["severity","split","case_id","task","item","detail"])
    case_df.to_csv(OUTDIR/"case_task_audit.csv", index=False)
    mask_df.to_csv(OUTDIR/"mask_audit.csv", index=False)
    pair_df.to_csv(OUTDIR/"pairwise_rater_metrics.csv", index=False)
    issues_df.to_csv(OUTDIR/"issues.csv", index=False)

    summary = {
        "scope": "QUBIQ2021 prostate, labeled train+validation only",
        "n_unique_cases": int(case_df.case_id.nunique()),
        "n_case_task_rows": int(len(case_df)),
        "n_masks": int(len(mask_df)),
        "n_pairwise_comparisons": int(len(pair_df)),
        "issues": dict(Counter(issues_df.item)) if len(issues_df) else {},
        "tasks": {},
    }
    for task, g in case_df.groupby("task"):
        pg = pair_df[pair_df.task == task]
        mg = mask_df[mask_df.task == task]
        summary["tasks"][task] = {
            "n_cases": int(g.case_id.nunique()),
            "rater_count_distribution": {str(k): int(v) for k,v in sorted(Counter(g.n_raters).items())},
            "shape_distribution": {str(k): int(v) for k,v in sorted(Counter(g.image_shape).items())},
            "spacing_distribution": {str(k): int(v) for k,v in sorted(Counter(g.spacing_mm).items())},
            "orientation_distribution": {str(k): int(v) for k,v in sorted(Counter(g.orientation).items())},
            "empty_masks": int(mg["empty"].sum()),
            "nonbinary_masks": int((mg.is_binary == 0).sum()),
            "duplicate_rater_pairs": int(g.exact_duplicate_rater_pairs.sum()),
            "mask_volume_cc": robust_stats(mg.volume_cc.tolist()),
            "pairwise_dice": robust_stats(pg.dice.tolist()),
            "pairwise_hd95_mm": robust_stats(pg.hd95_mm.tolist()),
            "case_mean_dice": robust_stats(g.pairwise_dice_mean.tolist()),
            "case_min_dice": robust_stats(g.pairwise_dice_min.tolist()),
            "case_mean_hd95_mm": robust_stats(g.pairwise_hd95_mean_mm.tolist()),
            "case_max_hd95_mm": robust_stats(g.pairwise_hd95_max_mm.tolist()),
            "case_volume_cv": robust_stats(g.volume_cv.tolist()),
        }
    with open(OUTDIR/"summary.json","w") as f:
        json.dump(summary, f, indent=2)

    # Markdown report with top-disagreement cases.
    lines = [
        "# QUBIQ 2021 Prostate Data Audit", "",
        "Read-only audit of labeled QUBIQ 2021 prostate data (official train + validation archives).",
        "All geometry-aware distances use NIfTI header spacing. In the audited prostate files spacing is uniformly 1x1x1, so HD95 values numerically equal pixel/voxel distances and should not be interpreted as verified acquisition-space millimetres without external metadata.", "",
        "## Scope", "",
        f"- Unique labeled cases: **{summary['n_unique_cases']}**",
        f"- Case-task records: **{summary['n_case_task_rows']}**",
        f"- Expert masks audited: **{summary['n_masks']}**",
        f"- Pairwise expert comparisons: **{summary['n_pairwise_comparisons']}**",
        "- Official test images are excluded because the uploaded test archive contains no ground-truth masks.", "",
        "## Integrity findings", "",
        f"- Empty masks: **{int(mask_df['empty'].sum())}**",
        f"- Non-binary masks: **{int((mask_df.is_binary==0).sum())}**",
        f"- Image/mask geometry issues: **{len(issues_df)}** recorded rows",
        f"- Exact duplicate expert-mask pairs: **{int(case_df.exact_duplicate_rater_pairs.sum())}**", "",
    ]
    for task in sorted(summary["tasks"]):
        s=summary["tasks"][task]
        lines += [f"## Task {task}", "",
                  f"- Cases: **{s['n_cases']}**; rater counts: `{s['rater_count_distribution']}`",
                  f"- Image shapes: `{s['shape_distribution']}`",
                  f"- Orientations: `{s['orientation_distribution']}`",
                  f"- Mask volume (cc), median [Q1, Q3]: **{fmt(s['mask_volume_cc']['median'])} [{fmt(s['mask_volume_cc']['q1'])}, {fmt(s['mask_volume_cc']['q3'])}]**",
                  f"- Pairwise Dice, median [Q1, Q3]: **{fmt(s['pairwise_dice']['median'])} [{fmt(s['pairwise_dice']['q1'])}, {fmt(s['pairwise_dice']['q3'])}]**",
                  f"- Pairwise HD95 (mm), median [Q1, Q3]: **{fmt(s['pairwise_hd95_mm']['median'])} [{fmt(s['pairwise_hd95_mm']['q1'])}, {fmt(s['pairwise_hd95_mm']['q3'])}]**",
                  f"- Case-level minimum Dice, median: **{fmt(s['case_min_dice']['median'])}**",
                  f"- Case-level maximum HD95 (mm), median: **{fmt(s['case_max_hd95_mm']['median'])}**",
                  "",
                  "### Highest-disagreement cases", "",
                  "| Case | Split | n raters | Mean Dice | Min Dice | Mean HD95 mm | Max HD95 mm | Volume CV |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        gg=case_df[case_df.task==task].sort_values(["pairwise_dice_min","pairwise_hd95_max_mm"], ascending=[True,False]).head(10)
        for _,r in gg.iterrows():
            lines.append(f"| {r.case_id} | {r.split} | {int(r.n_raters)} | {r.pairwise_dice_mean:.3f} | {r.pairwise_dice_min:.3f} | {r.pairwise_hd95_mean_mm:.2f} | {r.pairwise_hd95_max_mm:.2f} | {r.volume_cv:.3f} |")
        lines.append("")
    lines += ["## Files", "",
              "- `data/processed/audit_prostate/case_task_audit.csv`",
              "- `data/processed/audit_prostate/mask_audit.csv`",
              "- `data/processed/audit_prostate/pairwise_rater_metrics.csv`",
              "- `data/processed/audit_prostate/issues.csv`",
              "- `data/processed/audit_prostate/summary.json`", ""]
    REPORT.write_text("\n".join(lines))

    print(json.dumps(summary, indent=2))
    print(f"REPORT={REPORT.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
