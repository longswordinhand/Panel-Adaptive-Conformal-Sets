#!/usr/bin/env python3
"""Fold-independent preprocessing for labeled QUBIQ 2021 prostate cases.

Reads original NIfTI files from the manifest and writes one compressed NPZ per
case containing:
- normalized 2D image
- all Task01 expert masks + rater IDs
- all Task02 expert masks + rater IDs
- majority-vote consensus masks
- original/resized geometry metadata

No dataset-level statistics are fitted. Image normalization is per-case only.
"""
from __future__ import annotations

from pathlib import Path
import csv
import json
import re

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/processed/qubiq2021_manifest.csv"
OUTDIR = ROOT / "data/processed/prostate_512"
TARGET = (512, 512)


def load_2d(path: Path) -> np.ndarray:
    arr = np.asarray(nib.load(str(path)).dataobj)
    arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D after squeeze: {path} -> {arr.shape}")
    return arr


def normalize_image(arr: np.ndarray) -> tuple[np.ndarray, float, float]:
    x = np.asarray(arr, dtype=np.float32)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        raise ValueError("Image contains no finite pixels")
    nz = finite[finite != 0]
    ref = nz if nz.size >= 100 else finite
    lo, hi = np.percentile(ref, [1.0, 99.0])
    lo, hi = float(lo), float(hi)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(finite)), float(np.max(finite))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32), lo, hi
    x = np.clip(x, lo, hi)
    x = (x - lo) / (hi - lo)
    x[~np.isfinite(x)] = 0.0
    return x.astype(np.float32), lo, hi


def letterbox(arr: np.ndarray, target: tuple[int, int], order: int) -> tuple[np.ndarray, dict]:
    h, w = arr.shape
    th, tw = target
    scale = min(th / h, tw / w)
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    resized = zoom(arr, (nh / h, nw / w), order=order, mode="nearest", prefilter=(order > 1))
    # scipy rounding can differ by one pixel; crop/pad resized to requested intermediate size.
    resized = resized[:nh, :nw]
    if resized.shape != (nh, nw):
        tmp = np.zeros((nh, nw), dtype=resized.dtype)
        tmp[:resized.shape[0], :resized.shape[1]] = resized
        resized = tmp
    top = (th - nh) // 2
    bottom = th - nh - top
    left = (tw - nw) // 2
    right = tw - nw - left
    out = np.pad(resized, ((top, bottom), (left, right)), mode="constant", constant_values=0)
    if out.shape != target:
        raise AssertionError((arr.shape, resized.shape, out.shape, target))
    meta = {"scale": float(scale), "resized_h": nh, "resized_w": nw,
            "pad_top": top, "pad_bottom": bottom, "pad_left": left, "pad_right": right}
    return out, meta


def majority(masks: np.ndarray) -> np.ndarray:
    return (masks.mean(axis=0) >= 0.5).astype(np.uint8)


def parse_rater(path: str) -> str:
    m = re.search(r"seg(\d+)\.nii\.gz$", path)
    if not m:
        raise ValueError(f"Cannot parse rater ID from {path}")
    return m.group(1)


def main() -> None:
    rows = []
    with open(MANIFEST, newline="") as f:
        for r in csv.DictReader(f):
            if r["dataset_task"] == "prostate" and r["split"] in {"train", "valid"}:
                rows.append(r)
    by_case: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_case.setdefault(r["case_id"], {})[r["segmentation_task"].zfill(2)] = r
    if len(by_case) != 55:
        raise RuntimeError(f"Expected 55 labeled prostate cases, got {len(by_case)}")

    OUTDIR.mkdir(parents=True, exist_ok=True)
    index_rows = []
    for case_id in sorted(by_case):
        tasks = by_case[case_id]
        if set(tasks) != {"01", "02"}:
            raise RuntimeError(f"{case_id}: expected task01/task02, got {sorted(tasks)}")
        image_path = ROOT / tasks["01"]["image_path"]
        image_raw = load_2d(image_path)
        image_norm, clip_lo, clip_hi = normalize_image(image_raw)
        image_512, geom = letterbox(image_norm, TARGET, order=1)

        payload = {"image": image_512.astype(np.float32)}
        counts = {}
        for task in ("01", "02"):
            ann_paths = [p for p in tasks[task]["annotation_paths"].split(";") if p]
            raters = [parse_rater(p) for p in ann_paths]
            masks = []
            for p in ann_paths:
                m = load_2d(ROOT / p)
                if m.shape != image_raw.shape:
                    raise ValueError(f"{case_id} task{task}: mask/image shape mismatch")
                m = (m > 0).astype(np.uint8)
                m512, _ = letterbox(m, TARGET, order=0)
                masks.append((m512 > 0).astype(np.uint8))
            stack = np.stack(masks, axis=0)
            payload[f"task{task}_masks"] = stack
            payload[f"task{task}_consensus"] = majority(stack)
            payload[f"task{task}_rater_ids"] = np.asarray(raters, dtype="U8")
            counts[task] = len(raters)

        payload["case_id"] = np.asarray(case_id)
        payload["official_split"] = np.asarray(tasks["01"]["split"])
        payload["original_shape"] = np.asarray(image_raw.shape, dtype=np.int32)
        payload["clip_lo_hi"] = np.asarray([clip_lo, clip_hi], dtype=np.float32)
        payload["letterbox"] = np.asarray([
            geom["scale"], geom["resized_h"], geom["resized_w"],
            geom["pad_top"], geom["pad_bottom"], geom["pad_left"], geom["pad_right"],
        ], dtype=np.float32)

        out = OUTDIR / f"{case_id}.npz"
        np.savez_compressed(out, **payload)
        index_rows.append({
            "case_id": case_id,
            "official_split": tasks["01"]["split"],
            "npz_path": str(out.relative_to(ROOT)),
            "original_h": image_raw.shape[0],
            "original_w": image_raw.shape[1],
            "task01_n_raters": counts["01"],
            "task02_n_raters": counts["02"],
            "clip_lo": clip_lo,
            "clip_hi": clip_hi,
            **geom,
        })

    with open(OUTDIR / "index.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=index_rows[0].keys())
        writer.writeheader(); writer.writerows(index_rows)
    with open(OUTDIR / "preprocessing.json", "w") as f:
        json.dump({
            "source": "QUBIQ 2021 prostate labeled official train+validation",
            "n_cases": len(index_rows),
            "target_shape": list(TARGET),
            "image_normalization": "per-case nonzero 1st-99th percentile clip then [0,1] scaling",
            "resize": "aspect-ratio preserving scipy linear interpolation + zero letterbox",
            "mask_resize": "nearest-neighbor + threshold > 0",
            "consensus": "mean expert mask >= 0.5",
            "fold_independent": True,
        }, f, indent=2)
    print(f"Wrote {len(index_rows)} cases to {OUTDIR}")


if __name__ == "__main__":
    main()
