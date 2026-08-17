from __future__ import annotations

import math
import numpy as np
from scipy.ndimage import distance_transform_edt

R_UNIVERSAL = 724.0


def hard_prediction(prob: np.ndarray) -> np.ndarray:
    p = np.asarray(prob, dtype=float)
    if not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("probability map must be finite in [0,1]")
    pred = p >= 0.5
    if not np.any(pred):
        raise ValueError("morphological V2 requires non-empty hard prediction")
    return pred


def precompute_geometry(prob: np.ndarray) -> dict:
    pred = hard_prediction(prob)
    inside_depth = distance_transform_edt(pred).astype(np.float32)
    outside_dist = distance_transform_edt(~pred).astype(np.float32)
    return {"pred": pred, "inside_depth": inside_depth, "outside_dist": outside_dist}


def inclusion_radius(geom: dict, mask: np.ndarray) -> float:
    pred = geom["pred"]
    y = np.asarray(mask).astype(bool)
    if y.shape != pred.shape:
        raise ValueError("shape mismatch")
    r = 0.0
    fp_core = pred & ~y
    if np.any(fp_core):
        r = max(r, float(np.max(geom["inside_depth"][fp_core])))
    outside_target = y & ~pred
    if np.any(outside_target):
        r = max(r, float(np.max(geom["outside_dist"][outside_target])))
    return r


def band_metrics(geom: dict, radius: float, consensus_area: int) -> dict[str, float]:
    r = float(radius)
    if r >= R_UNIVERSAL:
        area = geom["pred"].size
    else:
        lower = geom["pred"] & (geom["inside_depth"] > r)
        upper = geom["outside_dist"] <= r
        area = int(np.count_nonzero(upper & ~lower))
    n = int(geom["pred"].size)
    return {
        "ambiguity_area_px": float(area),
        "ambiguity_fraction_image": float(area / n),
        "ambiguity_to_consensus_ratio": float(area / max(int(consensus_area),1)),
        "normalized_radius": float(min(r, R_UNIVERSAL) / 512.0),
    }


def split_quantile(scores, alpha: float) -> float:
    s=np.asarray(list(scores),dtype=float)
    if len(s)==0 or not np.all(np.isfinite(s)) or np.any(s<0):
        raise ValueError("invalid scores")
    k=int(math.ceil((len(s)+1)*(1-float(alpha))))
    if k>len(s):
        return R_UNIVERSAL
    return float(np.partition(s,k-1)[k-1])
