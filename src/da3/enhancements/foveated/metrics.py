"""Metrics for enhancement #2: depth errors against GT, pose AUC on view subsets, region metrics.

Depth predictions are scale-free, so every depth error states its alignment:
* per map ("view" / "region"): pred * median(gt) / median(pred) on the valid pixels (shape / detail);
* per scene: one scale for all views of a scene (cross-view scale consistency).
Pose: the paper's pairwise errors max(RRA, RTA) (da3.eval.metrics), AUC on chosen pairs of views
(focus-focus, context-context, focus-context, all).
"""

from __future__ import annotations

import cv2
import numpy as np

from da3.eval.metrics import pose_auc, reconstruction_metrics, relative_pose_errors

AUC_THRESHOLDS = (3, 30)


def resize_pred(pred: np.ndarray, hw) -> np.ndarray:
    pred = np.asarray(pred, np.float32)
    if tuple(pred.shape) == tuple(hw):
        return pred
    return cv2.resize(pred, (int(hw[1]), int(hw[0])), interpolation=cv2.INTER_LINEAR)


def valid_mask(pred, gt) -> np.ndarray:
    return (gt > 0) & np.isfinite(gt) & np.isfinite(pred) & (pred > 0)


def depth_errors(pred: np.ndarray, gt: np.ndarray, scale: float | None = None, min_valid: int = 100) -> dict | None:
    """AbsRel, delta < 1.25, delta < 1.05 of a depth map against GT (same size). scale: None = median
    alignment of this map; a number = that scale (e.g. the per-scene one)."""
    v = valid_mask(pred, gt)
    if v.sum() < min_valid:
        return None
    p, g = pred[v].astype(np.float64), gt[v].astype(np.float64)
    s = float(np.median(g) / np.median(p)) if scale is None else float(scale)
    p = p * s
    ratio = np.maximum(p / g, g / p)
    return {"abs_rel": float(np.mean(np.abs(p - g) / g)), "delta1": float(np.mean(ratio < 1.25)),
            "delta105": float(np.mean(ratio < 1.05)), "scale": s}


def pooled_scale(preds, gts, per_map: int = 20000, seed: int = 0) -> float:
    """One scale for a set of maps: median(gt / pred) over (sub-sampled) valid pixels of all maps."""
    rng = np.random.default_rng(seed)
    ratios = []
    for p, g in zip(preds, gts):
        if p is None or g is None:
            continue
        v = np.flatnonzero(valid_mask(p, g))
        if len(v) == 0:
            continue
        v = v if len(v) <= per_map else rng.choice(v, per_map, replace=False)
        ratios.append(g.ravel()[v].astype(np.float64) / p.ravel()[v].astype(np.float64))
    return float(np.median(np.concatenate(ratios))) if ratios else float("nan")


def pairwise_pose_errors(pred_w2c, gt_w2c) -> np.ndarray:
    """(S, S) symmetric matrix of max(RRA, RTA) in degrees (NaN on the diagonal)."""
    S = len(pred_w2c)
    E = np.full((S, S), np.nan)
    if S < 2:
        return E
    r, t = relative_pose_errors(np.asarray(pred_w2c), np.asarray(gt_w2c))
    i, j = np.triu_indices(S, k=1)
    E[i, j] = E[j, i] = np.maximum(r, t)
    return E


def subset_auc(E: np.ndarray, mask_a, mask_b=None, thresholds=AUC_THRESHOLDS) -> dict:
    """Pose AUC over the pairs (i < j) with both views in mask_a, or one in mask_a and one in mask_b."""
    S = E.shape[0]
    a = np.asarray(mask_a, bool)
    i, j = np.triu_indices(S, k=1)
    if mask_b is None:
        sel = a[i] & a[j]
    else:
        b = np.asarray(mask_b, bool)
        sel = (a[i] & b[j]) | (a[j] & b[i])
    e = E[i[sel], j[sel]]
    e = e[np.isfinite(e)]
    out = {"pairs": int(len(e))}
    for thr in thresholds:
        out[f"auc{thr:02d}"] = pose_auc(e, e, thr) if len(e) else float("nan")
    return out


def view_subset_aucs(E: np.ndarray, focus) -> dict:
    """AUC for all pairs, focus-focus, context-context and focus-context pairs."""
    f = np.asarray(focus, bool)
    out = {"all": subset_auc(E, np.ones_like(f))}
    if f.any():
        out["focus"] = subset_auc(E, f)
    if (~f).any():
        out["context"] = subset_auc(E, ~f)
        if f.any():
            out["cross"] = subset_auc(E, f, ~f)
    return out


def rot_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    M = np.asarray(Ra, np.float64) @ np.asarray(Rb, np.float64).T
    return float(np.degrees(np.arccos(np.clip((np.trace(M) - 1.0) / 2.0, -1.0, 1.0))))


def region_point_metrics(z_pred: np.ndarray, z_gt: np.ndarray, rays: np.ndarray,
                         rel_thresholds=(0.005, 0.01, 0.02)) -> dict:
    """Point-set metrics of a region: points z * ray (rays with unit z, source camera frame) of the
    prediction (already aligned) vs GT. Thresholds relative to the median GT depth; Chamfer reported
    in % of the median GT depth."""
    P = z_pred[:, None] * rays
    G = z_gt[:, None] * rays
    med = float(np.median(z_gt))
    out = {}
    for tau in rel_thresholds:
        m = reconstruction_metrics(P, G, threshold=tau * med)
        out[f"F1@{100 * tau:g}%"] = m["fscore"]
    out["chamfer_%"] = 100.0 * reconstruction_metrics(P, G, threshold=med)["overall"] / med
    return out
