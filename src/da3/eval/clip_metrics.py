"""Clip-level depth and trajectory metrics for posed sequences without dense GT (KITTI, enhancement #3).

Depth (vs sparse LiDAR), protocol of configs/lora_analysis.yaml:
* "clip scale" (primary): ONE median scale for all views of a clip, s = median(gt / pred) over the
  valid pixels of every view, so the metric also rewards multi-view scale consistency;
* "image scale" (secondary): one median scale per view (monocular style), metrics averaged over views.
AbsRel = mean |s·pred − gt| / gt and δ1 = fraction of max(s·pred/gt, gt/(s·pred)) < 1.25, pooled over
the valid pixels of the clip (resp. of each view).

Trajectory: predicted camera centres are Sim(3)-aligned to the GT centres (Umeyama, no RANSAC: a clip
is short and has no outliers by construction).
* ATE = RMSE of the aligned centres (metres);
* RPE-trans / RPE-rot = mean error of the relative motion between consecutive frames, translation in
  metres (prediction scaled by the Sim(3) scale), rotation in degrees.
"""

from __future__ import annotations

import numpy as np

from da3.eval.metrics import pose_metrics, umeyama


def _valid(gt, pred, min_depth, max_depth):
    return (gt > min_depth) & (gt <= max_depth) & np.isfinite(pred) & (pred > 0)


def _depth_errors(pred, gt):
    absrel = float(np.mean(np.abs(pred - gt) / gt))
    ratio = np.maximum(pred / gt, gt / pred)
    return absrel, float(np.mean(ratio < 1.25))


def clip_depth_metrics(pred: np.ndarray, gt: np.ndarray, min_depth: float = 1e-3, max_depth: float = 80.0) -> dict:
    """pred, gt: (S, H, W) at the same resolution; gt = 0 where there is no LiDAR point."""
    pred, gt = np.asarray(pred, np.float64), np.asarray(gt, np.float64)
    m = _valid(gt, pred, min_depth, max_depth)
    out = {"n_valid": int(m.sum())}
    if m.sum() == 0:
        return {**out, "absrel_clip": np.nan, "delta1_clip": np.nan, "absrel_image": np.nan,
                "delta1_image": np.nan, "scale_clip": np.nan}
    s = float(np.median(gt[m] / pred[m]))
    out["scale_clip"] = s
    out["absrel_clip"], out["delta1_clip"] = _depth_errors(s * pred[m], gt[m])
    per_view = []
    for p, g, mv in zip(pred, gt, m):
        if mv.sum() > 0:
            sv = np.median(g[mv] / p[mv])
            per_view.append(_depth_errors(sv * p[mv], g[mv]))
    out["absrel_image"] = float(np.mean([a for a, _ in per_view]))
    out["delta1_image"] = float(np.mean([d for _, d in per_view]))
    return out


def _inv(T):
    T = np.asarray(T, np.float64)
    if T.shape[-2:] == (3, 4):
        T = np.concatenate([T, np.broadcast_to([0, 0, 0, 1.0], T.shape[:-2] + (1, 4))], -2)
    return np.linalg.inv(T)


def trajectory_metrics(pred_w2c: np.ndarray, gt_w2c: np.ndarray) -> dict:
    """pred_w2c, gt_w2c: (S, 3|4, 4). Returns ATE (m), RPE-trans (m), RPE-rot (deg), Sim(3) scale, AUCs."""
    Pc, Gc = _inv(pred_w2c), _inv(gt_w2c)                  # c2w
    R, t, s = umeyama(Pc[:, :3, 3], Gc[:, :3, 3])
    aligned = (s * (R @ Pc[:, :3, 3].T)).T + t
    ate = float(np.sqrt(np.mean(np.sum((aligned - Gc[:, :3, 3]) ** 2, axis=1))))
    Pa = Pc.copy()
    Pa[:, :3, :3] = R @ Pc[:, :3, :3]
    Pa[:, :3, 3] = aligned
    rt, rr = [], []
    for i in range(len(Pa) - 1):
        dp = np.linalg.inv(Pa[i]) @ Pa[i + 1]
        dg = np.linalg.inv(Gc[i]) @ Gc[i + 1]
        e = np.linalg.inv(dg) @ dp
        rt.append(np.linalg.norm(e[:3, 3]))
        rr.append(np.degrees(np.arccos(np.clip((np.trace(e[:3, :3]) - 1) / 2, -1, 1))))
    path = float(np.sum(np.linalg.norm(np.diff(Gc[:, :3, 3], axis=0), axis=1)))
    return {"ate": ate, "rpe_trans": float(np.mean(rt)), "rpe_rot": float(np.mean(rr)), "sim3_scale": float(s),
            "path_length": path, **pose_metrics(pred_w2c, gt_w2c)}
