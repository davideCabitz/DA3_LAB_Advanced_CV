"""Benchmark metrics (Sec. 6.2), written to match the released benchmark's definitions.

* Pose: for all view pairs (i < j), relative rotation error (RRA) and relative translation-direction
  error (RTA, sign-ambiguous) in degrees. AUC@T = mean over 1-degree bins up to T of the cumulative
  fraction of pairs with max(RRA, RTA) below the bin edge (that is, the accuracy-threshold curve is
  built from the worse of the two errors).
* Alignment: Umeyama Sim(3) on camera centres, RANSAC variant (half-size subsets, inlier threshold
  = median nearest-neighbour distance after an initial fit, 10 iterations, refit on inliers).
* Reconstruction: accuracy = mean NN distance pred -> GT, completeness = GT -> pred,
  Chamfer = their mean, precision / recall at threshold d, F1 = harmonic mean.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


# ----------------------------------------------------------------------------- pose
def _to44(T: np.ndarray) -> np.ndarray:
    if T.shape[-2] == 4:
        return T.astype(np.float64)
    out = np.tile(np.eye(4), (len(T), 1, 1))
    out[:, :3, :4] = T
    return out


def _inv(T: np.ndarray) -> np.ndarray:
    R, t = T[:, :3, :3], T[:, :3, 3:]
    out = np.tile(np.eye(4), (len(T), 1, 1))
    out[:, :3, :3] = R.transpose(0, 2, 1)
    out[:, :3, 3:] = -R.transpose(0, 2, 1) @ t
    return out


def _quat_xyzw(R: np.ndarray) -> np.ndarray:
    import torch

    from da3.camera import mat_to_quat

    return mat_to_quat(torch.from_numpy(R)).numpy()


def relative_pose_errors(pred_w2c: np.ndarray, gt_w2c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-pair rotation / translation-angle errors (degrees) for all pairs i < j."""
    P, G = _to44(pred_w2c), _to44(gt_w2c)
    P, G = P @ _inv(P[:1]), G @ _inv(G[:1])             # express relative to the first camera
    i, j = np.triu_indices(len(P), k=1)
    rel_p = _inv(P[i]) @ P[j]
    rel_g = _inv(G[i]) @ G[j]
    # rotation angle via quaternions (as released): err = arccos(1 - 2(1 - <q1,q2>^2))
    qp, qg = _quat_xyzw(rel_p[:, :3, :3]), _quat_xyzw(rel_g[:, :3, :3])
    loss_q = np.clip(1 - (qp * qg).sum(1) ** 2, 1e-15, None)
    r_err = np.degrees(np.arccos(np.clip(1 - 2 * loss_q, -1, 1)))
    tp = rel_p[:, :3, 3] / (np.linalg.norm(rel_p[:, :3, 3], axis=1, keepdims=True) + 1e-15)
    tg = rel_g[:, :3, 3] / (np.linalg.norm(rel_g[:, :3, 3], axis=1, keepdims=True) + 1e-15)
    loss_t = np.clip(1 - (tp * tg).sum(1) ** 2, 1e-15, None)
    t_err = np.degrees(np.arccos(np.sqrt(1 - loss_t)))
    t_err = np.where(np.isfinite(t_err), t_err, 1e6)
    t_err = np.minimum(t_err, np.abs(180 - t_err))
    return r_err, t_err


def pose_auc(r_err: np.ndarray, t_err: np.ndarray, max_threshold: int) -> float:
    err = np.maximum(r_err, t_err)
    hist, _ = np.histogram(err, bins=np.arange(max_threshold + 1))
    return float(np.mean(np.cumsum(hist / len(err))))


def pose_metrics(pred_w2c, gt_w2c) -> dict:
    r, t = relative_pose_errors(pred_w2c, gt_w2c)
    return {f"auc{k:02d}": pose_auc(r, t, k) for k in (30, 15, 5, 3)}


# ----------------------------------------------------------------------------- Sim(3) alignment
def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Least-squares s, R, t with dst ≈ s R src + t (points (N, 3))."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = (D * np.diag(S)).sum() / (xs**2).sum(1).mean() if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return R, t, s


def align_sim3_ransac(gt_w2c, pred_w2c, ransac: bool = True, iters: int = 10, seed: int = 42):
    """Sim(3) mapping predicted camera centres onto GT ones. Returns R, t, s and aligned pred w2c."""
    Cg = _inv(_to44(gt_w2c))[:, :3, 3]
    Pc = _inv(_to44(pred_w2c))
    Cp = Pc[:, :3, 3]
    R, t, s = umeyama(Cp, Cg)
    if ransac:
        rng = np.random.default_rng(seed)
        N = len(Cp)
        sub = max(3, (N + 1) // 2)
        aligned0 = (s * (R @ Cp.T)).T + t
        thresh = float(np.median([np.linalg.norm(Cg - p, axis=1).min() for p in aligned0]))
        best, best_score, best_inl = (R, t, s), (-1, np.inf), None
        for _ in range(iters):
            idx = rng.choice(N, size=sub, replace=False)
            Ri, ti, si = umeyama(Cp[idx], Cg[idx])
            err = np.linalg.norm((si * (Ri @ Cp.T)).T + ti - Cg, axis=1)
            inl = err <= thresh
            k, me = int(inl.sum()), float(err[inl].mean()) if inl.any() else np.inf
            if k > best_score[0] or (k == best_score[0] and me < best_score[1]):
                best_score, best, best_inl = (k, me), (Ri, ti, si), inl
        R, t, s = umeyama(Cp[best_inl], Cg[best_inl]) if best_inl is not None and best_inl.sum() >= 3 else best
    c2w = Pc.copy()
    c2w[:, :3, :3] = R @ Pc[:, :3, :3]
    c2w[:, :3, 3] = (s * (R @ Pc[:, :3, 3].T)).T + t
    return R, t, s, _inv(c2w)


# ----------------------------------------------------------------------------- reconstruction
def voxel_downsample(P: np.ndarray, voxel: float) -> np.ndarray:
    """Average of the points falling in each voxel (same as Open3D voxel_down_sample)."""
    if voxel is None or voxel <= 0 or len(P) == 0:
        return P
    keys = np.floor((P - (P.min(0) - voxel / 2)) / voxel).astype(np.int64)  # Open3D grid origin
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    cnt = np.bincount(inv)
    return np.stack([np.bincount(inv, weights=P[:, k]) / cnt for k in range(3)], 1)


def reconstruction_metrics(pred: np.ndarray, gt: np.ndarray, threshold: float, down_sample: float | None = None) -> dict:
    pred, gt = voxel_downsample(pred, down_sample), voxel_downsample(gt, down_sample)
    if len(pred) == 0 or len(gt) == 0:
        return {"acc": np.inf, "comp": np.inf, "overall": np.inf, "precision": 0.0, "recall": 0.0, "fscore": 0.0}
    d_pg = cKDTree(gt).query(pred)[0]
    d_gp = cKDTree(pred).query(gt)[0]
    acc, comp = float(d_pg.mean()), float(d_gp.mean())
    prec, rec = float((d_pg < threshold).mean()), float((d_gp < threshold).mean())
    f = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return {"acc": acc, "comp": comp, "overall": (acc + comp) / 2, "precision": prec, "recall": rec, "fscore": f}
