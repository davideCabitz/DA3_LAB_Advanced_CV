"""Per-view Self-Audit signals (enhancement #1, docs/da3_enhancement_report.md §4.2).

DA3 estimates each camera twice (camera head D_C and ray map), its 6-channel ray map
over-parameterises a pinhole camera, and it can take its own cameras back through E_c. The
disagreement between these redundant estimates is a free reliability signal. All signals below
are oriented so that LARGER means LESS reliable:

(a) gap_rot_deg / gap_center / gap_focal: head pose vs ray-map pose (after a Sim(3) alignment of
    the ray poses onto the head poses, so only per-view inconsistencies remain);
(b) origin_spread: spread of the per-pixel ray origins (Eq. 1 assumes one centre), / scene scale;
(c) pinhole_resid_deg: mean angle between the predicted ray directions and those of the best
    pinhole camera fitted to them (Eq. 2, DLT + RQ), confidence-weighted;
(d) neg_logconf_*: minus log of DA3's own depth / ray confidences (the original DA3 signal);
(e) reproj_disagree: median |log(z / D_j)| of this view's points reprojected into its nearest
    views (occlusion-aware), i.e. cross-view depth consistency;
(f) ec_*: E_c fixed-point residual: how much depth, camera and rays move when DA3 is re-run
    conditioned on its own cameras (one extra pass). Rotations / rays are compared after removing
    one global rotation, because the posed pass works in the normalised input frame.

Inputs are numpy arrays of ONE scene: harness results (depth, conf, head extrinsics/intrinsics)
and the audit dump written by da3.enhancements.self_audit.dump.AuditDumper.
"""

from __future__ import annotations

import numpy as np
import torch

from da3.eval.metrics import align_sim3_ransac, relative_pose_errors
from da3.geometry import pixel_centers, rotation_intrinsics_from_rays

GAP_FEATURES = ["gap_rot_deg", "gap_center", "gap_focal"]
RAY_FEATURES = ["origin_spread", "pinhole_resid_deg"]
CONF_FEATURES = ["neg_logconf_depth_mean", "neg_logconf_depth_p10", "neg_logconf_ray_mean"]
REPROJ_FEATURES = ["reproj_disagree"]
EC_FEATURES = ["ec_depth_change", "ec_rot_change_deg", "ec_ray_change_deg"]
FEATURES = GAP_FEATURES + RAY_FEATURES + CONF_FEATURES + REPROJ_FEATURES + EC_FEATURES
BASELINE_SCORE = "neg_logconf_depth_mean"   # what the original DA3 offers: its own confidence


# ----------------------------------------------------------------------------- small geometry
def to44(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, np.float64)
    if T.shape[-2] == 4:
        return T
    out = np.tile(np.eye(4), T.shape[:-2] + (1, 1))
    out[..., :3, :4] = T
    return out


def rot_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    M = Ra @ np.swapaxes(Rb, -1, -2)
    cos = (np.trace(M, axis1=-2, axis2=-1) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def centers(w2c: np.ndarray) -> np.ndarray:
    """Camera centres C = -R^T t of (S, 3|4, 4) world-to-camera matrices."""
    w2c = np.asarray(w2c, np.float64)
    return -np.einsum("sji,sj->si", w2c[:, :3, :3], w2c[:, :3, 3])


def scene_scale(C: np.ndarray) -> float:
    d = np.linalg.norm(C - np.median(C, axis=0), axis=1)
    s = float(np.median(d))
    return s if s > 1e-8 else max(float(d.max()), 1e-8)


def focal(K: np.ndarray) -> np.ndarray:
    K = np.asarray(K, np.float64)
    return np.sqrt(np.abs(K[:, 0, 0] * K[:, 1, 1]))


def project_so3(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12, None)


def _weights(conf: np.ndarray | None, shape) -> np.ndarray:
    if conf is None:
        w = np.ones(shape)
    else:
        w = np.clip(np.asarray(conf, np.float64).reshape(shape), 1e-6, None)
    return w / w.sum(axis=1, keepdims=True)


# ----------------------------------------------------------------------------- signals
def align_onto(ref_w2c: np.ndarray, src_w2c: np.ndarray) -> np.ndarray:
    """Sim(3)-align the cameras `src` onto `ref` (camera centres, RANSAC when >= 10 views)."""
    S = len(src_w2c)
    if S < 3:
        return to44(src_w2c)
    _, _, _, aligned = align_sim3_ransac(ref_w2c, src_w2c, ransac=S >= 10, seed=42)
    return aligned


def head_ray_gap(head_w2c, head_K, ray_w2c, ray_K) -> dict:
    ray_al = align_onto(head_w2c, ray_w2c)
    head = to44(head_w2c)
    rot = rot_angle_deg(head[:, :3, :3], ray_al[:, :3, :3])
    C_h, C_r = centers(head), centers(ray_al)
    return {"gap_rot_deg": rot,
            "gap_center": np.linalg.norm(C_h - C_r, axis=1) / scene_scale(C_h),
            "gap_focal": np.abs(np.log(focal(head_K) / focal(ray_K)))}


def origin_spread(ray: np.ndarray, ray_conf: np.ndarray | None, scale: float) -> np.ndarray:
    S = ray.shape[0]
    o = np.asarray(ray[..., 0:3], np.float64).reshape(S, -1, 3)
    w = _weights(ray_conf, o.shape[:2])
    mu = (w[..., None] * o).sum(1)
    var = (w * ((o - mu[:, None]) ** 2).sum(-1)).sum(1)
    return np.sqrt(var) / scale


def pinhole_residual_deg(ray: np.ndarray, ray_conf: np.ndarray | None, image_hw) -> np.ndarray:
    """Angle between predicted directions and those of the DLT-fitted pinhole camera (Eq. 2)."""
    r = torch.from_numpy(np.asarray(ray, np.float64))
    S, h, w, _ = r.shape
    R, K = rotation_intrinsics_from_rays(r, image_hw)                       # R_w2c, K
    p = pixel_centers(h, w, image_hw[0], image_hw[1], dtype=torch.float64).reshape(-1, 3)
    d_cam = torch.einsum("sij,nj->sni", torch.linalg.inv(K), p)
    d_fit = torch.einsum("sji,snj->sni", R, d_cam).numpy()                 # R^T d_cam (world)
    d = r[..., 3:6].reshape(S, -1, 3).numpy()
    cos = np.clip(np.abs((_unit(d) * _unit(d_fit)).sum(-1)), 0.0, 1.0)
    return (_weights(ray_conf, cos.shape) * np.degrees(np.arccos(cos))).sum(1)


def confidence_stats(depth_conf: np.ndarray, ray_conf: np.ndarray | None) -> dict:
    S = depth_conf.shape[0]
    lc = np.log(np.clip(np.asarray(depth_conf, np.float64), 1e-6, None)).reshape(S, -1)
    out = {"neg_logconf_depth_mean": -lc.mean(1), "neg_logconf_depth_p10": -np.percentile(lc, 10, axis=1)}
    if ray_conf is not None:
        lr = np.log(np.clip(np.asarray(ray_conf, np.float64), 1e-6, None)).reshape(S, -1)
        out["neg_logconf_ray_mean"] = -lr.mean(1)
    else:
        out["neg_logconf_ray_mean"] = np.full(S, np.nan)
    return out


def reprojection_disagreement(depth, w2c, K, n_neighbors: int = 4, stride: int = 4, tau: float = 0.2,
                              min_points: int = 20) -> np.ndarray:
    """Median |log(z / D_j)| of each view's points (head cameras + depth) projected into its
    `n_neighbors` nearest views; points behind the surface seen by view j (occluded) are ignored."""
    depth = np.asarray(depth, np.float64)
    S, H, W = depth.shape
    T = to44(w2c)
    R, t = T[:, :3, :3], T[:, :3, 3]
    K = np.asarray(K, np.float64)
    Kinv = np.linalg.inv(K)
    ys, xs = np.arange(stride // 2, H, stride), np.arange(stride // 2, W, stride)
    vv, uu = np.meshgrid(ys, xs, indexing="ij")
    pix = np.stack([uu.ravel() + 0.5, vv.ravel() + 0.5, np.ones(uu.size)], 1)   # (N, 3)
    C = centers(T)
    out = np.full(S, np.nan)
    for i in range(S):
        d = depth[i, vv.ravel(), uu.ravel()]
        ok = np.isfinite(d) & (d > 0)
        if ok.sum() < min_points or S < 2:
            continue
        Xc = d[ok, None] * (pix[ok] @ Kinv[i].T)
        P = (Xc - t[i]) @ R[i]                                # R^T (Xc - t), row-vector form
        dist = np.linalg.norm(C - C[i], axis=1)
        dist[i] = np.inf
        res = []
        for j in np.argsort(dist)[: min(n_neighbors, S - 1)]:
            Xj = P @ R[j].T + t[j]
            z = Xj[:, 2]
            front = z > 1e-9
            uv = (Xj[front] @ K[j].T)
            u, v = uv[:, 0] / uv[:, 2], uv[:, 1] / uv[:, 2]
            inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            if not inside.any():
                continue
            zj = z[front][inside]
            Dj = depth[j, v[inside].astype(int), u[inside].astype(int)]
            vis = (Dj > 0) & (zj <= Dj * (1 + tau))
            res.append(np.abs(np.log(zj[vis] / Dj[vis])))
        r = np.concatenate(res) if res else np.zeros(0)
        if len(r) >= min_points:
            out[i] = float(np.median(r))
    return out


def ec_fixed_point_residuals(depth_ds, repass_depth_ds, head_w2c, repass_w2c, ray_ds=None, repass_ray_ds=None,
                             ray_conf_ds=None) -> dict:
    """How much DA3's prediction moves when it is conditioned on its own cameras (signal f)."""
    S = depth_ds.shape[0]
    d1, d2 = np.asarray(depth_ds, np.float64), np.asarray(repass_depth_ds, np.float64)
    valid = (d1 > 0) & (d2 > 0) & np.isfinite(d1) & np.isfinite(d2)
    diff = np.where(valid, np.log(np.where(valid, d2, 1.0)) - np.log(np.where(valid, d1, 1.0)), np.nan)
    shift = np.nanmedian(diff)
    depth_change = np.nanmedian(np.abs(diff - shift).reshape(S, -1), axis=1)
    # rotations: remove one global rotation G (c2w composes with world rotations on the left)
    Rin = np.swapaxes(to44(head_w2c)[:, :3, :3], -1, -2)
    Rout = np.swapaxes(to44(repass_w2c)[:, :3, :3], -1, -2)
    G = project_so3(np.einsum("sij,skj->ik", Rout, Rin))     # sum_s Rout_s Rin_s^T
    rot_change = rot_angle_deg(Rout, G[None] @ Rin)
    out = {"ec_depth_change": depth_change, "ec_rot_change_deg": rot_change,
           "ec_ray_change_deg": np.full(S, np.nan)}
    if ray_ds is not None and repass_ray_ds is not None:
        a = _unit(np.asarray(repass_ray_ds[..., 3:6], np.float64).reshape(S, -1, 3))
        b = _unit(np.asarray(ray_ds[..., 3:6], np.float64).reshape(S, -1, 3))
        w = _weights(ray_conf_ds, a.shape[:2])
        Gd = project_so3(np.einsum("sn,sni,snj->ij", w, a, b))
        cos = np.clip((a * (b @ Gd.T)).sum(-1), -1.0, 1.0)
        out["ec_ray_change_deg"] = (w * np.degrees(np.arccos(cos))).sum(1)
    return out


def compute_signals(results: dict, dump: dict) -> dict:
    """All per-view signals of one scene -> {feature: (S,) array}. Missing parts become NaN."""
    depth, conf = results["depth"], results["conf"]
    S, H, W = depth.shape
    head_w2c, head_K = dump["head_w2c"], dump["head_K"]
    if "ray_off_w2c" in dump:
        ray_w2c, ray_K = dump["ray_off_w2c"], dump["ray_off_K"]
    else:
        ray_w2c, ray_K = dump["ray_dlt_w2c"], dump["ray_dlt_K"]
    ray_ds = dump["ray_ds"].astype(np.float64)
    ray_conf_ds = dump["ray_conf_ds"].astype(np.float64) if "ray_conf_ds" in dump else None
    scale = scene_scale(centers(to44(head_w2c)))

    sig = {}
    sig.update(head_ray_gap(head_w2c, head_K, ray_w2c, ray_K))
    sig["origin_spread"] = origin_spread(ray_ds, ray_conf_ds, scale)
    sig["pinhole_resid_deg"] = pinhole_residual_deg(ray_ds, ray_conf_ds, (H, W))
    sig.update(confidence_stats(conf, ray_conf_ds))
    sig["reproj_disagree"] = reprojection_disagreement(depth, head_w2c, head_K)
    if "repass_depth_ds" in dump:
        sig.update(ec_fixed_point_residuals(dump["depth_ds"], dump["repass_depth_ds"], head_w2c,
                                            dump["repass_w2c"], ray_ds, dump.get("repass_ray_ds"), ray_conf_ds))
    else:
        sig.update({f: np.full(S, np.nan) for f in EC_FEATURES})
    return {f: np.asarray(sig[f], np.float64) for f in FEATURES}


# ----------------------------------------------------------------------------- labels (use GT)
def per_view_pose_error(pred_w2c: np.ndarray, gt_w2c: np.ndarray) -> dict:
    """Per-view errors from the paper's pairwise definition: for view i, the median over all other
    views j of max(RRA_ij, RTA_ij) (`err`, consistent with pose AUC) and of RRA_ij (`rot_err`)."""
    S = len(pred_w2c)
    r, t = relative_pose_errors(np.asarray(pred_w2c), np.asarray(gt_w2c))
    i, j = np.triu_indices(S, k=1)
    out = {}
    for name, e in (("err", np.maximum(r, t)), ("rot_err", r)):
        M = np.full((S, S), np.nan)
        M[i, j] = e
        M[j, i] = e
        out[name] = np.nanmedian(M, axis=1) if S > 1 else np.full(S, np.nan)
    return out
