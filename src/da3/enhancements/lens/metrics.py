"""Metrics of enhancement #4 (lens-consistent DA3).

* `scene_view_metrics`: per-scene diagnostics of one mixed-lens run, from the harness exports
  (export-frame depth / cameras), the GT cameras and the lens meta:
  - depth scale ratio zoomed / normal views: median of the per-view alignment scales
    median(gt) / median(pred) of the zoomed views over that of the normal views (1 = consistent;
    > 1 = zoomed views predicted too close);
  - pose AUC (the paper's max(RRA, RTA) pairs) on all pairs, normal-normal, zoom-zoom and
    zoom-normal (cross) pairs, cross pairs also with rotation and translation errors apart;
  - focal ratio: predicted / GT focal length in the frame of the dataset image (1 = DA3 understood
    the zoom); zoom-estimate error of canon_self.
* `reprojection_consistency`: ground-truth-free cross-view depth agreement of a reconstruction
  (any per-view sizes): points of view i projected into view j vs view j's depth. Used on the
  photo demo, where no GT exists.
"""

from __future__ import annotations

import numpy as np

from da3.enhancements.foveated.metrics import depth_errors, pairwise_rot_trans_errors, pooled_scale, subset_auc
from da3.io_utils import scale_intrinsics

from .geometry import KIND_NORMAL, KIND_ZOOM
from .plan import reference_kappa


def _nanmedian(x) -> float:
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else float("nan")


def scene_view_metrics(results: dict, gt_w2c, gt_K, meta: dict, gts=None) -> dict:
    """results: harness export (depth (S, h, w), extrinsics, intrinsics in export pixels);
    gt_w2c (S, 3|4, 4); gt_K (S, 3, 3) in pixels of the dataset image; meta: lens_meta.npz;
    gts: optional GT depth maps already at the export size (None entries allowed)."""
    kinds = np.asarray(meta["kinds"], int)
    S = len(kinds)
    normal, zoom = kinds == KIND_NORMAL, kinds == KIND_ZOOM
    export_hw = tuple(int(t) for t in np.asarray(meta["export_hw"]))
    orig_hw = np.asarray(meta["orig_hw"], int)

    Er, Et = pairwise_rot_trans_errors(results["extrinsics"], gt_w2c)
    E = np.fmax(Er, Et)
    ones = np.ones(S, bool)
    row = {"n_views": S, "n_zoom": int(zoom.sum()), "kinds": kinds.tolist(),
           "auc": {"all": subset_auc(E, ones), "normal": subset_auc(E, normal)},
           "auc_rot": {"all": subset_auc(Er, ones)}, "auc_trans": {"all": subset_auc(Et, ones)}}
    if zoom.any():
        row["auc"]["zoom"] = subset_auc(E, zoom)
        row["auc"]["cross"] = subset_auc(E, normal, zoom)
        row["auc_rot"]["cross"] = subset_auc(Er, normal, zoom)
        row["auc_trans"]["cross"] = subset_auc(Et, normal, zoom)

    # focal length in the frame of the dataset image (export size)
    Kp = np.asarray(results["intrinsics"], np.float64)
    fp = 0.5 * (Kp[:, 0, 0] + Kp[:, 1, 1])
    fg = np.array([0.5 * (k[0, 0] + k[1, 1]) for k in
                   (scale_intrinsics(np.asarray(gt_K[i]), orig_hw[i], export_hw) for i in range(S))])
    ratio = fp / fg
    row["focal_ratio"] = ratio.tolist()
    row["focal_ratio_normal"] = _nanmedian(ratio[normal])
    row["focal_ratio_zoom"] = _nanmedian(ratio[zoom]) if zoom.any() else float("nan")

    # canon_self zoom estimates vs the true zoom relative to the reference view
    z_est = np.asarray(meta.get("z_est", np.full(S, np.nan)), np.float64)
    if np.isfinite(z_est).any():
        true_rel = np.asarray(meta["zoom"], np.float64) / reference_kappa(np.asarray(meta["zoom"], np.float64))
        err = np.abs(z_est / true_rel - 1.0)
        row["z_est"] = z_est.tolist()
        row["z_err"] = err.tolist()
        row["z_err_zoom"] = _nanmedian(err[zoom]) if zoom.any() else float("nan")
        row["z_err_normal"] = _nanmedian(err[normal])

    if gts is not None:
        preds = [np.asarray(results["depth"][i], np.float32) for i in range(S)]
        s_scene = pooled_scale([p if g is not None else None for p, g in zip(preds, gts)], gts)
        absrel, absrel_scene, scale = [], [], []
        for p, g in zip(preds, gts):
            e = depth_errors(p, g) if g is not None else None
            es = depth_errors(p, g, scale=s_scene) if (g is not None and np.isfinite(s_scene)) else None
            absrel.append(e["abs_rel"] if e else np.nan)
            absrel_scene.append(es["abs_rel"] if es else np.nan)
            scale.append(e["scale"] if e else np.nan)
        absrel, absrel_scene, scale = (np.asarray(v, np.float64) for v in (absrel, absrel_scene, scale))
        row.update({"abs_rel": absrel.tolist(), "abs_rel_scene": absrel_scene.tolist(), "scale": scale.tolist(),
                    "abs_rel_normal": _nanmedian(absrel[normal]),
                    "abs_rel_zoom": _nanmedian(absrel[zoom]) if zoom.any() else float("nan"),
                    "abs_rel_scene_all": float(np.nanmean(absrel_scene)) if np.isfinite(absrel_scene).any() else float("nan")})
        sn, sz = _nanmedian(scale[normal]), (_nanmedian(scale[zoom]) if zoom.any() else float("nan"))
        row["scale_ratio"] = sz / sn if np.isfinite(sz) and np.isfinite(sn) and sn > 0 else float("nan")
    return row


# ----------------------------------------------------------------------------- GT-free consistency
def _w2c44(w2c) -> np.ndarray:
    w = np.asarray(w2c, np.float64)
    out = np.tile(np.eye(4), (len(w), 1, 1))
    out[:, :3, :4] = w[:, :3, :4]
    return out


def reprojection_consistency(depths, Ks, w2cs, stride: int = 4, neighbours: int = 6, occlusion: float = 0.25,
                             valid=None) -> dict:
    """Cross-view depth agreement of one reconstruction (scale-free, no GT).

    For every view i and its `neighbours` nearest views j (camera centres), the pixels of i (every
    `stride`-th) are lifted with depth_i, K_i, w2c_i and projected into j; where they land inside j
    with positive depth, r = log(z_ij / depth_j) compares the projected depth with j's own depth at
    that pixel (nearest). Pixels with |r| > log(1 + occlusion) are treated as occlusions / misses and
    left out of the error, but counted. Returns the median |r| (in %, ~ relative depth disagreement),
    the share of co-visible pixels within 1 / 2 / 5 %, the occlusion share and the per-view medians.
    depths: list of (h_i, w_i); Ks: (N, 3, 3) in each view's pixels; w2cs: (N, 3|4, 4)."""
    W = _w2c44(w2cs)
    C = np.stack([-W[i, :3, :3].T @ W[i, :3, 3] for i in range(len(W))])
    n = len(depths)
    per_view, all_r, n_occ, n_tot = [], [], 0, 0
    thr = np.log1p(occlusion)
    for i in range(n):
        d = np.asarray(depths[i], np.float64)
        h, w = d.shape
        vv, uu = np.mgrid[0:h:stride, 0:w:stride]
        z = d[vv, uu]
        ok = np.isfinite(z) & (z > 0)
        if valid is not None:
            ok &= np.asarray(valid[i], bool)[vv, uu]
        if not ok.any():
            per_view.append(float("nan"))
            continue
        u, v, z = uu[ok] + 0.5, vv[ok] + 0.5, z[ok]
        Ki = np.asarray(Ks[i], np.float64)
        Xc = np.stack([(u - Ki[0, 2]) / Ki[0, 0] * z, (v - Ki[1, 2]) / Ki[1, 1] * z, z], 1)
        Rc2w, Ci = W[i, :3, :3].T, C[i]
        Xw = Xc @ Rc2w.T + Ci
        order = np.argsort(np.linalg.norm(C - Ci, axis=1))
        rs = []
        for j in [k for k in order if k != i][:neighbours]:
            dj = np.asarray(depths[j], np.float64)
            hj, wj = dj.shape
            Kj = np.asarray(Ks[j], np.float64)
            Xj = Xw @ W[j, :3, :3].T + W[j, :3, 3]
            zj = Xj[:, 2]
            front = zj > 1e-9
            uj = np.where(front, Kj[0, 0] * Xj[:, 0] / np.where(front, zj, 1) + Kj[0, 2], -1)
            vj = np.where(front, Kj[1, 1] * Xj[:, 1] / np.where(front, zj, 1) + Kj[1, 2], -1)
            inside = front & (uj >= 0) & (uj < wj) & (vj >= 0) & (vj < hj)
            if not inside.any():
                continue
            dd = dj[vj[inside].astype(int), uj[inside].astype(int)]
            good = np.isfinite(dd) & (dd > 0)
            if valid is not None:
                good &= np.asarray(valid[j], bool)[vj[inside].astype(int), uj[inside].astype(int)]
            r = np.log(zj[inside][good] / dd[good])
            n_tot += len(r)
            occ = np.abs(r) > thr
            n_occ += int(occ.sum())
            rs.append(np.abs(r[~occ]))
        if rs:
            rv = np.concatenate(rs)
            all_r.append(rv)
            per_view.append(float(100 * np.median(rv)) if len(rv) else float("nan"))
        else:
            per_view.append(float("nan"))
    if not all_r or not len(np.concatenate(all_r)):
        return {"median_pct": float("nan"), "within_1pct": float("nan"), "within_2pct": float("nan"),
                "within_5pct": float("nan"), "occluded_share": float("nan"), "per_view_median_pct": per_view}
    r = np.concatenate(all_r)
    return {"median_pct": float(100 * np.median(r)), "within_1pct": float(np.mean(r < np.log(1.01))),
            "within_2pct": float(np.mean(r < np.log(1.02))), "within_5pct": float(np.mean(r < np.log(1.05))),
            "occluded_share": float(n_occ / max(n_tot, 1)), "per_view_median_pct": per_view}
