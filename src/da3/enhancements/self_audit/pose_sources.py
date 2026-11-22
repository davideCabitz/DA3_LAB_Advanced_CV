"""Per-view pose source: camera head (API default), ray map, or a per-view choice / fusion of both.

Head and ray-map poses come from the same forward pass. Before mixing them per view, the ray poses
are Sim(3)-aligned onto the head poses (camera centres), so that both live in the frame and scale
of the predicted depth (which the harness fuses with the cameras).

* `select`: per view, take the source predicted to be more accurate;
* `fuse`: per view, inverse-variance weighted average (rotation via sign-aligned quaternion
  average, centre and intrinsics linearly), weights from predicted errors;
* `oracle_choice`: per view, the source with the lower GT error (an upper bound, not a method).
"""

from __future__ import annotations

import numpy as np
import torch

from da3.camera import mat_to_quat, quat_to_mat

from .signals import align_onto, centers, to44


def _w2c_from(R_c2w: np.ndarray, C: np.ndarray) -> np.ndarray:
    Rw = np.swapaxes(R_c2w, -1, -2)
    out = np.tile(np.eye(4), (len(C), 1, 1))
    out[:, :3, :3] = Rw
    out[:, :3, 3] = -np.einsum("sij,sj->si", Rw, C)
    return out


def aligned_ray(head_w2c: np.ndarray, ray_w2c: np.ndarray) -> np.ndarray:
    return align_onto(head_w2c, ray_w2c)


def fuse(head_w2c, head_K, ray_w2c, ray_K, w_ray) -> tuple[np.ndarray, np.ndarray]:
    """Per-view convex combination with weight `w_ray` in [0, 1] on the (aligned) ray pose."""
    w = np.clip(np.asarray(w_ray, np.float64), 0.0, 1.0)
    H, Rr = to44(head_w2c), aligned_ray(head_w2c, ray_w2c)
    qh = mat_to_quat(torch.from_numpy(np.swapaxes(H[:, :3, :3], -1, -2).copy())).numpy()
    qr = mat_to_quat(torch.from_numpy(np.swapaxes(Rr[:, :3, :3], -1, -2).copy())).numpy()
    qr = qr * np.sign((qh * qr).sum(-1, keepdims=True) + 1e-12)
    q = (1 - w)[:, None] * qh + w[:, None] * qr
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    R_c2w = quat_to_mat(torch.from_numpy(q)).numpy()
    C = (1 - w)[:, None] * centers(H) + w[:, None] * centers(Rr)
    K = (1 - w)[:, None, None] * np.asarray(head_K, np.float64) + w[:, None, None] * np.asarray(ray_K, np.float64)
    return _w2c_from(R_c2w, C), K


def select(head_w2c, head_K, ray_w2c, ray_K, use_ray) -> tuple[np.ndarray, np.ndarray]:
    use_ray = np.asarray(use_ray, bool)
    H, Rr = to44(head_w2c), aligned_ray(head_w2c, ray_w2c)
    w2c = np.where(use_ray[:, None, None], Rr, H)
    K = np.where(use_ray[:, None, None], np.asarray(ray_K, np.float64), np.asarray(head_K, np.float64))
    return w2c, K


def inverse_variance_weight(err_head_pred, err_ray_pred) -> np.ndarray:
    """w_ray = s_h^2 / (s_h^2 + s_r^2) from predicted per-view errors (degrees, > 0)."""
    sh = np.clip(np.asarray(err_head_pred, np.float64), 1e-3, None) ** 2
    sr = np.clip(np.asarray(err_ray_pred, np.float64), 1e-3, None) ** 2
    return sh / (sh + sr)


def oracle_choice(err_head, err_ray) -> np.ndarray:
    return np.asarray(err_ray) < np.asarray(err_head)
