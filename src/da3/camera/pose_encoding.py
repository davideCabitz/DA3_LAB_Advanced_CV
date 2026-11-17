"""Camera <-> 9-D pose encoding [t (3), q (4, xyzw), fov_h, fov_w] (Sec. 3.1 / 3.2).

Conventions used throughout the project:
* extrinsics `w2c` are OpenCV world-to-camera (x right, y down, z forward), `c2w` is its inverse;
* the pose encoding stores the camera-to-world rotation / translation;
* intrinsics are in pixels for an image of size (H, W), principal point at the center.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    """(..., 4) xyzw quaternion (not necessarily unit) -> (..., 3, 3) rotation."""
    i, j, k, r = q.unbind(-1)
    two_s = 2.0 / (q * q).sum(-1)
    m = torch.stack(
        (
            1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
            two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
            two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return m.reshape(q.shape[:-1] + (3, 3))


def mat_to_quat(m: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) rotation -> (..., 4) unit xyzw quaternion with non-negative real part."""
    batch = m.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = m.reshape(batch + (9,)).unbind(-1)
    sq = torch.stack(
        [1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22], -1
    )
    # sqrt(max(0, x)) with a zero (not inf) gradient at 0
    q_abs = torch.where(sq > 0, sq.clamp_min(1e-12).sqrt(), torch.zeros_like(sq))
    cand = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], -1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], -1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], -1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], -1),
        ],
        -2,
    )  # rijk candidates
    cand = cand / (2.0 * q_abs[..., None].clamp_min(0.1))
    best = F.one_hot(q_abs.argmax(-1), 4).bool()
    q = cand[best].reshape(batch + (4,))[..., [1, 2, 3, 0]]  # rijk -> xyzw
    return torch.where(q[..., 3:4] < 0, -q, q)


def affine_inverse(T: torch.Tensor) -> torch.Tensor:
    """Inverse of rigid (..., 3, 4) or (..., 4, 4) transforms, same shape as input."""
    R, t = T[..., :3, :3], T[..., :3, 3:]
    Rt = R.transpose(-1, -2)
    out = torch.cat([Rt, -Rt @ t], dim=-1)
    if T.shape[-2] == 4:
        out = torch.cat([out, T[..., 3:, :]], dim=-2)
    return out


def to_homogeneous(T: torch.Tensor) -> torch.Tensor:
    if T.shape[-2] == 4:
        return T
    bottom = torch.zeros(T.shape[:-2] + (1, 4), dtype=T.dtype, device=T.device)
    bottom[..., 0, 3] = 1
    return torch.cat([T, bottom], dim=-2)


def encode_pose(c2w: torch.Tensor, K: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    """c2w (..., 3|4, 4), K (..., 3, 3) -> (..., 9)."""
    H, W = hw
    q = mat_to_quat(c2w[..., :3, :3])
    fov_h = 2 * torch.atan((H / 2) / K[..., 1, 1])
    fov_w = 2 * torch.atan((W / 2) / K[..., 0, 0])
    return torch.cat([c2w[..., :3, 3], q, fov_h[..., None], fov_w[..., None]], -1).float()


def decode_pose(enc: torch.Tensor, hw: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """(..., 9) -> c2w (..., 3, 4), K (..., 3, 3) with principal point at the image center."""
    H, W = hw
    R = quat_to_mat(enc[..., 3:7])
    c2w = torch.cat([R, enc[..., :3, None]], dim=-1)
    K = torch.zeros(enc.shape[:-1] + (3, 3), dtype=enc.dtype, device=enc.device)
    K[..., 0, 0] = (W / 2.0) / torch.tan(enc[..., 8] / 2.0).clamp_min(1e-6)
    K[..., 1, 1] = (H / 2.0) / torch.tan(enc[..., 7] / 2.0).clamp_min(1e-6)
    K[..., 0, 2], K[..., 1, 2], K[..., 2, 2] = W / 2, H / 2, 1.0
    return c2w, K


def normalize_extrinsics(w2c: torch.Tensor) -> torch.Tensor:
    """Express (B, S, 4, 4) w2c relative to view 0 and scale translations by the median camera
    distance (clamped at 0.1). This is how the released API prepares poses for E_c."""
    w2c = to_homogeneous(w2c)
    rel = w2c @ affine_inverse(w2c[:, :1])
    dist = affine_inverse(rel)[..., :3, 3].norm(dim=-1)
    scale = dist.flatten(1).median(dim=1).values.clamp_min(0.1)
    rel = rel.clone()
    rel[..., :3, 3] = rel[..., :3, 3] / scale[:, None, None]
    return rel
