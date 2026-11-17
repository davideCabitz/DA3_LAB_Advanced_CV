"""Depth-ray geometry (Sec. 3.1).

Ray map layout (paper): M[..., 0:3] = ray origin t, M[..., 3:6] = direction d = R K^-1 p.
d is not normalized (its camera-frame z is 1), so P = t + D(u, v) d with D the z-depth.
Pixel centers are at (u + 0.5, v + 0.5). A ray map of size (h, w) for an H x W image samples the
image at pixel centers ((j + 0.5) W / w, (i + 0.5) H / h).

Camera recovery from a ray map:
* Eq. 1: camera center t_c = mean of the per-pixel ray origins;
* Eq. 2: H* = argmin_{|H|=1} sum_p |H p x M(p, 3:)|, solved by DLT (with Hartley normalization of
  the pixel coordinates), then RQ-decomposed into K (upper triangular) and R.
  Here H maps pixels to world-frame directions, H = R_c2w K^-1; we fit its inverse
  p ~ (K R_w2c) d, which is the same least-squares problem with the roles swapped and gives K and
  R_w2c directly from an RQ decomposition.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

ORIG, DIR = slice(0, 3), slice(3, 6)


def pixel_centers(h: int, w: int, H: int, W: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """Homogeneous pixel coordinates (in the H x W image frame) of an h x w grid -> (h, w, 3)."""
    ys = (torch.arange(h, device=device, dtype=dtype) + 0.5) * (H / h)
    xs = (torch.arange(w, device=device, dtype=dtype) + 0.5) * (W / w)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)


def make_ray_map(K: torch.Tensor, c2w: torch.Tensor, image_hw, ray_hw=None) -> torch.Tensor:
    """K (..., 3, 3), c2w (..., 3|4, 4) -> ray map (..., h, w, 6) = [t, d]."""
    H, W = image_hw
    h, w = ray_hw or image_hw
    p = pixel_centers(h, w, H, W, K.device, K.dtype)
    cam_dirs = torch.einsum("...ij,hwj->...hwi", torch.linalg.inv(K), p)
    dirs = torch.einsum("...ij,...hwj->...hwi", c2w[..., :3, :3], cam_dirs)
    orig = c2w[..., None, None, :3, 3].expand(dirs.shape)
    return torch.cat([orig, dirs], dim=-1)


def resize_ray_map(ray: torch.Tensor, hw) -> torch.Tensor:
    """Bilinearly resample a (..., h, w, C) map to (..., H, W, C)."""
    if tuple(ray.shape[-3:-1]) == tuple(hw):
        return ray
    lead = ray.shape[:-3]
    x = ray.reshape(-1, *ray.shape[-3:]).permute(0, 3, 1, 2)
    x = F.interpolate(x, size=tuple(hw), mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1).reshape(*lead, *hw, ray.shape[-1])


def points_from_depth_ray(depth: torch.Tensor, ray: torch.Tensor) -> torch.Tensor:
    """P = t + D * d. depth (..., H, W); ray (..., h, w, 6) (resampled to H x W if needed)."""
    ray = resize_ray_map(ray, depth.shape[-2:])
    return ray[..., ORIG] + depth[..., None] * ray[..., DIR]


def unproject_depth(depth: torch.Tensor, K: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
    """World points from z-depth (..., H, W), intrinsics and camera-to-world poses."""
    return points_from_depth_ray(depth, make_ray_map(K, c2w, depth.shape[-2:]))


# ----------------------------------------------------------------------------- ray map -> camera
def camera_center_from_rays(ray: torch.Tensor) -> torch.Tensor:
    """Eq. 1: mean of the per-pixel ray origins. (..., h, w, 6) -> (..., 3)."""
    return ray[..., ORIG].mean(dim=(-3, -2))


def rq_decomposition(M: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """M (..., 3, 3) = K @ R with K upper triangular (positive diagonal) and R orthogonal."""
    P = torch.flip(torch.eye(3, dtype=M.dtype, device=M.device), dims=[0])
    Q_, R_ = torch.linalg.qr((P @ M).transpose(-1, -2))
    K = P @ R_.transpose(-1, -2) @ P
    R = P @ Q_.transpose(-1, -2)
    S = torch.diag_embed(torch.sign(torch.diagonal(K, dim1=-2, dim2=-1)))
    return K @ S, S @ R  # S @ S = I


def _dlt(dirs: torch.Tensor, pix: torch.Tensor) -> torch.Tensor:
    """DLT for pix ~ M dirs: minimizes sum |p x (M d)| with |M| = 1. dirs (B,N,3), pix (B,N,2)."""
    x, y = pix[..., :1], pix[..., 1:]
    z = torch.zeros_like(dirs)
    r1 = torch.cat([z, -dirs, y * dirs], -1)   # y (m3.d) - (m2.d) = 0
    r2 = torch.cat([dirs, z, -x * dirs], -1)   # (m1.d) - x (m3.d) = 0
    A = torch.cat([r1, r2], dim=1)
    _, _, Vh = torch.linalg.svd(A.transpose(-1, -2) @ A)
    return Vh[..., -1, :].reshape(-1, 3, 3)


def rotation_intrinsics_from_rays(ray: torch.Tensor, image_hw) -> tuple[torch.Tensor, torch.Tensor]:
    """Eq. 2 for a batch of ray maps (B, h, w, 6) -> R_w2c (B, 3, 3), K (B, 3, 3) in pixels of
    the H x W image."""
    H, W = image_hw
    B, h, w, _ = ray.shape
    dt = torch.float64
    dirs = ray[..., DIR].reshape(B, -1, 3).to(dt)
    p = pixel_centers(h, w, H, W, ray.device, dt).reshape(-1, 3)
    s = 2.0 / max(H, W)  # Hartley normalization of pixel coordinates (standard DLT practice)
    T = torch.tensor([[s, 0, -s * W / 2], [0, s, -s * H / 2], [0, 0, 1]], dtype=dt, device=ray.device)
    pn = (p @ T.T)[None, :, :2].expand(B, -1, -1)
    M = torch.linalg.inv(T) @ _dlt(dirs, pn)          # p ~ M d with M = K R_w2c (up to scale)
    M = M * torch.sign(torch.linalg.det(M))[:, None, None]
    K, R = rq_decomposition(M)
    K = K / K[:, 2:3, 2:3]
    return R.to(ray.dtype), K.to(ray.dtype)


def pose_from_ray_map(ray: torch.Tensor, image_hw) -> tuple[torch.Tensor, torch.Tensor]:
    """Ray map (..., h, w, 6) -> (w2c (..., 3, 4), K (..., 3, 3)) via Eq. 1 + Eq. 2."""
    lead = ray.shape[:-3]
    r = ray.reshape(-1, *ray.shape[-3:])
    R, K = rotation_intrinsics_from_rays(r, image_hw)
    c = camera_center_from_rays(r).to(R.dtype)
    w2c = torch.cat([R, -(R @ c[..., None])], dim=-1)
    return w2c.reshape(*lead, 3, 4), K.reshape(*lead, 3, 3)
