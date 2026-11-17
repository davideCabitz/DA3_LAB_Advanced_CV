"""DA3 training objective (Sec. 3.3):

    L = L_D(D̂, D) + L_M(R̂, M) + L_P(D̂ ⊙ d + t, P) + β L_C(ĉ, v) + α L_grad(D̂, D),  α = β = 1

L_D(D̂, D; D_c) = 1/Z_Ω Σ_p m_p (D_c,p |D̂_p − D_p| − λ_c log D_c,p). All terms are ℓ1.
Ground truth is first expressed in the frame of view 0 and divided by one scale per scene: the
mean ℓ2 norm of the valid points (Sec. 3.3). Choices the paper leaves open are LossConfig fields
(see docs/PLAN.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from da3.camera import affine_inverse, encode_pose, to_homogeneous
from da3.geometry import DIR, ORIG, make_ray_map, points_from_depth_ray, unproject_depth


@dataclass
class LossConfig:
    alpha_grad: float = 1.0
    beta_cam: float = 1.0
    w_depth: float = 1.0
    w_ray: float = 1.0
    w_point: float = 1.0
    lambda_conf: float = 0.2          # λ_c: value not given in the paper (DUSt3R ConfLoss default)
    point_rays: str = "pred"          # d, t in L_P: "pred" (D̂ ⊙ d̂ + t̂) or "gt"; the paper is ambiguous


# ----------------------------------------------------------------------------- GT normalization
@torch.no_grad()
def normalize_gt(depth, valid, K, w2c, ray_hw):
    """
    depth (B,S,H,W), valid (B,S,H,W) bool, K (B,S,3,3), w2c (B,S,3|4,4) in any world frame.
    Returns a dict with normalized depth/points/ray map/pose encoding, relative w2c/c2w, the scale.
    """
    B, S, H, W = depth.shape
    valid = valid & torch.isfinite(depth) & (depth > 0)
    depth = torch.where(valid, depth.float(), torch.zeros_like(depth, dtype=torch.float32))
    w2c = to_homogeneous(w2c.float())
    w2c_rel = w2c @ affine_inverse(w2c[:, :1])            # view 0 becomes the world frame
    c2w_rel = affine_inverse(w2c_rel)
    pts = unproject_depth(depth.float(), K.float(), c2w_rel)
    vf = valid.float()
    norm = (pts.norm(dim=-1) * vf).flatten(1).sum(1) / vf.flatten(1).sum(1).clamp_min(1)
    scale = norm.clamp_min(1e-6)                          # (B,)
    s = scale[:, None, None, None]
    c2w_n = c2w_rel.clone()
    c2w_n[..., :3, 3] = c2w_n[..., :3, 3] / scale[:, None, None]
    return {
        "depth": depth / s,
        "points": pts / s[..., None],
        "valid": valid,
        "ray": make_ray_map(K.float(), c2w_n, (H, W), ray_hw),
        "pose_enc": encode_pose(c2w_n, K.float(), (H, W)),
        "c2w": c2w_n,
        "w2c": affine_inverse(c2w_n),
        "K": K.float(),
        "scale": scale,
    }


# ----------------------------------------------------------------------------- terms
def _masked_mean(x, m):
    m = m.float()
    return (x * m).sum() / m.sum().clamp_min(1.0)


def conf_l1(pred, gt, conf, mask, lambda_conf):
    """Confidence-weighted ℓ1 (L_D form). pred/gt (..., C) or (...), conf/mask (...)."""
    err = (pred - gt).abs()
    if err.dim() > conf.dim():
        err = err.sum(-1)
    return _masked_mean(conf * err - lambda_conf * torch.log(conf), mask)


def grad_l1(pred, gt, mask):
    """Eq. 3: ℓ1 between horizontal / vertical finite differences, where both pixels are valid."""
    def d(x, dim):
        return x.diff(dim=dim)

    mx = mask[..., :, 1:] & mask[..., :, :-1]
    my = mask[..., 1:, :] & mask[..., :-1, :]
    lx = _masked_mean((d(pred, -1) - d(gt, -1)).abs(), mx)
    ly = _masked_mean((d(pred, -2) - d(gt, -2)).abs(), my)
    return lx + ly


class DA3Loss(torch.nn.Module):
    def __init__(self, cfg: LossConfig | None = None):
        super().__init__()
        self.cfg = cfg or LossConfig()

    def forward(self, pred: dict, gt: dict) -> tuple[torch.Tensor, dict]:
        c = self.cfg
        valid = gt["valid"]
        D_hat, D = pred["depth"].float(), gt["depth"]
        terms = {}
        terms["depth"] = conf_l1(D_hat, D, pred["depth_conf"].float(), valid, c.lambda_conf)
        terms["grad"] = grad_l1(D_hat, D, valid)

        ray_hat, ray_gt = pred["ray"].float(), gt["ray"]
        terms["ray"] = (ray_hat - ray_gt).abs().sum(-1).mean()  # L_M: ℓ1 over all ray pixels

        rays = ray_hat if c.point_rays == "pred" else ray_gt
        P_hat = points_from_depth_ray(D_hat, rays)
        terms["point"] = _masked_mean((P_hat - gt["points"]).abs().sum(-1), valid)

        if "pose_enc" in pred:
            terms["cam"] = (pred["pose_enc"].float() - gt["pose_enc"]).abs().sum(-1).mean()

        total = (
            c.w_depth * terms["depth"]
            + c.w_ray * terms["ray"]
            + c.w_point * terms["point"]
            + c.alpha_grad * terms["grad"]
            + c.beta_cam * terms.get("cam", torch.zeros((), device=D.device))
        )
        terms["total"] = total
        return total, {k: v.detach() for k, v in terms.items()}


# ----------------------------------------------------------------------------- eval helpers
@torch.no_grad()
def ray_errors(pred_ray, gt_ray):
    """Mean ℓ1 of direction and origin channels separately (diagnostics)."""
    return {
        "ray_dir_l1": (pred_ray[..., DIR] - gt_ray[..., DIR]).abs().mean().item(),
        "ray_orig_l1": (pred_ray[..., ORIG] - gt_ray[..., ORIG]).abs().mean().item(),
    }
