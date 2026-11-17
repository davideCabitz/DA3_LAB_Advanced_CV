import math

import torch

from da3.camera import quat_to_mat


def random_camera(B=2, H=240, W=320, seed=0, dtype=torch.float64):
    """Random K (off-center principal point, fx != fy), rotation and translation."""
    g = torch.Generator().manual_seed(seed)
    fx = 200 + 200 * torch.rand(B, generator=g, dtype=dtype)
    fy = fx * (0.9 + 0.2 * torch.rand(B, generator=g, dtype=dtype))
    cx = W / 2 + 20 * (torch.rand(B, generator=g, dtype=dtype) - 0.5)
    cy = H / 2 + 20 * (torch.rand(B, generator=g, dtype=dtype) - 0.5)
    K = torch.zeros(B, 3, 3, dtype=dtype)
    K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = fx, fy, cx, cy, 1
    q = torch.randn(B, 4, generator=g, dtype=dtype)
    R = quat_to_mat(q / q.norm(dim=-1, keepdim=True))
    t = torch.randn(B, 3, generator=g, dtype=dtype) * 2
    c2w = torch.eye(4, dtype=dtype).repeat(B, 1, 1)
    c2w[:, :3, :3], c2w[:, :3, 3] = R, t
    return K, c2w


def rot_angle_deg(R1, R2):
    cos = ((R1.transpose(-1, -2) @ R2).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    return torch.rad2deg(torch.arccos(cos.clamp(-1, 1)))


def tiny_model(preset="paper-small", **overrides):
    """A random-init model with a ViT-S preset (fast on CPU)."""
    from da3 import DA3
    from da3.config import PRESETS, ModelConfig

    cfg = ModelConfig.from_dict({**PRESETS[preset].to_dict(), **overrides})
    torch.manual_seed(0)
    return DA3(cfg).eval()
