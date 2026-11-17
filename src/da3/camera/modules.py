"""Camera encoder E_c (known poses -> camera tokens) and camera head D_C (tokens -> pose)."""

from __future__ import annotations

import torch
import torch.nn as nn

from da3.backbone.layers import Attention, LayerScale, Mlp
from .pose_encoding import affine_inverse, encode_pose


class _TrunkBlock(nn.Module):
    """Transformer block over the S camera tokens of a scene (LayerNorm eps 1e-5, LayerScale)."""

    def __init__(self, dim: int, num_heads: int, init_values: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.ls1 = LayerScale(dim, init_values)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, dim * 4)
        self.ls2 = LayerScale(dim, init_values)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class CameraEncoder(nn.Module):
    """E_c(f, q, t): the paper's lightweight MLP on the 9-D camera encoding (trunk_depth=0).

    trunk_depth > 0 reproduces the released encoder (MLP -> LN -> transformer blocks across views
    -> LN) and is used only to load the released checkpoints.
    """

    def __init__(self, dim: int, trunk_depth: int = 0, num_heads: int = 16, init_values: float = 0.01):
        super().__init__()
        self.pose_branch = Mlp(9, dim // 2, dim)
        if trunk_depth > 0:
            self.token_norm = nn.LayerNorm(dim)
            self.trunk = nn.Sequential(*[_TrunkBlock(dim, num_heads, init_values) for _ in range(trunk_depth)])
            self.trunk_norm = nn.LayerNorm(dim)
        else:
            self.trunk = None

    def forward(self, w2c: torch.Tensor, K: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
        """w2c (B, S, 4, 4) normalized extrinsics, K (B, S, 3, 3) -> (B, S, dim)."""
        x = self.pose_branch(encode_pose(affine_inverse(w2c), K, hw))
        if self.trunk is None:
            return x
        return self.trunk_norm(self.trunk(self.token_norm(x)))


class CameraHead(nn.Module):
    """D_C: camera-token features (B, S, C) -> pose encoding (B, S, 9) = [t, q_xyzw, fov_h, fov_w].

    The paper calls D_C a lightweight transformer on the camera tokens but gives no depth; its
    Tab. 8 parameter count (~1M for Small) matches the released 2-layer MLP, which we use.
    """

    def __init__(self, dim_in: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(dim_in, dim_in), nn.ReLU(), nn.Linear(dim_in, dim_in), nn.ReLU()
        )
        self.fc_t = nn.Linear(dim_in, 3)
        self.fc_qvec = nn.Linear(dim_in, 4)
        self.fc_fov = nn.Sequential(nn.Linear(dim_in, 2), nn.ReLU())

    def forward(self, tok: torch.Tensor) -> torch.Tensor:
        h = self.backbone(tok).float()
        return torch.cat([self.fc_t(h), self.fc_qvec(h), self.fc_fov(h)], dim=-1)
