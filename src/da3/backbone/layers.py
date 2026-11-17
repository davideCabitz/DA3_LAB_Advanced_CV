"""Transformer building blocks for the DA3 backbone.

Parameter names (qkv, proj, norm1, ls1, mlp.fc1, ...) follow DINOv2 so that DINOv2 pretrained
weights load without renaming. QK-norm and 2D RoPE are off by default and enabled per block
(DA3 uses them from block `alt_start` on, see docs/PLAN.md).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding2D(nn.Module):
    """2D rotary position embedding.

    The head dimension is split in two halves: the first half is rotated by the y position, the
    second by the x position. Inside each half the standard "rotate half" form is used with
    inverse frequencies base**(-2i/d).
    """

    def __init__(self, base: float = 100.0):
        super().__init__()
        self.base = base

    def _cos_sin(self, dim: int, max_pos: int, device, dtype):
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        angles = torch.arange(max_pos, device=device, dtype=inv_freq.dtype)[:, None] * inv_freq
        angles = angles.to(dtype)
        angles = torch.cat([angles, angles], dim=-1)
        return angles.cos().to(dtype), angles.sin().to(dtype)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """x: (B, heads, N, head_dim); pos: (B, N, 2) integer (y, x) positions."""
        half = x.shape[-1] // 2
        cos, sin = self._cos_sin(half, int(pos.max()) + 1, x.device, x.dtype)
        out = []
        for axis, feats in enumerate(x.chunk(2, dim=-1)):
            c = F.embedding(pos[..., axis], cos)[:, None]
            s = F.embedding(pos[..., axis], sin)[:, None]
            out.append(feats * c + self._rotate_half(feats) * s)
        return torch.cat(out, dim=-1)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        rope: RotaryEmbedding2D | None = None,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)
        self.rope = rope

    def forward(self, x: torch.Tensor, pos: torch.Tensor | None = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None and pos is not None:
            q, k = self.rope(q, pos), self.rope(k, pos)
        if q.dtype != v.dtype:  # LayerNorm under autocast returns fp32
            q, k = q.to(v.dtype), k.to(v.dtype)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int | None = None):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features or in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class Block(nn.Module):
    """Pre-norm transformer block with LayerScale (DINOv2 style)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        init_values: float | None = 1.0,
        qk_norm: bool = False,
        rope: RotaryEmbedding2D | None = None,
        ln_eps: float = 1e-6,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=ln_eps)
        self.attn = Attention(dim, num_heads, qk_norm=qk_norm, rope=rope)
        self.ls1 = LayerScale(dim, init_values) if init_values else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=ln_eps)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim, init_values) if init_values else nn.Identity()

    def forward(self, x: torch.Tensor, pos: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x), pos=pos))
        return x + self.ls2(self.mlp(self.norm2(x)))


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int = 14, in_chans: int = 3, embed_dim: int = 384):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[-2:]
        assert H % self.patch_size == 0 and W % self.patch_size == 0, (H, W)
        return self.proj(x).flatten(2).transpose(1, 2)
