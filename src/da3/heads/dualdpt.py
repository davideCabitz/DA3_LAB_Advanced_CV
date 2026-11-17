"""Dual-DPT head (Sec. 3.2, Fig. 3).

Shared reassemble modules (per-level 1x1 projection, resize to a 4-level pyramid x4 / x2 / x1 / /2
of the patch grid, 3x3 adapters), then two separate fusion stacks and output layers:

* depth branch -> H x W -> depth (exp) + confidence (1 + exp)
* ray branch   -> H/2 x W/2 -> 6-channel ray map [origin (3), direction (3)] (linear)

The returned ray map always uses the paper's channel order [origin, direction].

`ray_head="released"` reproduces the released checkpoints instead: the ray branch stays at the
fusion resolution (8x the patch grid), has 5 plain convs before a LayerNorm output layer, predicts
an extra ray confidence and emits [direction, origin] (reordered here). `token_norm` / `pos_embed`
(LayerNorm on input tokens, UV sin-cos embeddings) are also released-only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _interp(x: torch.Tensor, size=None, scale_factor=None) -> torch.Tensor:
    return F.interpolate(x, size=size, scale_factor=scale_factor, mode="bilinear", align_corners=True)


def uv_embedding(x: torch.Tensor, img_w: int, img_h: int, omega_0: float = 100.0) -> torch.Tensor:
    """Released-only: sin/cos embedding of a centered, diagonal-normalized UV grid -> (C, h, w)."""
    C, h, w = x.shape[1], x.shape[2], x.shape[3]
    aspect = img_w / img_h
    diag = (aspect**2 + 1.0) ** 0.5
    span_x, span_y = aspect / diag, 1.0 / diag
    xs = torch.linspace(-span_x * (w - 1) / w, span_x * (w - 1) / w, w, dtype=x.dtype, device=x.device)
    ys = torch.linspace(-span_y * (h - 1) / h, span_y * (h - 1) / h, h, dtype=x.dtype, device=x.device)
    uu, vv = torch.meshgrid(xs, ys, indexing="xy")

    def sincos(pos, dim):
        omega = torch.arange(dim // 2, dtype=torch.float32, device=x.device) / (dim / 2.0)
        out = pos.reshape(-1, 1).float() * (1.0 / omega_0**omega)
        return torch.cat([out.sin(), out.cos()], dim=1)

    emb = torch.cat([sincos(uu, C // 2), sincos(vv, C // 2)], dim=-1)
    return emb.reshape(h, w, C).permute(2, 0, 1)


class ResidualConvUnit(nn.Module):
    def __init__(self, features: int):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, 1, 1)
        self.conv2 = nn.Conv2d(features, features, 3, 1, 1)

    def forward(self, x):
        return x + self.conv2(F.relu(self.conv1(F.relu(x))))


class FusionBlock(nn.Module):
    """DPT fusion: (top + RCU(lateral)) -> RCU -> bilinear upsample -> 1x1 conv."""

    def __init__(self, features: int, has_residual: bool = True):
        super().__init__()
        self.res1 = ResidualConvUnit(features) if has_residual else None
        self.res2 = ResidualConvUnit(features)
        self.out_conv = nn.Conv2d(features, features, 1)

    def forward(self, top, lateral=None, size=None):
        y = top
        if self.res1 is not None and lateral is not None:
            y = y + self.res1(lateral)
        y = self.res2(y)
        y = _interp(y, size=size) if size is not None else _interp(y, scale_factor=2)
        return self.out_conv(y)


class FusionStack(nn.Module):
    """Four fusion blocks from the coarsest (level 4) to the finest (level 1) pyramid level."""

    def __init__(self, features: int):
        super().__init__()
        self.blocks = nn.ModuleList([FusionBlock(features, has_residual=False)] + [FusionBlock(features) for _ in range(3)])

    def forward(self, l1, l2, l3, l4):
        out = self.blocks[0](l4, size=l3.shape[-2:])
        out = self.blocks[1](out, l3, size=l2.shape[-2:])
        out = self.blocks[2](out, l2, size=l1.shape[-2:])
        return self.blocks[3](out, l1)


class ChannelLayerNorm(nn.LayerNorm):
    """LayerNorm over the channel dim of an NCHW tensor."""

    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class DualDPT(nn.Module):
    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        features: int = 64,
        out_channels: tuple[int, int, int, int] = (48, 96, 192, 384),
        ray_head: str = "paper",
        token_norm: bool = False,
        pos_embed: bool = False,
        head_hidden: int = 32,
        chunk_size: int | None = 8,
    ):
        super().__init__()
        assert ray_head in ("paper", "released")
        self.patch_size, self.ray_mode, self.pos_embed, self.chunk_size = patch_size, ray_head, pos_embed, chunk_size
        oc, f, f2 = out_channels, features, features // 2
        # ---- shared reassemble modules
        self.norm = nn.LayerNorm(dim_in) if token_norm else nn.Identity()
        self.projects = nn.ModuleList([nn.Conv2d(dim_in, c, 1) for c in oc])
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(oc[0], oc[0], 4, stride=4),
            nn.ConvTranspose2d(oc[1], oc[1], 2, stride=2),
            nn.Identity(),
            nn.Conv2d(oc[3], oc[3], 3, stride=2, padding=1),
        ])
        self.adapters = nn.ModuleList([nn.Conv2d(c, f, 3, 1, 1, bias=False) for c in oc])
        # ---- depth branch: fusion + output layer -> [depth, confidence]
        self.depth_fusion = FusionStack(f)
        self.depth_neck = nn.Conv2d(f, f2, 3, 1, 1)
        self.depth_out = nn.Sequential(nn.Conv2d(f2, head_hidden, 3, 1, 1), nn.ReLU(inplace=True),
                                       nn.Conv2d(head_hidden, 2, 1))
        # ---- ray branch: fusion + output layer -> 6-ch ray map
        self.ray_fusion = FusionStack(f)
        if ray_head == "paper":
            self.ray_neck = nn.Conv2d(f, f2, 3, 1, 1)
            self.ray_out = nn.Sequential(nn.Conv2d(f2, head_hidden, 3, 1, 1), nn.ReLU(inplace=True),
                                         nn.Conv2d(head_hidden, 6, 1))
        else:
            self.ray_neck = nn.Sequential(  # 5 plain convs, as released
                nn.Conv2d(f, f2, 3, 1, 1), nn.Conv2d(f2, f, 3, 1, 1), nn.Conv2d(f, f2, 3, 1, 1),
                nn.Conv2d(f2, f, 3, 1, 1), nn.Conv2d(f, f2, 3, 1, 1),
            )
            self.ray_out = nn.Sequential(nn.Conv2d(f2, head_hidden, 3, 1, 1), ChannelLayerNorm(head_hidden),
                                         nn.ReLU(inplace=True), nn.Conv2d(head_hidden, 7, 1))

    def ray_hw(self, H: int, W: int) -> tuple[int, int]:
        if self.ray_mode == "paper":
            return H // 2, W // 2
        return 8 * (H // self.patch_size), 8 * (W // self.patch_size)

    def _add_pe(self, x, W, H):
        return x + 0.1 * uv_embedding(x, W, H)[None].to(x.dtype) if self.pos_embed else x

    def _forward_views(self, feats: list[torch.Tensor], H: int, W: int) -> dict:
        n = feats[0].shape[0]
        ph, pw = H // self.patch_size, W // self.patch_size
        pyramid = []
        for i, tok in enumerate(feats):
            x = self.norm(tok).permute(0, 2, 1).reshape(n, -1, ph, pw)
            x = self._add_pe(self.projects[i](x), W, H)
            pyramid.append(self.adapters[i](self.resize_layers[i](x)))

        d = self.depth_neck(self.depth_fusion(*pyramid))
        d = self._add_pe(_interp(d, size=(ph * self.patch_size, pw * self.patch_size)), W, H)
        d = self.depth_out(d)
        out = {"depth": torch.exp(d[:, 0]), "depth_conf": torch.exp(d[:, 1]) + 1.0}

        r = self.ray_neck(self.ray_fusion(*pyramid))
        if self.ray_mode == "paper":
            r = self.ray_out(_interp(r, size=self.ray_hw(H, W))).permute(0, 2, 3, 1)
            out["ray"] = r
        else:
            r = self.ray_out(self._add_pe(r, W, H)).permute(0, 2, 3, 1)
            out["ray"] = torch.cat([r[..., 3:6], r[..., 0:3]], dim=-1)  # [dir, orig] -> [orig, dir]
            out["ray_conf"] = torch.exp(r[..., 6]) + 1.0
        return out

    def forward(self, feats: list[torch.Tensor], H: int, W: int) -> dict:
        """feats: 4 x (B, S, h*w, C). Returns depth/depth_conf (B,S,H,W) and ray (B,S,h',w',6)
        [+ ray_conf (B,S,h',w') for the released head]."""
        B, S = feats[0].shape[:2]
        flat = [f.flatten(0, 1) for f in feats]
        n, cs = B * S, self.chunk_size or B * S
        chunks = [self._forward_views([f[i : i + cs] for f in flat], H, W) for i in range(0, n, cs)]
        return {k: torch.cat([c[k] for c in chunks]).reshape(B, S, *chunks[0][k].shape[1:]) for k in chunks[0]}
