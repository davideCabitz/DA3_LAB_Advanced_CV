"""DINOv2 ViT with DA3's input-adaptive cross-view attention (Sec. 3.2).

Tokens are kept as (B, S, N, C): B scenes, S views, N tokens per view.

* within-view attention: reshape to (B*S, N, C), so each view attends only to itself;
* cross-view attention: reshape to (B, S*N, C), so all tokens of all views attend jointly.

Paper layout (`paper-*` presets): the first L_s blocks are within-view, the following L_g blocks
alternate cross-view / within-view (starting with cross-view), L_s : L_g = 2 : 1. Each view is
prepended with a camera token (E_c(f, q, t) when poses are known, else a learnable token) that takes
part in all attention operations. Learnable tokens: one for the reference view (view 0) and one
shared by all other views (the paper says "shared"; see config.py for why view 0 needs its own). With S = 1 the cross-view reshape is the identity,
so the model reduces to a monocular one at no extra cost.

The released checkpoints (`da3-*` presets) use a different arrangement, kept only to load them:
alternation from block 4 with cross-view at odd indices, the cls slot overwritten by the camera
token at that block, separate reference / other-view tokens, QK-norm, 2D RoPE, reference-view
reordering and concatenated head features (docs/PLAN.md).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .layers import Block, PatchEmbed, RotaryEmbedding2D

VIT_PRESETS = {
    "vits": dict(embed_dim=384, depth=12, num_heads=6),
    "vitb": dict(embed_dim=768, depth=12, num_heads=12),
    "vitl": dict(embed_dim=1024, depth=24, num_heads=16),
}
REF_SELECTION_MIN_VIEWS = 3  # released THRESH_FOR_REF_SELECTION


# ----------------------------------------------------------------------------- token reordering
def to_within_view(x: torch.Tensor) -> torch.Tensor:
    """(B, S, N, C) -> (B*S, N, C)."""
    B, S, N, C = x.shape
    return x.reshape(B * S, N, C)


def to_cross_view(x: torch.Tensor) -> torch.Tensor:
    """(B, S, N, C) -> (B, S*N, C)."""
    B, S, N, C = x.shape
    return x.reshape(B, S * N, C)


def from_within_view(x: torch.Tensor, B: int, S: int) -> torch.Tensor:
    return x.reshape(B, S, *x.shape[1:])


def from_cross_view(x: torch.Tensor, B: int, S: int) -> torch.Tensor:
    return x.reshape(B, S, x.shape[1] // S, x.shape[2])


# ----------------------------------------------------------------------------- released-only helpers
def select_reference_view(x: torch.Tensor, strategy: str) -> torch.Tensor:
    """Released reference-view selection from cls tokens x[:, :, 0]. Returns (B,) indices."""
    B, S = x.shape[:2]
    if S <= 1 or strategy == "first":
        return torch.zeros(B, dtype=torch.long, device=x.device)
    if strategy == "middle":
        return torch.full((B,), S // 2, dtype=torch.long, device=x.device)
    if strategy != "saddle_balanced":
        raise ValueError(strategy)
    cls = x[:, :, 0].float()
    feat = cls / cls.norm(dim=-1, keepdim=True)
    sim = feat @ feat.transpose(1, 2) - torch.eye(S, device=x.device)
    scores = [sim.sum(-1) / (S - 1), cls.norm(dim=-1), feat.var(dim=-1)]

    def to_unit(m):
        lo, hi = m.min(1, keepdim=True).values, m.max(1, keepdim=True).values
        return (m - lo) / (hi - lo + 1e-8)

    return sum((to_unit(m) - 0.5).abs() for m in scores).argmin(dim=1)


def reference_permutation(ref: torch.Tensor, S: int) -> torch.Tensor:
    """(B, S) gather indices that move view `ref` to position 0, keeping the others in order."""
    pos = torch.arange(S, device=ref.device).expand(ref.shape[0], S)
    perm = torch.where((pos > 0) & (pos <= ref[:, None]), pos - 1, pos)
    perm[:, 0] = ref
    return perm


def gather_views(x: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    return x[torch.arange(x.shape[0], device=x.device)[:, None], perm]


# ----------------------------------------------------------------------------- backbone
class MultiViewViT(nn.Module):
    def __init__(
        self,
        name: str = "vits",
        patch_size: int = 14,
        out_layers: tuple[int, ...] = (5, 7, 9, 11),
        alt_start: int = 8,
        cross_first: bool = True,
        cam_token_mode: str = "prepend",
        num_cam_tokens: int = 1,
        cat_token: bool = False,
        qknorm_start: int = -1,
        rope_start: int = -1,
        rope_base: float = 100.0,
        ref_view_selection: bool = False,
        pretrain_img_size: int = 518,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        assert cam_token_mode in ("prepend", "replace_cls")
        p = VIT_PRESETS[name]
        D, depth = p["embed_dim"], p["depth"]
        self.embed_dim, self.depth, self.patch_size = D, depth, patch_size
        self.out_layers = tuple(out_layers)
        self.alt_start, self.cross_first = alt_start, cross_first
        self.cam_token_mode, self.cat_token = cam_token_mode, cat_token
        self.rope_start, self.ref_view_selection = rope_start, ref_view_selection
        self.grad_checkpointing = grad_checkpointing
        # tokens per view before the patches: [cam, cls] (prepend) or [cls -> cam] (replace_cls)
        self.n_special = 2 if cam_token_mode == "prepend" else 1

        self.patch_embed = PatchEmbed(patch_size, 3, D)
        n_patches = (pretrain_img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, D))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, D))
        self.camera_token = nn.Parameter(torch.randn(1, num_cam_tokens, D) * 0.02)
        self.rope = RotaryEmbedding2D(rope_base) if rope_start >= 0 else None
        self.blocks = nn.ModuleList(
            [
                Block(
                    D,
                    p["num_heads"],
                    qk_norm=0 <= qknorm_start <= i,
                    rope=self.rope if 0 <= rope_start <= i else None,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(D, eps=1e-6)

    @property
    def out_dim(self) -> int:
        return self.embed_dim * (2 if self.cat_token else 1)

    def is_cross_view(self, i: int) -> bool:
        if self.alt_start < 0 or i < self.alt_start:
            return False
        return (i - self.alt_start) % 2 == (0 if self.cross_first else 1)

    # -- embeddings
    def interpolate_pos_embed(self, gh: int, gw: int) -> torch.Tensor:
        pe = self.pos_embed
        M = int(math.sqrt(pe.shape[1] - 1))
        if gh == M and gw == M:
            return pe
        cls_pe, patch_pe = pe[:, :1].float(), pe[:, 1:].float()
        patch_pe = patch_pe.reshape(1, M, M, -1).permute(0, 3, 1, 2)
        # The +0.1 offset reproduces the DINOv2 interpolation exactly (scale-factor based).
        patch_pe = F.interpolate(
            patch_pe, scale_factor=((gh + 0.1) / M, (gw + 0.1) / M), mode="bicubic", antialias=False
        )
        assert patch_pe.shape[-2:] == (gh, gw)
        patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, gh * gw, -1)
        return torch.cat([cls_pe, patch_pe], dim=1).to(pe.dtype)

    def learnable_cam_tokens(self, B: int, S: int) -> torch.Tensor:
        """(B, S, D): one shared token (paper), or [reference, others] (released)."""
        if self.camera_token.shape[1] == 1:
            return self.camera_token.expand(B, S, -1)
        return torch.cat([self.camera_token[:, :1].expand(B, 1, -1),
                          self.camera_token[:, 1:2].expand(B, S - 1, -1)], dim=1)

    def rope_positions(self, B: int, S: int, gh: int, gw: int, device):
        """Released: per-view (y, x) grid for within-view blocks, constant positions for cross-view
        blocks. Special tokens have position 0, patches are shifted by +1."""
        yy, xx = torch.meshgrid(torch.arange(gh, device=device), torch.arange(gw, device=device), indexing="ij")
        grid = torch.stack([yy, xx], dim=-1).reshape(1, 1, gh * gw, 2) + 1
        special = torch.zeros(1, 1, self.n_special, 2, dtype=grid.dtype, device=device)
        local = torch.cat([special, grid], dim=2).expand(B, S, -1, -1)
        const = torch.cat([special, torch.ones_like(grid)], dim=2).expand(B, S, -1, -1)
        return local, const

    def _run_block(self, blk, x, pos):
        if self.grad_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(blk, x, pos, use_reentrant=False)
        return blk(x, pos)

    # -- forward
    def forward(self, images: torch.Tensor, cam_token: torch.Tensor | None = None, ref_strategy: str = "first"):
        """
        images: (B, S, 3, H, W), ImageNet-normalized. cam_token: (B, S, D) from E_c, or None.
        Returns (feats, cam_feats, ref_index):
          feats: list of (B, S, h*w, out_dim) patch features at `out_layers` (final-normed)
          cam_feats: list of (B, S, out_dim) camera-token features at `out_layers`
          ref_index: (B,) reference view if released reordering happened, else None
        """
        B, S, _, H, W = images.shape
        gh, gw = H // self.patch_size, W // self.patch_size
        x = self.patch_embed(images.flatten(0, 1))
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = (x + self.interpolate_pos_embed(gh, gw)).reshape(B, S, gh * gw + 1, -1)
        posed = cam_token is not None
        if cam_token is None:
            cam_token = self.learnable_cam_tokens(B, S)
        if self.cam_token_mode == "prepend":
            x = torch.cat([cam_token[:, :, None].to(x.dtype), x], dim=2)

        local_pos = const_pos = None
        if self.rope is not None:
            local_pos, const_pos = self.rope_positions(B, S, gh, gw, x.device)

        perm = ref = None
        local_x = x
        feats, cam_feats = [], []
        for i, blk in enumerate(self.blocks):
            if (self.ref_view_selection and i == self.alt_start - 1 and S >= REF_SELECTION_MIN_VIEWS
                    and not posed):
                ref = select_reference_view(x, ref_strategy)
                perm = reference_permutation(ref, S)
                x, local_x = gather_views(x, perm), gather_views(local_x, perm)
            if self.cam_token_mode == "replace_cls" and i == self.alt_start:
                x = torch.cat([cam_token[:, :, None].to(x.dtype), x[:, :, 1:]], dim=2)

            use_rope = self.rope is not None and i >= self.rope_start
            if self.is_cross_view(i):
                pos = to_cross_view(const_pos) if use_rope else None
                x = from_cross_view(self._run_block(blk, to_cross_view(x), pos), B, S)
            else:
                pos = to_within_view(local_pos) if use_rope else None
                x = from_within_view(self._run_block(blk, to_within_view(x), pos), B, S)
                local_x = x

            if i in self.out_layers:
                if self.cat_token:  # released: [within-view features, normed current features]
                    out = torch.cat([local_x, x], dim=-1)
                    if perm is not None:
                        out = gather_views(out, torch.argsort(perm, dim=1))
                    cam_feats.append(out[:, :, 0])
                    out = torch.cat([out[..., : self.embed_dim], self.norm(out[..., self.embed_dim:])], -1)
                else:
                    out = self.norm(x)
                    if perm is not None:
                        out = gather_views(out, torch.argsort(perm, dim=1))
                    cam_feats.append(out[:, :, 0])
                feats.append(out[:, :, self.n_special:])
        return feats, cam_feats, ref
