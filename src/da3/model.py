"""Depth Anything 3 network: single ViT + Dual-DPT + camera encoder E_c / camera head D_C (Fig. 2)."""

from __future__ import annotations

import torch
import torch.nn as nn

from da3.backbone import MultiViewViT
from da3.camera import CameraEncoder, CameraHead, affine_inverse, decode_pose, normalize_extrinsics
from da3.config import ModelConfig, load_model_config
from da3.geometry import pose_from_ray_map
from da3.heads import DualDPT

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DA3(nn.Module):
    def __init__(self, cfg: ModelConfig | str | dict = "paper-small"):
        super().__init__()
        cfg = load_model_config(cfg)
        self.cfg = cfg
        self.backbone = MultiViewViT(
            name=cfg.name,
            patch_size=cfg.patch_size,
            out_layers=cfg.out_layers,
            alt_start=cfg.alt_start,
            cross_first=cfg.cross_first,
            cam_token_mode=cfg.cam_token_mode,
            num_cam_tokens=cfg.num_cam_tokens,
            cat_token=cfg.cat_token,
            qknorm_start=cfg.qknorm_start,
            rope_start=cfg.rope_start,
            rope_base=cfg.rope_base,
            ref_view_selection=cfg.ref_view_selection,
            grad_checkpointing=cfg.grad_checkpointing,
        )
        D, Dout = self.backbone.embed_dim, self.backbone.out_dim
        self.head = DualDPT(
            Dout,
            patch_size=cfg.patch_size,
            features=cfg.head_features,
            out_channels=cfg.head_out_channels,
            ray_head=cfg.ray_head,
            token_norm=cfg.head_token_norm,
            pos_embed=cfg.head_pos_embed,
            chunk_size=cfg.head_chunk_size,
        )
        self.cam_enc = CameraEncoder(D, trunk_depth=cfg.cam_enc_depth) if cfg.use_cam_enc else None
        self.cam_head = CameraHead(Dout) if cfg.use_cam_head else None

    def ray_hw(self, H: int, W: int) -> tuple[int, int]:
        return self.head.ray_hw(H, W)

    def forward(
        self,
        images: torch.Tensor,
        w2c: torch.Tensor | None = None,
        K: torch.Tensor | None = None,
        ref_strategy: str = "first",
        normalize_poses: bool = True,
    ) -> dict:
        """
        images: (B, S, 3, H, W) ImageNet-normalized; S = 1 is monocular.
        w2c, K: optional known poses (B, S, 4|3, 4) and intrinsics (B, S, 3, 3) -> camera tokens via E_c.
        Returns depth, depth_conf (B,S,H,W); ray (B,S,h,w,6) = [origin, direction];
        pose_enc (B,S,9) from D_C; ref_index (released reordering only, else None).
        """
        H, W = images.shape[-2:]
        cam_token = None
        if w2c is not None:
            if self.cam_enc is None:
                raise ValueError("model has no camera encoder")
            with torch.autocast(images.device.type, enabled=False):
                w2c_n = normalize_extrinsics(w2c.float()) if normalize_poses else w2c.float()
                cam_token = self.cam_enc(w2c_n, K.float(), (H, W))

        feats, cam_feats, ref = self.backbone(images, cam_token=cam_token, ref_strategy=ref_strategy)
        with torch.autocast(images.device.type, enabled=False):
            out = self.head([f.float() for f in feats], H, W)
            if self.cam_head is not None:
                out["pose_enc"] = self.cam_head(cam_feats[-1].float())
        out["ref_index"] = ref
        return out

    @staticmethod
    def cameras_from_head(pose_enc: torch.Tensor, hw) -> tuple[torch.Tensor, torch.Tensor]:
        """D_C output -> (w2c (..., 3, 4), K)."""
        c2w, K = decode_pose(pose_enc, hw)
        return affine_inverse(c2w), K

    @staticmethod
    def cameras_from_rays(out: dict, hw):
        """Ray map -> (w2c, K) via Eq. 1 + Eq. 2 (DLT + RQ)."""
        return pose_from_ray_map(out["ray"], hw)


def normalize_images(img01: torch.Tensor) -> torch.Tensor:
    """(..., 3, H, W) in [0, 1] -> ImageNet-normalized."""
    mean = img01.new_tensor(IMAGENET_MEAN)[:, None, None]
    std = img01.new_tensor(IMAGENET_STD)[:, None, None]
    return (img01 - mean) / std
